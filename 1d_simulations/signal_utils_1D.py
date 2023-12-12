from functools import partial
from cleanplots import *
import jax.numpy as np
import numpy as onp

from tqdm import tqdm
from jax import value_and_grad, jit, vmap
import jax
# from scipy.signal import resample as scipy_resample
from jax.scipy.special import digamma
import optax

import imageio
from mpl_toolkits.mplot3d import Axes3D

NUM_NYQUIST_SAMPLES = 32
UPSAMPLED_SIGNAL_LENGTH = 8 * NUM_NYQUIST_SAMPLES
OBJECT_LENGTH = 8 * NUM_NYQUIST_SAMPLES


def optimize_PSF_and_estimate_mi(objects_fn, noise_sigma, erasure_mask=None, initial_kernel=None,
                                 learning_rate=1e-2, learning_rate_decay=0.999, verbose=True,
                                 loss_improvement_patience=2000, max_epochs=5000, num_nyquist_samples=NUM_NYQUIST_SAMPLES
                                    , nyquist_sample_output=True
                                 ):
  objects = objects_fn()
  if initial_kernel is None:
    initial_kernel = bandlimited_nonnegative_signal(nyquist_samples=generate_random_bandlimited_signal(num_nyquist_samples=num_nyquist_samples),
                                                        num_nyquist_samples=num_nyquist_samples)
  initial_params = np.concatenate(params_from_signal(initial_kernel, num_nyquist_samples=num_nyquist_samples))

  if erasure_mask is None:
    # no erasure mask, so we just use all the pixels
    erasure_mask = np.ones(num_nyquist_samples)
    erasure_mask = np.array(erasure_mask, dtype=bool)

  loss_fn = make_convolutional_forward_model_with_mi_loss_and_erasure(
      objects, erasure_mask, noise_sigma=noise_sigma, num_nyquist_samples=num_nyquist_samples, nyquist_sample_output=nyquist_sample_output)
  optimized_params = run_optimzation(loss_fn, lambda x : signal_prox_fn(x, num_nyquist_samples=num_nyquist_samples),
                          initial_params, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, verbose=verbose,
                          loss_improvement_patience=loss_improvement_patience, max_epochs=max_epochs,
                          key=jax.random.PRNGKey(onp.random.randint(100000)))
  test_objects = objects_fn()                        
  optimized_mi = -make_convolutional_forward_model_with_mi_loss_and_erasure(test_objects, erasure_mask, noise_sigma=noise_sigma,
                                                                            num_nyquist_samples=num_nyquist_samples, nyquist_sample_output=nyquist_sample_output
                                                                            )(optimized_params, jax.random.PRNGKey(0))
  initial_mi = -make_convolutional_forward_model_with_mi_loss_and_erasure(test_objects, erasure_mask, noise_sigma=noise_sigma,
                                                                          num_nyquist_samples=num_nyquist_samples, nyquist_sample_output=nyquist_sample_output
                                                                          )(initial_params, jax.random.PRNGKey(0))
  return initial_kernel, initial_params, optimized_params, objects, initial_mi, optimized_mi


@partial(jit, static_argnums=(1, 2))
def compute_gaussian_differential_entropy(output_signals, ev_threshold=1e-10, average_values=True):
    mean_subtracted = output_signals.T - np.mean(output_signals.T, axis=1, keepdims=True)
    cov_mat = np.cov(mean_subtracted)
    eig_vals = np.linalg.eigvalsh(cov_mat)
    if ev_threshold is not None:
        eig_vals = np.clip(eig_vals, ev_threshold, None)
    log_evs = np.log(eig_vals)
    log_ev_term = np.log(2 * np.pi * np.e) + log_evs
    d = eig_vals.size
    if average_values:
        return 0.5 * np.sum(log_ev_term) / d  # Normalized by the number of dimensions
    else: 
        return 0.5 * log_ev_term 

@jit
def compute_mutual_information_per_pixel(noisy_output_signals, noise_sigma):
    """
    compute mutual information per pixel in bits assuming noise was generated from an additive gaussian
    """
    entropy = compute_gaussian_differential_entropy(noisy_output_signals, average_values=False)
    entropy -= 0.5 * np.log(2 * np.pi * np.e * noise_sigma**2)
    # convert to bits
    entropy /= np.log(2)
    return np.mean(entropy)

def filter_l1_ball_samples_to_positive(l1_ball_samples, return_everything=False):
    positive_targets = []
    negative_targets = []
    positive_indices = []
    negative_indices = []
    num_nyquist_samples = l1_ball_samples.shape[-1]
    for i, t in tqdm(enumerate(l1_ball_samples), total=l1_ball_samples.shape[0]):
        upsampled = upsample_signal(t, num_nyquist_samples=num_nyquist_samples)
        if upsampled.min() < 0:
            negative_targets.append(t)
            negative_indices.append(i)
        else:
            positive_targets.append(t)
            positive_indices.append(i)
    if return_everything:
        return np.array(positive_targets), np.array(negative_targets), np.array(positive_indices), np.array(negative_indices)
    else:
        return np.array(positive_targets)

def get_sampling_interval(num_samples):
    return 1 / num_samples

def sample_amplitude_object(type, seed=None, object_size=OBJECT_LENGTH, num_deltas=1, sin_freq_range=[0, 1],
                            gaussian_mixture_position=False, num_mixture_components=3, gaussian_mixture_seed=12345):
    """
    Generate a random object of a given type. Objects sum to one 
    and have values between 0 and 1.
    """

    # create a gaussian mixture model over the object
    if gaussian_mixture_position:
        onp.random.seed(gaussian_mixture_seed)
        means = onp.random.rand(num_mixture_components) * object_size
        stds = onp.random.rand(num_mixture_components) * object_size / 6
        weights = onp.random.rand(num_mixture_components)
        weights /= weights.sum()
        onp.random.seed(None)

    def get_random_position():
        if not gaussian_mixture_position:
            return onp.random.randint(object_size)
        else:
            component = onp.random.choice(num_mixture_components, p=weights)
            return int(onp.random.normal(loc=means[component], scale=stds[component])) % object_size

    if seed is not None:
        onp.random.seed(seed)
    if type == 'delta':
        delta_function = onp.zeros(object_size)
        for i in range(num_deltas):
            delta_function[get_random_position()] += 1 / num_deltas
        return delta_function
    elif type == 'random_amplitude_delta':
        delta_function = onp.zeros(object_size)
        for i in range(num_deltas):
            delta_function[get_random_position()] += onp.random.rand() / num_deltas
        return delta_function
    elif type == 'pink_noise':
        magnitude = onp.concatenate([onp.array([1]), 1 / np.sqrt(onp.fft.rfftfreq(OBJECT_LENGTH, 1/OBJECT_LENGTH)[1:])])
        random_phase = onp.concatenate([onp.array([1]), np.exp(1j * 2*np.pi*onp.random.rand(magnitude.shape[0] - 1))])
        object = onp.fft.irfft(magnitude * random_phase) 
        if np.min(object) < 0:
            object -= np.min(object)
        return object / onp.sum(object)
    elif type == 'white_noise':
        obj = onp.random.rand(object_size)
        if np.min(obj) < 0:
            obj -= np.min(obj)
        return obj / onp.sum(obj)
    elif type == 'masked_white_noise':
        obj = onp.random.rand(object_size)
        if onp.min(obj) < 0:
            obj -= onp.min(obj)
        obj = obj * (onp.random.rand(object_size) > 0.8)
        return obj / onp.sum(obj)
    elif type == 'sinusoid':
        obj = onp.sin(np.linspace(0, 2*np.pi, object_size) * onp.random.uniform(*sin_freq_range) + onp.random.rand() * 100000) + 1
        return obj / onp.sum(obj)

def optimize_towards_target_signals(target_signals, object, sampling_indices=None,
                                     initial_kernel=None, learning_rate=3e-3,
                                    learning_rate_decay=0.999, tolerance=1e-5, 
                                    transition_begin=800, verbose=False):                                    
    """
    Optimize a kernel to match a target signal
    """

    num_nyquist_samples = target_signals[0].shape[-1]
    if initial_kernel is None:
        initial_kernel = bandlimited_nonnegative_signal(nyquist_samples=generate_random_bandlimited_signal(num_nyquist_samples=num_nyquist_samples), 
                                                        num_nyquist_samples=num_nyquist_samples)

    optimized_kernels = []
    iter = tqdm(target_signals) if verbose else target_signals 
    for target_signal in iter:
        loss_fn = make_convolutional_forward_model_and_target_signal_MSE_loss_fn(object, target_signal, 
                                                               sampling_indices=sampling_indices)
        optimized_params = run_optimzation(loss_fn, lambda x : signal_prox_fn(x, num_nyquist_samples=num_nyquist_samples, ),
                                params_from_signal(initial_kernel, num_nyquist_samples=num_nyquist_samples),
                                transition_begin=transition_begin, learning_rate_decay=learning_rate_decay, tolerance=tolerance,
                                loss_improvement_patience=1000, max_epochs=10000,
                                  learning_rate=learning_rate, verbose=verbose)
        optimized_kernel = signal_from_params(optimized_params, num_nyquist_samples=num_nyquist_samples, )

        optimized_kernels.append(optimized_kernel)
    optimized_kernels = np.array(optimized_kernels)


    conv_mats = [make_convolutional_encoder(kernel, num_nyquist_samples=num_nyquist_samples) for kernel in optimized_kernels]
    output_signals = np.stack([conv_mat @ object for conv_mat in conv_mats], axis=0)
    return optimized_kernels, output_signals

def add_gaussian_noise_numpy(signal, noise_sigma):
    return signal + onp.random.normal(0, noise_sigma, signal.shape)

@partial(jit, static_argnums=(3, 4, 5))
def conv_forward_model_with_erasure(parameters, objects, erasure_mask, align_center=False, nyquist_sample_output=True,
                                    num_nyquist_samples=NUM_NYQUIST_SAMPLES):
  kernel = signal_from_params(parameters, num_nyquist_samples=num_nyquist_samples)
  if align_center:
    kernel = np.roll(kernel, kernel.size // 2 - np.argmax(kernel))
  conv_mat = make_convolutional_encoder(kernel, sample=nyquist_sample_output, num_nyquist_samples=num_nyquist_samples)
  output_signals = (objects @ conv_mat.T)
  return output_signals * erasure_mask.reshape(1, -1)

@partial(jit, static_argnums=(2, 3, 4))
def conv_forward_model(parameters, objects, align_center=False, nyquist_sample_output=False, num_nyquist_samples=NUM_NYQUIST_SAMPLES):
  kernel = signal_from_params(parameters, num_nyquist_samples=num_nyquist_samples)
  if align_center:
    kernel = np.roll(kernel, kernel.size // 2 - np.argmax(kernel))
  conv_mat = make_convolutional_encoder(kernel, num_nyquist_samples=num_nyquist_samples, sample=nyquist_sample_output)
  output_signals = (objects @ conv_mat.T)

  return output_signals


def make_convolutional_forward_model_with_mi_loss(objects, noise_sigma, nyquist_sample_output=True, 
                                                  num_nyquist_samples=NUM_NYQUIST_SAMPLES):
    
    @jit
    def convolve_and_loss(parameters, key):
        output_signals = conv_forward_model(parameters, objects,
                                                         nyquist_sample_output=nyquist_sample_output, 
                                                         num_nyquist_samples=num_nyquist_samples)

        if noise_sigma is None:
            raise Exception('Noise sigma must be specified')
        
        noisy_output_signals = output_signals + jax.random.normal(key, output_signals.shape) * noise_sigma
    
        return -compute_mutual_information_per_pixel(noisy_output_signals, noise_sigma=noise_sigma)
    
    return convolve_and_loss

def make_convolutional_forward_model_with_mi_loss_and_erasure(objects, erasure_mask, noise_sigma, 
                                                              nyquist_sample_output=True, num_nyquist_samples=NUM_NYQUIST_SAMPLES):
    
    if np.sum(erasure_mask) == 0:
        raise Exception('Erasure mask is empty')
    @jit
    def convolve_and_loss(parameters, key):
        output_signals = conv_forward_model_with_erasure(parameters, objects, erasure_mask, 
                                                         nyquist_sample_output=nyquist_sample_output, 
                                                         num_nyquist_samples=num_nyquist_samples)

        # don't include erased pixels in loss to avoid numerical errors
        output_signals = output_signals[:, erasure_mask]

        if noise_sigma is None:
            raise Exception('Noise sigma must be specified')
        
        noisy_output_signals = output_signals + jax.random.normal(key, output_signals.shape) * noise_sigma
    
        return -compute_mutual_information_per_pixel(noisy_output_signals, noise_sigma=noise_sigma)
    
    return convolve_and_loss

@partial(jit, static_argnums=(1, ))
def signal_from_params(params, num_nyquist_samples=NUM_NYQUIST_SAMPLES):
    """
    Generate signal with unit energy from real and imaginary parts of the spectrum
    """
    real, imag, amplitude_logit = params
    if real.shape[-1] != imag.shape[-1] + 1:
        raise Exception('Real must be one bigger than imaginary (DC)')
    # construct full signal from trainable parameters
    full_spectrum = np.concatenate([np.array([0]), imag]) *1j + real
    signal = np.fft.irfft(full_spectrum, num_nyquist_samples)

    # get energy normalized positive signal
    signal = bandlimited_nonnegative_signal(signal, num_nyquist_samples=num_nyquist_samples)
    # make amplitude always be between 0 and 1 using a sigmoid function
    # def sigmoid(x):
    #     return 1 / (1 + np.exp(-x))
    
    def tanh01(x):
        return (np.tanh(x) + 1) / 2
    
    def capped_relu(x):
        return np.clip(x, 0, 1)

    return signal * capped_relu(amplitude_logit)

@partial(jit, static_argnums=(1,))
def params_from_signal(signal, num_nyquist_samples=NUM_NYQUIST_SAMPLES):
    """
    Get nonzero real and imaginary parts of the spectrum from a signal
    """
    amplitude = np.sum(signal, keepdims=True)
    # avoid numerical errors
    amplitude = np.clip(amplitude, 0, 1)
    
    # def inverse_sigmoid(x):
    #     return np.clip(np.log(x / (1 - x)), -1e3, 1e3)

    def inverse_modified_tanh(y):
        return np.clip(np.arctanh(2 * y - 1), -1e3, 1e3)
    
    def inverse_capped_relu(x):
        return np.clip(x, 0, 1)

    amplitude_logit = inverse_capped_relu(amplitude)
    signal /= np.sum(signal)
    ft = np.fft.rfft(signal)
    real = ft[:num_nyquist_samples // 2 + 1].real
    imag = ft[1:num_nyquist_samples // 2 + 1].imag
    return (real, imag, amplitude_logit)

def random_unnormalized_signal(N=1, num_nyquist_samples=NUM_NYQUIST_SAMPLES, seed=None):
    """
    Generate a random, bandlimited signal by
    Sampling random nyquist samples between 0 and 1
    """
    if seed is not None:
        onp.random.seed(seed)
    return np.array(onp.random.rand(N, num_nyquist_samples))

@jit
def check_if_nonnegative(nyquist_samples):
    upsampled = upsample_signal(nyquist_samples)
    return np.all(upsampled >= 0, axis=-1)

@partial(jit, static_argnums=(1, 2, ))
def bandlimited_nonnegative_signal(nyquist_samples, 
                                   upsampled_signal_length=UPSAMPLED_SIGNAL_LENGTH, num_nyquist_samples=NUM_NYQUIST_SAMPLES):
    """
    Make sure that the signal is non-negative, has unit energy
    """
    assert num_nyquist_samples == nyquist_samples.shape[-1]
    nyquist_samples /= np.sum(nyquist_samples, axis=-1, keepdims=True)
    
    upsampled = upsample_signal(nyquist_samples, upsampled_signal_length=upsampled_signal_length,
                                num_nyquist_samples=num_nyquist_samples)

    # Make positive
    nyquist_samples = np.where(np.any(upsampled < 0), 
            nyquist_samples - np.min(upsampled, axis=-1, keepdims=True),
            nyquist_samples)
    # in case of numerical error
    nyquist_samples = np.where(np.any(nyquist_samples < 0), 
            nyquist_samples - np.min(nyquist_samples, axis=-1, keepdims=True),
            nyquist_samples)
    nyquist_samples /= np.sum(nyquist_samples, axis=-1, keepdims=True)

    return np.squeeze(nyquist_samples)

@partial(jit, static_argnums=(1, 2, 3))
def make_convolutional_encoder(kernel, object_length=OBJECT_LENGTH, sample=True, num_nyquist_samples=NUM_NYQUIST_SAMPLES):
    # upsample because we wan the size to be based on the input signal (i.e. the object)
    if object_length != num_nyquist_samples:
        kernel = upsample_signal(kernel, object_length, num_nyquist_samples=num_nyquist_samples)
    conv_mat = make_circulant_matrix(kernel)
    if sample:
        sampling_locations = np.linspace(0, 1, num_nyquist_samples, endpoint=False) + get_sampling_interval(num_nyquist_samples) / 2
        indices = (sampling_locations * kernel.size).astype(int)
        conv_mat = conv_mat[indices, :]
    return conv_mat

@jit
def make_circulant_matrix(kernel):
  """
  Generate a circulant matrix (for convolutions) from a vector
  """
  if len(kernel.shape) == 1:
    return make_circulant_matrix(kernel[:, None])
  n = kernel.shape[-2]
  m = kernel.shape[-1]
  if m == n:
    return kernel
  if m > n:
    return kernel[..., :n]
  r  = np.roll(kernel, m, axis=-2)
  return make_circulant_matrix(np.concatenate([kernel, r], axis=-1)) 

def make_intensity_coordinate_sampling_grid(sampling_indices, sample_n=40, num_nyquist_samples=NUM_NYQUIST_SAMPLES,
                                            randomize_non_sampling_indices=False):
    """
    Make a grid of the potential intensity coordinates (sum to 1, non-negative)
    sampling_indices: where the target signal is nonzero and will instead have values from the grid
    """
    if len(sampling_indices) == 2:
        y_grid, x_grid = np.meshgrid(np.linspace(0, 1, sample_n), np.linspace(0, 1, sample_n))
        yx = np.stack([y_grid.ravel(), x_grid.ravel()], axis=1)
        # mask out the ones with sum > 1
        yx = yx[np.sum(yx, axis=-1) <= 1]
        target_signal = onp.zeros((yx.shape[0], num_nyquist_samples))
        target_signal[:, sampling_indices[0]] = yx[:, 0]
        target_signal[:, sampling_indices[1]] = yx[:, 1]
        # make sure the signal sums to 1
    elif len(sampling_indices) == 3:
        y_grid, x_grid, z_grid = np.meshgrid(np.linspace(0, 1, sample_n), np.linspace(0, 1, sample_n), np.linspace(0, 1, sample_n))
        yxz = np.stack([y_grid.ravel(), x_grid.ravel(), z_grid.ravel()], axis=1)
        # mask out the ones with sum > 1
        yxz = yxz[np.sum(yxz, axis=-1) <= 1]
        target_signal = onp.zeros((yxz.shape[0], num_nyquist_samples))
        target_signal[:, sampling_indices[0]] = yxz[:, 0]
        target_signal[:, sampling_indices[1]] = yxz[:, 1]
        target_signal[:, sampling_indices[2]] = yxz[:, 2]
    else:
        raise Exception('Not implemented')
    


    if not randomize_non_sampling_indices:
        for signal in tqdm(target_signal):
            remainder = 1 - signal[np.array(sampling_indices)].sum()
            for index in range(signal.size):
                if index not in sampling_indices:
                    signal[index] = remainder / (signal.size - len(sampling_indices))
    else:
        remainder = 1 - target_signal[:, np.array(sampling_indices)].sum(axis=-1, keepdims=True)
        non_sampling_indices = [i for i in range(target_signal.shape[1]) if i not in sampling_indices]
        l1_ball_samples = np.load(".cache/{}_dimensional_random_L1_ball_samples.npy".format(target_signal.shape[-1]), allow_pickle=True)
        l1_ball_samples = np.array(l1_ball_samples)

        for i, signal in tqdm(enumerate(target_signal)):
            mask = l1_ball_samples[:, non_sampling_indices].sum(axis=-1) < remainder[i]
            # pick one from the mask
            valid_indices = np.arange(l1_ball_samples.shape[0])[mask]
            if valid_indices.size == 0:
                # make it uniform
                signal[non_sampling_indices] = remainder[i] / len(non_sampling_indices)
                print('Warning: no valid indices for this signal')
            else:
                index = onp.random.choice(valid_indices)
            signal[non_sampling_indices] = l1_ball_samples[index, non_sampling_indices]
            
    return np.array(target_signal)

def generate_random_bandlimited_signal(num_nyquist_samples=NUM_NYQUIST_SAMPLES, seed=None):
    return bandlimited_nonnegative_signal(random_unnormalized_signal(num_nyquist_samples=num_nyquist_samples, seed=seed), num_nyquist_samples=num_nyquist_samples)

def generate_concentrated_signal(sampling_indices, num_nyquist_samples=NUM_NYQUIST_SAMPLES):
    """
    Make a signal that tries to concentrate the energy at the given sampling indices,
    but is still non-negative, bandlimited, and has unit energy
    """
    nyquist_samples = onp.zeros(num_nyquist_samples)
    for i in sampling_indices:
        nyquist_samples[i] = 1
    signal = bandlimited_nonnegative_signal(nyquist_samples=np.array(nyquist_samples), num_nyquist_samples=num_nyquist_samples)
    return signal

def _resample_real_signal(x, upsampled_signal_length=UPSAMPLED_SIGNAL_LENGTH, num_nyquist_samples=NUM_NYQUIST_SAMPLES):
    if len(x.shape) == 1:
        return _resample_real_signal_1d(x, upsampled_signal_length=upsampled_signal_length, num_nyquist_samples=num_nyquist_samples)
    else:
        return vmap(lambda s: _resample_real_signal_1d(s, upsampled_signal_length=upsampled_signal_length, 
                                                      num_nyquist_samples=num_nyquist_samples))(x)


@partial(jit, static_argnums=(1, 2))
def _resample_real_signal_1d(x, upsampled_signal_length=UPSAMPLED_SIGNAL_LENGTH, num_nyquist_samples=NUM_NYQUIST_SAMPLES):
    """
    Copied from scipy.signal.resample because it doesnt have a jax implementation
    Modified to work with fixed sampling numbers in this file
    always for a 1D signal
    """

    # Forward transform
    # if real_input:
    X = np.fft.rfft(x)
    # else:  # Full complex FFT
    #     X = np.fft.fft(x, axis=axis)


    # Placeholder array for output spectrum
    # if real_input:
    newshape = upsampled_signal_length // 2 + 1
    # else:
    #     newshape = UPSAMPLED_SIGNAL_LENGTH

    # Copy positive frequency components (and Nyquist, if present)
    nyq = num_nyquist_samples // 2 + 1  # Slice index that includes Nyquist if present
    additional_zero_columns = newshape - nyq
    if num_nyquist_samples % 2 == 0:
         Y = np.concatenate([X[:num_nyquist_samples // 2],
                             0.5 * np.array([X[num_nyquist_samples // 2]]),
                             np.zeros(additional_zero_columns, X.dtype)])
         
    else:
        Y = np.concatenate([X[:additional_zero_columns], np.zeros(additional_zero_columns, X.dtype)])

    # if not real_input:
    #     # Copy negative frequency components
    #     if NUM_NYQUIST_SAMPLES > 2:  # (slice expression doesn't collapse to empty array)
    #         sl[axis] = slice(nyq - NUM_NYQUIST_SAMPLES, None)
    #         Y[tuple(sl)] = X[tuple(sl)]

    # Split/join Nyquist component(s) if present
    # So far we have set Y[+N/2]=X[+N/2]
    # if NUM_NYQUIST_SAMPLES % 2 == 0:
        # if not real_input:
        #     temp = Y[tuple(sl)]
        #     # set the component at -N/2 equal to the component at +N/2
        #     sl[axis] = slice(UPSAMPLED_SIGNAL_LENGTH-NUM_NYQUIST_SAMPLES//2, UPSAMPLED_SIGNAL_LENGTH-NUM_NYQUIST_SAMPLES//2 + 1)
        #     Y[tuple(sl)] = temp

    # Inverse transform
    # if real_input:
    y = np.fft.irfft(Y, upsampled_signal_length)
    # else:
    #     y = np.fft.ifft(Y)

    y *= (float(upsampled_signal_length) / float(num_nyquist_samples))

    # if t is None:
    return y
    # else:
    #     new_t = np.arange(0, UPSAMPLED_SIGNAL_LENGTH) * (t[1] - t[0]) * NUM_NYQUIST_SAMPLES / float(UPSAMPLED_SIGNAL_LENGTH) + t[0]
    #     return y, new_t


@partial(jit, static_argnums=(1, 2, 3))
def upsample_signal(nyquist_samples, upsampled_signal_length=UPSAMPLED_SIGNAL_LENGTH, 
                    num_nyquist_samples=NUM_NYQUIST_SAMPLES, return_domain=False):
    """
    Upsample to restore a nyquist sampled signal, 
    with the sampled points lying at the center of the upsampled points.
    """

    x = np.linspace(0, 1, num_nyquist_samples, endpoint=False)
    x += get_sampling_interval(num_nyquist_samples) / 2
    upsampled_signal = _resample_real_signal(nyquist_samples, upsampled_signal_length=upsampled_signal_length, num_nyquist_samples=num_nyquist_samples)
        # upsampled_signal = scipy_resample(nyquist_samples, upsampled_signal_length, axis=-1)
    
    x_upsampled = np.linspace(0, 1, upsampled_signal_length, endpoint=False) 
    integer_shift = int((get_sampling_interval(num_nyquist_samples) / 2) / (1 / upsampled_signal_length) )
    upsampled_signal = np.roll(upsampled_signal, integer_shift, axis=-1)

    if return_domain:
        return x, x_upsampled, upsampled_signal
    else:
        return upsampled_signal

def make_convolutional_forward_model_and_target_signal_MSE_loss_fn(object, target_signal, sampling_indices=None):
    
    num_nyquist_samples = target_signal.shape[-1]
    if sampling_indices is None:
        sampling_indices = np.arange(target_signal.shape[-1])
    @jit
    def convolve_and_loss(parameters):
        kernel = signal_from_params(parameters, num_nyquist_samples=num_nyquist_samples)
        conv_mat = make_convolutional_encoder(kernel, num_nyquist_samples=num_nyquist_samples)
        output_signal = conv_mat @ object
        return np.sum((output_signal[..., sampling_indices] - target_signal[..., sampling_indices])**2)
    return convolve_and_loss


@partial(jit, static_argnums=(1, 2,))
def signal_prox_fn(parameters, num_nyquist_samples=NUM_NYQUIST_SAMPLES, unit_energy=False):
    """
    Apply proximal function to the parameters to generate a signal that is non-negative, bandlimited, and (optionally) has unit energy
    """
    real, imag, amplitude_logit = parameters
    signal = signal_from_params((real, imag, amplitude_logit if unit_energy else np.array([1e2])),
                                          num_nyquist_samples=num_nyquist_samples)
    # this makes a unit energy signal
    signal = bandlimited_nonnegative_signal(signal, num_nyquist_samples=num_nyquist_samples)
    real, imag, _ = params_from_signal(signal, num_nyquist_samples=num_nyquist_samples)
    return (real, imag, amplitude_logit if not unit_energy else np.array([1e2]))

def run_optimzation(loss_fn, prox_fn, parameters, learning_rate=1e0, verbose=False,
                     tolerance=1e-6, momentum=0.9, loss_improvement_patience=800, max_epochs=100000,
                     learning_rate_decay=0.999, transition_begin=500,
                       key=None):
    """
    Run optimization with optax, return optimized parameters
    """
   
    if key is not None:
        grad_fn = value_and_grad(loss_fn, argnums=0)
    else:
        grad_fn = value_and_grad(loss_fn)
    grad_fn = jit(grad_fn)

    learning_rate = learning_rate if learning_rate_decay is None else optax.exponential_decay(
        learning_rate, transition_steps=1, decay_rate=learning_rate_decay, transition_begin=transition_begin)
    optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum, nesterov=False)


    opt_state = optimizer.init(parameters)

    if verbose:
        print('initial loss', loss_fn(parameters) if key is None else loss_fn(parameters, key))

    @jit
    def tolerance_check(loss, loss_history):
        return np.abs(loss - loss_history.max()) < tolerance

    last_best_loss_index = 0
    best_loss = 1e10
    best_params = parameters

    loss_history = []
    for i in range(max_epochs):
        if key is not None:
            key, subkey = jax.random.split(key)
            loss, gradient = grad_fn(parameters, subkey)
        else:
            loss, gradient = grad_fn(parameters)

        loss_history.append(loss)
        if loss < best_loss:
            best_loss = loss
            last_best_loss_index = i
            best_params = parameters
        if i - last_best_loss_index > loss_improvement_patience:
            break
        # if the loss is improving less than the tolerance, stop
        if i % loss_improvement_patience == 0 and \
                len(loss_history) > loss_improvement_patience and \
                    tolerance_check(loss, np.array(loss_history[-loss_improvement_patience:])):
            break
        # check for nans in loss and gradient
        if np.isnan(loss) or np.any(np.isnan(np.concatenate(gradient))):
            print('nan detected, breaking')
            break
        # Update parameters using optax's update rule.
        updates, opt_state = optimizer.update(gradient, opt_state)
        parameters = optax.apply_updates(parameters, updates)

        # Apply proximal function if provided.
        parameters = prox_fn(parameters)

        if verbose == 'very':
            print(f'{i}: {loss:.7f}')
        elif verbose:
            print(f'{i}: {loss:.7f}\r', end='')

    return best_params
