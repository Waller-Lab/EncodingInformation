from functools import partial
from cleanplots import *
import jax.numpy as np
import numpy as onp

from tqdm import tqdm
from jax import value_and_grad, jit, vmap
import jax
from jax.scipy.special import digamma
import optax

import imageio
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import resample

from encoding_information.information_estimation import estimate_mutual_information

NUM_NYQUIST_SAMPLES = 8
UPSAMPLED_SIGNAL_LENGTH = 512


@jit
def unit_energy_normalize(x):
    """
    Normalize a signal to unit energy. It is assumed that the domain of the signal is np.arange(signal.size), which means that np.sum(signal) is its energy
    (since dx = 1) and that the signal is positive. This conveniently makes a delta function be a vector of a single 1 and all other 0s

    Parameters
    ----------
    x : array-like
        The signal to normalize
    """
    return x / np.sum(x) 

def integrate_pixels(signal, num_nyquist_samples=NUM_NYQUIST_SAMPLES):
    """
    Integrate a signal over its pixels
    """
    new_shape = signal.shape[:-1] + (num_nyquist_samples, -1)
    return signal.reshape(*new_shape).sum(axis=-1)


@partial(jit, static_argnums=(1, ))
def signal_from_params(params, upsampled_signal_length=UPSAMPLED_SIGNAL_LENGTH):
    """
    Generate signal with unit energy from real and imaginary parts of the spectrum

    Parameters
    ----------
    params : tuple
        (real, imag, amplitude_logit) where real and imag are the real and imaginary parts of the spectrum given by the rfft
             and amplitude_logit is the logit of the amplitude of the DC component
    upsampled_signal_length : int
        Length of the signal to generate
    """
    real, imag, amplitude_logit = params
    if real.shape[-1] != imag.shape[-1] + 1:
        raise Exception('Real must be one bigger than imaginary (DC)')
    # params are stored in upsampled form to avoid weird fft behavior, so enforce this here
    

    # construct full signal from trainable parameters
    full_spectrum = np.concatenate([np.array([0]), imag]) *1j + real
    signal = np.fft.irfft(full_spectrum, upsampled_signal_length)

    # Make positive
    signal -= np.min(signal)
    # make unit energy. Assumes that the total length of the signal is the number of nyquist samples
    signal = unit_energy_normalize(signal)

    # Now can make energy less than 1 by scaling down amplitude
    def capped_relu(x):
        return np.clip(x, 0, 1)

    return signal * capped_relu(amplitude_logit)

@partial(jit, static_argnums=(1,))
def params_from_signal(signal, num_nyquist_samples=NUM_NYQUIST_SAMPLES):
    """
    Convert an upsampled signal to a tuple of paramters describing it

    Parameters
    ----------
    signal : array-like
        The signal to convert to parameter representation
    num_nyquist_samples : int
        The number of nyquist samples in the signal (i.e. its bandwidth). Since signal is
        upsampled beyond its bandwidth, this is not known from the signal itself
    upsampled_signal_length : int
        The length of the upsampled signal (provided explictly for jit purposes). This is needed to properly normalize it to unit energy
    """
    # get the amplitude first
    amplitude = np.sum(signal, keepdims=True)
    # avoid numerical errors
    amplitude = np.clip(amplitude, 0, 1)
    def inverse_capped_relu(x):
        return np.clip(x, 0, 1)
    amplitude_logit = inverse_capped_relu(amplitude)
    # now energy normalize the signal to get the other parameters describing its shape
    signal = unit_energy_normalize(signal)
    ft = np.fft.rfft(signal)
    real = ft[:num_nyquist_samples // 2 + 1].real
    imag = ft[1:num_nyquist_samples // 2 + 1].imag
    return (real, imag, amplitude_logit)

def generate_params_of_random_signal(unit_energy=True, seed=None, upsampled_signal_length=UPSAMPLED_SIGNAL_LENGTH, num_nyquist_samples=NUM_NYQUIST_SAMPLES):
    """
    Generate parameters of a random positive, bandlimited signal. This doesn't guarentee a uniform distribution over all possible signals

    """
    onp.random.seed(seed)
    # generate a nonbandlimited, positive signal
    signal = onp.random.rand(upsampled_signal_length)
    # apply bandlimit
    ft = onp.fft.rfft(signal)
    ft[num_nyquist_samples // 2 + 1:] = 0
    ft[0] = np.abs(ft[0]) # DC component must be real
    signal = np.fft.irfft(ft)
    # ensure positive
    signal -= np.min(signal)
    signal = unit_energy_normalize(signal)
    if not unit_energy:
        signal *= onp.random.rand()
    return params_from_signal(signal, num_nyquist_samples=num_nyquist_samples)

@partial(jit, static_argnums=(1, 2))
def downsample(y, num_samples, upsampled_signal_length=UPSAMPLED_SIGNAL_LENGTH):
    """
    Sample signal(s) at regular intervals

    Parameters
    ----------
    y : array-like
        Signal(s) to be sampled
    num_samples : int
        Number of samples to take
    upsampled_signal_length : int
        The length of the signal(s) in y (provided explictly for jit purposes)
    """
    return y[..., upsampled_signal_length // num_samples // 2::upsampled_signal_length // num_samples ]

def generate_random_signal(unit_energy=True, seed=None, upsampled_signal_length=UPSAMPLED_SIGNAL_LENGTH, num_nyquist_samples=NUM_NYQUIST_SAMPLES):
    """
    Generate a random positive, bandlimited signal. This is is heuristic and doesn't guarentee a uniform distribution over all possible signals
    """
    return signal_from_params(generate_params_of_random_signal(unit_energy=unit_energy, seed=seed, upsampled_signal_length=upsampled_signal_length, num_nyquist_samples=num_nyquist_samples), upsampled_signal_length=upsampled_signal_length)

def generate_random_object(type, seed=None, object_size=UPSAMPLED_SIGNAL_LENGTH, num_deltas=1, sin_freq_range=[0, 1],
                            gaussian_mixture_position=False, num_mixture_components=3, gaussian_mixture_seed=12345):
    """
    Generate a random object of a given type. Objects must be positive and have unit energy, but they do not need to sum to 1
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
        magnitude = onp.concatenate([onp.array([1]), 1 / np.sqrt(onp.fft.rfftfreq(UPSAMPLED_SIGNAL_LENGTH, 1/UPSAMPLED_SIGNAL_LENGTH)[1:])])
        random_phase = onp.concatenate([onp.array([1]), np.exp(1j * 2*np.pi*onp.random.rand(magnitude.shape[0] - 1))])
        object = onp.fft.irfft(magnitude * random_phase) 
        if np.min(object) < 0:
            object -= np.min(object)
        return object / onp.sum(object)
    elif type == 'white_noise':
        freqs = onp.fft.rfftfreq(object_size, 1/object_size)
        random_phase = onp.concatenate([onp.array([1]), np.exp(1j * 2*np.pi*onp.random.rand(freqs.shape[0] - 1))])
        object = onp.fft.irfft( random_phase) 
        if np.min(object) < 0:
            object -= np.min(object)
        # return object 
        return object / onp.sum(object)







################################################################
########## Functions for encoders and optimization ############
################################################################

@jit
def make_convolutional_encoder(conv_kernel):
    """
    Make a convolutional encoder from a 1D signal kernel

    Parameters
    ----------
    conv_kernel : array-like
        The convolutional kernel to use
    """
    conv_mat = make_circulant_matrix(conv_kernel)
    return conv_mat

def run_optimzation(loss_fn, prox_fn, parameters, learning_rate=1e0, verbose=False,
                     tolerance=1e-6, momentum=0.9, loss_improvement_patience=800, max_epochs=100000,
                     learning_rate_decay=0.999, transition_begin=500, return_param_history=False,
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
    param_history = []
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

        if return_param_history:
            param_history.append(parameters)

        if verbose == 'very':
            print(f'{i}: {loss:.7f}')
        elif verbose:
            print(f'{i}: {loss:.7f}\r', end='')

    if return_param_history:
        return best_params, param_history
    return best_params

def make_convolutional_forward_model_and_target_signal_MSE_loss_fn(object, target_integrated_signal, sampling_indices=None):
    
    num_nyquist_samples = target_integrated_signal.shape[-1]
    if sampling_indices is None:
        sampling_indices = np.arange(target_integrated_signal.shape[-1])
    @jit
    def convolve_and_loss(parameters):
        kernel = signal_from_params(parameters)
        conv_mat = make_convolutional_encoder(kernel)
        output_signal = conv_mat @ object
        integrated_output_signal = integrate_pixels(output_signal, num_nyquist_samples=num_nyquist_samples)

        return np.sum((integrated_output_signal[..., sampling_indices] - target_integrated_signal[..., sampling_indices])**2)
    return convolve_and_loss


@partial(jit, static_argnums=(1, 2,))
def signal_prox_fn(parameters, num_nyquist_samples=NUM_NYQUIST_SAMPLES, unit_energy=False):
    """
    Apply proximal function to the parameters to generate a signal that is non-negative, bandlimited, and (optionally) has unit energy

    The parameterization makes it inherently bandlimited since higher frequencies are not represented
    It also makes it have <= unit energy since it is parameterizied by an amplitude logit
    But it needs to made explicitly non-negative 
    """
    real, imag, amplitude_logit = parameters
    signal = signal_from_params((real, imag, amplitude_logit if unit_energy else np.array([1e2])))
    # make strictly positive
    signal -= np.min(signal)
    real, imag, _ = params_from_signal(signal, num_nyquist_samples=num_nyquist_samples)
    return (real, imag, amplitude_logit if not unit_energy else np.array([1e2]))




def optimize_towards_target_signals(target_integrated_signals, object, initial_kernel, sampling_indices=None,
                                      learning_rate=3e-3, learning_rate_decay=0.999, tolerance=1e-5, transition_begin=800,
                                      return_params=False,
                                        verbose=False):                                    
    """
    Optimize a kernel to match a target integrated signal (i.e. it has been integrated over pixels and is length num_nyquist_samples)
    """

    if target_integrated_signals[0].size == initial_kernel.size:
        raise Exception('Target integrated signal is the same size as the kernel, so it is not integrated over pixels')
    num_nyquist_samples = target_integrated_signals[0].shape[-1]

    optimized_kernels = []
    iter = tqdm(target_integrated_signals) if verbose else target_integrated_signals 
    for target_signal in iter:
        loss_fn = make_convolutional_forward_model_and_target_signal_MSE_loss_fn(object, target_signal, 
                                                               sampling_indices=sampling_indices)
        prox_fn = lambda x : signal_prox_fn(x, num_nyquist_samples=num_nyquist_samples)
        optimized_params = run_optimzation(loss_fn, prox_fn,
                                params_from_signal(initial_kernel, num_nyquist_samples=num_nyquist_samples),
                                transition_begin=transition_begin, learning_rate_decay=learning_rate_decay, tolerance=tolerance,
                                loss_improvement_patience=1000, max_epochs=10000,
                                  learning_rate=learning_rate, verbose=verbose)
        optimized_kernel = signal_from_params(optimized_params)

        optimized_kernels.append(optimized_kernel)
    optimized_kernels = np.array(optimized_kernels)


    conv_mats = [make_convolutional_encoder(kernel) for kernel in optimized_kernels]
    output_signals = np.stack([conv_mat @ object for conv_mat in conv_mats], axis=0)
    if return_params:
        return optimized_kernels, output_signals, optimized_params
    return optimized_kernels, output_signals

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
        # generate 1000 random samples within the positive orthant of the L1 ball using jax
        l1_ball_samples = jax.random.ball(key=jax.random.PRNGKey(onp.random.randint(10000)), d=target_signal.shape[-1], p=1, shape=(1000, ))
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


def make_convolutional_forward_model_with_mi_loss(objects, noise_sigma, integrate_output_signals=True, 
                                                  num_nyquist_samples=NUM_NYQUIST_SAMPLES, upsampled_signal_length=UPSAMPLED_SIGNAL_LENGTH):
    
    @jit
    def convolve_and_loss(parameters, key):
        output_signals = conv_forward_model(parameters, objects,
                                                         integrate_output_signals=integrate_output_signals, 
                                                         num_nyquist_samples=num_nyquist_samples,
                                                         upsampled_signal_length=upsampled_signal_length)
        if noise_sigma is None:
            raise Exception('Noise sigma must be specified')
        
        noisy_output_signals = output_signals + jax.random.normal(key, output_signals.shape) * noise_sigma
        return -compute_mutual_information_per_pixel(noisy_output_signals, noise_sigma=noise_sigma)
    
    return convolve_and_loss

@partial(jit, static_argnums=(2, 3, 4, 5))
def conv_forward_model(parameters, objects, align_center=False, integrate_output_signals=False, 
                       num_nyquist_samples=NUM_NYQUIST_SAMPLES, upsampled_signal_length=UPSAMPLED_SIGNAL_LENGTH):
  """
    Convolve objects with a kernel 

    Parameters
    ----------
    parameters : tuple
        (real, imag, amplitude_logit) where real and imag are the real and imaginary parts of the spectrum given by the rfft
             and amplitude_logit is the logit of the amplitude of the DC component
    objects : array-like
        The objects to convolve with the kernel
    align_center : bool
    nyquist_sample_output : bool
    num_nyquist_samples : int
         
  """
  kernel = signal_from_params(parameters, upsampled_signal_length=upsampled_signal_length)
  if align_center:
    kernel = np.roll(kernel, kernel.size // 2 - np.argmax(kernel))
  conv_mat = make_convolutional_encoder(kernel)
  output_signals = (objects @ conv_mat.T)
  if integrate_output_signals:
    output_signals = integrate_pixels(output_signals, num_nyquist_samples=num_nyquist_samples)
  return output_signals


def optimize_PSF_and_estimate_mi(objects_fn, noise_sigma, initial_kernel=None,
                                 learning_rate=1e-2, learning_rate_decay=0.999, verbose=True,
                                 estimate_with_pixel_cnn=True,
                                 loss_improvement_patience=2000, max_epochs=5000, num_nyquist_samples=NUM_NYQUIST_SAMPLES, 
                                 upsampled_signal_length=UPSAMPLED_SIGNAL_LENGTH):
  if estimate_with_pixel_cnn:
      # make sure num_nyquist_samples is a perfect square and upsampled_signal_length is a multiple of it
        if not np.sqrt(num_nyquist_samples) % 1 == 0:
            raise Exception('num_nyquist_samples must be a perfect square')
        if upsampled_signal_length % num_nyquist_samples != 0:
            raise Exception('upsampled_signal_length must be a multiple of num_nyquist_samples')

  objects = objects_fn()
  if initial_kernel is None:
    initial_kernel = generate_random_signal(num_nyquist_samples=num_nyquist_samples)
  initial_params = params_from_signal(initial_kernel, num_nyquist_samples=num_nyquist_samples)

  loss_fn = make_convolutional_forward_model_with_mi_loss(
      objects, noise_sigma=noise_sigma,  num_nyquist_samples=num_nyquist_samples,
      upsampled_signal_length=upsampled_signal_length)
  optimized_params = run_optimzation(loss_fn, lambda x : signal_prox_fn(x, num_nyquist_samples=num_nyquist_samples),
                          initial_params, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, verbose=verbose,
                          loss_improvement_patience=loss_improvement_patience, max_epochs=max_epochs,
                          key=jax.random.PRNGKey(onp.random.randint(100000)))
  test_objects = objects_fn()   

  if not estimate_with_pixel_cnn:
    optimized_mi = -make_convolutional_forward_model_with_mi_loss(test_objects, noise_sigma=noise_sigma, num_nyquist_samples=num_nyquist_samples,
                                                                    )(optimized_params, jax.random.PRNGKey(0))
    initial_mi = -make_convolutional_forward_model_with_mi_loss(test_objects, noise_sigma=noise_sigma, num_nyquist_samples=num_nyquist_samples, 
                                                                )(initial_params, jax.random.PRNGKey(0))
  else:   
    scale_factor = 100000 # because these signals are 0-1 but pixel cnn is designed for photon counts
    # output_signals = conv_forward_model(initial_params, test_objects,
    #                                             integrate_output_signals=True, 
    #                                             num_nyquist_samples=num_nyquist_samples,
    #                                             upsampled_signal_length=upsampled_signal_length)
    # noisy_output_signals = output_signals + jax.random.normal(jax.random.PRNGKey(onp.random.randint(10000)), output_signals.shape) * noise_sigma
    # fake_images = noisy_output_signals.reshape(-1, int(np.sqrt(num_nyquist_samples)), int(np.sqrt(num_nyquist_samples))) * scale_factor
    # if verbose:
    #     print('computing initial mi')
    # initial_mi = estimate_mutual_information(fake_images, gaussian_noise_sigma=noise_sigma * scale_factor, verbose=False)
    initial_mi = None

    output_signals = conv_forward_model(optimized_params, test_objects,
                                                integrate_output_signals=True, 
                                                num_nyquist_samples=num_nyquist_samples,
                                                upsampled_signal_length=upsampled_signal_length)
    noisy_output_signals = output_signals + jax.random.normal(jax.random.PRNGKey(onp.random.randint(10000)), output_signals.shape) * noise_sigma
    fake_images = noisy_output_signals.reshape(-1, int(np.sqrt(num_nyquist_samples)), int(np.sqrt(num_nyquist_samples))) * scale_factor
    if verbose:
        print('computing optimized mi')
    optimized_mi = estimate_mutual_information(fake_images, gaussian_noise_sigma=noise_sigma * scale_factor, verbose=False)
                                                        

  return initial_kernel, initial_params, optimized_params, objects, initial_mi, optimized_mi



def filter_l1_ball_samples_to_valid_signals(l1_ball_samples, upsampled_signal_length=UPSAMPLED_SIGNAL_LENGTH):
    signals = np.array([resample(s, upsampled_signal_length) for s in l1_ball_samples])
    
    min_val = signals.min(axis=-1)
    total_energy = signals.sum(axis=-1)

    nonnegative = min_val >= 0
    valid_energy = total_energy <= 1
    
    # print(f"of {len(l1_ball_samples)} samples, {nonnegative.sum()} are nonnegative and {valid_energy.sum()} have valid energy")

    return signals[nonnegative & valid_energy]

@partial(jax.jit, static_argnums=(1, 2))
def random_l1_vector(key, N, d):
    # Generate a random radius and a random direction
    vectors = np.abs(jax.random.ball(key, d, p=1, shape=(N,)))
    return vectors, jax.random.split(key)[1]   

def generate_uniform_random_bandlimited_signals(num_nyquist_samples, num_signals, batch_size=256,           
                                                upsampled_signal_length=UPSAMPLED_SIGNAL_LENGTH):    
    signals = []
    pbar = tqdm(total=num_signals, desc="Generating signals")
    key = jax.random.PRNGKey(onp.random.randint(0, 1000000))
    while len(signals) < num_signals:
        l1_ball_samples, key = random_l1_vector(key, batch_size, num_nyquist_samples)
        l1_ball_samples = l1_ball_samples / (upsampled_signal_length / num_nyquist_samples)
        valid_signals = filter_l1_ball_samples_to_valid_signals(l1_ball_samples, upsampled_signal_length=upsampled_signal_length)
        valid_signals = np.array(valid_signals)
        valid_signals = np.roll(valid_signals, upsampled_signal_length // num_nyquist_samples // 2, axis=-1)

        signals.extend(valid_signals)
        pbar.update(valid_signals.shape[0])
    pbar.close()
    print('concatenating...')
    return np.array(signals)[:num_signals]
