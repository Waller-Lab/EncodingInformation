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
OBJECT_LENGTH = 4 * NUM_NYQUIST_SAMPLES

@partial(jit, static_argnums=(1, 2))
def compute_gaussian_differential_entropy_per_pixel(output_signals, ev_threshold=1e-10, average_values=True):
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
    entropy = compute_gaussian_differential_entropy_per_pixel(noisy_output_signals, average_values=False)
    entropy -= 0.5 * np.log(2 * np.pi * np.e * noise_sigma**2)
    return np.mean(entropy)

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

def optimize_towards_target_signals(target_signals, input_signal, sampling_indices, initial_kernel=None, learning_rate=1e-3, ):                                    
    """
    Optimize a kernel to match a target signal
    """
    # TODO: add options for different types of convolutionals
    if initial_kernel is None:
        initial_kernel = bandlimited_nonnegative_signal(nyquist_samples=generate_random_bandlimited_signal())

    optimized_kernels = []
    for target_signal in tqdm(target_signals):
        loss_fn = make_convolutional_forward_model_and_loss_fn(input_signal, target_signal, sampling_indices=sampling_indices)
        optimized_params = run_optimzation(loss_fn, real_imag_bandlimit_energy_norm_prox_fn, 
                                np.concatenate(real_imag_params_from_signal(initial_kernel)), learning_rate=learning_rate, verbose=False)
        optimized_kernels.append(signal_from_real_imag_params(*param_vector_to_real_imag(optimized_params)))
    optimized_kernels = np.array(optimized_kernels)


    conv_mats = [make_convolutional_encoder(kernel) for kernel in optimized_kernels]
    output_signals = np.stack([conv_mat @ input_signal for conv_mat in conv_mats], axis=0)
    return optimized_kernels, output_signals

@jit
def signal_from_real_imag_param_vec(parameters):
  # optimization is done with respect to real and imaginary parts of the fourier spectrum
  real = parameters[:NUM_NYQUIST_SAMPLES // 2 + 1]
  imag = parameters[NUM_NYQUIST_SAMPLES // 2 + 1:]
  return signal_from_real_imag_params(real, imag)

def add_gaussian_noise_numpy(signal, noise_sigma):
    return signal + onp.random.normal(0, noise_sigma, signal.shape)

@partial(jit, static_argnums=(3, 4))
def conv_forward_model_with_erasure(parameters, objects, erasure_mask, align_center=False, nyquist_sample_output=True):
  kernel = signal_from_real_imag_param_vec(parameters)
  if align_center:
    kernel = np.roll(kernel, kernel.size // 2 - np.argmax(kernel))
  conv_mat = make_convolutional_encoder(kernel, sample=nyquist_sample_output)
  output_signals = (objects @ conv_mat.T)
  return output_signals * erasure_mask.reshape(1, -1)

@partial(jit, static_argnums=(2,))
def conv_forward_model(parameters, objects, align_center=False):
  kernel = signal_from_real_imag_param_vec(parameters)
  if align_center:
    kernel = np.roll(kernel, kernel.size // 2 - np.argmax(kernel))
  conv_mat = make_convolutional_encoder(kernel)
  output_signals = (objects @ conv_mat.T)

  return output_signals


def make_convolutional_forward_model_with_mi_loss_and_erasure(objects, erasure_mask, nyquist_sample_output=True, noise_sigma=0):
    
    if np.sum(erasure_mask) == 0:
        raise Exception('Erasure mask is empty')
    @jit
    def convolve_and_loss(parameters, key):
        output_signals = conv_forward_model_with_erasure(parameters, objects, erasure_mask, nyquist_sample_output=nyquist_sample_output)

        # don't include erased pixels in loss to avoid numerical errors
        output_signals = output_signals[:, erasure_mask]

        noisy_output_signals = output_signals + jax.random.normal(key, output_signals.shape) * noise_sigma
    
        return -compute_mutual_information_per_pixel(noisy_output_signals, noise_sigma=noise_sigma)
    
    return convolve_and_loss

@jit
def signal_from_real_imag_params(real, imag):
    """
    Generate signal with unit energy from real and imaginary parts of the spectrum
    """
    if real.shape[-1] != imag.shape[-1] + 1:
        raise Exception('Real must be one bigger than imaginary (DC)')
    # construct full signal from trainable parameters
    full_spectrum = np.concatenate([np.array([0]), imag]) *1j + real
    signal = np.fft.irfft(full_spectrum, NUM_NYQUIST_SAMPLES)

    # get energy normalized positive signal
    signal = bandlimited_nonnegative_signal(signal)
    return signal

@jit
def real_imag_params_from_signal(signal):
    """
    Get nonzero real and imaginary parts of the spectrum from a signal
    """
    ft = np.fft.rfft(signal)
    real = ft[:NUM_NYQUIST_SAMPLES // 2 + 1].real
    imag = ft[1:NUM_NYQUIST_SAMPLES // 2 + 1].imag
    return real, imag

def random_unnormalized_signal(N=1):
    """
    Generate a random, bandlimited signal by
    Sampling random nyquist samples between 0 and 1
    """
    return np.array(onp.random.rand(N, NUM_NYQUIST_SAMPLES))

@jit
def check_if_nonnegative(nyquist_samples):
    upsampled = upsample_signal(nyquist_samples)
    return np.all(upsampled >= 0, axis=-1)

@partial(jit, static_argnums=(1, 2, 3))
def bandlimited_nonnegative_signal(nyquist_samples, normalize_energy=True, 
                                   upsampled_signal_length=UPSAMPLED_SIGNAL_LENGTH, num_nyquist_samples=NUM_NYQUIST_SAMPLES):
    """
    Make sure that the signal is non-negative, has unit energy
    """
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


    if normalize_energy:
        nyquist_samples /= np.sum(nyquist_samples, axis=-1, keepdims=True)
 
    return np.squeeze(nyquist_samples)

@partial(jit, static_argnums=(1, 2))
def make_convolutional_encoder(kernel, object_length=OBJECT_LENGTH, sample=True):
    # upsample because we wan the size to be based on the input signal (i.e. the object)
    if object_length != NUM_NYQUIST_SAMPLES:
        kernel = upsample_signal(kernel, object_length)
    conv_mat = make_circulant_matrix(kernel)
    if sample:
        sampling_locations = np.linspace(0, 1, NUM_NYQUIST_SAMPLES, endpoint=False) + get_sampling_interval(NUM_NYQUIST_SAMPLES) / 2
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

@jit
def param_vector_to_real_imag(params):
    return params[:NUM_NYQUIST_SAMPLES // 2 + 1], params[NUM_NYQUIST_SAMPLES // 2 + 1:]

def make_intensity_coordinate_sampling_grid(sampling_indices, sample_n=40):
    """
    Make a grid of the potential intensity coordinates (sum to 1, non-negative)
    sampling_indices: where the target signal is nonzero and will instead have values from the grid
    """
    if len(sampling_indices) == 2:
        y_grid, x_grid = np.meshgrid(np.linspace(0, 1, sample_n), np.linspace(0, 1, sample_n))
        yx = np.stack([y_grid.ravel(), x_grid.ravel()], axis=1)
        # mask out the ones with sum > 1
        yx = yx[np.sum(yx, axis=-1) <= 1]
        target_signal = onp.zeros((yx.shape[0], NUM_NYQUIST_SAMPLES))
        target_signal[:, sampling_indices[0]] = yx[:, 0]
        target_signal[:, sampling_indices[1]] = yx[:, 1]
        # make sure the signal sums to 1
    elif len(sampling_indices) == 3:
        y_grid, x_grid, z_grid = np.meshgrid(np.linspace(0, 1, sample_n), np.linspace(0, 1, sample_n), np.linspace(0, 1, sample_n))
        yxz = np.stack([y_grid.ravel(), x_grid.ravel(), z_grid.ravel()], axis=1)
        # mask out the ones with sum > 1
        yxz = yxz[np.sum(yxz, axis=-1) <= 1]
        target_signal = onp.zeros((yxz.shape[0], NUM_NYQUIST_SAMPLES))
        target_signal[:, sampling_indices[0]] = yxz[:, 0]
        target_signal[:, sampling_indices[1]] = yxz[:, 1]
        target_signal[:, sampling_indices[2]] = yxz[:, 2]
    else:
        raise Exception('Not implemented')
    for signal in target_signal:
            remainder = 1 - signal[np.array(sampling_indices)].sum()
            for index in range(signal.size):
                if index not in sampling_indices:
                    signal[index] = remainder / (signal.size - len(sampling_indices))
    return np.array(target_signal)

def generate_random_bandlimited_signal():
    return bandlimited_nonnegative_signal(random_unnormalized_signal())

def generate_concentrated_signal(sampling_indices):
    """
    Make a signal that tries to concentrate the energy at the given sampling indices,
    but is still non-negative, bandlimited, and has unit energy
    """
    nyquist_samples = onp.zeros(NUM_NYQUIST_SAMPLES)
    for i in sampling_indices:
        nyquist_samples[i] = 1
    signal = bandlimited_nonnegative_signal(nyquist_samples=np.array(nyquist_samples))
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


def plot_in_spatial_coordinates(ax, signal, label=None, show_upsampled=True, show_samples=False, 
                                color_samples=False, vertical_line_indices=None, full_height_vertical_lines=False,
                                sample_point_indices=None, horizontal_line_indices=None, 
                                 num_nyquist_samples=NUM_NYQUIST_SAMPLES, upsampled_signal_length=UPSAMPLED_SIGNAL_LENGTH,
                                 markersize=8, marker='o', random_colors=False, center=False, plot_lim=1, color='k', 
                                 colors=None, erasure_mask=None,
                                 **kwargs):                                   

    num_nyquist_samples = NUM_NYQUIST_SAMPLES

    def plot_one_signal(signal, sample_point_indices=None, color=None, erasure_mask=erasure_mask):
        if signal.size == num_nyquist_samples:
            x, x_upsampled, upsampled_signal = upsample_signal(signal, num_nyquist_samples=num_nyquist_samples, 
                                                            upsampled_signal_length=upsampled_signal_length, return_domain=True)
        else:
            upsampled_signal = signal
            x_upsampled = np.linspace(0, 1, upsampled_signal_length, endpoint=False)
            x = np.linspace(0, 1, num_nyquist_samples, endpoint=False)
        if show_upsampled:
            if center:
                upsampled_signal = np.roll(upsampled_signal, upsampled_signal.size // 2 - np.argmax(upsampled_signal))

            if erasure_mask is not None:
                erasure_mask = np.repeat(erasure_mask, UPSAMPLED_SIGNAL_LENGTH // NUM_NYQUIST_SAMPLES)
                upsampled_signal *= erasure_mask.astype(float)
        

            ax.plot(x_upsampled, upsampled_signal, label=label, linewidth=2.1, color=color, **kwargs)
            # get the color used for the line
            color = ax.get_lines()[-1].get_color()
        if show_samples:
            if sample_point_indices is None:
                sample_point_indices = np.arange(num_nyquist_samples)
            sample_point_indices = np.array(sample_point_indices)
            ax.plot(x[sample_point_indices], signal[sample_point_indices], 
                                       marker,
                                        markersize=markersize, 
                                        color='k' if not color_samples else color,
                                        label=None if show_upsampled else label, 
                                        **kwargs)
            

    if vertical_line_indices is not None:
        x = upsample_signal(signal, return_domain=True)[0]
        # find highest value over all signals at verical_line_indices
        if full_height_vertical_lines:
            max_values = np.ones(len(vertical_line_indices))
        else:
            max_values = np.max(signal.reshape(-1, num_nyquist_samples)[..., vertical_line_indices], axis=0)
        for i, max_val in zip(vertical_line_indices, max_values):
            # plot line going from 0 to max_val
            ax.plot([x[i], x[i]], [0, max_val], 'k--')
    if horizontal_line_indices is not None:
        x = upsample_signal(signal, return_domain=True)[0]
        # find highest value over all signals at verical_line_indices
        y_values = np.max(signal.reshape(-1, num_nyquist_samples)[..., vertical_line_indices], axis=0)
        for x_index, y in zip(horizontal_line_indices, y_values):
            # plot line going from 0 to max_val
            ax.plot([0, x[x_index]], [y, y], 'k--')
            

    if len(signal.shape) == 1:
        plot_one_signal(signal, sample_point_indices=sample_point_indices, color=color)
    else:
        for i in range(signal.shape[0]):
            color = colors[i] if colors is not None else None
            plot_one_signal(signal[i], sample_point_indices=sample_point_indices, 
                                color=color if not random_colors else onp.random.rand(3))
                
    clear_spines(ax)
    ax.set(ylabel='Intensity', xlim=[0, 1], xlabel='Space', ylim=[0, plot_lim])




def plot_object(ax, signal, colors=None, **kwargs):
    for i, o in enumerate(signal.reshape(-1, signal.shape[-1])):
        ax.plot(np.linspace(0,1, o.size), o, **kwargs, color=colors[i] if colors is not None else None)
    ax.set(xlabel='Space', ylabel='Intensity', xlim=[0, 1], xticks=[0,1], ylim=[0, signal.max()], yticks=[0, signal.max()])
    sparse_ticks(ax)

    clear_spines(ax)
    

def plot_intensity_coord_histogram(ax, signals, sample_point_indices=(3, 4), **kwargs):
    bins = np.linspace(0, 1, 50)  
    h = ax.hist2d(signals[:, sample_point_indices[0]], signals[:, sample_point_indices[1]], 
              bins=[bins, bins], cmap='inferno', density=True)
    ax.set(xlabel='$I_1$', ylabel='$I_2$')
    # dash white 45 degree line
    ax.plot([0, 1], [1, 0], 'w--')
    # show colorbar with not ticks
    plt.colorbar(h[3], ax=ax, ticks=[])
    # make an axis label for the colorbar
    ax.text(1.2, 0.5, 'Probability', rotation=90, va='center', ha='left', transform=ax.transAxes)

    

def plot_in_intensity_coordinates(ax, signal, markersize=30, random_colors=False,
                                color=None, differentiate_colors=False, sample_point_indices=(3,4), plot_lim=1,
                                **kwargs):
    # plot the line y = -x + 
    # only plot this if there's nothing in the axes already
    if len(ax.lines) == 0:
        ax.plot([0, 1], [1, 0], 'k--', zorder=-1)
        ax.set(xlim=[0, 1], ylim=[0, 1], xlabel='$I_1$', ylabel='$I_2$')
    if differentiate_colors:
        color = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if len(signal.shape) == 1:
            color = color[0]
        else: 
            color = [color[i % len(color)] for i in range(signal.shape[0])]
    if random_colors:
        # generate a list of random colors of length signal.shape[0]
        color = onp.random.rand(signal.shape[0], 3)

    ax.scatter(signal[..., sample_point_indices[0]], signal[..., sample_point_indices[1]], s=markersize, 
               c=color, **kwargs)
               
            #    size=markersize, color=color)
    ax.set_aspect('equal')
    ax.set(xlim=[0, plot_lim], ylim=[0, plot_lim])

    clear_spines(ax)
    # plot again with x as marker

def make_convolutional_forward_model_and_loss_fn(input_signal, target_signal, sampling_indices=None):
    if sampling_indices is None:
        sampling_indices = np.arange(target_signal.shape[-1])
    @jit
    def convolve_and_loss(parameters):
        real = parameters[:NUM_NYQUIST_SAMPLES // 2 + 1]
        imag = parameters[NUM_NYQUIST_SAMPLES // 2 + 1:]
        kernel = signal_from_real_imag_params(real, imag)
        conv_mat = make_convolutional_encoder(kernel)
        output_signal = conv_mat @ input_signal
        return np.sum((output_signal[..., sampling_indices] - target_signal[..., sampling_indices])**2)
    return convolve_and_loss


def make_real_imag_loss_fn(target_signal, indices=None):
    if indices is None:
        indices = np.arange(target_signal.shape[-1])
    @jit
    def real_imag_loss_fn(parameters):
        real = parameters[:NUM_NYQUIST_SAMPLES // 2 + 1]
        imag = parameters[NUM_NYQUIST_SAMPLES // 2 + 1:]
        signal = signal_from_real_imag_params(real, imag)

        return np.sum((signal[..., indices] - target_signal[..., indices])**2)
    return real_imag_loss_fn

@jit
def real_imag_bandlimit_energy_norm_prox_fn(parameters):
    real, imag = param_vector_to_real_imag(parameters)
    signal = signal_from_real_imag_params(real, imag)
    signal = bandlimited_nonnegative_signal(signal)
    real, imag = real_imag_params_from_signal(signal)
    return np.concatenate([real, imag])



def run_optimzation(loss_fn, prox_fn, parameters, learning_rate=1e0, verbose=False,
                     tolerance=1e-6, momentum=0.9, loss_improvement_patience=500, max_epochs=100000,
                     learning_rate_decay=None, key=None):
    """
    Run optimization with optax, return optimized parameters
    """
   
    if key is not None:
        grad_fn = jit(value_and_grad(loss_fn, argnums=0))
    else:
        grad_fn = jit(value_and_grad(loss_fn))
    

    learning_rate = learning_rate if learning_rate_decay is None else optax.exponential_decay(
        learning_rate, transition_steps=1, decay_rate=learning_rate_decay, transition_begin=500)
    optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum, nesterov=False)


    
    opt_state = optimizer.init(parameters)

    if verbose:
        print('initial loss', loss_fn(parameters) if key is None else loss_fn(parameters, key))

    last_best_loss_index = 0
    best_loss = 1e10
    best_params = np.copy(parameters)
    for i in range(max_epochs):
        if key is not None:
            key, subkey = jax.random.split(key)
            loss, gradient = grad_fn(parameters, subkey)
        else:
            loss, gradient = grad_fn(parameters)
        if loss < best_loss - tolerance:
            best_loss = loss
            last_best_loss_index = i
            best_params = np.copy(parameters)
        if i - last_best_loss_index > loss_improvement_patience:
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
