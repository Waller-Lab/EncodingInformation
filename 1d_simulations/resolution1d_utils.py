
import os
import jax.numpy as np
import numpy as onp
from jax import random
from cleanplots import *
import cleanplots
import jax


def incoherent_psf_1d(wavelength, na, x_range):
    """
    Generate the 1D sinc^2 PSF for an incoherent imaging system.

    Parameters:
    wavelength (float): Wavelength of the light (in units consistent with x_range).
    na (float): Numerical aperture of the imaging system.
    x_range (array-like): Spatial coordinates in the image plane.

    Returns:
    psf (array-like): The 1D sinc^2 PSF evaluated at the given spatial coordinates.
    """
    x = np.asarray(x_range)
    
    # Normalize the input
    norm_x = (2 * np.pi / wavelength) * na * x
    
    # Calculate the sinc^2 pattern
    psf = np.sinc(norm_x / np.pi) ** 2
    
    # Normalize
    psf /= np.sum(psf)
    
    return psf


def nyquist_pixel_size(wavelength, na):
    """
    Calculate the size of a Nyquist-sampled pixel for the sinc function PSF.

    Parameters:
    wavelength (float): Wavelength of the light (in units consistent with the pixel size).
    na (float): Numerical aperture of the imaging system.

    Returns:
    pixel_size (float): The maximum size of a Nyquist-sampled pixel.
    """
    pixel_size = wavelength / (4 * na)

    return pixel_size

def make_one_point_object(signal_length):
    one_point_object = onp.zeros(signal_length)
    one_point_object[signal_length//2] = 1
    return np.array(one_point_object)

def make_two_point_object(signal_length, seperation_distance):
    two_point_object = onp.zeros(signal_length)
    two_point_object[signal_length//2 - int(seperation_distance/2)] += 0.5
    two_point_object[signal_length//2 + int(seperation_distance/2)] += 0.5
    return np.array(two_point_object)

get_noisy_measurements = lambda noiseless, snr, num_measurements: noiseless + (1 / (snr * noiseless.size)) * random.normal(
                                   random.PRNGKey(onp.random.randint(10000)), (num_measurements, noiseless.size), np.float32)

def gmm_mean_nll(X, means, covs, weights):
    """
    Compute the mean negative log-likelihood of data points X under a Gaussian Mixture Model
    with given means, covariance matrices, and weights.
    """
    n_components = len(means)
    
    # Compute log probabilities for each component and weight them
    log_likelihoods = np.array([np.log(weights[i]) + jax.scipy.stats.multivariate_normal.logpdf(X, mean=means[i], cov=covs[i])
                                 for i in range(n_components)])

    # Use logsumexp for numerical stability
    total_log_likelihood = jax.scipy.special.logsumexp(log_likelihoods, axis=0)

    return -np.mean(total_log_likelihood)

def estimate_2_point_1_point_mi(one_point_noiseless_pixels, two_point_noiseless_pixels, snr, num_measurements=int(1e6)):
    gaussian_noise_sigma = 1 / (snr * one_point_noiseless_pixels.size)
    num_pixels = one_point_noiseless_pixels.size
    noisy_one_point_measurements = get_noisy_measurements(one_point_noiseless_pixels, snr, num_measurements // 2)
    noisy_two_point_measurements = get_noisy_measurements(two_point_noiseless_pixels, snr, num_measurements // 2)

    all_measurements = np.concatenate([noisy_one_point_measurements, noisy_two_point_measurements], axis=0)

    means = [one_point_noiseless_pixels, two_point_noiseless_pixels]
    covs = [np.eye(num_pixels) * gaussian_noise_sigma**2 for _ in means]
    weights = np.array([0.5, 0.5])

    hy_mid_x = (0.5 * np.log(2 * np.pi * np.e * gaussian_noise_sigma**2)) * num_pixels

    h_y = gmm_mean_nll(all_measurements, means, covs, weights)

    mi = (h_y - hy_mid_x) / np.log(2) # convert to bits
    return mi

def simulate_optics(wavelength, NA, size, seperation_distance, pixel_size=None):
    dx_per_pixel = 32
    if pixel_size == None:
        pixel_size = nyquist_pixel_size(wavelength, NA) # in nanometers
    dx = pixel_size / dx_per_pixel
    # convert seperation distance to dx steps
    seperation_steps = seperation_distance / dx
    # compute num_pixels from size and pixel_size
    num_pixels = int(size / pixel_size)
    signal_length = num_pixels * dx_per_pixel
    spatial_extent = int(pixel_size * num_pixels)
    x = np.linspace(0, spatial_extent, signal_length)

    psf = incoherent_psf_1d(wavelength, NA, np.linspace(-spatial_extent, spatial_extent, 2*signal_length))

    one_point_object = make_one_point_object(signal_length)
    two_point_object = make_two_point_object(signal_length, seperation_steps)

    # convolve the objects with the sinc kernel
    one_point_convolved = np.convolve(one_point_object, psf, mode='valid')[:-1]
    two_point_convolved = np.convolve(two_point_object, psf, mode='valid')[:-1]

    # bin the convolved objects into pixels
    one_point_noiseless_pixels = one_point_convolved.reshape(num_pixels, dx_per_pixel).sum(axis=1)
    two_point_noiseless_pixels = two_point_convolved.reshape(num_pixels, dx_per_pixel).sum(axis=1)

    cropped_PSF = psf[signal_length//2:-signal_length//2]
    return one_point_object, two_point_object, one_point_convolved, two_point_convolved, one_point_noiseless_pixels, two_point_noiseless_pixels, x, cropped_PSF

def make_signal_and_measurement_plot(one_point_convolved, two_point_convolved, one_point_noiseless_pixels, two_point_noiseless_pixels,
                                      x, snr, show_pixelated=True, num_measurements=1, alpha=1, energy_coord_measurements=None, y_max=None,
                                      projection_vector=False):
    colors = [get_color_cycle()[1], get_color_cycle()[3]]
    
    extent = x.max() - x.min()
    pixel_size = extent / one_point_noiseless_pixels.size
    num_pixels = one_point_noiseless_pixels.size

    # create noisy measurements
    one_point_noisy_measurement = get_noisy_measurements(one_point_noiseless_pixels, snr, num_measurements)
    two_point_noisy_measurement = get_noisy_measurements(two_point_noiseless_pixels, snr, num_measurements)
    if num_measurements == 1:
        one_point_noisy_measurement = one_point_noisy_measurement.flatten()
        two_point_noisy_measurement = two_point_noisy_measurement.flatten()

    fig, ax = plt.subplots(1, 3 + 1 if energy_coord_measurements else 0,
                            figsize=(12 + 4 if energy_coord_measurements else 0, 4))
    # plot the convolved objects
    ax[0].plot(x, one_point_convolved, color=colors[0])
    ax[0].plot(x, two_point_convolved, color=colors[1])
    ax[0].set(title='Convolved Objects', xlim=(x.min(), x.max()), ylim=(0, one_point_convolved.max()), 
            yticks=[0, one_point_convolved.max()], xlabel='Position (nm)', ylabel='Intensity')

    # plot the pixelated objects
    pixel_domain = np.linspace(0, x.max(), num_pixels)
    if y_max is None:
        y_max = max(one_point_noisy_measurement.max(), two_point_noisy_measurement.max()) 
        # round up to nearest .01
        y_max = np.ceil(y_max * 100) / 100

    if show_pixelated:
        ax[1].plot(pixel_domain, one_point_noiseless_pixels, drawstyle='steps-mid', color=colors[0])
        ax[1].plot(pixel_domain, two_point_noiseless_pixels, drawstyle='steps-mid', color=colors[1])
    else:
        ax[1].plot(pixel_domain, one_point_noiseless_pixels, color=colors[0])
        ax[1].plot(pixel_domain, two_point_noiseless_pixels, color=colors[1])
    ax[1].set(title='Pixelated clean signals', xlim=(pixel_domain.min(), pixel_domain.max()), ylim=(0, y_max), 
            yticks=[0, y_max], xlabel='Position (nm)', ylabel='Intensity')
    ax[1].legend(['one point', 'two point'])

    # plot noisy measurements
    gaussian_noise_sigma = 1 / (snr * num_pixels) 
    if num_measurements == 1:
        one_point_noisy_measurement = [one_point_noisy_measurement]
        two_point_noisy_measurement = [two_point_noisy_measurement]
        
    for i, (one_point_measurement, two_point_measurement) in enumerate(zip(one_point_noisy_measurement, two_point_noisy_measurement)):
        if show_pixelated:
            ax[2].plot(pixel_domain, one_point_measurement, drawstyle='steps-mid', label='one point' if i == 0 else None, alpha=alpha, color=colors[0])
            ax[2].plot(pixel_domain, two_point_measurement, drawstyle='steps-mid', label='two point' if i == 0 else None, alpha=alpha, color=colors[1])
        else:
            ax[2].plot(pixel_domain, one_point_measurement, label='one point' if i == 0 else None, alpha=alpha, color=colors[0])
            ax[2].plot(pixel_domain, two_point_measurement, label='two point' if i == 0 else None, alpha=alpha, color=colors[1])
            
    ax[2].set(title='Noisy Measurements', xlim=(pixel_domain.min(), pixel_domain.max()), ylim=(0, y_max), 
                yticks=[], xlabel='Position (nm)', ylabel='Intensity')
    # ax[2].legend()

    if energy_coord_measurements:
        # make more noisy measurements
        one_point_noisy_measurement = get_noisy_measurements(one_point_noiseless_pixels, snr, energy_coord_measurements)
        two_point_noisy_measurement = get_noisy_measurements(two_point_noiseless_pixels, snr, energy_coord_measurements)

        if projection_vector:
            diff = one_point_noiseless_pixels - two_point_noiseless_pixels
            diff = diff / np.linalg.norm(diff)
            midpoint = (one_point_noiseless_pixels + two_point_noiseless_pixels) / 2
            midpoint = np.dot(midpoint, diff)

            one_point_noisy_measurement = np.dot(one_point_noisy_measurement, diff)
            two_point_noisy_measurement = np.dot(two_point_noisy_measurement, diff)
            bins = np.linspace(midpoint - 1, midpoint + 1, 100)
        else:
            # take center pixel
            center_pixel_index = num_pixels // 2
            one_point_noisy_measurement = one_point_noisy_measurement[:, center_pixel_index]
            two_point_noisy_measurement = two_point_noisy_measurement[:, center_pixel_index]

            bins = np.linspace(0, y_max, 100)

        ax[3].hist(one_point_noisy_measurement, bins=bins, alpha=0.5, color=colors[0])
        ax[3].hist(two_point_noisy_measurement, bins=bins, alpha=0.5, color=colors[1])
        ax[3].set(title='Energy Coordinate Measurements', xlabel='Photons', ylabel='Probability',
                  xticks=[], yticks=[])
        if projection_vector:
            # find midpoint along projection vector

            ax[3].axvline(midpoint, color='black', linestyle='--', linewidth=2)
            ax[3].legend(['optimal classifier','one point', 'two point'])
            # make xlims twice the range of the means
            ax[3].set(xlim=(bins.min(), bins.max()))
                           
        else:
            ax[3].legend(['one point', 'two point'])

    _ = [clear_spines(ax[i]) for i in range(len(ax))]




# def make_intenisty_coord_plot(wavelength, NA_value, size, num_pixels, seperation_distance, snr):
#     num_noisy_measurements = int(1e5)
#     pixel_size = size / num_pixels
#     (one_point_object, two_point_object, one_point_convolved, two_point_convolved, 
#      one_point_noiseless_pixels, two_point_noiseless_pixels, x, PSF) = simulate_optics(wavelength, NA_value, 
#                                                                                        size, seperation_distance, pixel_size)
#     center_pixel_index = num_pixels // 2
#     dx_per_pixel = one_point_object.size // num_pixels
#     seperation_distance_pixels = int(np.round(seperation_distance / (x[1] - x[0]) / dx_per_pixel))

#     one_point_noisy_measurements = get_noisy_measurements(one_point_noiseless_pixels, snr, num_noisy_measurements)
#     x1_one_point = one_point_noisy_measurements[:, center_pixel_index]
#     x2_one_point = one_point_noisy_measurements[:, center_pixel_index + seperation_distance_pixels]

#     two_point_noisy_measurements = get_noisy_measurements(two_point_noiseless_pixels, snr, num_noisy_measurements)
#     x1_two_point = two_point_noisy_measurements[:, center_pixel_index]
#     x2_two_point = two_point_noisy_measurements[:, center_pixel_index + seperation_distance_pixels]

#     # plot hist2d

#     # TODO: get the color histrogram blending code    

#     bin_max = max(x1_one_point.mean(), x2_one_point.mean(), x1_two_point.mean(), x2_two_point.mean())
#     bins_x1 = np.linspace(0, bin_max, 50)
#     bins_x2 = np.linspace(0, bin_max, 50)
#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#     ax[0].hist2d(x1_one_point, x2_one_point, bins=[bins_x1, bins_x2], cmap='viridis')
#     ax[1].hist2d(x1_two_point, x2_two_point, bins=[bins_x1, bins_x2], cmap='viridis')

