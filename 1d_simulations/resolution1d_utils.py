
import os
import jax.numpy as np
import numpy as onp
from jax import random
from cleanplots import *
import cleanplots
import jax
from tqdm import tqdm
import matplotlib

from encoding_information.information_estimation import estimate_mutual_information
from encoding_information.plot_utils import plot_intensity_coord_histogram
from matplotlib.colors import LinearSegmentedColormap
from encoding_information.plot_utils import add_multiple_colorbars



def pcnn_mi_estimate(noisy_measurements, snr, bias=40, scale=300, max_epochs=10, num_samples_to_return=10):
    """
    Convenience function for estimating MI of 1D noisy measurements in arbitrary units using PCNN.
    """
    num_pixels = noisy_measurements.shape[-1]

    # make square for pcnn
    noisy_measurements = noisy_measurements.reshape(-1, int(np.sqrt(noisy_measurements.shape[-1])), int(np.sqrt(noisy_measurements.shape[-1])))
    gaussian_noise_sigma = 1 / (snr * num_pixels)

    # rescale because pcnn expects photons
    photon_sigma = gaussian_noise_sigma * scale
    photon_measurements = noisy_measurements * scale + bias
    mi, pcnn = estimate_mutual_information(photon_measurements, return_entropy_model=True, max_epochs=max_epochs, num_val_samples=2000,
                                 entropy_model='pixelcnn', gaussian_noise_sigma=photon_sigma, verbose=True)
    total_mi = mi * num_pixels
    print(total_mi)

    samples = pcnn.generate_samples(num_samples_to_return).reshape(-1, num_pixels)

    samples -= bias
    samples /= scale

    samples = samples.reshape(-1, num_pixels)

    return total_mi, samples

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

def get_noisy_measurements(noiseless, snr, num_measurements, batch_size=1024):
    """
    Add gaussian noise to noiseless measurements to achieve a given SNR.
    If noisless is a single image, make a distribution of different noisy measurements added to this
    If noiseless is a distribution of images, add noise to each measurement in the distribution.
    """
    num_batches = (num_measurements + batch_size - 1) // batch_size
    noisy_measurements = []
    
    # only show progress bar if there are over ten batches
    # if num_batches > 10:
    #     iter = tqdm(range(num_batches), desc="Adding noise")
    # else:
    iter = range(num_batches)
    for batch in iter:
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_measurements)
        batch_size_actual = end_idx - start_idx

        if noiseless.ndim == 2 and noiseless.shape[0] > 1:
            # make sure its same size as number of measurements requested
            assert noiseless.shape[0] == num_measurements
            noiseless_batch = noiseless[start_idx:end_idx]
        else:
            noiseless_batch = noiseless
        
        noise = (1 / (snr * noiseless.shape[-1])) * random.normal(
            random.PRNGKey(onp.random.randint(10000)), (batch_size_actual, noiseless_batch.shape[-1]), np.float32)
        noisy_batch = noiseless_batch + noise
        
        noisy_measurements.append(noisy_batch)
    
    noisy_measurements = np.concatenate(noisy_measurements, axis=0)
    return noisy_measurements

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
                                      x, snr, show_pixelated=True, num_measurements=1, alpha=1, 
                                      energy_coord_pixel_indices=None, num_energy_coord_noisy_measurements=int(1e6),
                                      y_max=None,
                                      show_extent=None):
    colors = [get_color_cycle()[1], get_color_cycle()[3]]
    
    extent = x.max() - x.min()
    if show_extent is not None:
        xlim = (extent / 2 - show_extent / 2, extent / 2 + show_extent / 2)
    else:
        xlim = (x.min(), x.max())
    pixel_size = extent / one_point_noiseless_pixels.size
    num_pixels = one_point_noiseless_pixels.size

    # create noisy measurements
    one_point_noisy_measurement = get_noisy_measurements(one_point_noiseless_pixels, snr, num_measurements)
    two_point_noisy_measurement = get_noisy_measurements(two_point_noiseless_pixels, snr, num_measurements)

    # figur eout scale for y axis
    pixel_domain = np.linspace(0, x.max(), num_pixels)
    if y_max is None:
        y_max = max(one_point_noisy_measurement.max(), two_point_noisy_measurement.max()) 
        # round up to nearest .01
        y_max = np.ceil(y_max * 100) / 100
    # make the peaks of the conolved objects align with the pixelated ones
    # interpolate the pixelated objects to the same x values as the convolved objects
    one_point_noiseless_pixels_upsampled = np.interp(x, np.linspace(0, x.max(), num_pixels), one_point_noiseless_pixels)
    ratio = one_point_convolved.mean() / one_point_noiseless_pixels_upsampled.mean()
    y_max_convolved = y_max * ratio


    if num_measurements == 1:
        one_point_noisy_measurement = one_point_noisy_measurement.flatten()
        two_point_noisy_measurement = two_point_noisy_measurement.flatten()

    fig, ax = plt.subplots(1, 2 + (1 if energy_coord_pixel_indices else 0),
                            figsize=(7 + (3 if energy_coord_pixel_indices else 0), 3))

    # plot the convolved objects
    ax[0].plot(x, one_point_convolved, color=colors[0], linewidth=3)
    ax[0].plot(x, two_point_convolved, color=colors[1], linewidth=3)
    ax[0].set(title='Convolved Objects', xlim=xlim, ylim=(0, y_max_convolved), 
            yticks=[0, y_max_convolved], xlabel='Position (nm)', ylabel='Intensity')

    # plot the pixelated objects
    # if show_pixelated:
    #     ax[1].plot(pixel_domain, one_point_noiseless_pixels, drawstyle='steps-mid', color=colors[0])
    #     ax[1].plot(pixel_domain, two_point_noiseless_pixels, drawstyle='steps-mid', color=colors[1])
    # else:
    #     ax[1].plot(pixel_domain, one_point_noiseless_pixels, color=colors[0], linewidth=3)
    #     ax[1].plot(pixel_domain, two_point_noiseless_pixels, color=colors[1], linewidth=3)
    # ax[1].set(title='Pixelated clean signals', xlim=xlim, ylim=(0, y_max), 
    #         yticks=[0, y_max], xlabel='Position (nm)', ylabel='Intensity')
    # ax[1].legend(['one point', 'two point'])

    # plot noisy measurements
    if num_measurements == 1:
        one_point_noisy_measurement = [one_point_noisy_measurement]
        two_point_noisy_measurement = [two_point_noisy_measurement]
        
    last_index = len(one_point_noisy_measurement) - 1
    for i, (one_point_measurement, two_point_measurement) in enumerate(zip(one_point_noisy_measurement, two_point_noisy_measurement)):
        if show_pixelated:
            ax[1].plot(pixel_domain, one_point_measurement, drawstyle='steps-mid', 
                       label='one point' if i == 0 else None, alpha=alpha, color=colors[0], linewidth=3)
            ax[1].plot(pixel_domain, two_point_measurement, drawstyle='steps-mid', 
                       label='two point' if i == 0 else None, alpha=alpha, color=colors[1], linewidth=3)
        else:
            alpha_to_use = 1 if i == last_index else alpha
            ax[1].plot(pixel_domain, one_point_measurement, label='one point' if i == 0 else None,
                        alpha=alpha_to_use, color=colors[0], linewidth=3)
            ax[1].plot(pixel_domain, two_point_measurement, label='two point' if i == 0 else None,
                        alpha=alpha_to_use, color=colors[1], linewidth=3)
            
    ax[1].set(title='Noisy Measurements', xlim=xlim, ylim=(0, y_max), 
                yticks=[], xlabel='Position (nm)', ylabel='Intensity')
    # ax[1].legend()




    if energy_coord_pixel_indices:
        # plot dotted lines at sampling locations
        for i in energy_coord_pixel_indices:
            ax[1].axvline(i * pixel_size, color='k', linestyle='--', alpha=0.5)

        # make more noisy measurements
        one_point_x1x2_noiseless = one_point_noiseless_pixels[..., np.round(np.array(energy_coord_pixel_indices)).astype(int)]
        two_point_x1x2_noiseless = two_point_noiseless_pixels[..., np.round(np.array(energy_coord_pixel_indices)).astype(int)]
        
        one_point_noisy_measurement = get_noisy_measurements(one_point_x1x2_noiseless,
                                                              snr * (one_point_noiseless_pixels.size / 2), num_energy_coord_noisy_measurements)
        two_point_noisy_measurement = get_noisy_measurements(two_point_x1x2_noiseless,
                                                                snr * (two_point_noiseless_pixels.size / 2), num_energy_coord_noisy_measurements)


        # plot 2 2d histograms
        plot_intensity_coord_histogram(ax[2], [one_point_noisy_measurement[:, 0], two_point_noisy_measurement[:, 0]],
                                        [one_point_noisy_measurement[:, 1], two_point_noisy_measurement[:, 1]],
                                       bins=100, max=y_max,
                                       colors=colors
                                )
        # # plot circles for_one_point_x1x2_noiseless
        # for i, (x1, x2) in enumerate(zip(two_point_x1x2_noiseless, one_point_x1x2_noiseless)):
        #     ax[2].add_patch(matplotlib.patches.Circle((x1, x2), .01, facecolor=colors[1 - i], alpha=1, edgecolor='black', linewidth=1))

        ax[2].set(title='Energy Coordinate Measurements', xlabel='Photons at pixel 1', ylabel='Photons at pixel 2')

    _ = [clear_spines(ax[i]) for i in range(len(ax))]



def plot_meaurement_sptial_distributions(x, one_point_convolved, two_point_convolved, SNR_value,  y_max, show_extent, 
                                  hist_density_max=0.3,
                                  num_intensity_bins=100, num_measurements= int(1e4), ):
    """
    Make image with two histograms of the measurements of two objects
    """

    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        
    
    # create noisy measurements
    one_point_noisy_measurement = get_noisy_measurements(one_point_convolved, SNR_value, num_measurements)
    two_point_noisy_measurement = get_noisy_measurements(two_point_convolved, SNR_value, num_measurements)

    colors = [get_color_cycle()[1], get_color_cycle()[3]]
    cmaps = [ LinearSegmentedColormap.from_list(f'cmap{i}', [(1,1,1), colors[i]]) for i in range(2)]
            
    extent = x.max() - x.min()
    xlim = (extent / 2 - show_extent / 2, extent / 2 + show_extent / 2)

    # mask show_extent from center in pixels
    show_extent_mask = (x > xlim[0]) & (x < xlim[1])
    x = x[show_extent_mask]


    # plot the clean signals
    ax[0].plot(x, one_point_convolved[show_extent_mask], color=colors[0], linewidth=3)
    ax[0].plot(x, two_point_convolved[show_extent_mask], color=colors[1], linewidth=3)
    ax[0].set(sta xlim=xlim, ylim=(0, y_max), 
            yticks=[], xlabel='Position (nm)', ylabel='Intensity', 
            xticks = [x.min(), x.max()], xticklabels=[0, show_extent])
    

    # repeat for number of measurements
    x = np.tile(x, (num_measurements, 1))

    y_one_point = one_point_noisy_measurement[:, show_extent_mask]
    y_two_point = two_point_noisy_measurement[:, show_extent_mask]

    x_bins = np.linspace(x.min(), x.max(), show_extent_mask.sum() + 1)
    intensity_bins = np.linspace(0, y_max, num_intensity_bins)


    hist_counts_one_point = np.histogram2d(x.flatten(), y_one_point.flatten(), bins=(x_bins, intensity_bins), density=True)[0].T
    hist_counts_two_point = np.histogram2d(x.flatten(), y_two_point.flatten(), bins=(x_bins, intensity_bins), density=True)[0].T

    # hist_counts_one_point = hist_counts_one_point / hist_density_max
    # hist_counts_two_point = hist_counts_two_point / hist_density_max

    hist_counts_one_point /= hist_counts_one_point.max()
    hist_counts_two_point /= hist_counts_two_point.max()

    hists = [hist_counts_one_point, hist_counts_two_point]
    blended_color = np.prod(np.stack([cmap(hist) for cmap, hist in zip(cmaps, hists)], axis=0), axis=0)

    ax[1].imshow(blended_color, origin='lower', extent=(x.min(), x.max(), intensity_bins.min(), intensity_bins.max()), aspect='auto')
    ax[1].set(xticks=[x.min(), x.max()], xticklabels=[0, show_extent], xlabel='Position (nm)', ylabel='Intensity', yticks=[])

    # add another ax for the colorbar, smaller than the main ax
    # cax = fig.add_axes([0.85, 0.15, 0.02, 0.7])

    add_multiple_colorbars(ax[1], cmaps)

    clear_spines(ax[0])
    clear_spines(ax[1])

