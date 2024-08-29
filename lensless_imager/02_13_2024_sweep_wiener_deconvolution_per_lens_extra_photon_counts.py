# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: info_jax_flax_23
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Sweeping non-unsupervised Wiener Deconvolution with hand-tuned parameter, 01/29/2024
#
# Using a fixed seed (10) for consistency.

# %%
# %load_ext autoreload
# %autoreload 2

import os
from jax import config
config.update("jax_enable_x64", True)
import sys
sys.path.insert(0, '/home/lkabuli_waller/workspace/EncodingInformation/')
sys.path.append('/home/lkabuli_waller/workspace/EncodingInformation/imager_experiments/')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from encoding_information.gpu_utils import limit_gpu_memory_growth
limit_gpu_memory_growth()

# from image_distribution_models import PixelCNN

from cleanplots import *
#import jax.numpy as np
from jax.scipy.special import logsumexp
import numpy as np

from leyla_fns import *

# %%
from encoding_information.image_utils import add_noise, extract_patches
from encoding_information.models.gaussian_process import StationaryGaussianProcess
from encoding_information.models.pixel_cnn import PixelCNN
from encoding_information.information_estimation import estimate_mutual_information

# %%
from skimage.restoration import wiener, unsupervised_wiener, richardson_lucy
import skimage.metrics as skm


# %%
# load the PSFs

diffuser_psf = load_diffuser_32()
one_psf = load_single_lens_uniform(32)
two_psf = load_two_lens_uniform(32)
three_psf = load_three_lens_uniform(32)
four_psf = load_four_lens_uniform(32)
five_psf = load_five_lens_uniform(32)
aperture_psf = np.copy(diffuser_psf)
aperture_psf[:5] = 0
aperture_psf[-5:] = 0
aperture_psf[:,:5] = 0
aperture_psf[:,-5:] = 0


# %%
def compute_skm_metrics(gt, recon):
    # takes in already normalized gt
    mse = skm.mean_squared_error(gt, recon)
    psnr = skm.peak_signal_noise_ratio(gt, recon)
    nmse = skm.normalized_root_mse(gt, recon)
    ssim = skm.structural_similarity(gt, recon, data_range=1)
    return mse, psnr, nmse, ssim


# %%
# set seed values for reproducibility
seed_values_full = np.arange(1, 4)

# set photon properties 
#mean_photon_count_list = [20, 40, 60, 80, 100, 150, 200, 250, 300]
mean_photon_count_list = [160, 320]

# set eligible psfs

psf_patterns = [None, one_psf, two_psf, three_psf, four_psf, five_psf, diffuser_psf, aperture_psf]
psf_names = ['uc', 'one', 'two', 'three', 'four', 'five', 'diffuser', 'aperture']

# MI estimator parameters 
patch_size = 32
num_patches = 10000
bs = 500
max_epochs = 50

# %%
reg_value_best = 10**-2
print(reg_value_best)

# %% [markdown]
# ## Regular Wiener Deconvolution including fixed seed 10

# %%
psf_patterns_use = [one_psf, two_psf, three_psf, four_psf, five_psf, diffuser_psf, aperture_psf]
psf_names_use = ['one', 'two', 'three', 'four', 'five', 'diffuser', 'aperture']

#mean_photon_count_list = [20, 40, 60, 80, 100, 150, 200, 250, 300]
mean_photon_count_list = [160, 320]

seed_value = 10


for photon_count in mean_photon_count_list:
    for psf_idx, psf_use in enumerate(psf_patterns_use):
        print('PSF: {}, Photon Count: {}'.format(psf_names_use[psf_idx], photon_count))
        # make the data and scale by the photon count 
        (x_train, y_train), (x_test, y_test) = tfk.datasets.cifar10.load_data()
        data = np.concatenate((x_train, x_test), axis=0) # make one big glob of data
        data = data.astype(np.float64)
        data /= np.mean(data)
        data *= photon_count # convert to photons with mean value of photon_count
        max_val = np.max(data)
        labels = np.concatenate((y_train, y_test), axis=0) # make one big glob of labels. 
        # for CIFAR 100, need to convert images to grayscale
        if len(data.shape) == 4:
            data = tf.image.rgb_to_grayscale(data).numpy() # convert to grayscale
            data = data.squeeze()
        # zero pad data to be 96 x 96
        data_padded = np.zeros((data.shape[0], 96, 96))
        data_padded[:, 32:64, 32:64] = data

        convolved_data = convolved_dataset(psf_use, data_padded)
        convolved_data_noise = add_noise(convolved_data, seed=seed_value)
        # output of this noisy data is a jax array of float32, correct to regular numpy and float64
        convolved_data_noise = np.array(convolved_data_noise).astype(np.float64)

        mse_psf = []
        psnr_psf = []
        for i in range(convolved_data_noise.shape[0]):
            recon, _ = unsupervised_wiener(convolved_data_noise[i] / max_val, psf_use)
            recon = recon[17:49, 17:49] #this is the crop window to look at
            mse = skm.mean_squared_error(data[i] / max_val, recon)
            psnr = skm.peak_signal_noise_ratio(data[i] / max_val, recon)
            mse_psf.append(mse)
            psnr_psf.append(psnr)
        print('PSF: {}, Mean MSE: {}, Mean PSNR: {}'.format(psf_names_use[psf_idx], np.mean(mse_psf), np.mean(psnr_psf)))
        #np.save('unsupervised_wiener_deconvolution_fixed_seed/recon_{}_photon_count_{}_psf.npy'.format(photon_count, psf_names_use[psf_idx]), [mse_psf, psnr_psf])



# %%
for photon_count in mean_photon_count_list:
    for psf_idx, psf_use in enumerate(psf_patterns_use):
        print('PSF: {}, Photon Count: {}'.format(psf_names_use[psf_idx], photon_count))
        # make the data and scale by the photon count 
        (x_train, y_train), (x_test, y_test) = tfk.datasets.cifar10.load_data()
        data = np.concatenate((x_train, x_test), axis=0) # make one big glob of data
        data = data.astype(np.float64)
        data /= np.mean(data)
        data *= photon_count # convert to photons with mean value of photon_count
        max_val = np.max(data)
        labels = np.concatenate((y_train, y_test), axis=0) # make one big glob of labels. 
        # for CIFAR 100, need to convert images to grayscale
        if len(data.shape) == 4:
            data = tf.image.rgb_to_grayscale(data).numpy() # convert to grayscale
            data = data.squeeze()
        # zero pad data to be 96 x 96
        data_padded = np.zeros((data.shape[0], 96, 96))
        data_padded[:, 32:64, 32:64] = data

        convolved_data = convolved_dataset(psf_use, data_padded)
        convolved_data_noise = add_noise(convolved_data, seed=seed_value)
        # output of this noisy data is a jax array of float32, correct to regular numpy and float64
        convolved_data_noise = np.array(convolved_data_noise).astype(np.float64)

        mse_psf = []
        psnr_psf = []
        for i in range(convolved_data_noise.shape[0]):
            recon = wiener(convolved_data_noise[i] / max_val, psf_use, reg_value_best)
            recon = recon[17:49, 17:49] #this is the crop window to look at
            mse = skm.mean_squared_error(data[i] / max_val, recon)
            psnr = skm.peak_signal_noise_ratio(data[i] / max_val, recon)
            mse_psf.append(mse)
            psnr_psf.append(psnr)
        print('PSF: {}, Mean MSE: {}, Mean PSNR: {}'.format(psf_names_use[psf_idx], np.mean(mse_psf), np.mean(psnr_psf)))
        #np.save('regular_wiener_deconvolution_fixed_seed/recon_{}_photon_count_{}_psf.npy'.format(photon_count, psf_names_use[psf_idx]), [mse_psf, psnr_psf])

# %%
        

