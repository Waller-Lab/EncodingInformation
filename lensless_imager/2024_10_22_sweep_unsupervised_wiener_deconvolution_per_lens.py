# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: infotransformer
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Sweeping both unsupervised Wiener Deconvolution and non-unsupervised Wiener Deconvolution with hand-tuned paramete
#
# Using a fixed seed (10) for consistency.

# %%
# %load_ext autoreload
# %autoreload 2

import os
from jax import config
config.update("jax_enable_x64", True)
import sys 
sys.path.append('/home/your_username/EncodingInformation/src')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from encoding_information.gpu_utils import limit_gpu_memory_growth
limit_gpu_memory_growth()


from cleanplots import *
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

from lensless_helpers import *
from tqdm import tqdm

# %%
from encoding_information.image_utils import add_noise
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
mean_photon_count_list = [20, 40, 80, 160, 320]

# set eligible psfs

psf_patterns_use = [one_psf, four_psf, diffuser_psf]
psf_names_use = ['one', 'four', 'diffuser']

save_dir = '/home/your_username/EncodingInformation/lensless_imager/deconvolutions/'


# MI estimator parameters 
patch_size = 32
num_patches = 10000
test_set_size = 1500 
bs = 500
max_epochs = 50

seed_value = 10

reg_value_best = 10**-2

# %%
# data generation process 

for photon_count in mean_photon_count_list:
    for psf_idx, psf_pattern in enumerate(psf_patterns_use):
        # load dataset 
        (x_train, y_train), (x_test, y_test) = tfk.datasets.cifar10.load_data()
        data = np.concatenate((x_train, x_test), axis=0)
        data = data.astype(np.float64)
        labels = np.concatenate((y_train, y_test), axis=0) # make one big glob of labels. 
        # convert data to grayscale before converting to photons
        if len(data.shape) == 4:
            data = tf.image.rgb_to_grayscale(data).numpy()
            data = data.squeeze()
        # convert to photons with mean value of photon_count
        data /= np.mean(data)
        data *= photon_count
        # get maximum value in this data 
        max_val = np.max(data)
        # make tiled data 
        random_data, random_labels = generate_random_tiled_data(data, labels, seed_value)
        # only keep the middle part of the data 
        data_padded = np.zeros((data.shape[0], 96, 96))
        data_padded[:, 32:64, 32:64] = random_data[:, 32:64, 32:64]
        # save the middle part of the data as the gt for metric computation, include only the test set portion.
        gt_data = data_padded[:, 32:64, 32:64]
        gt_data = gt_data[-test_set_size:]
        # extract the test set before doing convolution 
        test_data = data_padded[-test_set_size:]
        # convolve the data 
        convolved_data = convolved_dataset(psf_pattern, test_data) 
        convolved_data_noisy = add_noise(convolved_data, seed=seed_value)
        # output of add_noise is a jax array that's float32, convert to regular numpy array and float64.
        convolved_data_noisy = np.array(convolved_data_noisy).astype(np.float64)

        # compute metrics using unsupervised wiener deconvolution 
        mse_psf = []
        psnr_psf = [] 
        ssim_psf = []
        for i in tqdm(range(convolved_data_noisy.shape[0])):
            recon, _ = unsupervised_wiener(convolved_data_noisy[i] / max_val, psf_pattern)
            recon = recon[17:49, 17:49] #this is the crop window to look at
            mse = skm.mean_squared_error(gt_data[i] / max_val, recon)
            psnr = skm.peak_signal_noise_ratio(gt_data[i] / max_val, recon)
            ssim = skm.structural_similarity(gt_data[i] / max_val, recon, data_range=1)
            mse_psf.append(mse)
            psnr_psf.append(psnr)
            ssim_psf.append(ssim)
            
        print('PSF: {}, Mean MSE: {}, Mean PSNR: {}, Mean SSIM: {}'.format(psf_names_use[psf_idx], np.mean(mse_psf), np.mean(psnr_psf), np.mean(ssim_psf)))
        np.save(save_dir + 'unsupervised_wiener_recon_{}_photon_count_{}_psf.npy'.format(photon_count, psf_names_use[psf_idx]), [mse_psf, psnr_psf, ssim_psf])

        # repeat with regular deconvolution
        mse_psf = []
        psnr_psf = []
        ssim_psf = []
        for i in tqdm(range(convolved_data_noisy.shape[0])): 
            recon = wiener(convolved_data_noisy[i] / max_val, psf_pattern, reg_value_best)
            recon = recon[17:49, 17:49] #this is the crop window to look at
            mse = skm.mean_squared_error(gt_data[i] / max_val, recon)
            psnr = skm.peak_signal_noise_ratio(gt_data[i] / max_val, recon)
            ssim = skm.structural_similarity(gt_data[i] / max_val, recon, data_range=1)
            mse_psf.append(mse)
            psnr_psf.append(psnr)
            ssim_psf.append(ssim)
        print('PSF: {}, Mean MSE: {}, Mean PSNR: {}, Mean SSIM: {}'.format(psf_names_use[psf_idx], np.mean(mse_psf), np.mean(psnr_psf), np.mean(ssim_psf)))
        np.save(save_dir + 'regular_wiener_recon_{}_photon_count_{}_psf.npy'.format(photon_count, psf_names_use[psf_idx]), [mse_psf, psnr_psf, ssim_psf])





# %%


