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

# %%
# %load_ext autoreload
# %autoreload 2

# Final MI estimation script for lensless imager, used in paper.

import os
from jax import config
config.update("jax_enable_x64", True)
import sys 
sys.path.append('/home/your_username/EncodingInformation/src')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from encoding_information.gpu_utils import limit_gpu_memory_growth
limit_gpu_memory_growth()

from cleanplots import *
import jax.numpy as np
import numpy as onp
import tensorflow as tf
import tensorflow.keras as tfk


from lensless_helpers import *

# %%
from encoding_information import extract_patches
from encoding_information.models import PixelCNN
from encoding_information.plot_utils import plot_samples
from encoding_information.models import PoissonNoiseModel
from encoding_information.image_utils import add_noise
from encoding_information import estimate_information

# %% [markdown]
# ### Sweep Photon Count and Diffusers

# %%
diffuser_psf = load_diffuser_32()
one_psf = load_single_lens_uniform(32)
two_psf = load_two_lens_uniform(32)
three_psf = load_three_lens_uniform(32)
four_psf = load_four_lens_uniform(32)
five_psf = load_five_lens_uniform(32)

# %%
# set seed values for reproducibility
seed_values_full = np.arange(1, 5)

# set photon properties 
bias = 10 # in photons
mean_photon_count_list = [20, 40, 80, 160, 320]

# set eligible psfs

psf_patterns = [diffuser_psf, four_psf, one_psf] 
psf_names = ['diffuser', 'four', 'one']

# MI estimator parameters 
patch_size = 32
num_patches = 10000
val_set_size = 1000
test_set_size = 1500 
num_samples = 8
learning_rate = 1e-3  # using 5x iterations per epoch, using smaller lr, and using less patience since it should be a smoother curve. 
num_iters_per_epoch = 500
patience_val = 20


save_dir = '/home/your_username/EncodingInformation/lensless_imager/mi_estimates_smaller_lr/'


# %%
for photon_count in mean_photon_count_list:
    for index, psf_pattern in enumerate(psf_patterns):
        val_loss_log = []
        mi_estimates = []
        lower_bounds = []
        upper_bounds = []
        for seed_value in seed_values_full:
            # load dataset 
            (x_train, y_train), (x_test, y_test) = tfk.datasets.cifar10.load_data() 
            data = onp.concatenate((x_train, x_test), axis=0)
            labels = np.concatenate((y_train, y_test), axis=0)
            data = data.astype(np.float32)
            # convert data to grayscale before converting to photons 
            if len(data.shape) == 4:
                data = tf.image.rgb_to_grayscale(data).numpy()
                data = data.squeeze()
            # convert to photons with mean value of photon_count 
            data /= onp.mean(data)
            data *= photon_count 
            # make tiled data
            random_data, random_labels = generate_random_tiled_data(data, labels, seed_value)

            if psf_pattern is None:
                start_idx = data.shape[-1] // 2 
                end_idx = data.shape[-1] // 2 - 1
                psf_data = random_data[:, start_idx:-end_idx, start_idx:-end_idx]
            else:
                psf_data = convolved_dataset(psf_pattern, random_data) 
            # add small bias to data 
            psf_data += bias 
            # make patches for training and testing splits, random patching 
            patches = extract_patches(psf_data[:-test_set_size], patch_size=patch_size, num_patches=num_patches, seed=seed_value, verbose=True)
            test_patches = extract_patches(psf_data[-test_set_size:], patch_size=patch_size, num_patches=test_set_size, seed=seed_value, verbose=True)  
            # put all the clean patches together for use in MI estimatino function later
            full_clean_patches = onp.concatenate([patches, test_patches])
            # add noise to both sets 
            patches_noisy = add_noise(patches, seed=seed_value)
            test_patches_noisy = add_noise(test_patches, seed=seed_value)

            # initialize pixelcnn 
            pixel_cnn = PixelCNN() 
            # fit pixelcnn to noisy patches. defaults to 10% val samples which will be 1k as desired. 
            # using smaller lr this time and adding seeding, letting it go for full training time.
            val_loss_history = pixel_cnn.fit(patches_noisy, seed=seed_value, learning_rate=learning_rate, do_lr_decay=False, steps_per_epoch=num_iters_per_epoch, patience=patience_val)
            # generate samples, not necessary for MI sweeps
            # pixel_cnn_samples = pixel_cnn.generate_samples(num_samples=num_samples)
            # # visualize samples
            # plot_samples([pixel_cnn_samples], test_patches, model_names=['PixelCNN'])

            # instantiate noise model
            noise_model = PoissonNoiseModel()
            # estimate information using the fit pixelcnn and noise model, with clean data
            pixel_cnn_info, pixel_cnn_lower_bound, pixel_cnn_upper_bound = estimate_information(pixel_cnn, noise_model, patches_noisy, 
                                                                                                test_patches_noisy, clean_data=full_clean_patches, 
                                                                                                confidence_interval=0.95)
            print("PixelCNN estimated information: ", pixel_cnn_info)
            print("PixelCNN lower bound: ", pixel_cnn_lower_bound)
            print("PixelCNN upper bound: ", pixel_cnn_upper_bound)
            # append results to lists
            val_loss_log.append(val_loss_history)
            mi_estimates.append(pixel_cnn_info)
            lower_bounds.append(pixel_cnn_lower_bound)
            upper_bounds.append(pixel_cnn_upper_bound)
            np.save(save_dir + 'pixelcnn_mi_estimate_{}_photon_count_{}_psf.npy'.format(photon_count, psf_names[index]), np.array([mi_estimates, lower_bounds, upper_bounds]))
            np.save(save_dir + 'pixelcnn_val_loss_{}_photon_count_{}_psf.npy'.format(photon_count, psf_names[index]), np.array(val_loss_log, dtype=object))
        np.save(save_dir + 'pixelcnn_mi_estimate_{}_photon_count_{}_psf.npy'.format(photon_count, psf_names[index]), np.array([mi_estimates, lower_bounds, upper_bounds]))
        np.save(save_dir + 'pixelcnn_val_loss_{}_photon_count_{}_psf.npy'.format(photon_count, psf_names[index]), np.array(val_loss_log, dtype=object))
