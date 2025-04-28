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

# %%
# %load_ext autoreload
# %autoreload 2

import os
from jax import config
config.update("jax_enable_x64", True)
import sys
sys.path.insert(0, '/home/your_username/EncodingInformation/')
sys.path.append('/home/your_username/EncodingInformation/imager_experiments/')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from encoding_information.gpu_utils import limit_gpu_memory_growth
limit_gpu_memory_growth()

# import tensorflow_datasets as tfds  # TFDS for MNIST #TODO INSTALL AGAIN LATER
#import tensorflow as tf             # TensorFlow operations



# from image_distribution_models import PixelCNN

from cleanplots import *
import jax.numpy as np
from jax.scipy.special import logsumexp
import numpy as onp

from leyla_fns import *

# %%
from encoding_information.image_utils import add_noise, extract_patches
from encoding_information.models.gaussian_process import StationaryGaussianProcess
from encoding_information.models.pixel_cnn import PixelCNN
from encoding_information.information_estimation import estimate_mutual_information

# %% [markdown]
# ### Sweep Photon Count and Diffusers

# %%
# load the PSFs

diffuser_psf = load_diffuser_32()
four_psf = load_four_lens_32()
one_psf = load_single_lens_32()

# %%
# set seed values for reproducibility
seed_values_full = np.arange(1, 4)

# set photon properties 
bias = 10 # in photons
mean_photon_count_list = [20, 40, 60, 80, 100, 150, 200, 250, 300]

# set eligible psfs

psf_patterns = [None, one_psf, four_psf, diffuser_psf]
psf_names = ['uc', 'one', 'four', 'diffuser']

# MI estimator parameters 
patch_size = 32
num_patches = 10000
bs = 500
max_epochs = 50

# %%
for photon_count in mean_photon_count_list[:1]: #TODO change to more photons 
    for index, psf_pattern in enumerate(psf_patterns):
        gaussian_mi_estimates = []
        pixelcnn_mi_estimates = []
        print('Mean photon count: {}, PSF: {}'.format(photon_count, psf_names[index]))
        for seed_value in seed_values_full:
            # load dataset
            (x_train, y_train), (x_test, y_test) = tfk.datasets.cifar10.load_data()
            data = onp.concatenate((x_train, x_test), axis=0) # make one big glob of data
            data = data.astype(np.float32)
            data /= onp.mean(data)
            data *= photon_count # convert to photons with arbitrary mean value of 1000 photons
            labels = np.concatenate((y_train, y_test), axis=0) # make one big glob of labels. 
            # for CIFAR 100, need to convert images to grayscale
            if len(data.shape) == 4:
                data = tf.image.rgb_to_grayscale(data).numpy() # convert to grayscale
                data = data.squeeze()
            # make tiled data
            random_data, random_labels = generate_random_tiled_data(data, labels, seed_value)
         
            if psf_pattern is None:
                start_idx = data.shape[-1] // 2
                end_idx = data.shape[-1] // 2 - 1  
                psf_data = random_data[:, start_idx:-end_idx, start_idx:-end_idx]
            else:
                psf_data = convolved_dataset(psf_pattern, random_data)
            # omitting zero cutoff step TODO verify
            # add bias to data 
            psf_data += bias
            # make patches and add noise
            psf_data_patch = extract_patches(psf_data, patch_size=patch_size, num_patches=num_patches, seed=seed_value)
            psf_data_shot_patch = add_noise(psf_data_patch, seed=seed_value, batch_size=bs)
            # compute gaussian MI estimate, use comparison clean images
            mi_gaussian_psf = estimate_mutual_information(psf_data_shot_patch, clean_images=psf_data_patch, entropy_model='gaussian',
                                                            max_epochs=max_epochs, verbose=True)
            # compute PixelCNN MI estimate, use comparison clean images
            mi_pixelcnn_psf = estimate_mutual_information(psf_data_shot_patch, clean_images=psf_data_patch, entropy_model='pixelcnn', num_val_samples=1000,
                                                            max_epochs=max_epochs, do_lr_decay=True, verbose=True)
            gaussian_mi_estimates.append(mi_gaussian_psf)
            pixelcnn_mi_estimates.append(mi_pixelcnn_psf)
            np.save('cifar10_mi_estimates/pixelcnn_mi_estimate_{}_photon_count_{}_psf.npy'.format(photon_count, psf_names[index]), np.array(pixelcnn_mi_estimates))
            np.save('cifar10_mi_estimates/gaussian_mi_estimate_{}_photon_count_{}_psf.npy'.format(photon_count, psf_names[index]), np.array(gaussian_mi_estimates))
        # save the results once the seeds are done, file includes photon count and psf name
        np.save('cifar10_mi_estimates/pixelcnn_mi_estimate_{}_photon_count_{}_psf.npy'.format(photon_count, psf_names[index]), np.array(pixelcnn_mi_estimates))
        np.save('cifar10_mi_estimates/gaussian_mi_estimate_{}_photon_count_{}_psf.npy'.format(photon_count, psf_names[index]), np.array(gaussian_mi_estimates))
