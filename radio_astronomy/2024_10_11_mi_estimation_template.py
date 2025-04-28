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

import os
from jax import config
config.update("jax_enable_x64", True)
import argparse
import warnings

# include argparse 
parser = argparse.ArgumentParser(description='Generate black hole MI estimates from measurements for array index')
parser.add_argument('--array_idx', type=int, required=True, help='index of the array to use, 0 through 69')
parser.add_argument('--gpu_idx', type=int, required=False, default=0, help='which GPU to use, defaults to 0')
args = parser.parse_args() 

array_idx = args.array_idx # set based on command line argument
print(array_idx, " is the array index")
# check if there is a gpu_idx entry 
gpu_idx = args.gpu_idx


import sys 
sys.path.append('/home/your_username/EncodingInformation/src')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
from encoding_information.gpu_utils import limit_gpu_memory_growth
limit_gpu_memory_growth()

import jax.numpy as np

from cleanplots import *

from tqdm import tqdm

# %%
from encoding_information.models import AnalyticComplexPixelGaussianNoiseModel
from encoding_information.models import FullGaussianProcess
from encoding_information import estimate_information




# %% [markdown]
# Data loading functions

# %%
def load_single_black_hole_measurement(path):
    measurement = np.load(path)
    real_part = np.real(measurement) 
    real_part = real_part + 1 
    real_part = real_part * (255) / 2
    imag_part = np.imag(measurement)
    imag_part = imag_part + 1
    imag_part = imag_part * (255) / 2
    # take the two vectors and concatenate them into one long vector
    return np.concatenate([real_part, imag_part], axis=-1)

def load_all_black_hole_measurements(folder, num_images, start_idx=0, scale_sigma=False):
    warnings.warn("The current way things are computed, the sigma scaling isn't done, but in the future it needs to.")
    measurements = []
    sigma = np.load(folder + 'sigmas/0.npy')
    if scale_sigma:
        # include scaling on the sigma values in future versions of data loading. this case isn't used for now and it's corrected for elsewhere after MI estimation is done
        sigma *= 255.0 / 2.0
    for image in range(start_idx, start_idx + num_images):
        path = folder + 'visibilities_s/visibilities/{}.npy'.format(image)
        measurements.append(load_single_black_hole_measurement(path))
    measurements = np.array(measurements)
    return measurements, sigma 



# %%
def plot_samples(samples, ground_truth, model_names=['Samples']):
    # plot inital samples, final samples, and ground truth
    # padding for better visualization
    padding_amount = np.zeros(50 - ground_truth.shape[-1] % 50)

    #vmin, vmax = np.percentile(ground_truth.flatten()[:5000], 1), np.percentile(ground_truth.flatten()[:5000], 99)
    vmin, vmax = 0, 256

    if type(samples) is not list:
        samples = [samples]

    fig, axs = plt.subplots(len(samples) + 1, 8, figsize=(20, 2*(len(samples)+1)))
    
    for row_index, model_samples in enumerate(samples):
        for i, ax in enumerate(axs[row_index]):
            model_sample_to_plot = np.concatenate([model_samples[i], padding_amount], axis=-1).reshape((50, -1))
            ax.imshow(model_sample_to_plot, cmap='inferno', vmin=vmin, vmax=vmax)
            ax.axis('off')

    # plot ground truth
    for i, ax in enumerate(axs[-1]):
        ground_truth_to_plot = np.concatenate([ground_truth[i], padding_amount], axis=-1).reshape((50, -1))
        ax.imshow(ground_truth_to_plot, cmap='inferno', vmin=vmin, vmax=vmax)
        ax.axis('off')

    # set y labels to left of each row by adding new axes
    for i, (ax, name) in enumerate(zip(axs[:, 0], model_names + ['Ground Truth'])):
        ax.text(-0.1, 0.5, name,  transform=ax.transAxes, rotation=90, va='center', ha='center')


# %%
def estimate_total_mi(full_gp, noise_model, train_measurements, test_measurements, average_nll=True):
    all_measurements = np.concatenate([test_measurements, train_measurements])
    # compute total NLL on the test set (and compare to training set)
    estimated_nll = full_gp.compute_negative_log_likelihood(test_measurements, average=True)
    # convert to total NLL instead of the above per-pixel NLL
    estimated_total_nll = estimated_nll * train_measurements.shape[1] 
    print("total NLL: ", estimated_total_nll, "per pixel NLL: ", estimated_nll)
    # compute the conditional entropy (which is calculated as total over all pixels conditional entropy)
    total_conditional_entropy = noise_model.estimate_conditional_entropy(all_measurements) # computed in loge space
    print("total conditional entropy: ", total_conditional_entropy)
    total_mutual_information = (estimated_total_nll - total_conditional_entropy) / np.log(2)
    print("total mutual information: ", total_mutual_information)
    if not average_nll:
        # compute non-averaged NLL on test set
        each_nll = full_gp.compute_negative_log_likelihood(test_measurements, average=average_nll)
        # convert to total NLL instead of per-pixel NLL 
        each_nll = each_nll * train_measurements.shape[1]
        return total_mutual_information, full_gp, each_nll 
    return total_mutual_information, full_gp


# %%
def compute_confidence_interval(list_of_items, confidence_interval=0.95):
    assert confidence_interval > 0 and confidence_interval < 1
    mean_value = np.mean(list_of_items)
    lower_bound = np.percentile(list_of_items, 50 * (1 - confidence_interval))
    upper_bound = np.percentile(list_of_items, 50 * (1 + confidence_interval))
    return mean_value, lower_bound, upper_bound


# %% [markdown]
# # Add in bootstrapping to the Gaussian Fit - Sweeps

# %%
test_set_options = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
num_bootstraps = 100

# %%
base_folder = '/home/your_username/black_holes/'

telescope_name = 'observations_combination_{}'.format(array_idx)
folder = base_folder + telescope_name + '/'

save_dir = '/home/your_username/black_holes/mi_estimates_all_combinations/'

add_uniform_noise = False # always no uniform noise now. 

# %%
num_measurements = 50000
train_start_idx = 10000

print("mi estimation for ", folder) 
mi_bootstraps_across_test_set_sizes = []
for test_set_size in test_set_options:
    total_measurements = num_measurements + test_set_size
    assert total_measurements <= 100000
    # load the data from that folder, doing it each time just to be safe.
    train_measurements, sigma = load_all_black_hole_measurements(folder, num_measurements, start_idx = train_start_idx)
    test_measurements, _ = load_all_black_hole_measurements(folder, test_set_size, start_idx = 0)
    all_measurements = np.concatenate([test_measurements, train_measurements]) # intentional ordering

    # set up NLL computation model 
    full_gp = FullGaussianProcess(train_measurements, add_uniform_noise=add_uniform_noise)
    # fit model to data
    full_gp.fit(train_measurements)
    # set up noise model
    noise_model = AnalyticComplexPixelGaussianNoiseModel(sigma) 

    mi_estimate, lower_bound, upper_bound = estimate_information(full_gp, noise_model, train_measurements, test_measurements, scale_total_mi=True, confidence_interval=0.95)
    print("Estimated MI: ", mi_estimate, "Confidence interval: ", lower_bound, upper_bound)
    mi_bootstraps_across_test_set_sizes.append(([mi_estimate, lower_bound, upper_bound]))
    # save the results each time 
    np.save(save_dir + '{}_main_code_{}_noise.npy'.format(telescope_name, 'uniform' if add_uniform_noise else 'no_uniform'), np.array(mi_bootstraps_across_test_set_sizes))
# save the results once all the test sets are done 
np.save(save_dir + '{}_main_code_{}_noise.npy'.format(telescope_name, 'uniform' if add_uniform_noise else 'no_uniform'), np.array(mi_bootstraps_across_test_set_sizes))
