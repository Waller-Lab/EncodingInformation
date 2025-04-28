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
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from encoding_information.gpu_utils import limit_gpu_memory_growth
limit_gpu_memory_growth()

from cleanplots import *
import jax.numpy as np
from jax.scipy.special import logsumexp
import numpy as onp

from lensless_helpers import *
from encoding_information.image_utils import add_noise

# %%
# %%
# load the 4 psfs
diffuser_psf = load_diffuser_32()
four_psf = load_four_lens_32()
one_psf = load_single_lens_32()
two_psf = load_two_lens()
two_psf = np.pad(two_psf, ((2,2),(2,2)), 'constant', constant_values=0)
rml_psf = load_rml_new_psf()
rml_psf = np.pad(rml_psf, ((2,2),(2,2)), 'constant', constant_values=0)


# %%
def test_system(mean_photon_count=300, psf_pattern=None, bias=10, model_name='cnn', seed_value=1, training_fraction=0.8, testing_fraction=0.1):
    # clear cache
    seed_value = int(seed_value)
    tfk.backend.clear_session()
    gc.collect()
    tfk.utils.set_random_seed(seed_value)
    # load cifar10 data
    (x_train, y_train), (x_test, y_test) = tfk.datasets.cifar10.load_data()
    data = np.concatenate((x_train, x_test), axis=0) # make one big glob of data
    data = data.astype(np.float32)
    data /= np.mean(data)
    data *= mean_photon_count # convert to photons with arbitrary mean value of 1000 photons
    labels = np.concatenate((y_train, y_test), axis=0) # make one big glob of labels. 
    # for CIFAR 100, need to convert images to grayscale
    if len(data.shape) == 4:
        data = tf.image.rgb_to_grayscale(data).numpy() # convert to grayscale
        data = data.squeeze()
    # generate random tiled data for entire dataset
    random_data, random_labels = generate_random_tiled_data(data, labels, seed_value)
    # generate convolution/patterned data
    if psf_pattern is None:
        start_idx = data.shape[-1] // 2
        end_idx = data.shape[-1] // 2 - 1
        psf_data = random_data[:, start_idx:-end_idx, start_idx:-end_idx]
    else:
        psf_data = convolved_dataset(psf_pattern, random_data)
    # add bias
    psf_data += bias 
    # add noise
    psf_data_noisy = add_noise(psf_data, seed=seed_value, batch_size=500)
    # run network
    _, _, test_loss, test_acc = run_network_cifar(psf_data_noisy, random_labels, seed_value, training_fraction, testing_fraction, mode=model_name)
    return test_acc



# %%
# set training, testing, validation fractions
training_fraction = 0.8
testing_fraction = 0.1
validation_fraction = 0.1

# set seed values for reproducibility
seed_values_full = np.arange(1, 11)

# set photon properties 
bias = 10 # in photons
#mean_photon_count_list = [20, 40, 60, 80, 100, 150, 200, 250, 300]
mean_photon_count_list = [20, 40, 60, 80, 100, 150, 300]
# set eligible psfs

psf_patterns = [one_psf, four_psf, diffuser_psf]
psf_names = ['one', 'four', 'diffuser']

# %%
for mean_photon_count in mean_photon_count_list:
    for model_name in ['cnn']:
        for index, psf_name in enumerate(psf_names):
            print(f'Training for {psf_name} psf with {model_name} model and {mean_photon_count} photon count')
            psf_pattern = psf_patterns[index]
            classification_results = []
            for seed_value in seed_values_full:
                test_accuracy_result = test_system(mean_photon_count, psf_pattern, bias, model_name, seed_value, training_fraction=training_fraction, testing_fraction=testing_fraction)
                classification_results.append(test_accuracy_result)
            #np.save('classifier_results/cifar_test_accuracy_{}_mean_photon_count_{}_psf_{}_bias_{}_model.npy'.format(mean_photon_count, psf_name, bias, model_name), classification_results)
