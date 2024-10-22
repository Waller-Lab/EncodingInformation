"""
Script that trains a single model based on the info in a provided config file
"""

# this only works on startup!
from jax import config
config.update("jax_enable_x64", True)



print("~~~~~~~~~~~~~running train script~~~~~~~~~~~~~~")
from cookie_monster_backend_lib import train_script_setup, train_script_complete
config_file_path, saving_dir, config, hyperparameters, already_elapsed_time, \
    tensorboard_dir, logging_dir, model_dir, resume_backup_dir  = train_script_setup()


#######################################################
### Enter your training code here #####################

# replace 'Analysis_' with ''
model_dir = model_dir.replace('Analysis_', '')



from cleanplots import *
from tqdm import tqdm
from encoding_information.information_estimation import *
from encoding_information.image_utils import *
from encoding_information.models.gaussian_process import StationaryGaussianProcess

from encoding_information.datasets.bsccm_utils import *
from bsccm import BSCCM
from jax import jit
import numpy as np
import yaml
from led_array.tf_util import prepare_test_dataset
import tensorflow.keras as tfk

from encoding_information.models import PixelCNN, FullGaussianProcess, StationaryGaussianProcess 
from encoding_information.models import PoissonNoiseModel
from encoding_information import estimate_information


def get_marker_index(target_row):
    return np.flatnonzero(np.logical_not(np.isnan(target_row)))[0]

#compute negative log_likelihood over test set
def compute_nlls(model, test_dataset, max_num, markers):
    negative_log_likelihoods = []
    marker_indices = []
    for i, (image, target) in tqdm(enumerate(test_dataset), total=max_num):
        if max_num is not None and i > max_num:
            break
        marker_index = get_marker_index(target)
        marker_indices.append(marker_index)
        marker = markers[marker_index]
        mixture = model(image[None])[marker]
        nll = -mixture.log_prob(target[marker_index]).numpy() 
        negative_log_likelihoods.append(nll)
    return np.array(negative_log_likelihoods), np.array(marker_indices)


def estimate_mi(model_name, config, patch_size, num_images=5000, num_patches=10000, test_set_fraction=0.1, confidence=0.95):
    saving_name = f'{model_name}_{patch_size}patch_mi_estimates'

    # # check if already cached
    # if os.path.exists(f'{saving_dir}/analysis/{saving_name}.npz'):
    #     print(f'Loading cached results for {model_name} MI estimates')
    #     return np.load(f'{saving_dir}/analysis/{saving_name}.npz')

    median_filter = config['data']['synthetic_noise']['median_filter']

    markers, image_target_generator, dataset_size, display_range, indices = get_bsccm_image_marker_generator(bsccm, **config['data'])
    images = load_bsccm_images(bsccm, indices=indices[:num_images], channel=config['data']['channels'][0], 
                convert_units_to_photons=True, edge_crop=config['data']['synthetic_noise']['edge_crop'],
                use_correction_factor=config['data']['synthetic_noise']['use_correction_factor'],
                median_filter=median_filter)

    mean_photons_per_pixel = np.mean(images)
    rescale_fraction = config['data']['synthetic_noise']['photons_per_pixel'] / mean_photons_per_pixel
    if rescale_fraction > 1:
        raise Exception('Rescale fraction must be less than 1')

    patches = extract_patches(images, patch_size=patch_size, num_patches=num_patches, strategy='random')

    if median_filter:
        # assume noiseless
        noisy_patches = add_noise(patches * rescale_fraction)
        clean_patches = patches * rescale_fraction
    else:
        noisy_patches = add_shot_noise_to_experimenal_data(patches, rescale_fraction)
    
    
    pixel_cnn = PixelCNN()
    # stationary_gp = StationaryGaussianProcess(noisy_patches)

    pixel_cnn.fit(noisy_patches, verbose=False, max_epochs=1000, patience=300)
    # stationary_gp.fit(noisy_patches, verbose=False)


    noise_model = PoissonNoiseModel()

    train_patches = noisy_patches[:int(len(noisy_patches) * (1 - test_set_fraction))]
    test_patches = noisy_patches[int(len(noisy_patches) * (1 - test_set_fraction)):]
    

    pixel_cnn_info, pixel_cnn_lower_bound, pixel_cnn_upper_bound = estimate_information(pixel_cnn, noise_model, train_patches, test_patches, confidence_interval=confidence)
    # full_gp_info, full_gp_lower_bound, full_gp_upper_bound = estimate_information(stationary_gp, noise_model, train_patches, test_patches, confidence_interval=confidence)


    # mi_pixel_cnn = estimate_mutual_information(noisy_patches, clean_images=clean_patches if median_filter else None, 
    #                 entropy_model='pixel_cnn', verbose=True, max_epochs=500, patience=100)
    # mi_gp = estimate_mutual_information(noisy_patches, clean_images=clean_patches if median_filter else None,
    #                  entropy_model='gaussian', verbose=True)

    # save the cached results (both nlls and marker indices in a single file)
    # create save directory if it doesn't exist
    if not os.path.exists(f'{saving_dir}/analysis'):
        os.makedirs(f'{saving_dir}/analysis')
    np.savez(f'{saving_dir}/analysis/{saving_name}', mi_pixel_cnn=pixel_cnn_info, 
             pixel_cnn_lower_bound=pixel_cnn_lower_bound, pixel_cnn_upper_bound=pixel_cnn_upper_bound)
            #  mi_gp=full_gp_info, full_gp_lower_bound=full_gp_lower_bound, full_gp_upper_bound=full_gp_upper_bound)
    return np.load(f'{saving_dir}/analysis/{saving_name}.npz')
    

def test_set_phenotyping_nll(model_name, config):
    saving_name = f'{model_name}_phenotyping_nll'

    # # check if already cached
    # if os.path.exists(f'{saving_dir}/analysis/{saving_name}.npz'):
    #     print(f'Loading cached results for {model_name} phenotyping nlls')
    #     return np.load(f'{saving_dir}/analysis/{saving_name}.npz')
    
    markers, image_target_generator, dataset_size, display_range, indices = get_bsccm_image_marker_generator(bsccm, **config['data'])
    test_dataset, test_dataset_size = prepare_test_dataset(config['hyperparameters']['test_fraction'], image_target_generator, dataset_size)
    
    model = tfk.models.load_model(model_dir + '/saved_model.h5', compile=False)

    nlls, marker_indices = compute_nlls(model, test_dataset, max_num=test_dataset_size, markers=markers)

    # save the cached results (both nlls and marker indices in a single file)
    # create save directory if it doesn't exist
    if not os.path.exists(f'{saving_dir}/analysis'):
        os.makedirs('f{saving_dir}/analysis')
    np.savez(f'{saving_dir}/analysis/{saving_name}', nlls=nlls, marker_indices=marker_indices)
    return np.load(f'{saving_dir}/analysis/{saving_name}.npz')
    



bsccm = BSCCM('/home/hpinkard_waller/data/BSCCM/')
# remove the .yaml and take the file name
model_name = config_file_path.split('/')[-1].split('.')[0]
# remove leading 'Analysis_' from model name
model_name = model_name.split('Analysis_')[-1]
patch_size = config['patch_size']

estimate_mi(model_name, config, patch_size)
test_set_phenotyping_nll(model_name, config)




#######################################################
##### Training complete file flag for scheduler #######
#######################################################
train_script_complete(saving_dir)