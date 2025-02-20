import numpy as np
import os
from tqdm import tqdm
from skimage import metrics
import warnings


def compute_confidence_interval(list_of_items, confidence_interval=0.95):
    assert confidence_interval > 0 and confidence_interval < 1
    mean_value = np.mean(list_of_items)
    lower_bound = np.percentile(list_of_items, 50 * (1 - confidence_interval))
    upper_bound = np.percentile(list_of_items, 50 * (1 + confidence_interval))
    return mean_value, lower_bound, upper_bound



def check_max_value_in_folder(folder, num_imgs, start_idx=0, file_prefix=''): 
    directory_min = 100000
    directory_max = 0
    for image_idx in range(start_idx, start_idx + num_imgs):
        path = os.path.join(folder, '{}{}.npy'.format(file_prefix, image_idx))
        image = np.load(path)
        directory_min = min(np.min(image), directory_min)
        directory_max = max(np.max(image), directory_max)
    print("Min: {}, Max: {}".format(directory_min, directory_max))
    return directory_min, directory_max



def compute_metrics_for_test_set(data_folder, gt_folder, data_range, start_idx = 0, test_set_length = 2000):
    mses = [] 
    psnrs = []
    ssims = [] 
    for image_idx in range(start_idx, start_idx + test_set_length): 
        gt_image_path = os.path.join(gt_folder, 'black_hole_{}.npy'.format(image_idx))
        in_image_path = os.path.join(data_folder, '{}.npy'.format(image_idx))
        # load the gt image
        gt_image = np.load(gt_image_path)
        # Load the recon image
        in_image = np.load(in_image_path)

        mse = metrics.mean_squared_error(gt_image, in_image)
        psnr = metrics.peak_signal_noise_ratio(gt_image, in_image)
        ssim = metrics.structural_similarity(gt_image, in_image, data_range=data_range)

        mses.append(mse)
        psnrs.append(psnr)
        ssims.append(ssim)
    mses = np.array(mses)
    psnrs = np.array(psnrs)
    ssims = np.array(ssims)
    return mses, psnrs, ssims

def compute_bootstraps(mses, psnrs, ssims, test_set_length, num_bootstraps=100): 
    bootstrap_mses = []
    bootstrap_psnrs = []
    bootstrap_ssims = []
    for bootstrap_idx in tqdm(range(num_bootstraps), desc='Bootstrapping to compute confidence interval'):
        # select indices for sampling
        bootstrap_indices = np.random.choice(test_set_length, test_set_length, replace=True) 
        # take the metric values at those indices
        bootstrap_selected_mses = mses[bootstrap_indices]
        bootstrap_selected_psnrs = psnrs[bootstrap_indices]
        bootstrap_selected_ssims = ssims[bootstrap_indices]
        # accumulate the mean of the selected metric values
        bootstrap_mses.append(np.mean(bootstrap_selected_mses))
        bootstrap_psnrs.append(np.mean(bootstrap_selected_psnrs))
        bootstrap_ssims.append(np.mean(bootstrap_selected_ssims))
    bootstrap_mses = np.array(bootstrap_mses)
    bootstrap_psnrs = np.array(bootstrap_psnrs)
    bootstrap_ssims = np.array(bootstrap_ssims) 
    return bootstrap_mses, bootstrap_psnrs, bootstrap_ssims




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
    if not scale_sigma:
        warnings.warn("The current way things are computed, the sigma scaling isn't done, but in the future it needs to.")
    measurements = []
    sigma = np.load(folder + 'sigmas/0.npy')
    if scale_sigma:
        # include scaling on the sigma values for the most correct approach. for now, in this framework it'll be scaled outside.
        sigma *= 255.0 / 2.0
    for image in range(start_idx, start_idx + num_images):
        path = folder + 'visibilities_s/visibilities/{}.npy'.format(image)
        measurements.append(load_single_black_hole_measurement(path))
    measurements = np.array(measurements)
    return measurements, sigma 


def find_file_with_values(directory, search_values, verbose=False):
    for filename in os.listdir(directory):
        if filename.endswith('.txt') and filename.startswith('combination'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                    if all(value in content for value in search_values):
                        if verbose:
                            print(f'File found: {filename} with values: {search_values}')
                        return filename 
            except Exception as e:
                print(f'Error reading file {filename}: {e}')
    print(f'File not found with values: {search_values}')
    return None


def correct_for_sigma_scaling(sigma_length):
    # H(Y | X) adds a 2 * sum log(sigma * 255 / 2) / log(2) 
    # so here, have 2 * sigma_length * log(255/2) / log(2) 
    # this term will be subtracted from (H(Y) - H(Y | X) / log(2): - correction_factor
    return 2 * sigma_length * np.log(255.0/2.0) / np.log(2.0)