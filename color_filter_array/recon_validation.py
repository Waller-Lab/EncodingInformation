import os
import yaml
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.utils import resample
from utils import *
from recon import FullModel

def patch_image(image, patch_size, stride):
    """
    Patches the image with a specified stride and patch size.

    Args:
        image (np.ndarray): The input image to be patched.
        patch_size (int): The size of each patch.
        stride (int): The stride between patches.

    Returns:
        List[np.ndarray]: A list of image patches.
    """
    patches = []
    h, w, _ = image.shape
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i + patch_size, j:j + patch_size, :]
            patches.append(patch)
    return patches

def rescale_for_display(x):
    return np.clip(x ** (1 / 2.2), 0, 1)

def main(config_path, gpu_index):
    # Set GPU device based on index provided
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    # prevent preallocation of memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # Load config
    with open(config_path, 'r') as file:
        h_params = yaml.safe_load(file)

    # Initialize the model
    model_key = jax.random.PRNGKey(h_params['model_key'])
    P, F, K = h_params['P'], h_params['F'], h_params['K']
    sensor_shape = (h_params['mask_size'], h_params['mask_size'], h_params['num_channels'])
    gamma = h_params['gamma']
    model = FullModel(sensor_shape, P, F, K, gamma, model_key)
    

    # Load the trained model
    model_dir = os.path.join("models", h_params['model_name'])
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.eqx')]
    latest_model = max(model_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    if h_params['mask_path']:
        mask = np.load(h_params['mask_path'])
        model = eqx.tree_at(lambda m: m.sensor_layer, model, replace=model.sensor_layer.update_w(mask))
    model = eqx.tree_deserialise_leaves(os.path.join(model_dir, latest_model), model)

    data_key = jax.random.PRNGKey(h_params['data_key'])
    mean_photons = h_params['mean_photons']
    data_mean = np.load(h_params['data_mean_path'])

    image_path = '/home/anonymous_user/Workspace/PYTHON/EncodingInformation/color_filter_array/data/npy_files/test'
    images = glob.glob(os.path.join(image_path, '0*.npy'))

    patch_size = 24
    stride = 8

    # Initialize lists to store metrics for all images
    all_image_mse_list = []
    all_image_ssim_list = []
    all_image_psnr_list = []

    # Initialize lists to store metrics for all patches
    all_patch_mse_list = []
    all_patch_ssim_list = []
    all_patch_psnr_list = []

    for image_path in tqdm(images):
        clean_image = np.load(image_path)
        scale_factor = mean_photons / np.mean(data_mean)
        image = clean_image * scale_factor
        noise_key, subkey = jax.random.split(data_key)
        image = image + jax.random.normal(subkey, shape=image.shape) * jnp.sqrt(image)
        image = image / scale_factor
        image = image.clip(0, 1)
        image_patches = patch_image(image, patch_size, stride)
        image_patches = jnp.stack(image_patches)
        print(f"Generated {len(image_patches)} patches from the image.")

        # define a dataloader for the patches
        class PatchLoader:
            def __init__(self, patches, batch_size):
                self.patches = patches
                self.batch_size = batch_size
                self.indices = jnp.arange(len(self.patches))

            def __iter__(self):
                return self

            def __next__(self):
                if len(self.indices) == 0:
                    raise StopIteration
                batch_indices = self.indices[:self.batch_size]
                self.indices = self.indices[self.batch_size:]
                return self.patches[batch_indices]

            def __len__(self):
                return len(self.patches)    

        patch_loader = PatchLoader(image_patches, 1024)

        # reconstruct the patches with the model
        predictions_list = []
        for patches in patch_loader:
            alpha = jnp.full((len(patches), 1), 1e8)
            predictions = jax.vmap(model)(patches, alpha)
            predictions_list.append(np.array(predictions))

        predictions = np.concatenate(predictions_list, axis=0)

        ii_range = image.shape[0]//8-2
        jj_range = image.shape[1]//8-2
        reconstructed_image = np.zeros((ii_range*8, jj_range*8, 3))
        patch_index = 0
        for ii in range(ii_range):
            for jj in range(jj_range):
                try:
                    reconstructed_image[ii*8:(ii+1)*8, jj*8:(jj+1)*8, :] = predictions[patch_index]*3
                except Exception as e:
                    print(f"Error at patch index {patch_index}: {e}")
                patch_index += 1

        # calculate the target image accounting for the crop to patch to 24
        target_image = clean_image[8:reconstructed_image.shape[0]+8,8:reconstructed_image.shape[1]+8,:3]*3
        target_image = np.array(target_image)

        # calculate ssim, psnr, and mse for the full image
        image_mse = np.mean((target_image - reconstructed_image) ** 2)
        image_ssim_value = ssim(target_image, reconstructed_image, data_range=1.0, channel_axis=-1)
        image_psnr_value = psnr(target_image, reconstructed_image, data_range=1.0)

        # Store metrics for the full image
        all_image_mse_list.append(image_mse)
        all_image_ssim_list.append(image_ssim_value)
        all_image_psnr_list.append(image_psnr_value)

        # patch the target image and reconstructed image into 64 x 64 tiles
        target_image_tiles = patch_image(target_image, 64, 64)
        reconstructed_image_tiles = patch_image(reconstructed_image, 64, 64)

        # calculate ssim, psnr, and mse for each tile
        patch_mse_list = []
        patch_ssim_list = []
        patch_psnr_list = []

        for target_tile, reconstructed_tile in zip(target_image_tiles, reconstructed_image_tiles):
            patch_mse = np.mean((target_tile - reconstructed_tile) ** 2)
            patch_ssim_value = ssim(target_tile, reconstructed_tile, data_range=1.0, channel_axis=-1)
            patch_psnr_value = psnr(target_tile, reconstructed_tile, data_range=1.0)
            patch_mse_list.append(patch_mse)
            patch_ssim_list.append(patch_ssim_value)
            patch_psnr_list.append(patch_psnr_value)

        # Store patch metrics
        all_patch_mse_list.extend(patch_mse_list)
        all_patch_ssim_list.extend(patch_ssim_list)
        all_patch_psnr_list.extend(patch_psnr_list)

        # save the lists of metrics each iteration as a single dictionary
        metrics = {
            'image_mse': image_mse,
            'image_ssim': image_ssim_value,
            'image_psnr': image_psnr_value,
            'patch_mse': patch_mse_list,
            'patch_ssim': patch_ssim_list,
            'patch_psnr': patch_psnr_list
        }

        # save the metrics to a npz file
        np.savez(os.path.join('/home/anonymous_user/Workspace/PYTHON/EncodingInformation/color_filter_array/recon_results', h_params['model_name']+'.npz'), **metrics)


    # calculate the mean ssim, psnr, and mse for full images
    mean_image_ssim = np.mean(all_image_ssim_list)
    mean_image_psnr = np.mean(all_image_psnr_list)
    mean_image_mse = np.mean(all_image_mse_list)

    # calculate the mean ssim, psnr, and mse for patches
    mean_patch_ssim = np.mean(all_patch_ssim_list)
    mean_patch_psnr = np.mean(all_patch_psnr_list)
    mean_patch_mse = np.mean(all_patch_mse_list)

    # bootstrap the ssim, psnr, and mse by sampling with replacement

    # Number of bootstrap samples
    n_bootstrap_samples = 100

    # Initialize lists to store bootstrap results for images
    bootstrap_image_ssims = []
    bootstrap_image_psnrs = []
    bootstrap_image_mses = []

    # Initialize lists to store bootstrap results for patches
    bootstrap_patch_ssims = []
    bootstrap_patch_psnrs = []
    bootstrap_patch_mses = []

    # Perform bootstrap sampling and calculate metrics for images
    for _ in range(n_bootstrap_samples):
        sample_indices = resample(np.arange(len(all_image_ssim_list)), replace=True)
        bootstrap_image_ssims.append(np.mean([all_image_ssim_list[i] for i in sample_indices]))
        bootstrap_image_psnrs.append(np.mean([all_image_psnr_list[i] for i in sample_indices]))
        bootstrap_image_mses.append(np.mean([all_image_mse_list[i] for i in sample_indices]))

    # Perform bootstrap sampling and calculate metrics for patches
    for _ in range(n_bootstrap_samples):
        sample_indices = resample(np.arange(len(all_patch_ssim_list)), replace=True)
        bootstrap_patch_ssims.append(np.mean([all_patch_ssim_list[i] for i in sample_indices]))
        bootstrap_patch_psnrs.append(np.mean([all_patch_psnr_list[i] for i in sample_indices]))
        bootstrap_patch_mses.append(np.mean([all_patch_mse_list[i] for i in sample_indices]))

    # Calculate confidence intervals for images
    bootstrap_image_ssims_ci = np.percentile(bootstrap_image_ssims, [2.5, 97.5])
    bootstrap_image_psnrs_ci = np.percentile(bootstrap_image_psnrs, [2.5, 97.5])
    bootstrap_image_mses_ci = np.percentile(bootstrap_image_mses, [2.5, 97.5])


    # Calculate confidence intervals for patches
    bootstrap_patch_ssims_ci = np.percentile(bootstrap_patch_ssims, [2.5, 97.5])
    bootstrap_patch_psnrs_ci = np.percentile(bootstrap_patch_psnrs, [2.5, 97.5])
    bootstrap_patch_mses_ci = np.percentile(bootstrap_patch_mses, [2.5, 97.5])

    # Save all the calculations and metrics to a single dictionary
    results = {
        'image_mse': mean_image_mse,
        'image_mse_ci': bootstrap_image_mses_ci,
        'image_ssim': mean_image_ssim,
        'image_ssim_ci': bootstrap_image_ssims_ci,
        'image_psnr': mean_image_psnr,
        'image_psnr_ci': bootstrap_image_psnrs_ci,
        'patch_mse': mean_patch_mse,
        'patch_mse_ci': bootstrap_patch_mses_ci,
        'patch_ssim': mean_patch_ssim,
        'patch_ssim_ci': bootstrap_patch_ssims_ci,
        'patch_psnr': mean_patch_psnr,
        'patch_psnr_ci': bootstrap_patch_psnrs_ci,
        'patch_list_mse': all_patch_mse_list,
        'patch_list_ssim': all_patch_ssim_list,
        'patch_list_psnr': all_patch_psnr_list,
        'image_list_mse': all_image_mse_list,
        'image_list_ssim': all_image_ssim_list,
        'image_list_psnr': all_image_psnr_list
    }
    

    # save the results to an npz file
    np.savez(os.path.join('/home/anonymous_user/Workspace/PYTHON/EncodingInformation/color_filter_array/recon_results', h_params['model_name']+'.npz'), **results)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <config_path> <gpu_index>")
        sys.exit(1)

    config_path = sys.argv[1]
    gpu_index = sys.argv[2]

    main(config_path, gpu_index)
