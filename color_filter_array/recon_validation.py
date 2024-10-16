import os
import argparse
import yaml
import jax
import jax.numpy as jnp
import equinox as eqx
from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pickle

from utils import *
from recon import FullModel

def calculate_metrics(prediction, target):
    mse = jnp.mean((prediction - target) ** 2)
    ssim_value = ssim(prediction, target, data_range=1.0, channel_axis=-1)
    psnr_value = psnr(target, prediction, data_range=1.0)
    return mse, ssim_value, psnr_value

def evaluate_model(model, test_loader, h_params):
    mse_list, ssim_list, psnr_list = [], [], []
    scale_factor = h_params['mean_photons'] / 0.06824377 
    h_params['scale_factor'] = scale_factor
    noise_key = jax.random.PRNGKey(0)

    for images, targets in tqdm(test_loader, desc="Evaluating"):
        # Apply noise to images
        images = images * scale_factor
        noise_key, subkey = jax.random.split(noise_key)
        images = images + jax.random.normal(subkey, shape=images.shape) * jnp.sqrt(images)
        images = (images / scale_factor).clip(0, 1)

        # Generate predictions
        alpha = jnp.full((images.shape[0], 1), 1 + (h_params['gamma'] * h_params['num_steps']) ** 2)
        predictions = jax.vmap(model)(images, alpha)

        # Calculate metrics for each prediction-target pair
        for prediction, target in zip(predictions, targets):
            mse, ssim_value, psnr_value = calculate_metrics(np.array(prediction)*3, 3*np.array(target))
            mse_list.append(mse)
            ssim_list.append(ssim_value)
            psnr_list.append(psnr_value)

    return mse_list, ssim_list, psnr_list

def main(config_dirs, gpu_idx):
    # Set up GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

    results = []
    previous_h_params = None
    test_loader = None

    for config_dir in config_dirs:
        for config_file in os.listdir(config_dir):
            if config_file.endswith('.yaml'):
                config_path = os.path.join(config_dir, config_file)
                
                # Load config
                with open(config_path, 'r') as file:
                    h_params = yaml.safe_load(file)

                # Initialize the model
                key = jax.random.PRNGKey(h_params['key'])
                P, F, K = h_params['P'], h_params['F'], h_params['K']
                sensor_shape = (h_params['mask_size'], h_params['mask_size'], h_params['num_channels'])
                gamma = h_params['gamma']
                model = FullModel(sensor_shape, P, F, K, gamma, key)
                if h_params['mask_path']:
                    mask = np.load(h_params['mask_path'])
                    model = eqx.tree_at(lambda m: m.sensor_layer, model, replace=model.sensor_layer.update_w(mask))

                # Load the trained model
                model_dir = os.path.join("models", h_params['model_name'])
                if not os.path.exists(model_dir):
                    print(f"Model directory {model_dir} does not exist. Skipping config {config_file}.")
                    continue

                model_files = [f for f in os.listdir(model_dir) if f.endswith('.eqx')]
                if not model_files:
                    print(f"No model files found in {model_dir}. Skipping config {config_file}.")
                    continue

                latest_model = max(model_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
                if h_params['mask_path']:
                    mask = np.load(os.path.join(model_dir, latest_model))
                    model = eqx.tree_at(lambda m: m.sensor_layer, model, replace=model.sensor_layer.update_w(mask))
                model = eqx.tree_deserialise_leaves(os.path.join(model_dir, latest_model), model)

                # Create test loader only if parameters change
                if previous_h_params is None or h_params['data_path'] != previous_h_params['data_path'] or h_params['n_batch'] != previous_h_params['n_batch'] or h_params['val_size'] != previous_h_params['val_size']:
                    _, test_loader, _, _ = split_train_test_predetermined(
                        h_params['data_path'], 
                        h_params['n_batch'], 
                        (F, F), 
                        key, 
                        num_patches_per_image=10,
                        train_split=(1-h_params['val_size']),
                        num_workers=24
                    )
                    previous_h_params = h_params

                # Evaluate model
                mse_list, ssim_list, psnr_list = evaluate_model(model, test_loader, h_params)

                # Store results
                result = {
                    'Model': h_params['model_name'],
                    'MSE': mse_list,
                    'SSIM': ssim_list,
                    'PSNR': psnr_list,
                    'Mean Photons': h_params['mean_photons'],
                    'E2E': h_params['e2e'],
                    'Gamma': h_params['gamma'],
                    'Mask LR': h_params['mask_lr'],
                    'LR': h_params['lr'],
                    'Scale Factor': h_params['scale_factor']
                }

                # Save individual result to a pickle file
                result_filename = f"{h_params['model_name']}_evaluation_results.pkl"
                save_path = '/home/emarkley/Workspace/PYTHON/EncodingInformation/color_filter_array/recon_results'
                os.makedirs(save_path, exist_ok=True)
                with open(os.path.join(save_path, result_filename), 'wb') as f:
                    pickle.dump(result, f)
                print(f"Results for {h_params['model_name']} saved to {result_filename}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate reconstruction models")
    parser.add_argument("--config_dirs", nargs='+', default=['e2e_configs', 'recon_configs'], help="Directories containing config files")
    parser.add_argument("--gpu_idx", type=int, default=0, help="GPU index to use")
    args = parser.parse_args()

    main(args.config_dirs, args.gpu_idx)