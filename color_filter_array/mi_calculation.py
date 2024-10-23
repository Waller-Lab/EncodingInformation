import os
import yaml
import jax
import jax.random as random
import jax.numpy as jnp
from utils import *

def load_config(config_file):
    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data

def run_calculations(config_file):
    # Load parameters from config file
    config_data = load_config(config_file)

    # Dataset parameters
    zarr_path = config_data['dataset']['zarr_path']
    mean_photons = config_data['dataset']['mean_photons']
    data_mean_path = config_data['dataset']['data_mean_path']
    total_patches = config_data['dataset']['total_patches']
    val_size = config_data['dataset']['val_size']
    model_path = config_data['dataset']['model_path']
    patch_size = config_data['dataset']['patch_size']
    data_key = config_data['dataset']['data_key']
    data_key = jax.random.PRNGKey(data_key)
    model_key = config_data['dataset']['model_key']
    model_key = jax.random.PRNGKey(model_key)
    batch_size = config_data['dataset']['batch_size']
    key = jax.random.PRNGKey(config_data['seeds']['data_seed'])

    # Training parameters
    epochs = config_data['training']['epochs']
    N_models = config_data['training']['N_models']
    num_bootstrap_samples = config_data['training']['num_bootstrap_samples']
    data_seed = config_data['seeds']['data_seed']

    # Output parameters
    results_dir = config_data['output']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    save_name = config_data['output']['save_name']

    # Split train and test data
    test_loader, test_indices = create_patch_loader(
            zarr_path = os.path.join(zarr_path, 'test'), 
            batch_size=batch_size, 
            patch_size=(patch_size, patch_size),
            key=data_key, 
            total_patches=total_patches,
            num_workers=24
    )

    mean_last_channel = np.load(data_mean_path)
    scale_factor = mean_photons / mean_last_channel
    config_data['dataset']['scale_factor'] = scale_factor

    # Define the model
    sensor_model = SensorLayer((patch_size // 3, patch_size // 3), key=model_key)
    sensor_pattern = np.load(model_path)
    sensor_pattern = jnp.array(sensor_pattern, dtype=jnp.float64)
    sensor_model = sensor_model.update_w(sensor_pattern)

    # Collect the dataset
    dataset = []
    for images, _ in test_loader:
        measurements = jax.vmap(sensor_model)(images, 1e8 * jnp.ones((images.shape[0],))) * scale_factor
        dataset.append(measurements)
    dataset = np.concatenate(dataset, axis=0)

    # Squeeze the dataset
    dataset = dataset.squeeze()

    # Add Poisson noise to the dataset
    noisy_dataset = dataset + jax.random.poisson(key, dataset, shape=dataset.shape)

    # Split the noisy dataset into train and test sets
    train_set = noisy_dataset[:int(len(noisy_dataset) * (1 - val_size))]
    test_set = noisy_dataset[int(len(noisy_dataset) * (1 - val_size)):]

    # Initialize empty lists and variables
    models = []
    nlls = []
    partial_test_set_nlls_by_model = []
    best_model = None
    best_average_nll = float('inf')  # Initialize with a large value

    # Prepare test set sizes only once
    test_set_sizes = np.linspace(10, test_set.shape[0], 20).astype(int)
    
    # Loop through model training and save after each iteration
    for i in tqdm(range(N_models)):
        model_seed = i
        model = PixelCNN()
        

        # Train model
        model.fit(train_set, model_seed=model_seed, data_seed=data_seed, max_epochs=epochs, steps_per_epoch=1000, verbose=False, learning_rate=1e-3, patience=5)
        models.append(model)

        # Compute NLL and store it
        nll = model.compute_negative_log_likelihood(test_set, data_seed=data_seed, average=False)
        nlls.append(nll)

        # Calculate partial NLLs for the current model
        partial_test_set_nlls = [np.mean(nll[:size]) for size in test_set_sizes]
        partial_test_set_nlls_by_model.append(partial_test_set_nlls)

        # Calculate the average NLL for all trained models so far
        average_nlls = np.mean(np.array(nlls), axis=1)

        # Check if the current model has the lowest average NLL
        current_best_index = np.argmin(average_nlls)
        current_best_nll = average_nlls[current_best_index]

        if current_best_nll < best_average_nll:
            print(f"New best model found at iteration {i}")
            # If a new best model is found, update the best model and recalculate all related values
            best_model = models[current_best_index]
            best_average_nll = current_best_nll

            # Bootstrap NLL distribution for the best model to calculate confidence intervals
            mean_nll_dist_by_size = []
            for size in test_set_sizes:
                means = [onp.random.choice(nlls[current_best_index], size=size, replace=True).mean() for _ in range(num_bootstrap_samples)]
                mean_nll_dist_by_size.append(means)

            # Convert the list to a NumPy array
            mean_nll_dist_by_size = np.array(mean_nll_dist_by_size)

            # Calculate confidence intervals on the mean NLL
            conf_low = np.percentile(mean_nll_dist_by_size, 2.5, axis=1)
            conf_high = np.percentile(mean_nll_dist_by_size, 97.5, axis=1)

            # Calculate mutual information for the new best model
            noise_model = PoissonNoiseModel()
            mi, lb, up = estimate_information(best_model, noise_model, train_set, test_set, confidence_interval=.95, num_bootstraps=num_bootstrap_samples)

            # Filter matrix (assuming static)
            filter_matrix = sensor_pattern

        # Save the current state after each model training
        results = {
            'average_nlls': average_nlls,
            'partial_test_set_nlls_by_model': partial_test_set_nlls_by_model,
            'test_set_sizes': test_set_sizes,
            'mean_nll_dist_by_size': mean_nll_dist_by_size,
            'mean_nll_dist_by_size_conf_low': conf_low,
            'mean_nll_dist_by_size_conf_high': conf_high,
            'mutual_information': mi,
            'mutual_information_conf_low': lb,
            'mutual_information_conf_high': up,
            'filter_matrix': filter_matrix
        }

        # Save after each iteration
        np.savez(os.path.join(results_dir, f"tuned_model_{save_name}"), **results)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python calculation.py <config_file> <gpu_index>")
        sys.exit(1)

    config_file = sys.argv[1]
    gpu_index = sys.argv[2]

    from jax import config
    config.update("jax_enable_x64", True)

    from encoding_information.gpu_utils import limit_gpu_memory_growth
    limit_gpu_memory_growth()
    # Set GPU device based on index provided
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    import jax.numpy as np
    import numpy as onp
    from tqdm import tqdm
    from encoding_information.information_estimation import *
    from encoding_information.image_utils import *
    from encoding_information.datasets.cfa_dataset import ColorFilterArrayDataset
    from encoding_information import extract_patches
    from encoding_information.models import PoissonNoiseModel

    run_calculations(config_file)
