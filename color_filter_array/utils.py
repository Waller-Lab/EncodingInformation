import os
import jax
import jax.numpy as jnp
import equinox as eqx
import zarr
import random
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import jax.tree_util as jtu
from typing import Any, List, Tuple

# Define the sensor layer
class SensorLayer(eqx.Module):
    w: jnp.ndarray  # Learnable parameter for the multiplexing pattern
    replicates: int  # Number of replicates in the multiplexing pattern

    def __init__(self, shape, replicates = 3, key=None):
        key = jax.random.PRNGKey(0) if key is None else key
        self.w = jax.random.uniform(key, shape, minval=-1, maxval=1)
        self.replicates = replicates

    def __call__(self, x, alpha):
        # Apply the softmax to the mask
        I = jnp.tile(jax.nn.softmax(alpha*self.w, axis=-1), (self.replicates, self.replicates, 1))
        # Apply the mask to the image
        s = jnp.sum(I * x, axis=-1)
        return s[None, ...]  # Add channel dimension for compatibility
    
    def update_w(self, w):
        return eqx.tree_at(lambda c: c.w, self, w)
    
    
# Define the reconstruction network
class InterpolationPath(eqx.Module):
    log_shift: float
    fc: eqx.nn.Linear
    conv: eqx.nn.Conv2d
    P: int
    K: int

    def __init__(self, P, K, key):
        self.log_shift = 1e-8
        self.P = P
        self.K = K
        fc_key, conv_key = jax.random.split(key)
        self.fc = eqx.nn.Linear(
            3 * P * 3 * P, P * P * 3 * K, key=fc_key, use_bias=False
        )
        self.fc = eqx.tree_at(
            lambda c: c.weight,
            self.fc,
            replace=jax.random.normal(fc_key, self.fc.weight.shape) * 0.001,
        )
        self.conv = eqx.nn.Conv2d(3 * K, 3 * K, kernel_size=(1, 1), key=conv_key)
        self.conv = eqx.tree_at(
            lambda c: c.weight,
            self.conv,
            replace=jax.random.normal(conv_key, self.conv.weight.shape) * 0.001,
        )
        self.conv = eqx.tree_at(
            lambda c: c.bias, self.conv, replace=jnp.zeros(self.conv.bias.shape)
        )

    def __call__(self, sensor):
        log_sensor = jnp.log(sensor + self.log_shift).flatten()
        fc_out = self.fc(log_sensor)
        exp_out = jnp.exp(fc_out)
        reshape_out = jnp.reshape(exp_out, (3 * self.K, self.P, self.P))
        conv_out = self.conv(reshape_out)
        return conv_out

class SelectorPath(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    fc: eqx.nn.Linear
    relu: jax.nn.relu
    P: int
    F: int
    K: int

    def __init__(self, P, F, K, key):
        self.P = P
        self.F = F
        self.K = K
        conv1_key, conv2_key, conv3_key, fc_key = jax.random.split(key, 4)

        self.conv1 = eqx.nn.Conv2d(
            1, F, kernel_size=(P, P), stride=(P, P), key=conv1_key
        )
        self.conv1 = eqx.tree_at(
            lambda c: c.weight,
            self.conv1,
            replace=jax.random.normal(conv1_key, self.conv1.weight.shape) * 0.001,
        )
        self.conv1 = eqx.tree_at(
            lambda c: c.bias, self.conv1, replace=jnp.zeros(self.conv1.bias.shape)
        )

        self.conv2 = eqx.nn.Conv2d(F, F, kernel_size=(2, 2), key=conv2_key)
        self.conv2 = eqx.tree_at(
            lambda c: c.weight,
            self.conv2,
            replace=jax.random.normal(conv2_key, self.conv2.weight.shape) * 0.001,
        )
        self.conv2 = eqx.tree_at(
            lambda c: c.bias, self.conv2, replace=jnp.zeros(self.conv2.bias.shape)
        )

        self.conv3 = eqx.nn.Conv2d(F, F, kernel_size=(2, 2), key=conv3_key)
        self.conv3 = eqx.tree_at(
            lambda c: c.weight,
            self.conv3,
            replace=jax.random.normal(conv3_key, self.conv3.weight.shape) * 0.001,
        )
        self.conv3 = eqx.tree_at(
            lambda c: c.bias, self.conv3, replace=jnp.zeros(self.conv3.bias.shape)
        )

        self.fc = eqx.nn.Linear(F, P * P * 3 * K, key=fc_key)
        self.fc = eqx.tree_at(
            lambda c: c.weight,
            self.fc,
            replace=jax.random.normal(fc_key, self.fc.weight.shape) * 0.001,
        )
        self.fc = eqx.tree_at(
            lambda c: c.bias, self.fc, replace=jnp.zeros(self.fc.bias.shape)
        )

        self.relu = lambda x: jax.nn.relu(x)

    def __call__(self, sensor):
        x = self.conv1(sensor)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x).flatten()
        x = self.fc(x).reshape((3, self.P, self.P, self.K))
        return x

class ReconstructionModel(eqx.Module):
    interpolation_path: InterpolationPath
    selector_path: SelectorPath
    P: int
    K: int

    def __init__(self, P, F, K, key):
        self.P = P
        self.K = K
        interp_key, select_key = jax.random.split(key)
        self.interpolation_path = InterpolationPath(P, K, interp_key)
        self.selector_path = SelectorPath(P, F, K, select_key)

    def __call__(self, sensor):
        sensor = sensor.clip(0, 1)
        interp_out = self.interpolation_path(sensor).reshape(3, self.P, self.P, self.K)
        select_out = self.selector_path(sensor).reshape(3, self.P, self.P, self.K)
        combined = interp_out * select_out
        return jnp.sum(combined, axis=-1).clip(0, 1).transpose((1, 2, 0))
    
class FullModel(eqx.Module):
    sensor_layer: SensorLayer
    reconstruction_model: ReconstructionModel
    gamma: float
    iteration: int 

    def __init__(self, sensor_shape, P, F, K, gamma=1e-3, key=None):
        key = jax.random.PRNGKey(0) if key is None else key
        subkeys = jax.random.split(key, 3)
        self.sensor_layer = SensorLayer(sensor_shape, key=subkeys[0])
        self.reconstruction_model = ReconstructionModel(P, F, K, subkeys[1])
        self.gamma = gamma
        self.iteration = 0

    def __call__(self, image, alpha):
        # Apply sensor layer
        # alpha = 1 + (self.gamma * self.iteration) ** 2
        sensor_output = self.sensor_layer(image, alpha)
        # Reconstruction model expects sensor_output with shape [H, W, 1]
        reconstructed = self.reconstruction_model(sensor_output)
        return reconstructed.clip(0, 1/3)
    
class PredeterminedPatchLoader:
    def __init__(self, zarr_path, batch_size, patch_size, key, indices=None, num_workers=24):
        self.dataset = zarr.open(zarr_path, mode='r')
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        
        self.patch_indices = indices if indices is not None else []
        self.key = key
        self.shuffle_indices()
        
        self.current_idx = 0
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def shuffle_indices(self):
        self.key, subkey = jax.random.split(self.key)
        self.patch_indices = jax.random.permutation(subkey, jnp.array(self.patch_indices)).tolist()

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.patch_indices):
            raise StopIteration

        batch_indices = self.patch_indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size

        patches = list(self.executor.map(self._load_patch, batch_indices))
        images = jnp.array(patches) / 3.0

        h, w = images.shape[1], images.shape[2]
        targets = images[:, h // 3:2 * h // 3, w // 3:2 * w // 3, :3]

        return images, targets

    def _load_patch(self, indices):
        img_idx, top, left = indices
        ph, pw = self.patch_size
        return self.dataset[img_idx, top:top+ph, left:left+pw, :]

    def get_indices(self):
        return self.patch_indices

def generate_patch_indices(dataset_shape: Tuple[int, int, int, int], 
                           patch_size: Tuple[int, int], 
                           total_patches: int, 
                           key: Any) -> List[Tuple[int, int, int]]:
    indices = []
    num_images = dataset_shape[0]
    num_patches_per_image = -(-total_patches // num_images)  # Ceiling division
    for img_idx in range(num_images):
        for _ in range(num_patches_per_image):
            key, subkey = jax.random.split(key)
            top = jax.random.randint(subkey, (), 0, dataset_shape[1] - patch_size[0])
            key, subkey = jax.random.split(key)
            left = jax.random.randint(subkey, (), 0, dataset_shape[2] - patch_size[1])
            indices.append((img_idx, int(top), int(left)))
    return indices

def create_patch_loader(zarr_path: str, 
                        batch_size: int, 
                        patch_size: Tuple[int, int], 
                        key: Any, 
                        total_patches: int, 
                        num_workers: int = 24) -> Tuple[PredeterminedPatchLoader, List[Tuple[int, int, int]]]:
    dataset = zarr.open(zarr_path, mode='r')
    dataset_shape = dataset.shape

    # Generate all patch indices
    all_indices = generate_patch_indices(dataset_shape, patch_size, total_patches, key)

    # Shuffle indices
    key, subkey = jax.random.split(key)
    all_indices = jax.random.permutation(subkey, jnp.array(all_indices)).tolist()

    # Create data loader
    patch_loader = PredeterminedPatchLoader(
        zarr_path, batch_size, patch_size, key, indices=all_indices, num_workers=num_workers
    )

    return patch_loader, all_indices

def load_mask(path):
    # Load the mask from a file (e.g., .npy file)
    return np.load(path)

def generate_bayer_filter():
    """
    Generate an 8x8x4 Bayer filter with RGGBW pattern using np.tile.
    
    Returns
    -------
    np.ndarray
        The generated Bayer filter.
    """
    # Define the base RGGBW pattern (2x2)
    base_pattern = np.array([
        [[1, 0, 0, 0], [0, 1, 0, 0]],  # Red, Green
        [[0, 1, 0, 0], [0, 0, 1, 0]]   # Green, Blue
    ])
    
    # Use np.tile to create an 8x8 filter by repeating the base pattern
    bayer_filter = np.tile(base_pattern, (4, 4, 1))  # 4x4 repetition
    
    return bayer_filter

def generate_random_filter():
    """
    Generate an 8x8x4 random filter with one-hot encoding based on integers 0-3.
    
    Returns
    -------
    np.ndarray
        The generated random filter.
    """
    # Generate an 8x8 filter of integers 0-3 inclusive
    random_integers = np.random.randint(0, 4, (8, 8))
    
    # One-hot encode the integers to create an 8x8x4 binary filter
    random_filter = np.eye(4)[random_integers]
    
    return random_filter

# Function to create filter spec
def create_filter_spec(model: eqx.Module, unlearnable_params: List[Any]) -> Any:
    unlearnable_ids = {id(param) for param in unlearnable_params}

    def filter_fn(param):
        # Check if the parameter is in the set of unlearnable parameters
        if id(param) in unlearnable_ids:
            return False  # Non-learnable
        return eqx.is_array(param)  # Learnable if it's an array

    # Recursively apply the filter function
    return jtu.tree_map(filter_fn, model)