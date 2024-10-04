
"""
Functions for processing images in useful ways extracting patches, adding synthetic noise, etc.
"""
import numpy as onp
from tqdm import tqdm
from cleanplots import *
import jax
import jax.numpy as np


def _extract_random_patches(data, num_patches, patch_size, verbose):
    """
    Extract random patches from a dataset, used by both random and uniform_random strategies.
    
    Parameters
    ----------
    data : ndarray
        Input dataset to extract patches from.
    num_patches : int
        The number of patches to extract.
    patch_size : int
        Size of each patch.
    verbose : bool
        If True, show progress with tqdm.

    Returns
    -------
    ndarray
        Array of extracted patches.
    """
    image_indices = jax.random.randint(key=jax.random.PRNGKey(onp.random.randint(100000)), minval=0, maxval=data.shape[0], shape=(num_patches,))
    x_indices = jax.random.randint(key=jax.random.PRNGKey(onp.random.randint(100000)), minval=0, maxval=data.shape[1] - patch_size + 1, shape=(num_patches,))
    y_indices = jax.random.randint(key=jax.random.PRNGKey(onp.random.randint(100000)), minval=0, maxval=data.shape[2] - patch_size + 1, shape=(num_patches,))
        
    iterator = tqdm(range(num_patches)) if verbose else range(num_patches)
    patches = [data[image_indices[i], x_indices[i]:x_indices[i]+patch_size, y_indices[i]:y_indices[i]+patch_size] for i in iterator]
    
    return np.array(patches)



def extract_patches(data, num_patches=1000, patch_size=16, strategy='random', crop_location=None,
                    num_masked_pixels=256, seed=None, verbose=False) -> np.ndarray:
    """
    Extract patches from a dataset using various strategies.

    Parameters
    ----------
    data : ndarray
        Input data from which to extract patches. Can have shapes (N, W, H), (N, W, H, C), or (N, D).
    num_patches : int, optional
        Number of patches to extract.
    patch_size : int, optional
        Size of the square patches.
    strategy : str, optional
        Strategy for patch extraction ('random', 'uniform_random', 'tiled', 'cropped', 'masked').
    crop_location : tuple, optional
        Top-left corner of the patch for 'cropped' strategy. If None, a random location is chosen.
    num_masked_pixels : int, optional
        Number of pixels to mask in the 'masked' strategy.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        If True, show progress during patch extraction.

    Returns
    -------
    ndarray
        Extracted patches.
    """
    
    valid_strategies = ['random', 'uniform_random', 'tiled', 'cropped', 'masked']
    if strategy not in valid_strategies:
        raise ValueError(f'Invalid patch extraction strategy, must be one of: {valid_strategies}')

    if seed is not None:
        onp.random.seed(seed)

    has_channels = len(data.shape) == 4
    patches = []

    if strategy == 'random':
        patches = _extract_random_patches(data, num_patches, patch_size, verbose)
        
    elif strategy == 'uniform_random':
        pad_width = ((0, 0), (patch_size, patch_size), (patch_size, patch_size), (0, 0)) if has_channels else ((0, 0), (patch_size, patch_size), (patch_size, patch_size))
        padded_data = np.pad(data, pad_width, mode='constant', constant_values=np.mean(data))
        patches = _extract_random_patches(padded_data, num_patches, patch_size, verbose)

    elif strategy == 'tiled':
        num_tiles_x = data.shape[1] // patch_size
        num_tiles_y = data.shape[2] // patch_size
        image_indices = jax.random.randint(key=jax.random.PRNGKey(onp.random.randint(100000)), minval=0, maxval=data.shape[0], shape=(num_patches,))
        x_tile_indices = jax.random.randint(key=jax.random.PRNGKey(onp.random.randint(100000)), minval=0, maxval=num_tiles_x, shape=(num_patches,))
        y_tile_indices = jax.random.randint(key=jax.random.PRNGKey(onp.random.randint(100000)), minval=0, maxval=num_tiles_y, shape=(num_patches,))

        iterator = tqdm(range(num_patches)) if verbose else range(num_patches)
        patches = [
            data[image_indices[i], x_tile_indices[i] * patch_size:(x_tile_indices[i] + 1) * patch_size,
                 y_tile_indices[i] * patch_size:(y_tile_indices[i] + 1) * patch_size]
            for i in iterator
        ]

    elif strategy == 'cropped':
        if crop_location is not None:
            y_index, x_index = crop_location
        else:
            x_index = onp.random.randint(0, data.shape[1] - patch_size + 1)
            y_index = onp.random.randint(0, data.shape[2] - patch_size + 1)
        if num_patches > data.shape[0]:
            raise ValueError("Number of patches exceeds number of images.")
        patches = data[:, x_index:x_index+patch_size, y_index:y_index+patch_size]
        if num_patches < data.shape[0]:
            patches = patches[onp.random.choice(data.shape[0], num_patches, replace=False)]

    elif strategy == 'masked':
        data_size = data[0].size
        mask = onp.zeros(data_size)
        random_indices = onp.random.choice(data_size, num_masked_pixels, replace=False)
        mask[random_indices] = 1
        if num_patches > data.shape[0]:
            raise ValueError("Number of patches exceeds number of images.")
        patches = data.reshape(data.shape[0], -1)[:, mask.astype(bool)]
        if num_patches < data.shape[0]:
            patches = patches[onp.random.choice(data.shape[0], num_patches, replace=False)]

    return np.array(patches)

def add_noise(images, ensure_positive=True, gaussian_sigma=None, key=None, seed=None, batch_size=1000):
    """
    Add Poisson or Gaussian noise to a stack of images.
    
    Parameters
    ----------
    images : ndarray
        A stack of images (NxHxW) or patches (Nx(num pixels)).
    ensure_positive : bool, optional
        Whether to ensure all resulting pixel values are non-negative.
    gaussian_sigma : float, optional
        Standard deviation for Gaussian noise. If None, Poisson noise is added.
    key : jax.random.PRNGKey, optional
        PRNGKey for generating noise. If None, a key is generated based on the seed.
    seed : int, optional
        Seed for generating noise, if no key is provided.
    batch_size : int, optional
        Number of images to process in batches.

    Returns
    -------
    ndarray
        Noisy images.
    """
    if seed is None: 
        seed = onp.random.randint(0, 100000)
    if key is None:
        key = jax.random.PRNGKey(seed)
    images = images.astype(np.float32)

    num_images = images.shape[0]
    num_batches = int(np.ceil(num_images / batch_size))
    batches = np.array_split(images, num_batches)

    noisy_batches = []
    for batch in batches:
        if gaussian_sigma is not None:
            noisy_batch = batch + jax.random.normal(key, shape=batch.shape) * gaussian_sigma
        else:
            noisy_batch = jax.random.poisson(key, shape=batch.shape, lam=batch).astype(np.float32)
        key = jax.random.split(key)[1]
        if ensure_positive:
            noisy_batch = np.where(noisy_batch < 0, 0, noisy_batch)
        noisy_batches.append(noisy_batch)

    return np.concatenate(noisy_batches, axis=0)


def normalize_image_stack(stack):
    """
    Rescale pixel values to normalize the average energy across images.
    
    Parameters
    ----------
    stack : ndarray
        Stack of images to normalize.

    Returns
    -------
    ndarray
        Normalized image stack.
    """
    average_energy_per_pixel = np.sum(stack.astype(float)) / np.prod(stack.size)
    return stack.astype(float) / average_energy_per_pixel