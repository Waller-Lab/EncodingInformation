
"""
Functions for processing images in useful ways extracting patches, adding synthetic noise, etc.
"""

import numpy as onp
from tqdm import tqdm
from cleanplots import *
import jax
import jax.numpy as np


def add_noise(images, ensure_positive=True, gaussian_sigma=None, key=None, seed=None, batch_size=1000):
    """
    Add poisson noise or additive Gaussian noise to a stack of noiseless images

    images : ndarray NxHxW or Nx(num pixels) array of images or image patches
    ensure_positive : bool, whether to ensure the noisy images are nonnegative
    gaussian_sigma : float, if not None, add IID gaussian noise with this sigma. Otherwise add poisson noise.
    key : jax.random.PRNGKey, if not None, use this key to generate the noise. Otherwise, generate a key based on the seed
    seed: int, if key is None, use this seed to generate the key
    batch_size : int, if the number of images is large, split them into batches to avoid memory issues
    """
    if seed is None: 
        seed = onp.random.randint(0, 100000)
    if key is None:
        key = jax.random.PRNGKey(seed)
    images = images.astype(np.float32)

    # Split the images into batches
    num_images = images.shape[0]
    num_batches = int(np.ceil(num_images / batch_size))
    batches = np.array_split(images, num_batches)

    # Add noise to each batch
    noisy_batches = []
    for batch in batches:
        if gaussian_sigma is not None:
            # Additive gaussian
            noisy_batch = batch + jax.random.normal(key, shape=batch.shape) * gaussian_sigma
        else:
            # Poisson
            noisy_batch = jax.random.poisson(key, shape=batch.shape, lam=batch).astype(np.float32)
        key = jax.random.split(key)[1]
        if ensure_positive:
            noisy_batch = np.where(noisy_batch < 0, 0, noisy_batch)
        noisy_batches.append(noisy_batch)

    # Concatenate the noisy batches back into a single array
    noisy_images = np.concatenate(noisy_batches, axis=0)
    return noisy_images

def extract_patches(stack, patch_size, num_patches=1000, seed=None, verbose=False, return_locations=False):
    patches = []
    if seed is not None:
        onp.random.seed(seed)
    # image_indices = onp.random.randint(0, stack.shape[0], num_patches)
    # x_indices = onp.random.randint(0, stack.shape[1] - patch_size + 1, num_patches)
    # y_indices = onp.random.randint(0, stack.shape[2] - patch_size + 1, num_patches)
    
    image_indices = jax.random.randint(key=jax.random.PRNGKey(onp.random.randint(100000)), minval=0, maxval=stack.shape[0], shape=(num_patches,))
    x_indices = jax.random.randint(key=jax.random.PRNGKey( onp.random.randint(100000)), minval=0, maxval=stack.shape[1] - patch_size + 1, shape=(num_patches,))
    y_indices = jax.random.randint(key=jax.random.PRNGKey( onp.random.randint(100000)), minval=0, maxval=stack.shape[2] - patch_size + 1, shape=(num_patches,))
    center_locations = np.stack([x_indices + patch_size // 2, y_indices + patch_size // 2], axis=1)
    if verbose:
        iterator = tqdm(range(num_patches))
    else:
        iterator = range(num_patches)
    for i in iterator:
        patches.append(stack[image_indices[i], x_indices[i]:x_indices[i]+patch_size, y_indices[i]:y_indices[i]+patch_size])
    if return_locations:
        return jax.numpy.array(patches), center_locations.astype(np.float32)
    return jax.numpy.array(patches)

def normalize_image_stack(stack):
    """
    rescale pixel values to account for the fact that different images have different average energies
    (this also rescales noise)
    """
    # scale to account for different energy 
    average_energy_per_pixel = np.sum(stack.astype(float)) / np.prod(stack.size)
    stack = stack.astype(float) / average_energy_per_pixel
    return stack
