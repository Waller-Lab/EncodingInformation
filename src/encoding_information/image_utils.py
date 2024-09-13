
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

    """
    image_indices = jax.random.randint(key=jax.random.PRNGKey(onp.random.randint(100000)), minval=0, maxval=data.shape[0], shape=(num_patches,))
    x_indices = jax.random.randint(key=jax.random.PRNGKey(onp.random.randint(100000)), minval=0, maxval=data.shape[1] - patch_size + 1, shape=(num_patches,))
    y_indices = jax.random.randint(key=jax.random.PRNGKey(onp.random.randint(100000)), minval=0, maxval=data.shape[2] - patch_size + 1, shape=(num_patches,))
        
    if verbose:
        iterator = tqdm(range(num_patches))
    else:
        iterator = range(num_patches)
    patches = []
    for i in iterator:
        patches.append(data[image_indices[i], x_indices[i]:x_indices[i]+patch_size, y_indices[i]:y_indices[i]+patch_size])
    return np.array(patches)


def extract_patches(data, num_patches=1000, patch_size=16, strategy='random', num_masked_pixels=256, seed=None, verbose=False):
    """
    Extract patches from a dataset using various strategies.
    
    Parameters
    ----------
    data : ndarray
        Input data from which to extract patches. The input can have one of the following shapes:
        - (N, W, H): N images of size W x H.
        - (N, W, H, C): N images of size W x H with C channels.
        - (N, D): N data points of D-dimensional vectors (for random masking).
    
    num_patches : int, optional
        Number of patches to extract. 
    patch_size : int, optional
        Size of the square patches. 
    
    strategy : str, optional
        The strategy for patch extraction. Options are:
        - 'random': randomly sample patches from the data.
        - 'uniform_random': sample patches randomly but with padding, ensuring all original pixels 
          have equal probability of being included in a patch. This will result in many patches with
            constant values on one or two edges
        - 'tiled': split the data into a grid of non-overlapping patches.
        - 'cropped': select a single random location and extract patches centered on that location.
        - 'masked': use a random mask to extract patches. In this case, the patch is returned as a flattened vector.
    
    num_masked_pixels : int, optional
        Number of pixels to mask in the 'masked' strategy.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        If True, show progress during patch extraction. 

    Returns
    -------
    patches : ndarray
        The extracted patches. The shape of the returned patches depends on the input data shape and strategy:
        - For (N, W, H) or (N, W, H, C), patches will have shape (num_patches, patch_size, patch_size) or 
          (num_patches, patch_size, patch_size, C), respectively.
        - For 'masked' strategy with the data will be vectorized into an (N, D) array, patches will have shape
            (num_patches, num_masked_pixels).
    """
    
    valid_strategies = ['random', 'uniform_random', 'tiled', 'cropped', 'masked']
    if strategy not in valid_strategies:
        raise ValueError('Invalid patch extraction strategy, must be one of: {}'.format(valid_strategies))

    if seed is not None:
        onp.random.seed(seed)

    patches = []

    # Determine if the input data has channels or not
    has_channels = (len(data.shape) == 4)

    if strategy == 'random':
        patches = _extract_random_patches(data, num_patches, patch_size, verbose)
        
    elif strategy == 'uniform_random':
        pad_width = ((0, 0), (patch_size, patch_size), (patch_size, patch_size)) if not has_channels else ((0, 0), (patch_size, patch_size), (patch_size, patch_size), (0, 0))
        padded_data = np.pad(data, pad_width, mode='constant', constant_values=np.mean(data))
        patches =  _extract_random_patches(padded_data, num_patches, patch_size, verbose)

    elif strategy == 'tiled':
        num_tiles_x = data.shape[1] // patch_size
        num_tiles_y = data.shape[2] // patch_size
        # sample num_patches random tiles from random images
        image_indices = jax.random.randint(key=jax.random.PRNGKey(onp.random.randint(100000)), minval=0, maxval=data.shape[0], shape=(num_patches,))
        x_tile_indices = jax.random.randint(key=jax.random.PRNGKey(onp.random.randint(100000)), minval=0, maxval=num_tiles_x, shape=(num_patches,))
        y_tile_indices = jax.random.randint(key=jax.random.PRNGKey(onp.random.randint(100000)), minval=0, maxval=num_tiles_y, shape=(num_patches,))

        if verbose:
            iterator = tqdm(range(num_patches))
        else:
            iterator = range(num_patches)

        patches = []
        for i in iterator:
            image_index = image_indices[i]
            x_tile_index = x_tile_indices[i]
            y_tile_index = y_tile_indices[i]
            x_index = x_tile_index * patch_size
            y_index = y_tile_index * patch_size
            patches.append(data[image_index, x_index:x_index+patch_size, y_index:y_index+patch_size])

    elif strategy == 'cropped':
        x_index = onp.random.randint(0, data.shape[1] - patch_size + 1)
        y_index = onp.random.randint(0, data.shape[2] - patch_size + 1)
        # throw error if patch count exceeds image count
        if num_patches > data.shape[0]:
            raise ValueError("Number of patches exceeds number of images, which is not possible with cropped strategy")
        patches = data[:, x_index:x_index+patch_size, y_index:y_index+patch_size]
        # take a random subset, if necessary
        if num_patches < data.shape[0]:
            patches = patches[onp.random.choice(data.shape[0], num_patches, replace=False)]

    elif strategy == 'masked':
        # generate a random mask
        data_size = data[0].size
        mask = onp.zeros(data_size)
        random_indices = onp.random.choice(data_size, num_masked_pixels, replace=False)
        mask[random_indices] = 1
        # throw error if patch count exceeds image count
        if num_patches > data.shape[0]:
            raise ValueError("Number of patches exceeds number of images, which is not possible with masked strategy")
        patches = data.reshape(data.shape[0], -1)[:, mask.astype(bool)]
        # take a random subset, if necessary
        if num_patches < data.shape[0]:
            patches = patches[onp.random.choice(data.shape[0], num_patches, replace=False)]

    return np.array(patches)


def add_noise(images, ensure_positive=True, gaussian_sigma=None, key=None, seed=None, batch_size=1000):
    """
    Add poisson noise or additive Gaussian noise to a stack of noiseless images. This uses jax to speed performance

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


def normalize_image_stack(stack):
    """
    rescale pixel values to account for the fact that different images have different average energies
    (this also rescales noise)
    """
    # scale to account for different energy 
    average_energy_per_pixel = np.sum(stack.astype(float)) / np.prod(stack.size)
    stack = stack.astype(float) / average_energy_per_pixel
    return stack
