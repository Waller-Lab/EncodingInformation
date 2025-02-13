import equinox as eqx
import jax
from encoding_information import image_utils
import numpy as onp
import jax.numpy as jnp
from functools import partial
from encoding_information.models.model_base_class import MeasurementModel, make_dataset_generators
from encoding_information.models.gaussian_process import match_to_generator_data, generate_multivariate_gaussian_samples
import warnings

# Patching functions

@partial(jax.jit, static_argnames=('num_patches', 'patch_size'))
def _extract_random_patches(data, key, num_patches, patch_size):
    keys = jax.random.split(key, 3)
    image_indices = jax.random.randint(keys[0], shape=(num_patches,), 
                                     minval=0, maxval=data.shape[0])
    x_indices = jax.random.randint(keys[1], shape=(num_patches,), 
                                 minval=0, maxval=data.shape[1] - patch_size + 1)
    y_indices = jax.random.randint(keys[2], shape=(num_patches,), 
                                 minval=0, maxval=data.shape[2] - patch_size + 1)
    
    def get_patch(i):
        return jax.lax.dynamic_slice(
            data[image_indices[i]], 
            (x_indices[i], y_indices[i]), 
            (patch_size, patch_size)
        )
    
    return jax.vmap(get_patch)(jnp.arange(num_patches))

@partial(jax.jit, static_argnames=('num_patches', 'patch_size'))
def _extract_tiled_patches(data, key, num_patches, patch_size):
    num_tiles_x = data.shape[1] // patch_size
    num_tiles_y = data.shape[2] // patch_size
    
    keys = jax.random.split(key, 3)
    image_indices = jax.random.randint(keys[0], shape=(num_patches,), 
                                     minval=0, maxval=data.shape[0])
    x_tile_indices = jax.random.randint(keys[1], shape=(num_patches,), 
                                      minval=0, maxval=num_tiles_x)
    y_tile_indices = jax.random.randint(keys[2], shape=(num_patches,), 
                                      minval=0, maxval=num_tiles_y)
    
    def get_tile(i):
        return jax.lax.dynamic_slice(
            data[image_indices[i]], 
            (x_tile_indices[i] * patch_size, y_tile_indices[i] * patch_size), 
            (patch_size, patch_size)
        )
    
    return jax.vmap(get_tile)(jnp.arange(num_patches))

@partial(jax.jit, static_argnames=('num_patches', 'patch_size'))
def _extract_cropped_patches(data, key, num_patches, patch_size, crop_location):
    if crop_location is not None:
        y_index, x_index = crop_location
    else:
        key1, key2 = jax.random.split(key)
        x_index = jax.random.randint(key1, shape=(), 
                                   minval=0, maxval=data.shape[1] - patch_size + 1)
        y_index = jax.random.randint(key2, shape=(), 
                                   minval=0, maxval=data.shape[2] - patch_size + 1)
    
    patches = jax.lax.dynamic_slice(data, 
                                  (0, x_index, y_index), 
                                  (min(data.shape[0], num_patches), patch_size, patch_size))
    
    if num_patches > data.shape[0]:
        key3 = jax.random.split(key)[0]
        extra_indices = jax.random.randint(key3, shape=(num_patches - data.shape[0],), 
                                         minval=0, maxval=data.shape[0])
        patches = jnp.concatenate([patches, patches[extra_indices]])
    elif num_patches < data.shape[0]:
        key3 = jax.random.split(key)[0]
        indices = jax.random.permutation(key3, data.shape[0])[:num_patches]
        patches = patches[indices]
        
    return patches

@partial(jax.jit, static_argnames=('num_patches', 'patch_size', 'num_masked_pixels'))
def _extract_masked_patches(data, key, num_patches, patch_size, num_masked_pixels):
    data_size = data[0].size
    indices = jax.random.permutation(key, data_size)[:num_masked_pixels]
    
    def get_masked_data(img):
        return jnp.take(img.ravel(), indices)
    
    patches = jax.vmap(get_masked_data)(data)
    
    if num_patches > data.shape[0]:
        key2 = jax.random.split(key)[0]
        extra_indices = jax.random.randint(key2, shape=(num_patches - data.shape[0],), 
                                         minval=0, maxval=data.shape[0])
        patches = jnp.concatenate([patches, patches[extra_indices]])
    elif num_patches < data.shape[0]:
        key2 = jax.random.split(key)[0]
        sample_indices = jax.random.permutation(key2, data.shape[0])[:num_patches]
        patches = patches[sample_indices]
    
    if patch_size * patch_size == num_masked_pixels:
        patches = patches.reshape(num_patches, patch_size, patch_size)
    
    return patches

def extract_patches(data, key, num_patches=1000, patch_size=16, strategy='random', 
                   crop_location=None, num_masked_pixels=256, verbose=False) -> jnp.ndarray:
    """Extract patches from a dataset using various strategies, optimized for JAX."""
    strategies = {
        'random': lambda: _extract_random_patches(data, key, num_patches, patch_size),
        'uniform_random': lambda: _extract_random_patches(
            jnp.pad(data, 
                   ((0, 0), (patch_size, patch_size), 
                    (patch_size, patch_size)), 
                   mode='constant', 
                   constant_values=jnp.mean(data)),
            key, num_patches, patch_size
        ),
        'tiled': lambda: _extract_tiled_patches(data, key, num_patches, patch_size),
        'cropped': lambda: _extract_cropped_patches(data, key, num_patches, patch_size, crop_location),
        'masked': lambda: _extract_masked_patches(data, key, num_patches, patch_size, num_masked_pixels)
    }
    
    return strategies[strategy]()

# Noise functions

@jax.jit
def _add_noise_single(image, key, gaussian_sigma, ensure_positive):
    """Helper function to add noise to a single image."""
    if gaussian_sigma is not None:
        noisy_image = image + gaussian_sigma * jax.random.normal(key, shape=image.shape)
    else:
        noisy_image = image + jax.random.normal(key, shape=image.shape) * jnp.sqrt(jnp.maximum(image,1e-8))

    
    return jnp.where(ensure_positive, jnp.maximum(0, noisy_image), noisy_image)

def add_noise(images, ensure_positive=True, gaussian_sigma=None, key=None, seed=None, batch_size=None):
    """
    Add Poisson or Gaussian noise to a stack of images using JAX optimization.
    
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
        Deprecated. Included for backward compatibility but no longer needed.

    Returns
    -------
    ndarray
        Noisy images.
    """
    if seed is None:
        seed = onp.random.randint(0, 100000)
    if key is None:
        key = jax.random.PRNGKey(seed)
    
    images = images.astype(jnp.float32)
    
    # Create separate keys for each image
    keys = jax.random.split(key, images.shape[0])
    
    # Vectorize the noise addition operation across the batch dimension
    vectorized_noise = jax.vmap(_add_noise_single, in_axes=(0, 0, None, None))
    
    return vectorized_noise(images, keys, gaussian_sigma, ensure_positive)

def jax_crop2D(target_shape, mat):
    """
    Center crop the 2D or 3D input matrix to the target shape.

    Args:
        target_shape (tuple): Target shape for cropping.
        mat (np.array): Input matrix.

    Returns:
        onp.array: Cropped matrix.
    """
    y_margin = (mat.shape[-2] - target_shape[-2]) // 2
    x_margin = (mat.shape[-1] - target_shape[-1]) // 2
    if mat.ndim == 2:
        return mat[y_margin : -y_margin or None, x_margin : -x_margin or None]
    elif mat.ndim == 3:
        return mat[:, y_margin : -y_margin or None, x_margin : -x_margin or None]
    else:
        raise ValueError("jax_crop2D only supports 2D and 3D arrays.")
