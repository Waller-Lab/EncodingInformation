
"""
Functions for processing images in useful ways
    extracting patches, adding synthetic noise, etc.
"""
import jax.numpy as np
import numpy as onp
from tqdm import tqdm
import warnings



def add_shot_noise(images):
    """
    Add synthetic shot noise to a single or stack of noiselss images
    images should be in units of photons
    Images use the Gaussian approximation to the Poisson distribution
    """
    noisy_images = images + onp.random.randn(*images.shape) * onp.sqrt(images)
    # ensure non-negative
    noisy_images[noisy_images < 0] = 0
    return noisy_images

def compute_cov_mat(patches):
    """
    Take an NxWxH stack of patches, and compute the covariance matrix of the vectorized patches
    """
    patches = np.array(patches)
    vectorized_patches = patches.reshape(patches.shape[0], -1).T
    # center on 0
    vectorized_patches = vectorized_patches - np.mean(vectorized_patches, axis=1, keepdims=True)
    return np.cov(vectorized_patches)

def compute_eigenvalues(image_patches):   
    """
    Take an NxWxH stack of image patches, and compute the eigenvalues of the covariance matrix of the vectorized patches
    """
    cov_mat = compute_cov_mat(image_patches)
    evals = np.linalg.eigvalsh(cov_mat)
    # sometimes there are a few tiny negative EVs
    evals = np.abs(evals)
    return evals

def extract_patches(stack, patch_size, num_patches=1000, seed=12345):
    patches = []
    onp.random.seed(seed)
    image_indices = onp.random.randint(0, stack.shape[0], num_patches)
    x_indices = onp.random.randint(0, stack.shape[1] - patch_size, num_patches)
    y_indices = onp.random.randint(0, stack.shape[2] - patch_size, num_patches)
    for i in tqdm(range(num_patches)):
        patches.append(stack[image_indices[i], x_indices[i]:x_indices[i]+patch_size, y_indices[i]:y_indices[i]+patch_size])
    return np.array(patches)

def normalize_image_stack(stack):
    """
    rescale pixel values to account for the fact that different images have different average energies
    (this also rescales noise)
    """
    # scale to account for different energy 
    average_energy_per_pixel = np.sum(stack.astype(float)) / np.prod(stack.size)
    stack = stack.astype(float) / average_energy_per_pixel
    return stack
