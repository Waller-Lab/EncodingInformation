
"""
Functions for processing images in useful ways extracting patches, adding synthetic noise, etc.
"""
import jax.numpy as np
import numpy as onp
from tqdm import tqdm
import numpy as np
import scipy.linalg as sla
from cleanplots import *




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

##########################
## Functions for computing and sampling from Gaussian processes    


def sample_multivariate_gaussian(cholesky):
    """
    Generate a sample from multivariate gaussian with zero mean given the cholesky decomposition of its covariance matrix
    """
    sample = cholesky @ np.random.multivariate_normal(np.zeros(cholesky.shape[0]), np.eye(cholesky.shape[0]))
    sampled_image = sample.reshape((int(np.sqrt(sample.size)), int(np.sqrt(sample.size))))
    return sampled_image


