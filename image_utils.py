
"""
Functions for processing images in useful ways extracting patches, adding synthetic noise, etc.
"""

import numpy as onp
from tqdm import tqdm
from cleanplots import *
from jax.scipy.linalg import toeplitz
import jax.numpy as np
import jax
import warnings

def add_shot_noise(images):
    """
    Add synthetic shot noise to a single or stack of noiselss images
    images should be in units of photons
    Images use the Gaussian approximation to the Poisson distribution
    """
    noisy_images = images + onp.random.randn(*images.shape) * onp.sqrt(images)
    # ensure non-negative
    noisy_images = np.where(noisy_images < 0, 0, noisy_images)
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

def extract_patches(stack, patch_size, num_patches=1000, seed=None):
    patches = []
    if seed is not None:
        onp.random.seed(seed)
    image_indices = onp.random.randint(0, stack.shape[0], num_patches)
    x_indices = onp.random.randint(0, stack.shape[1] - patch_size + 1, num_patches)
    y_indices = onp.random.randint(0, stack.shape[2] - patch_size + 1, num_patches)
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


def compute_cov_mat(patches):
    """
    Take an NxWxH stack of patches, and compute the covariance matrix of the vectorized patches
    """
    # patches = np.array(patches)
    vectorized_patches = patches.reshape(patches.shape[0], -1).T
    # center on 0
    vectorized_patches = vectorized_patches - np.mean(vectorized_patches, axis=1, keepdims=True)
    return np.cov(vectorized_patches)

# trying to add jit to this somehow makes it run slower
def compute_stationary_cov_mat(patches, verbose=True):
    """
    Uses images patches to estimate the covariance matrix of a stationary 2D process.
    The covariance matrix of such a process will be doubly toeplitz--i.e. a toeplitz matrix
    of blocks, each of which itself is a toeplitz matrix. The raw covariance matrix calculated
    from the patches is not doubly toeplitz, so we need to average over all elements that should 
    be the same as each other within this structure.
    """
    cov_mat = compute_cov_mat(patches)
    # split it into individual blocks
    block_size = patches.shape[-1]
    # divide it into blocks
    blocks = [np.hsplit(row, cov_mat.shape[1]//block_size) for row in np.vsplit(cov_mat, cov_mat.shape[0]//block_size)]

    toeplitz_blocks = {}
    for i in range(len(blocks)):
        for j in range(len(blocks[0])):
            id = (i - j)
            if id not in toeplitz_blocks:
                toeplitz_blocks[id] = []
            toeplitz_blocks[id].append(blocks[i][j])

    # compute the mean of each block
    toeplitz_block_means = {id: np.mean(np.stack(blocks, axis=0), axis=0) for id, blocks in toeplitz_blocks.items()}

    # now repeat the process within each block
    j, i = np.meshgrid(np.arange(block_size), np.arange(block_size))
    differences = abs(i - j)
    for block_id, block in tqdm(dict(toeplitz_block_means.items()).items()) if verbose else dict(toeplitz_block_means.items()).items():
        diag_values = []
        for id in np.arange(block_size):
            # recompute mask each time to save memory
            mask = differences == id
            diag_values.append(np.sum(np.where(mask, block, 0)) / np.sum(mask))            
        # create a new topelitz matrix from the diagonal values
        toeplitz_block_means[block_id] = toeplitz(np.array(diag_values))

    # now reconstruct the full doubly toeplitz matrix from the blocks
    new_blocks = []
    for i in range(len(blocks)):
        row = []
        for j in range(len(blocks[0])):
            id = abs(i - j)
            row.append(toeplitz_block_means[id])
        new_blocks.append(np.hstack(row))
    stationary_cov_mat = np.vstack(new_blocks)

    return stationary_cov_mat

def sample_multivariate_gaussian(cholesky, key):
    """
    Generate a sample from multivariate gaussian with zero mean given the cholesky decomposition of its covariance matrix
    """
    mvn_sample = jax.random.multivariate_normal(key, np.zeros(cholesky.shape[0]), np.eye(cholesky.shape[0]))
    # mvn_sample = onp.random.normal(size=cholesky.shape[0])
    sample = np.array(cholesky) @ mvn_sample
    sampled_image = sample.reshape((int(np.sqrt(sample.size)), int(np.sqrt(sample.size))))
    return sampled_image

def make_positive_definite(A, eigenvalue_floor, show_plot=True, verbose=True):
    """
    Ensure the matrix A is positive definite by adding a small amount to the diagonal
    (Tikhonov regularization)

    A : matrix to make positive definite
    eigenvalue_floor : float, make all eigenvalues of the matrix at least this large
    show_plot : whether to show a plot of the eigenvalue spectrum and cutoff threshold
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.where(eigvals < eigenvalue_floor, eigenvalue_floor, eigvals)
    new_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
    if show_plot:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.semilogy(eigvals)
        ax.set(title='Eigenvalue spectrum', xlabel='Eigenvalue index', ylabel='Eigenvalue')
        new_evs = np.linalg.eigvalsh(new_matrix)
        ax.semilogy(new_evs)
        ax.set(title='Eigenvalue spectrum after regularization', xlabel='Eigenvalue index', ylabel='Eigenvalue')
    
    return new_matrix


def generate_stationary_gaussian_process_samples(cov_mat, sample_size, num_samples, mean=None, 
                                                 ensure_nonnegative=False,
                                                 prefer_iterative_sampling=False, seed=None):
    """
    Given a covariance matrix of a stationary Gaussian process, generate samples from it. If the sample_size
    is less than or equal to the patch size used to generate the covariance matrix, this will be relatively
    fast. If it is larger, it will be slower and linear in the number additional pixels, since each new
    pixel is sampled conditional on the previous ones.

    cov_mat : The covariance matrix of a stationary Gaussian process
    sample_size : int that is the one dimensional shape of a patch that the new
        covariance matrix represents the size of the covariance matrix is the patch size squared
    num_samples : int number of samples to generate
    mean : mean of the Gaussian process. If None, use zero mean
    ensure_nonnegative : bool if true, ensure that all pixel values of sampls are nonnegative
    prefer_iterative_sampling : bool if true, dont directly sample the first (patch_size, patch_size) pixels
        directly from the covariance matrix. Instead, sample them iteratively from the previous pixels.
        This is much slower
    """
    key = jax.random.PRNGKey(onp.random.randint(0, 10000) if seed is None else seed)

    # precompute everything that will be the same for all samples
    patch_size = int(np.sqrt(cov_mat.shape[0]))
    vectorized_masks = []
    variances = []
    mean_multipliers = []
    for i in tqdm(np.arange(sample_size), desc='precomputing masks and variances'):
        for j in np.arange(sample_size):
            if not prefer_iterative_sampling and i < patch_size - 1 and j < patch_size - 1:
                # Add placeholders since these get sampled from the covariance matrix directly
                variances.append(None)
                mean_multipliers.append(None)
                vectorized_masks.append(None)
            else:
                top_part = np.ones((min(i, patch_size - 1), patch_size), dtype=bool)
                left_part = np.ones((1, min(j, patch_size - 1)), dtype=bool)
                right_part = np.zeros((1, patch_size - min(j, patch_size - 1)), dtype=bool)
                bottom_part = np.zeros((patch_size - min(i, patch_size - 1) - 1, patch_size), dtype=bool)
                middle_row = np.hstack((left_part, right_part))
                conditioning_mask = np.vstack((top_part, middle_row, bottom_part))

                vectorized_mask = conditioning_mask.reshape(-1)
                vectorized_masks.append(vectorized_mask)
                # find the linear index in the covariance matrix of the pixel we want to predict
                pixel_to_predict_index = np.min(np.array([i, patch_size - 1])) * patch_size + np.min(np.array([j, patch_size - 1]))
                sigma_11 = cov_mat[vectorized_mask][:, vectorized_mask].reshape(pixel_to_predict_index, pixel_to_predict_index) 
                sigma_12 = cov_mat[vectorized_mask][:, pixel_to_predict_index].reshape(-1, 1)
                sigma_21 = sigma_12.reshape(1, -1)
                sigma_22 = cov_mat[pixel_to_predict_index, pixel_to_predict_index].reshape(1, 1)

                variances.append(sigma_22 - sigma_21 @ np.linalg.inv(sigma_11) @ sigma_12)
                mean_multipliers.append(sigma_21 @ np.linalg.inv(sigma_11))
                # print(i, j, np.linalg.det(sigma_11))

                # print(i,j, mean_multipliers[-1].mean())
                if variances[-1] < 0:
                    raise ValueError('Variance is negative {} {}'.format(i, j))


    samples = []
    print('generating samples')
    for i in range(num_samples):
        sample, key = _generate_sample(cov_mat, key, sample_size, vectorized_masks, variances, 
                                        mean_multipliers, prefer_iterative_sampling=prefer_iterative_sampling)
        if mean is not None:
            sample += mean
        if ensure_nonnegative:
            sample = np.where(sample < 0, 0, sample)
        samples.append(sample)
    return samples

def _generate_sample(cov_mat, key, sample_size, vectorized_masks, variances, mean_multipliers, prefer_iterative_sampling=False):
    patch_size = int(np.sqrt(cov_mat.shape[0]))
    if not prefer_iterative_sampling:
        cholesky = onp.linalg.cholesky(cov_mat)
        sampled_image = sample_multivariate_gaussian(cholesky, key)
        key = jax.random.split(key)[1]

        if sampled_image.shape[0] == sample_size:
            return sampled_image, key
        elif sampled_image.shape[0] > sample_size:
            return sampled_image[:sample_size, :sample_size], key

        # pad the right and bottom with zeros
        sampled_image = np.pad(sampled_image, ((0, sample_size - sampled_image.shape[0]), (0, sample_size - sampled_image.shape[1])))
    else:
        sampled_image = np.zeros((sample_size, sample_size))

    for i in tqdm(np.arange(sample_size), desc='generating sample'):
        for j in np.arange(sample_size):
            if not prefer_iterative_sampling and i < patch_size - 1 and j < patch_size - 1:
                # use existing values
                pass
            elif i == 0 and j == 0:
                # top left pixel is not conditioned on anything
                mean = 0
                sample = (jax.random.normal(key) * np.sqrt(cov_mat[0, 0]) + mean).reshape(-1)[0]
                sampled_image = sampled_image.at[i, j].set(sample)
                key = jax.random.split(key)[1]
            else:
                vectorized_mask = vectorized_masks[i * sample_size + j]
                # get the relevant window of previous values
                relevant_window = sampled_image[max(i - patch_size + 1, 0):max(i - patch_size + 1, 0) + patch_size, 
                                                max(j - patch_size + 1, 0):max(j - patch_size + 1, 0) + patch_size]
                previous_values = relevant_window.reshape(-1)[vectorized_mask].reshape(-1, 1)
                
                if i == 9 and j == 10:
                    pass
                mean = mean_multipliers[i * sample_size + j] @ previous_values
                variance = variances[i * sample_size + j]
                sample = (jax.random.normal(key) * np.sqrt(variance) + mean).reshape(-1)[0]
                sampled_image = sampled_image.at[i, j].set(sample)
                key = jax.random.split(key)[1]
            
    return sampled_image, jax.random.split(key)[1]
    
