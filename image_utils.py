
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
            noisy_batch = jax.random.poisson(key, shape=batch.shape, lam=batch)
        if ensure_positive:
            noisy_batch = np.where(noisy_batch < 0, 0, noisy_batch)
        noisy_batches.append(noisy_batch)

    # Concatenate the noisy batches back into a single array
    noisy_images = np.concatenate(noisy_batches, axis=0)
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

def extract_patches(stack, patch_size, num_patches=1000, seed=None, verbose=False):
    patches = []
    if seed is not None:
        onp.random.seed(seed)
    # image_indices = onp.random.randint(0, stack.shape[0], num_patches)
    # x_indices = onp.random.randint(0, stack.shape[1] - patch_size + 1, num_patches)
    # y_indices = onp.random.randint(0, stack.shape[2] - patch_size + 1, num_patches)
    
    image_indices = jax.random.randint(key=jax.random.PRNGKey(onp.random.randint(100000)), minval=0, maxval=stack.shape[0], shape=(num_patches,))
    x_indices = jax.random.randint(key=jax.random.PRNGKey( onp.random.randint(100000)), minval=0, maxval=stack.shape[1] - patch_size + 1, shape=(num_patches,))
    y_indices = jax.random.randint(key=jax.random.PRNGKey( onp.random.randint(100000)), minval=0, maxval=stack.shape[2] - patch_size + 1, shape=(num_patches,))
    if verbose:
        iterator = tqdm(range(num_patches))
    else:
        iterator = range(num_patches)
    for i in iterator:
        patches.append(stack[image_indices[i], x_indices[i]:x_indices[i]+patch_size, y_indices[i]:y_indices[i]+patch_size])
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
def compute_stationary_cov_mat(patches, verbose=False):
    """
    Uses images patches to estimate the covariance matrix of a stationary 2D process.
    The covariance matrix of such a process will be doubly toeplitz--i.e. a toeplitz matrix
    of blocks, each of which itself is a toeplitz matrix. The raw covariance matrix calculated
    from the patches is not doubly toeplitz, so we need to average over all elements that should 
    be the same as each other within this structure.
    """
    cov_mat = compute_cov_mat(patches)
    block_size = int(np.sqrt(cov_mat.shape[0]))
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
    if np.linalg.eigvalsh(new_matrix).min() < 0:
        raise ValueError('Covariance matrix is still not positive definite')
    if show_plot:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.semilogy(eigvals)
        ax.set(title='Eigenvalue spectrum', xlabel='Eigenvalue index', ylabel='Eigenvalue')
        new_evs = np.linalg.eigvalsh(new_matrix)
        ax.semilogy(new_evs)
        ax.set(title='Eigenvalue spectrum after regularization', xlabel='Eigenvalue index', ylabel='Eigenvalue')
    
    return new_matrix


def generate_multivariate_gaussian_samples(cov_mat, num_samples, mean=None, seed=None):
    """
    Generate samples from a 2D gaussian process with the given covariance matrix

    Args:
    cov_mat: jnp.ndarray, covariance matrix of the Gaussian distribution
    num_samples: int, number of samples to generate
    seed: int or jnp.ndarray, seed for the random number generator

    Returns:
    jnp.ndarray of shape (num_samples, cov_mat.shape[0]), samples from the multivariate Gaussian distribution
    """
    key = jax.random.PRNGKey(onp.random.randint(0, 100000) if seed is None else seed)
    samples = jax.random.multivariate_normal(key, mean if mean is not None else np.array([0]), cov_mat, (num_samples,))
    images = samples.reshape(num_samples, int(np.sqrt(cov_mat.shape[0])), int(np.sqrt(cov_mat.shape[0])))
    return images


def compute_stationary_log_likelihood(samples, cov_mat, mean, prefer_iterative=False):  
    """
    Compute the likelihood of a set of samples from a stationary process

    :param samples: N x H x W array of samples
    :param cov_mat: covariance matrix of the process
    :param mean: float mean of the process
    :param prefer_iterative: if True, compute likelihood iteratively, otherwise compute directly if possible

    :return: N x 1 array of log likelihoods
    """
    # samples is not going to be the same size as the covariance matrix
    # if sample is smaller than cov_mat, throw an excpetion
    # if sample is larger than cov_mat, then compute likelihood iteratively
    # if sample is the same size as cov_mat, then compute likelihood directly, unless prefer_iterative is True
    # check that mean if float or 1 element array
    if not isinstance(mean, float) and mean.shape != tuple():
        raise ValueError('Mean must be a float or a 1 element array')
    N_samples = samples.shape[0]
    # check for expected shape
    if samples.ndim != 3 or samples.shape[1] != samples.shape[2]:
        raise ValueError('Samples must be N x H x W')
    sample_size = samples.shape[1]

    if np.linalg.eigvalsh(cov_mat).min() < 0:
        raise ValueError('Covariance matrix is not positive definite')
    # precompute everything that will be the same for all samples
    patch_size = int(np.sqrt(cov_mat.shape[0]))
    vectorized_masks = []
    variances = []
    mean_multipliers = []
    for i in tqdm(np.arange(sample_size), desc='precomputing masks and variances'):
        for j in np.arange(sample_size):
            if not prefer_iterative and i < patch_size and j < patch_size:
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

                
                # more numerically stable
                if i == 0 and j == 0:
                    # top left pixel is not conditioned on anything
                    variance = sigma_22 
                    mean_multiplier = np.zeros((1, 1))
                else:
                    x = jax.scipy.linalg.solve(sigma_11, sigma_12)
                    variance = (sigma_22 - sigma_21 @ x) 
                    mean_multiplier = jax.scipy.linalg.solve(sigma_11, sigma_21.T)

                # sigma11_inv = np.linalg.inv(sigma_11)
                # variance = (sigma_22 - sigma_21 @ sigma11_inv @ sigma_12) 
                # mean_multiplier = sigma_21 @ sigma11_inv 
                
                variances.append(variance)
                mean_multipliers.append(mean_multiplier)

                if variances[-1] < 0:
                    raise ValueError('Variance is negative {} {}'.format(i, j))

    print('evaluating likelihood')

    log_likelihoods = []
    if not prefer_iterative:
        # compute the log_likelihood to the top left image subpatch of the image directly
        top_left_subpatch = samples[:, :patch_size, :patch_size].reshape(N_samples, -1)
        direct = []
        for sample in top_left_subpatch:
            direct.append(jax.scipy.stats.multivariate_normal.logpdf(sample.reshape(-1), mean=mean * np.ones(cov_mat.shape[0]), cov=cov_mat))
        direct = np.array(direct)
        log_likelihoods.append(direct)

    for i in tqdm(np.arange(sample_size), desc='computing log likelihoods'):
        for j in np.arange(sample_size):

            if not prefer_iterative and i < patch_size and j < patch_size :
                # already did this
                pass
            elif i == 0 and j == 0:
                # top left pixel is not conditioned on anything
                variance = cov_mat[0, 0]
                # compute likelihood of top left pixel
                log_likelihoods.append(jax.scipy.stats.norm.logpdf(samples[:, i, j], loc=mean, scale=np.sqrt(variance)))
            else:
                vectorized_mask = vectorized_masks[i * sample_size + j]
                # get the relevant window of previous values
                relevant_window = samples[:, max(i - patch_size + 1, 0):max(i - patch_size + 1, 0) + patch_size, 
                                                max(j - patch_size + 1, 0):max(j - patch_size + 1, 0) + patch_size]

                previous_values = relevant_window.reshape(N_samples, -1)[:, vectorized_mask].reshape(N_samples, -1)
                mean_to_use = mean + (mean_multipliers[i * sample_size + j].reshape(1, -1) @ (previous_values - mean).T).T
                variance = variances[i * sample_size + j]
                # print(variance)
                # compute likelihood of pixel
                # iterate over batch dimension
                batch_likelihoods = []
                
                scale = np.sqrt(float(variance))
                for k in np.arange(N_samples):
                    mean_for_sample = mean_to_use[k]
                    batch_likelihoods.append(jax.scipy.stats.norm.logpdf(samples[k, i, j], loc=mean_for_sample, scale=scale))                    
                log_likelihoods.append(np.array(batch_likelihoods).flatten())

    # print(np.array(log_likelihoods).reshape(N_samples, sample_size, sample_size))
    return np.sum(np.array(log_likelihoods), axis=0)


def generate_stationary_gaussian_process_samples(cov_mat, num_samples, sample_size=None, mean=None, 
                                                 ensure_nonnegative=False,
                                                 prefer_iterative_sampling=False, seed=None):
    """
    Given a covariance matrix of a stationary Gaussian process, generate samples from it. If the sample_size
    is less than or equal to the patch size used to generate the covariance matrix, this will be relatively
    fast. If it is larger, it will be slower and linear in the number additional pixels, since each new
    pixel is sampled conditional on the previous ones.

    cov_mat : The covariance matrix of a stationary Gaussian process
    num_samples : int number of samples to generate

    sample_size : int that is the one dimensional shape of a patch that the new
        covariance matrix represents the size of the covariance matrix is the patch size squared.
        if None, use the same patch size as the covariance matrix
    mean : mean of the Gaussian process. If None, use zero mean
    ensure_nonnegative : bool if true, ensure that all pixel values of sampls are nonnegative
    prefer_iterative_sampling : bool if true, dont directly sample the first (patch_size, patch_size) pixels
        directly from the covariance matrix. Instead, sample them iteratively from the previous pixels.
        This is much slower
    """
    if sample_size is None:
        sample_size = int(np.sqrt(cov_mat.shape[0]))
    if np.linalg.eigvalsh(cov_mat).min() < 0:
        raise ValueError('Covariance matrix is not positive definite')
    key = jax.random.PRNGKey(onp.random.randint(0, 100000) if seed is None else seed)
    # Use jax to do it all at once if possible
    if not prefer_iterative_sampling and sample_size <= int(np.sqrt(cov_mat.shape[0])):
        samples = jax.random.multivariate_normal(key, np.zeros(cov_mat.shape[0]), cov_mat, shape=(num_samples,))
        # crop if needed
        if sample_size < int(np.sqrt(cov_mat.shape[0])):
            samples = samples[:, :sample_size, :sample_size]
        # add mean
        if mean is not None:
            samples += mean
        if ensure_nonnegative:
            samples = np.where(samples < 0, 0, samples)
        return samples.reshape(num_samples, sample_size, sample_size)
    # precompute everything that will be the same for all samples
    patch_size = int(np.sqrt(cov_mat.shape[0]))
    vectorized_masks = []
    variances = []
    mean_multipliers = []
    for i in tqdm(np.arange(sample_size), desc='precomputing masks and variances'):
        for j in np.arange(sample_size):
            if not prefer_iterative_sampling and i < patch_size - 1 and j < patch_size - 1:
                raise Exception('why is there a -1 here? double check')

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

                # more numerically stable
                if i == 0 and j == 0:
                    # top left pixel is not conditioned on anything
                    variance = sigma_22 
                    mean_multiplier = np.zeros((1, 1))
                else:
                    x = jax.scipy.linalg.solve(sigma_11, sigma_12)
                    variance = (sigma_22 - sigma_21 @ x) 
                    mean_multiplier = jax.scipy.linalg.solve(sigma_11, sigma_21.T)

                # sigma11_inv = np.linalg.inv(sigma_11)
                # variance = (sigma_22 - sigma_21 @ sigma11_inv @ sigma_12) 
                # mean_multiplier = sigma_21 @ sigma11_inv 
                
                variances.append(variance)
                mean_multipliers.append(mean_multiplier)


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
        raise Exception('this is broken, change to generate_multivariate_gaussian_samples')
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
                raise Exception('why is there a -1 here? double check')
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
                
                mean = mean_multipliers[i * sample_size + j] @ previous_values
                variance = variances[i * sample_size + j]
                sample = (jax.random.normal(key) * np.sqrt(variance) + mean).reshape(-1)[0]
                sampled_image = sampled_image.at[i, j].set(sample)
                key = jax.random.split(key)[1]
            
    return sampled_image, jax.random.split(key)[1]
    
