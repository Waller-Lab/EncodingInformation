"""
Functions for computing and sampling from Gaussian processes    
"""
from jax.scipy.linalg import toeplitz
import jax.numpy as np
from tqdm import tqdm
import jax
import matplotlib.pyplot as plt
from jax import grad, jit
import numpy as onp
from functools import partial
import optax
import warnings

def estimate_cov_mat(patches):
    """
    Take an NxWxH stack of patches, and compute the covariance matrix of the vectorized patches
    """
    # patches = np.array(patches)
    vectorized_patches = patches.reshape(patches.shape[0], -1).T
    # center on 0
    vectorized_patches = vectorized_patches - np.mean(vectorized_patches, axis=1, keepdims=True)
    return np.cov(vectorized_patches).reshape(vectorized_patches.shape[0], vectorized_patches.shape[0])

def _plugin_estimate_stationary_cov_mat(patches, eigenvalue_floor, verbose=False, suppress_warning=False):
    cov_mat = estimate_cov_mat(patches)
    block_size = int(np.sqrt(cov_mat.shape[0]))

    stationary_cov_mat = average_diagonals_to_make_doubly_toeplitz(cov_mat, block_size, verbose=verbose)

    # try to make both stationary and positive definite
    if eigenvalue_floor is not None:
        # make positive definite
        eigvals, eig_vecs = np.linalg.eigh(stationary_cov_mat)
        eigvals = np.where(eigvals < eigenvalue_floor, eigenvalue_floor, eigvals)
        stationary_cov_mat = eig_vecs @ np.diag(eigvals) @ eig_vecs.T
        if np.linalg.eigvalsh(stationary_cov_mat).min() < 0:
            raise ValueError('Covariance matrix is not positive definite event after applying eigenvalue floor. This indicates numerical error.' +
                             'Try raising the eigenvalue floor')
        doubly_toeplitz = average_diagonals_to_make_doubly_toeplitz(stationary_cov_mat, block_size, verbose=verbose)
        dt_eigs = np.linalg.eigvalsh(doubly_toeplitz)
        if np.any(dt_eigs < 0) and not suppress_warning:
            warnings.warn('Cannot make both doubly toeplitz and positive definite. Using positive definite matrix.'
                          'Smallest eigenvalue is {}'.format(dt_eigs.min()))


    return stationary_cov_mat

# trying to add jit to this somehow makes it run slower
def estimate_stationary_cov_mat(patches, eigenvalue_floor=1e-3, optimize=False, verbose=False, return_mean=False,
                                patience=20, num_validation=100, batch_size=12,
                           gradient_clip=1, learning_rate=1e2, momentum=0.9, max_iters=100):
    """
    Uses images patches to estimate the covariance matrix of a stationary 2D process.
    The covariance matrix of such a process will be doubly toeplitz--i.e. a toeplitz matrix
    of blocks, each of which itself is a toeplitz matrix. The raw covariance matrix calculated
    from the patches is not doubly toeplitz, so we need to average over all elements that should 
    be the same as each other within this structure.
    """
    if not optimize:
        cov_mat = _plugin_estimate_stationary_cov_mat(patches, eigenvalue_floor=eigenvalue_floor, verbose=verbose)
        mean_vec = np.mean(patches) * np.ones(cov_mat.shape[0])
    else:
        mean_vec, cov_mat = fit_optimized_gaussian(patches, eigenvalue_floor=eigenvalue_floor, verbose=verbose, 
                                                    patience=patience, num_validation=num_validation, batch_size=batch_size,
                                                    gradient_clip=gradient_clip, learning_rate=learning_rate,
                                                momentum=momentum, max_iters=max_iters)
    if return_mean:
        return mean_vec, cov_mat
    return cov_mat


def generate_multivariate_gaussian_samples(mean_vec, cov_mat, num_samples, seed=None, key=None):
    """
    Generate samples from a 2D gaussian process with the given covariance matrix

    Args:
    cov_mat: jnp.ndarray, covariance matrix of the Gaussian distribution
    num_samples: int, number of samples to generate
    seed: int or jnp.ndarray, seed for the random number generator
    key: a jax.random.PRNGKey, if None, use the seed to generate one

    Returns:
    jnp.ndarray of shape (num_samples, cov_mat.shape[0]), samples from the multivariate Gaussian distribution
    """
    if key is None:
        key = jax.random.PRNGKey(onp.random.randint(0, 100000) if seed is None else seed)
    samples = jax.random.multivariate_normal(key, mean_vec, cov_mat, (num_samples,))
    images = samples.reshape(num_samples, int(np.sqrt(cov_mat.shape[0])), int(np.sqrt(cov_mat.shape[0])))
    return images


def compute_stationary_log_likelihood(samples, cov_mat, mean, prefer_iterative=False):  
    """
    Compute the likelihood of a set of samples from a stationary process

    :param samples: N x H x W array of samples
    :param cov_mat: covariance matrix of the process
    :param mean: float mean of the process
    :param prefer_iterative: if True, compute likelihood iteratively, otherwise compute directly if possible

    :return: (N,) array of log likelihoods for each sample
    """
    # samples is not going to be the same size as the covariance matrix
    # if sample is smaller than cov_mat, throw an excpetion
    # if sample is larger than cov_mat, then compute likelihood iteratively
    # if sample is the same size as cov_mat, then compute likelihood directly, unless prefer_iterative is True
    # check that mean if float or 1 element array
    if not isinstance(mean, float) and mean.shape != tuple():
        if np.unique(mean).size != 1:
            raise ValueError('Mean for stationary process cannot be an array with more than one unique value')
        mean = mean[0]
        
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

    # sum over all pixels to get a length N vector of log likelihoods for each sample
    return np.sum(np.array(log_likelihoods), axis=0)


def generate_stationary_gaussian_process_samples(mean_vec, cov_mat, num_samples, sample_size=None,
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
        samples = jax.random.multivariate_normal(key, mean_vec, cov_mat, shape=(num_samples,))
        # crop if needed
        if sample_size < int(np.sqrt(cov_mat.shape[0])):
            samples = samples[:, :sample_size, :sample_size]
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
            if not prefer_iterative_sampling and i < patch_size and j < patch_size:
                # raise Exception('why is there a -1 here? double check')

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


    
    print('generating samples')
    samples = _generate_samples(num_samples, cov_mat, mean_vec, key, sample_size, vectorized_masks, variances, 
                                    mean_multipliers, prefer_iterative_sampling=prefer_iterative_sampling)
    if ensure_nonnegative:
        samples = np.where(samples < 0, 0, samples)
    
    return samples

def _generate_samples(num_samples, cov_mat, mean_vec, key, sample_size, vectorized_masks, variances, mean_multipliers, prefer_iterative_sampling=False):
    patch_size = int(np.sqrt(cov_mat.shape[0]))
    if not prefer_iterative_sampling:
        # sample the first (patch_size, patch_size) pixels directly from the covariance matrix
        sampled_images = generate_multivariate_gaussian_samples(mean_vec, cov_mat, num_samples, key=key)

        # if the directly sampled image is sufficiently large for the sample size requested, return it
        if sampled_images.shape[1] == sample_size:
            return sampled_images
        elif sampled_images.shape[1] > sample_size:
            return sampled_images[..., :sample_size, :sample_size]

        # pad the right and bottom with zeros
        sampled_images = np.pad(sampled_images, ((0, 0), (0, sample_size - sampled_images.shape[-2]), (0, sample_size - sampled_images.shape[-1])))
    else:
        sampled_images = np.zeros((num_samples, sample_size, sample_size))

    for i in tqdm(np.arange(sample_size)):
        for j in np.arange(sample_size):
            if not prefer_iterative_sampling and i < patch_size  and j < patch_size :
                # raise Exception('why is there a -1 here? double check')
                # use existing values
                pass
            elif i == 0 and j == 0:
                # top left pixel is not conditioned on anything                
                samples = jax.random.normal(key, shape=(num_samples,)) * np.sqrt(cov_mat[0, 0]) + mean_vec[0]
                sampled_images = sampled_images.at[:, i, j].set(samples)
                key = jax.random.split(key)[1]
            else:
                vectorized_mask = vectorized_masks[i * sample_size + j]
                # get the relevant window of previous values
                relevant_window = sampled_images[..., 
                                                max(i - patch_size + 1, 0):max(i - patch_size + 1, 0) + patch_size, 
                                                max(j - patch_size + 1, 0):max(j - patch_size + 1, 0) + patch_size]
                previous_values = relevant_window.reshape(num_samples, -1)[:, vectorized_mask].reshape(num_samples, vectorized_mask.sum(), 1)
                
                mean = (mean_multipliers[i * sample_size + j].reshape(1, -1) @ (previous_values - mean_vec[0]) + mean_vec[0]).flatten()
                variance = variances[i * sample_size + j]
                samples = (jax.random.normal(key, shape=(num_samples,)) * np.sqrt(variance) + mean)
                sampled_images = sampled_images.at[:, i, j].set(samples.flatten())
                key = jax.random.split(key)[1]
            
    return sampled_images
    

#####################################################
####### Optimizing a stationary gaussian fit ########
#####################################################
@partial(jit, static_argnums=(1, 2))
def average_diagonals_to_make_doubly_toeplitz(cov_mat, patch_size, verbose=False):
    # divide it into blocks
    blocks = [np.hsplit(row, cov_mat.shape[1]//patch_size) for row in np.vsplit(cov_mat, cov_mat.shape[0]//patch_size)]

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
    j, i = np.meshgrid(np.arange(patch_size), np.arange(patch_size))
    differences = abs(i - j)
    for block_id, block in tqdm(dict(toeplitz_block_means.items()).items(), desc='building toeplitz mat') if verbose else dict(toeplitz_block_means.items()).items():
        diag_values = []
        for id in np.arange(patch_size):
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
    doubly_toeplitz = np.vstack(new_blocks)
    return doubly_toeplitz

def gaussian_likelihood(cov_mat, mean_vec, batch):
    """
    Evaluate the log likelihood of a multivariate gaussian
    for a batch of NxWXH samples.
    """
    log_likelihoods = []
    for sample in batch:
        ll = jax.scipy.stats.multivariate_normal.logpdf(sample.reshape(-1), mean=mean_vec, cov=cov_mat)
        log_likelihoods.append(ll)
    return np.array(log_likelihoods)

def batch_nll(log_likelihoods):
    return -np.mean(log_likelihoods)

def loss_function(eigvals, eig_vecs, mean_vec, data, num_pixels):
    """
    Negative log likelihood of a multivariate gaussian per pixel
    """
    cov_mat = eig_vecs @ np.diag(eigvals) @ eig_vecs.T
    ll = gaussian_likelihood(cov_mat, mean_vec, data)
    return batch_nll(ll) / num_pixels

def make_positive_definite(cov_mat, eigenvalue_floor):
    eigvals, eig_vecs = np.linalg.eigh(cov_mat)
    eigvals = np.where(eigvals < eigenvalue_floor, eigenvalue_floor, eigvals)
    return eig_vecs @ np.diag(eigvals) @ eig_vecs.T

def try_to_make_doubly_toeplitz_and_positive_definite(eigvals, eig_vecs, eigenvalue_floor, patch_size):
    """
    Average along diagonals and block diagonals to make a doubly toeplitz matrix,
    then make sure it is positive definite by setting all eigenvalues below
    eigenvalue_floor to eigenvalue_floor to get rid of negative eigenvalues.

    This won't neccesarily return a doubly toeplitz matrix, but it will be positive definite.
    """
    cov_mat = eig_vecs @ np.diag(eigvals) @ eig_vecs.T
    dt_cov_mat = average_diagonals_to_make_doubly_toeplitz(cov_mat, patch_size)
    eigvals, eig_vecs = np.linalg.eigh(dt_cov_mat)
    eigvals = np.where(eigvals < eigenvalue_floor, eigenvalue_floor, eigvals)
    return eigvals, eig_vecs

def fit_optimized_gaussian(data, batch_size=12, eigenvalue_floor=1e-3, patience=20, num_validation=100, 
                           gradient_clip=1, learning_rate=1e2, momentum=0.9, max_iters=100, return_everything=False,
                        verbose=True):
    optimizer = optax.chain(
        # don't let update size exceed approx parameter size * Learning rate.
        # this prevents tiny, incorrect eigenvalues from making things diverge
        optax.clip(gradient_clip), 
        optax.sgd(learning_rate, momentum=momentum, nesterov=False)
    )
    
    patch_size = int(np.sqrt(np.prod(np.array(data.shape)[1:])))
    # Initialize parameters, hyperparameters
    mean_vec = np.ones(patch_size**2) * np.mean(data)
    patch_size = int(np.sqrt(np.prod(np.array(data.shape)[1:])))

    @jit
    def optmization_step(opt_state, eigvals, eig_vecs, data, mean_vec, eigenvalue_floor):
        grad_fn = grad(loss_function, argnums=0)
        eigenvalues_grad = grad_fn(eigvals, eig_vecs, mean_vec, data, patch_size**2)
        updates, opt_state = optimizer.update(eigenvalues_grad, opt_state, eigvals)
        eigvals = optax.apply_updates(eigvals, updates)
        # prox operator: make sure make sure positive definite, make sure doubly toeplitz
        eigvals, eig_vecs = try_to_make_doubly_toeplitz_and_positive_definite(eigvals, eig_vecs, eigenvalue_floor, patch_size=patch_size)
        loss = loss_function(eigvals, eig_vecs, mean_vec, data, patch_size**2)
        return eigvals, eig_vecs, opt_state, loss


    # Split data into training and validation sets
    if num_validation >= len(data):
        # default to a 75/25 split
        num_validation = int(len(data) * 0.25)
        warnings.warn('Number of validation samples is larger than the number of samples. Defaulting to a 75/25 split')
        num_validation = max(num_validation, 1)
        
    num_train = len(data) - num_validation
    train_data = data[:num_train]
    validation_data = data[num_train:]

    # initialize covariance matrix so likelihood is not nan
    cov_mat_initial = _plugin_estimate_stationary_cov_mat(train_data, eigenvalue_floor=eigenvalue_floor, suppress_warning=True)

    initial_evs, initial_eig_vecs = np.linalg.eigh(cov_mat_initial)
    initial_loss = loss_function(initial_evs, initial_eig_vecs, mean_vec, validation_data, patch_size**2)
    if verbose:
        print('Initial loss: ', initial_loss)

    cov_mat_initial = initial_eig_vecs @ np.diag(initial_evs) @ initial_eig_vecs.T

    if np.isnan(jax.scipy.stats.multivariate_normal.logpdf(train_data[0].flatten(), mean=mean_vec, cov=cov_mat_initial)):
        raise ValueError("Initial likelihood is nan")
    
    # Training loop
    eigenvalues = initial_evs
    opt_state = optimizer.init(eigenvalues)
    eig_vecs = initial_eig_vecs
    best_loss = np.inf
    key = jax.random.PRNGKey(onp.random.randint(0, 100000))
    best_loss_iter = 0
    train_loss_history = []
    validation_loss_history = []
    for i in range(max_iters):
        # select a random batch
        batch_indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=num_train)
        key, subkey = jax.random.split(key)
        batch = train_data[batch_indices]
        
        eigenvalues, eig_vecs, opt_state, train_loss = optmization_step(
            opt_state, eigenvalues, eig_vecs, batch, mean_vec, eigenvalue_floor)

        train_loss_history.append(train_loss)

        validation_loss = loss_function(eigenvalues, eig_vecs, mean_vec, validation_data, patch_size**2)   
        validation_loss_history.append(validation_loss)

        if verbose:
            print(f"Iteration {i+1}, validation loss: {validation_loss}", end='\r')
        if validation_loss < best_loss:
            best_loss_iter = i
            best_loss = validation_loss
            best_eigvals = eigenvalues
            best_eig_vecs = eig_vecs
        if i - best_loss_iter > patience:
            break

    eigenvalues, eig_vecs = try_to_make_doubly_toeplitz_and_positive_definite(best_eigvals, best_eig_vecs, eigenvalue_floor, patch_size=patch_size)
    cov_mat_optimized = eig_vecs @ np.diag(eigenvalues) @ eig_vecs.T
    final_loss = loss_function(eigenvalues, eig_vecs, mean_vec, validation_data, patch_size**2)
    if initial_loss < final_loss:
        warnings.warn("Optimization did not improve the validation loss, returning initial covariance matrix")
        cov_mat_optimized = cov_mat_initial

    if verbose:
        print('\Optimized loss: ', final_loss)

    if not return_everything:
        return mean_vec, cov_mat_optimized
    final_cov_mat = eig_vecs @ np.diag(eigenvalues) @ eig_vecs.T

    return cov_mat_optimized, cov_mat_initial, mean_vec, best_loss, train_loss_history, validation_loss_history, final_cov_mat
    

