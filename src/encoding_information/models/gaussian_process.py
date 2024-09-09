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
import flax.linen as nn
from flax.training.train_state import TrainState
from encoding_information.models.image_distribution_models import ProbabilisticImageModel, train_model, make_dataset_generators


def match_to_generator_data(images, seed=None):
    """
    Important: during the training process, noise is added to the pixel values to account for the 
    fact that discrete pixel values are used with a continuous density in the model. This is handled by the 
    make_dataset_generator function in the image_distribution_models module. So we call this here on a the images
    to ensure that the same noise is added to the images here as was added during training, and then convert back
    to a jax array
    """
    _, dataset_fn = make_dataset_generators(images, batch_size=images.shape[0], num_val_samples=images.shape[0], seed=seed)
    return next(dataset_fn())

def estimate_full_cov_mat(patches):
    """
    Take an NxWxH stack of patches, and compute the covariance matrix of the vectorized patches
    """
    # patches = np.array(patches)
    vectorized_patches = patches.reshape(patches.shape[0], -1).T
    # center on 0
    vectorized_patches = vectorized_patches - np.mean(vectorized_patches, axis=1, keepdims=True)
    return np.cov(vectorized_patches).reshape(vectorized_patches.shape[0], vectorized_patches.shape[0])

def plugin_estimate_stationary_cov_mat(patches, eigenvalue_floor, verbose=False, suppress_warning=False):
    cov_mat = estimate_full_cov_mat(patches)
    block_size = int(np.sqrt(cov_mat.shape[0]))

    stationary_cov_mat = average_diagonals_to_make_doubly_toeplitz(cov_mat, block_size, verbose=verbose)

    # try to make both stationary and positive definite
    if eigenvalue_floor is not None:
        if verbose:
            print('trying to make doubly toeplitz and positive definite')
        # make positive definite
        eigvals, eig_vecs = np.linalg.eigh(stationary_cov_mat)
        eigvals = np.where(eigvals < eigenvalue_floor, eigenvalue_floor, eigvals)
        stationary_cov_mat = eig_vecs @ np.diag(eigvals) @ eig_vecs.T
        while np.linalg.eigvalsh(stationary_cov_mat).min() < 0:
            warnings.warn('Covariance matrix is not positive definite even after applying eigenvalue floor. This indicates numerical error.' +
                             'Try raising the eigenvalue floor than the current value of {}'.format(eigenvalue_floor))
            eigenvalue_floor *= 10
            print('trying eigenvalue floor of {}'.format(eigenvalue_floor))
            eigvals = np.where(eigvals < eigenvalue_floor, eigenvalue_floor, eigvals)
            stationary_cov_mat = eig_vecs @ np.diag(eigvals) @ eig_vecs.T
        if verbose:
            print('made positive definite, smallest ev: ' + str(np.linalg.eigvalsh(stationary_cov_mat).min()))
    
        doubly_toeplitz = average_diagonals_to_make_doubly_toeplitz(stationary_cov_mat, block_size, verbose=verbose)
        dt_eigs = np.linalg.eigvalsh(doubly_toeplitz)
        if np.any(dt_eigs < 0) and not suppress_warning:
            warnings.warn('Cannot make both doubly toeplitz and positive definite. Using positive definite matrix.'
                          'Smallest eigenvalue is {}'.format(dt_eigs.min()))


    return stationary_cov_mat


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


def _compute_stationary_log_likelihood(samples, cov_mat, mean, prefer_iterative=False, verbose=False):  
    """
    Compute the log likelihood per pixel of a set of samples from a stationary process

    :param samples: N x H x W array of samples
    :param cov_mat: covariance matrix of the process
    :param mean: float mean of the process
    :param prefer_iterative: if True, compute likelihood iteratively, otherwise compute directly if possible
    :param verbose: if True, print progress

    :return: average log_likelihood per pixel
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
    # drop trailing channel dim
    if samples.ndim == 4 and samples.shape[3] == 1:
        samples = samples.reshape(samples.shape[0], samples.shape[1], samples.shape[2])
    # check for expected shape
    if samples.ndim == 4 and samples.shape[3] == 1:
        samples = samples[..., 0] # remove trailing channel dimension
    if samples.ndim != 3 or samples.shape[1] != samples.shape[2]:
        raise ValueError('Samples must be N x H x W, but got {}'.format(samples.shape))
    sample_size = samples.shape[1]

    if np.linalg.eigvalsh(cov_mat).min() < 0:
        raise ValueError('Covariance matrix is not positive definite')
    # precompute everything that will be the same for all samples
    patch_size = int(np.sqrt(cov_mat.shape[0]))
    vectorized_masks = []
    variances = []
    mean_multipliers = []
    iter = tqdm(np.arange(sample_size), desc='precomputing masks and variances') if verbose else np.arange(sample_size)
    for i in iter:
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

    if verbose:
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

    iter = tqdm(np.arange(sample_size), desc='computing log likelihoods') if verbose else np.arange(sample_size)
    for i in iter:
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

    # return average log likelihood per pixel
    return np.mean(np.array(log_likelihoods)) / cov_mat.shape[0]


def generate_stationary_gaussian_process_samples(mean_vec, cov_mat, num_samples, sample_size=None,
                                                 ensure_nonnegative=False,
                                                 prefer_iterative_sampling=False, seed=None, verbose=False):
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
    seed : int , seed for the random number generator
    verbose : bool if true, print progress
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
    iter = tqdm(np.arange(sample_size), desc='precomputing masks and variances') if verbose else np.arange(sample_size)
    for i in iter:
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


    if verbose:
        print('generating stationary gaussian process samples')
    samples = _generate_samples(num_samples, cov_mat, mean_vec, key, sample_size, vectorized_masks, variances, 
                                    mean_multipliers, prefer_iterative_sampling=prefer_iterative_sampling, verbose=verbose)
    if ensure_nonnegative:
        samples = np.where(samples < 0, 0, samples)
    
    return samples

def _generate_samples(num_samples, cov_mat, mean_vec, key, sample_size, vectorized_masks, variances, 
                      mean_multipliers, prefer_iterative_sampling=False, verbose=False):
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

    iter = tqdm(np.arange(sample_size), desc='generating samples') if verbose else np.arange(sample_size)
    for i in iter:
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

def nll_per_pixel_from_cov_mat(cov_mat, mean_vec, data, num_pixels):
    """
    Negative log likelihood of a multivariate gaussian per pixel
    """
    ll = gaussian_likelihood(cov_mat, mean_vec, data)
    nll = -np.mean(ll) # average over batch
    return nll / num_pixels

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

 

#####################################################
# Flax implementation of Gaussian process ######

class _StationaryGaussianProcessFlaxImpl(nn.Module):
    
    size: int

    def setup(self):
        self.eig_vals = self.param('eig_vals',  nn.initializers.zeros, (self.size,))
        self.eig_vecs = self.param('eig_vecs',  nn.initializers.zeros, (self.size, self.size))
        self.mean_vec = self.param('mean_vec', nn.initializers.zeros, (self.size,))

    def __call__(self):
        """
        return the mean and covariance matrix of the Gaussian process as a function of the optimizable parameters
        """
        cov_mat = self.eig_vecs @ np.diag(self.eig_vals) @ self.eig_vecs.T
        return self.mean_vec, cov_mat
    

    def compute_loss(self, mean_vec, cov_mat, images):
        """ 
        Compute average negative log likelihood per pixel averaged over batch
        """
        return nll_per_pixel_from_cov_mat(cov_mat, mean_vec, images, np.prod(np.array(images.shape[1:])))


##################################################################################################
#### Wrapper for the Flax implementation of Gaussian processes to the probabilistic image model API ######

class StationaryGaussianProcess(ProbabilisticImageModel):

    def __init__(self, images, eigenvalue_floor=1e-3, seed=None, verbose=False):
        """
        Create a StationaryGaussianProcess model and initialize it to the plugin estimate of the stationary covariance matrix
        """
        self.image_shape = images.shape[1:]

        self._flax_model = _StationaryGaussianProcessFlaxImpl(size=np.prod(np.array(self.image_shape)))
        self.initial_params = self._flax_model.init(jax.random.PRNGKey(0)) # Note: this RNG doesnt actually matter because there's no random initialization

        # initialize parameters
        self.images = images
        data_generator_matched = match_to_generator_data(images, seed=seed)
        initial_cov_mat = plugin_estimate_stationary_cov_mat(data_generator_matched, eigenvalue_floor=eigenvalue_floor, suppress_warning=True, verbose=verbose)
        mean_vec = np.ones(self.image_shape[0]**2) * np.mean(data_generator_matched)        
        
        eig_vals, eig_vecs = np.linalg.eigh(initial_cov_mat)
        self.initial_params['params']['eig_vals'] = eig_vals
        self.initial_params['params']['eig_vecs'] = eig_vecs
        self.initial_params['params']['mean_vec'] = mean_vec
        self._state = None
        self._eigenvalue_floor = eigenvalue_floor



    def fit(self, train_images=None, data_seed=None,
            learning_rate=1e2, max_epochs=60, steps_per_epoch=1,  patience=15, 
            batch_size=12, num_val_samples=None, percent_samples_for_validation=0.1,
            eigenvalue_floor=1e-3, gradient_clip=1, momentum=0.9,
            precondition_gradient=False, verbose=True):
        
        if train_images is None:
            train_images = self.images

        num_val_samples = int(train_images.shape[0] * percent_samples_for_validation) if num_val_samples is None else num_val_samples
        
        
        self._optimizer = optax.chain(
            # don't let update size exceed approx parameter size * Learning rate.
            # this prevents tiny, incorrect eigenvalues from making things diverge
            optax.clip(gradient_clip), 
            optax.sgd(learning_rate, momentum=momentum, nesterov=False)
        )

        @jax.jit
        def _train_step(state, imgs):
            loss_fn = lambda params, imgs: state.apply_fn(params, imgs)
            loss, grads = jax.value_and_grad(loss_fn, 0)(state.params, imgs)
            # jax.grad(loss_fn, 0)(state.params, imgs)

            if precondition_gradient:
                # Define a function that computes the log-likelihood from eigenvalues
                def nll_from_eigenvalues(eigenvalues, imgs):
                    params = {'params': {'eig_vecs': state.params['params']['eig_vecs'], 'mean_vec': state.params['params']['mean_vec'],
                                'eig_vals': eigenvalues}}
                    return state.apply_fn(params, imgs)

                fi_fn = jax.jit(jax.hessian(nll_from_eigenvalues, 0))
                # Now compute the Hessian for the initial eigenvalues to get the Fisher Information Matrix
                fisher_information_matrix = fi_fn(state.params['params']['eig_vals'], imgs)

                # precondition the eig_vals gradient
                grads['params']['eig_vals'] = np.linalg.solve(fisher_information_matrix, grads['params']['eig_vals'])


            # mean vec and eig vecs are not updated via gradient descent, but instead by proximal step
            grads['params']['eig_vecs'] = np.zeros_like(grads['params']['eig_vecs'])  
            grads['params']['mean_vec'] = np.zeros_like(grads['params']['mean_vec'])  

            state = state.apply_gradients(grads=grads)

            # proximal step
            eig_vals, eig_vecs = state.params['params']['eig_vals'], state.params['params']['eig_vecs']
            state.params['params']['eig_vals'], state.params['params']['eig_vecs'] = try_to_make_doubly_toeplitz_and_positive_definite(
                eig_vals, eig_vecs, eigenvalue_floor, patch_size=self.image_shape[0])  
            return state, loss
                                                
        if self._state is None:
            def apply_fn(params, x):
                output = self._flax_model.apply(params)
                return self._flax_model.compute_loss(*output, x)
            
            self._state = TrainState.create(apply_fn=apply_fn, params=self.initial_params, tx=self._optimizer)
        else:
            # Fit has already been called and now we're optimizing some more
            self._state = self._state.replace(tx=self._optimizer)


        best_params, val_loss_history = train_model(train_images=train_images, state=self._state, batch_size=batch_size, num_val_samples=int(num_val_samples),
                                                    steps_per_epoch=steps_per_epoch, num_epochs=max_epochs, patience=patience, train_step=_train_step, 
                                                    seed=data_seed,
                                                    verbose=verbose)
        # ensure that eigenvalues are positive definite
        if best_params['params']['eig_vals'].min() < 0:
            warnings.warn('Covariance matrix is not positive definite after running optimization')
            best_params['params']['eig_vals'] = np.where(best_params['params']['eig_vals'] < eigenvalue_floor, eigenvalue_floor, best_params['params']['eig_vals'])
        self._state = self._state.replace(params=best_params)
        # ensure that evs remain positive when converting to cov_mat and back
        while True:
            cov_mat = self.get_cov_mat()
            eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
            if eig_vals.min() > 0:
                break
            warnings.warn('Covariance matrix is not positive definite after running optimization. Increasing eigenvalue floor')
            best_params['params']['eig_vals'] = np.where(best_params['params']['eig_vals'] < eigenvalue_floor, eigenvalue_floor, best_params['params']['eig_vals'])
            self._state = self._state.replace(params=best_params)
            eigenvalue_floor *= 2

        return val_loss_history


    def compute_negative_log_likelihood(self, images, data_seed=None, verbose=True, seed=None):
        if seed is not None:
            warnings.warn('seed argument is deprecated. Use data_seed instead')
            data_seed = seed
        eig_vals, eig_vecs, mean_vec = self._get_current_params()
        cov_mat = eig_vecs @ np.diag(eig_vals) @ eig_vecs.T
        
        while np.linalg.eigvalsh(cov_mat).min() < 0:
            if eig_vals.min() <= 0:
                raise ValueError('Covariance matrix is not positive definite. This should not have happened')
            warnings.warn('Covariance matrix does not retain positive definiteness after after eigenvalue decomposition and recomposition. '
                    'This likely indicates numerical error. Trying to boost the smallest EVs to fix this.')
            floor = eig_vals.min() * 2
            eig_vals = np.where(eig_vals < floor, floor, eig_vals)
            cov_mat = eig_vecs @ np.diag(eig_vals) @ eig_vecs.T
            
        images = match_to_generator_data(images, seed=data_seed)

        lls = _compute_stationary_log_likelihood(images, cov_mat, mean_vec, verbose=verbose)
        return -lls.mean()
    
        
    def generate_samples(self, num_samples, sample_shape=None, ensure_nonnegative=True, seed=None, verbose=True):
        eig_vals, eig_vecs, mean_vec = self._get_current_params()
        cov_mat = eig_vecs @ np.diag(eig_vals) @ eig_vecs.T
        samples = generate_stationary_gaussian_process_samples( 
                    mean_vec, cov_mat, num_samples, sample_shape, ensure_nonnegative=ensure_nonnegative, seed=seed, verbose=verbose)
        return samples

    def get_cov_mat(self):
        eig_vals, eig_vecs, mean_vec = self._get_current_params()
        cov_mat = eig_vecs @ np.diag(eig_vals) @ eig_vecs.T
        return cov_mat

    def get_mean_vec(self):
        eig_vals, eig_vecs, mean_vec = self._get_current_params()
        return mean_vec

    def compute_analytic_entropy(self):
        """
        Compute the differential entropy per pixel of the Gaussian process
        """
        eig_vals, eig_vecs, mean_vec = self._get_current_params()
        D = eig_vals.size
        sum_log_evs = np.sum(np.log(eig_vals))
        gaussian_entropy = 0.5 *(sum_log_evs + D * np.log(2* np.pi * np.e)) / D
        return gaussian_entropy


    def _get_current_params(self):
        """
        Return the initial or optimized params
        """
        if self._state is None:
            eig_vals, eig_vecs = self.initial_params['params']['eig_vals'], self.initial_params['params']['eig_vecs']
            mean_vec = self.initial_params['params']['mean_vec']
        else:
            eig_vals, eig_vecs = self._state.params['params']['eig_vals'], self._state.params['params']['eig_vecs']
            mean_vec = self._state.params['params']['mean_vec']
        return eig_vals, eig_vecs, mean_vec
    




##################################################################################################
#### Full (non-stationary Gaussian Process) ######

class FullGaussianProcess(ProbabilisticImageModel):

    def __init__(self, images, eigenvalue_floor=1e-3, seed=None, verbose=False):
        """
        Estiamte mean and covariance matrix of a full Gaussian process from images
        """
        self.image_shape = images.shape[1:]

        images = match_to_generator_data(images, seed=seed)
        # initialize parameters
        if verbose:
            print('computing full covariance matrix')
        self.cov_mat = estimate_full_cov_mat(images)
        # ensure positive definiteness
        eigvals, eig_vecs = np.linalg.eigh( self.cov_mat)
        eigvals = np.where(eigvals < eigenvalue_floor, eigenvalue_floor, eigvals)
        self.cov_mat = eig_vecs @ np.diag(eigvals) @ eig_vecs.T
        while np.linalg.eigvalsh( self.cov_mat).min() < 0:
            warnings.warn('Covariance matrix is not positive definite even after applying eigenvalue floor. This indicates numerical error.' +
                             'Try raising the eigenvalue floor than the current value of {}'.format(eigenvalue_floor))
            eigenvalue_floor *= 10
            print('trying eigenvalue floor of {}'.format(eigenvalue_floor))
            eigvals = np.where(eigvals < eigenvalue_floor, eigenvalue_floor, eigvals)
            self.cov_mat = eig_vecs @ np.diag(eigvals) @ eig_vecs.T

        if verbose:
            print('computing mean vector')
        self.mean_vec = np.mean(images, axis=0).flatten()     
        

    def fit(self, *args, **kwargs):
        warnings.warn('Full Gaussian process does not require fitting. Skipping fit method.')


    def compute_negative_log_likelihood(self, images, data_seed=None, verbose=True, seed=None):
        if seed is not None:
            warnings.warn('seed argument is deprecated. Use data_seed instead')
            data_seed = seed
        images = match_to_generator_data(images, seed=data_seed)
        # average nll per pixel
        return -gaussian_likelihood(self.cov_mat, self.mean_vec, images).mean() / np.prod(np.array(images.shape[1:]))

    
        
    def generate_samples(self, num_samples, sample_shape=None, ensure_nonnegative=True, seed=None, verbose=True):
        if sample_shape is not None and sample_shape != int(np.sqrt(self.cov_mat.shape[0])):
            raise ValueError('Sample shape must match the shape of training images')
        samples = generate_multivariate_gaussian_samples(self.mean_vec, self.cov_mat, num_samples, seed=seed)
        return samples

