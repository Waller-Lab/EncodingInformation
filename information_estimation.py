"""
Functions for estimating entropy and mutual information
"""
from image_utils import *
from jax import jit
from jax.scipy.special import digamma, gammaln
from functools import partial
import jax.numpy as np

 
def nearest_neighbors_entropy_estimate(X, k=3):
    """
    Estimate the entropy (in nats) of a dsitribution from samples usiing the KL 
    nearest neighbors estimator. 
    
    X : ndarray, shape (n_samples, n_dimensions)
    k : int The k in k-nearest neighbors
    base : base for the logarithms
    """
    return _do_nearest_neighbors_entropy_estimate(X, X.shape[0], X.shape[1], k)

@partial(jit, static_argnums=(1, 2, 3))
def _do_nearest_neighbors_entropy_estimate(X, N, d, k=3):
    """
    Just-in-time compiled helper function for nearest_neighbors_entropy_estimate.
    """
    nn = nearest_neighbors_distance(X, k)

    # compute the log volume of the d-dimensional ball with raidus of the nearest neighbor distance
    log_vd = d * np.log(nn) + d/2 * np.log(np.pi) - gammaln(d/2 + 1)
    # h = np.mean(log_vd + np.log(N - 1) - digamma(k))
    h = np.log(k) - digamma(k) + np.mean(log_vd + np.log(N) - np.log(k))
    return h 


@partial(jit, static_argnums=1)
def nearest_neighbors_distance(X, k):
    """
    Compute the distance to the kth nearest neighbor for each point in X by
    exhaustively searching all points in X.
    
    X : ndarray, shape (n_samples, W, H) or (n_samples, num_features)
    k : int
    """
    X = X.reshape(X.shape[0], -1)
    distance_matrix = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    kth_nn_index = np.argsort(distance_matrix, axis=-1)[:, k]
    kth_nn = X[kth_nn_index, :]
    kth_nn_dist = np.sqrt(np.sum((X - kth_nn)**2, axis=-1))
    return kth_nn_dist


def gaussian_entropy_estimate(X, stationary=True, cutoff_percentile=25, show_plot=False, return_cov_mat=False,
                              verbose=False):
    """
    Estimate the entropy (in nats) of samples from a distribution of images by approximating 
    the distribution as a Gaussian.  i.e. Taking its covariance matrix and 
    computing the entropy of the Gaussian distribution with that same covariance matrix.

    X : ndarray, shape (n_samples, W, H) or (n_samples, num_features)
    stationary : bool, whether to assume the distribution is stationary
    cutoff_percentile : float, if estimating a stationary covariance matrix and it
        is not positive definite, enforce this by setting making all eigenvalues
        below this percentile of the eigenvalue distribution to be this percentile.
    show_plot : bool, whether to show a plot of the eigenvalues and threshold used 
        to make the covariance matrix positive definite.
    return_cov_mat : bool, whether to return the estimated covariance matrix
    """
    X = X.reshape(X.shape[0], -1)
    return _do_gaussian_entropy_estimate(X, X.shape[1], stationary=stationary, 
                                          cutoff_percentile=cutoff_percentile, show_plot=show_plot, 
                                          return_cov_mat=return_cov_mat, verbose=verbose)

# Cant JIT this one because make_positive_definite cant be jitted do to conditionals
# @partial(jit, static_argnums=(1,2,3))
def _do_gaussian_entropy_estimate(X, D, stationary=True, cutoff_percentile=25, show_plot=False, return_cov_mat=False, verbose=False):
    """
    Just-in-time compiled helper function for gaussian_entropy_estimate.
    """
    # np.cov takes D x N shaped data but compute stationary cov mat takes N x D
    zero_centered = X.T - np.mean(X.T, axis=1, keepdims=True)
    if not stationary:
        try:
            cov_mat = np.cov(zero_centered)
        except:
            raise Exception("Couldn't compute covariance matrix")
            
        evs = np.linalg.eigvalsh(cov_mat)
        if np.any(evs < 0):
            warnings.warn("Covariance matrix is not positive definite. This indicates numerical error.")
        sum_log_evs = np.sum(np.log(np.where(evs < 0, 1e-15, evs)))                        
    else:
        cov_mat = compute_stationary_cov_mat(zero_centered.T, verbose=verbose)
        cov_mat = make_positive_definite(cov_mat, cutoff_percentile=cutoff_percentile, show_plot=show_plot, verbose=verbose)

        sum_log_evs = np.sum(np.log(np.linalg.eigvalsh(cov_mat)))
    gaussian_entropy = 0.5 *(sum_log_evs + D * np.log(2* np.pi * np.e))
    if return_cov_mat:
        return gaussian_entropy, cov_mat
    else:
        return gaussian_entropy

@partial(jit, static_argnums=(1,))
def compute_conditional_entropy(images, gaussian_noise_sigma=None):
    """
    Compute the conditional entropy H(Y | X) in nats,
    where Y is a random noisy realization of a random clean image X

    images : ndarray clean image HxW or images NxHxW
    gaussian_noise_sigma : float, if not None, assume gaussian noise with this sigma.
            otherwise assume poisson noise.
    """
    # vectorize
    images = images.reshape(-1, images.shape[-2] * images.shape[-1])
         
    images = np.where(images <= 0, .1, images) #always at least .1 photon

    if gaussian_noise_sigma is None:
        # conditional entropy H(Y | x) for Poisson noise (see derivation in paper)
        return np.mean(0.5 * (images.shape[-1] * np.log(2 * np.pi * np.e) + np.sum(np.log(images), axis=1)))
    else:
        # conditional entropy H(Y | x) for Gaussian noise
        # only depends on the gaussian sigma
        return np.mean(np.sum(images.shape[-1] * 0.5 * np.log(2 * np.pi * np.e * gaussian_noise_sigma**2), axis=1))
    
def estimate_mutual_information(noisy_images, clean_images=None, use_stationary_model=True, 
                                cutoff_percentile=10, show_eigenvalue_plot=False, confidence_interval=None, 
                                num_bootstrap_samples=1000):
    """
    Estimate the mutual information (in bits per pixel) of a stack of noisy images, by making a Gaussian approximation
    to the distribution of noisy images, and subtracting the conditional entropy of the clean images
    If clean_images is not provided, instead compute the conditional entropy of the noisy images.

    noisy : ndarray NxHxW array of images or image patches
    clean_images : ndarray NxHxW array of images or image patches
    use_stationary_model : bool, whether to assume the distribution is stationary
    cutoff_percentile : float, if estimating a stationary covariance matrix and it
        is not positive definite, enforce this by setting making all eigenvalues
        below this percentile of the eigenvalue distribution to be this percentile.
    show_eigenvalue_plot : bool, whether to show a plot of the eigenvalues of the estimated
        stationary covariance matrix and the correction applied to make it positive definite.
    confidence_interval : float, if not None, compute the confidence interval for the
        mutual information estimate using the bootstrap method.
    """
    clean_images_if_available = clean_images if clean_images is not None else noisy_images
    if np.any(clean_images_if_available < 0):   
        warnings.warn(f"{np.sum(clean_images_if_available < 0) / clean_images_if_available.size:.2%} of pixels are negative.")

    def do_the_estimating(noisy_images, clean_images_if_available, verbose=False):
        h_y_given_x = compute_conditional_entropy(clean_images_if_available)
        h_y_given_x_per_pixel_bits = h_y_given_x / (np.log(2) * (clean_images_if_available.shape[-2] * clean_images_if_available.shape[-1]))
        h_y_gaussian = gaussian_entropy_estimate(noisy_images, stationary=use_stationary_model, 
                                                cutoff_percentile=cutoff_percentile,
                                                show_plot=show_eigenvalue_plot)
        h_y_gaussian_per_pixel_bits = h_y_gaussian / (np.log(2) * (noisy_images.shape[-2] * noisy_images.shape[-1]))
        
        mutual_info = (h_y_gaussian_per_pixel_bits - h_y_given_x_per_pixel_bits)
        return mutual_info 

    # compute per pixels entropies in bits
    if confidence_interval is None:        
        return do_the_estimating(noisy_images, clean_images_if_available, verbose=True)
    else:
        if show_eigenvalue_plot:
            warnings.warn("Showing eigenvalue plots with bootstrapped confidence intervals would open, like, a lot of plots. "
                    "So I'm going to go ahead and not do that.")
            show_eigenvalue_plot = False
        # compute bootstrap confidence interval
        mutual_infos = []
        key = jax.random.PRNGKey(0)

        for i in tqdm(range(num_bootstrap_samples), desc="Running bootstraps"):
            key, subkey = jax.random.split(key)
            noisy_images_sample = jax.random.choice(key, noisy_images, shape=(noisy_images.shape[0],), replace=True)
            key, subkey = jax.random.split(key)
            clean_images_sample = jax.random.choice(key, clean_images_if_available, shape=(clean_images_if_available.shape[0],), replace=True)
            mutual_infos.append(do_the_estimating(noisy_images_sample, clean_images_sample, verbose=False))        
            
        mutual_infos.append(do_the_estimating(noisy_images_sample, clean_images_sample, verbose=False))
        mutual_infos = np.array(mutual_infos)
        mutual_info = np.mean(mutual_infos)
        conf_int = [np.percentile(mutual_infos, 50 - confidence_interval/2),
                    np.percentile(mutual_infos, 50 + confidence_interval/2)]

        return mutual_info, conf_int
