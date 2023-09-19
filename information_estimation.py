"""
Functions for estimating entropy and mutual information
"""
from image_utils import *
from jax import jit
from jax.scipy.special import digamma
from functools import partial
import jax.numpy as np

 
def nearest_neighbors_entropy_estimate(X, k=3, base=2):
    """
    Estimate the entropy of a dsitribution from samples usiing the KL 
    nearest neighbors estimator. 
    
    X : ndarray, shape (n_samples, n_dimensions)
    k : int The k in k-nearest neighbors
    base : base for the logarithms
    """
    return _do_nearest_neighbors_entropy_estimate(X, X.shape[1], k, base)

@partial(jit, static_argnums=(1, 2, 3))
def _do_nearest_neighbors_entropy_estimate(X, N, k=3, base=2):
    """
    Just-in-time compiled helper function for nearest_neighbors_entropy_estimate.
    """
    nn = nearest_neighbors_distance(X, k)
    cons = digamma(N) - digamma(k) + N * np.log(2)
    return (cons + N * np.log(nn).mean()) / np.log(base)


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
    kth_nn_dist = np.sum((X - kth_nn)**2, axis=-1)
    return kth_nn_dist


def gaussian_entropy_estimate(X, stationary=True, base=2):
    """
    Estimate the entropy of samples from a distribution of images by approximating 
    the distribution as a Gaussian.  i.e. Taking its covariance matrix and 
    computing the entropy of the Gaussian distribution with that same covariance matrix.

    X : ndarray, shape (n_samples, W, H) or (n_samples, num_features)
    stationary : bool, whether to assume the distribution is stationary
    base : float base for the logarithms
    """
    X = X.reshape(X.shape[0], -1)
    return _do_gaussian_entropy_estimate(X, X.shape[1], stationary=stationary, 
                                         base=base)

# @partial(jit, static_argnums=(1,2,3))
def _do_gaussian_entropy_estimate(X, D, stationary=True, base=2):
    """
    Just-in-time compiled helper function for gaussian_entropy_estimate.
    """
    # np.cov takes D x N shaped data but compute stationary cov mat takes N x D
    zero_centered = X.T - np.mean(X.T, axis=1, keepdims=True)
    if not stationary:
        cov_mat = np.cov(zero_centered)
    else:
        cov_mat = compute_stationary_cov_mat(zero_centered.T)
        cov_mat = make_positive_definite(cov_mat, show_plot=False)

    sum_log_evs = np.sum(np.log2(np.abs(np.linalg.eigvalsh(cov_mat))))
    gaussian_entropy = 0.5 *(sum_log_evs + D * np.log2(2* np.pi * np.e))
    return gaussian_entropy
