"""
Functions for estimating entropy and mutual information
"""
from .image_utils import *
from jax import jit
from jax.scipy.special import digamma, gammaln
from .models.gaussian_process import StationaryGaussianProcess, FullGaussianProcess
from .models.pixel_cnn import PixelCNN

from functools import partial
import jax.numpy as np
import numpy as onp
import warnings


def estimate_information(measurement_model, noise_model, train_set, test_set, 
                         confidence_interval=None, num_bootstraps=100):
    """
    Estimate mutual information in bits per pixel given a probabilistic model of the measurement process p(y)
    and a probabilistic model of the noise process p(y|x). Optionally, estimate a confidence interval using bootstrapping,
    which represents the uncertainty in the estimate due to the finite size of the test set.

    Parameters
    ----------
    measurement_model : MeasurementModel
        A probabilistic model of the measurement process p(y|x) (e.g. PixelCNN, FullGaussian, etc.).
    noise_model : NoiseModel
        A probabilistic model of the noise process p(y|x) (e.g. GaussianNoiseModel, PoissonNoiseModel).
    train_set : ndarray, shape (n_samples, ...)
        The training set of noisy measurements, with different shapes depending on data type/model.
    test_set : ndarray, shape (n_samples, ...)
        The test set of noisy measurements, which should have the same shape as the training set.
    confidence_interval : float, optional
        Confidence interval for the mutual information, estimated via bootstrapping.
    num_bootstraps : int, optional
        Number of times to resample the test set to estimate the confidence interval.

    Returns
    -------
    mutual_info : float
        The mutual information in bits per pixel.
    (lower_bound, upper_bound) : tuple of floats, optional
        Lower and upper bounds of the confidence interval for the mutual information (if confidence_interval is provided).
    """
    # make sure confidence interval is between 0 and 1
    if confidence_interval is not None:
        if confidence_interval <= 0 or confidence_interval >= 1:
            raise ValueError("Confidence interval must be between 0 and 1")
    
    full_dataset = np.concatenate([train_set, test_set])
    nll = measurement_model.compute_negative_log_likelihood(test_set)
    hy_given_x = noise_model.estimate_conditional_entropy(full_dataset)
    mutual_info = (nll - hy_given_x) / np.log(2)
    if confidence_interval is None:
        return mutual_info    
    
    # calculate this way for confidence intervals so it is faster
    nll = measurement_model.compute_negative_log_likelihood(test_set, average=False)

    # estimate confidence interval by bootstrapping data
    nlls = []
    hy_given_xs = []
    for i in tqdm(range(num_bootstraps), desc='Bootstrapping to compute confidence interval'):
        # resample test set
        bootstrap_indices = onp.random.choice(len(test_set), len(test_set), replace=True)
        bootstrapped_nlls = nll[bootstrap_indices]
        nlls.append(bootstrapped_nlls.mean())

        # resample full dataset for conditional entropy
        bootstrap_indices = onp.random.choice(len(full_dataset), len(full_dataset), replace=True)
        bootstrap_full_dataset = full_dataset[bootstrap_indices]
        hy_given_xs.append(noise_model.estimate_conditional_entropy(bootstrap_full_dataset))

    nlls = np.array(nlls)
    hy_given_xs = np.array(hy_given_xs)
    # take all random combinations of nlls and hy_given_xs
    mutual_infos = (nlls[None, :] - hy_given_xs[:, None]) / np.log(2)
    mutual_infos = mutual_infos.flatten()
    # compute confidence interval
    lower_bound = np.percentile(mutual_infos, 50*(1-confidence_interval))
    upper_bound = np.percentile(mutual_infos, 50*(1+confidence_interval))
    return mutual_info, lower_bound, upper_bound

def analytic_multivariate_gaussian_entropy(cov_matrix):
    """
    Compute the entropy of a multivariate Gaussian distribution in a numerically stable manner.

    Parameters
    ----------
    cov_matrix : ndarray, shape (d, d)
        Covariance matrix of the multivariate Gaussian distribution.

    Returns
    -------
    entropy : float
        The entropy of the multivariate Gaussian in bits.
    """

    d = cov_matrix.shape[0]
    entropy = 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * np.sum(np.log(np.linalg.eigvalsh(cov_matrix)))
    return entropy / d
 
def nearest_neighbors_entropy_estimate(X, k=3):
    """
    Estimate the entropy (in nats) of a distribution using the k-nearest neighbors estimator.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_dimensions)
        The samples from the distribution to estimate the entropy for.
    k : int, optional
        The number of nearest neighbors to use in the entropy estimate (default is 3).

    Returns
    -------
    entropy : float
        The estimated entropy in nats.
    """

    return _do_nearest_neighbors_entropy_estimate(X, X.shape[0], X.shape[1], k)

@partial(jit, static_argnums=(1, 2, 3))
def _do_nearest_neighbors_entropy_estimate(X, N, d, k=3):
    """
    Just-in-time compiled helper function for nearest_neighbors_entropy_estimate.
    """
    nn = _nearest_neighbors_distance(X, k)

    # compute the log volume of the d-dimensional ball with raidus of the nearest neighbor distance
    log_vd = d * np.log(nn) + d/2 * np.log(np.pi) - gammaln(d/2 + 1)
    # h = np.mean(log_vd + np.log(N - 1) - digamma(k))
    h = np.log(k) - digamma(k) + np.mean(log_vd + np.log(N) - np.log(k))
    return h 


@partial(jit, static_argnums=1)
def _nearest_neighbors_distance(X, k):
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


@partial(jit, static_argnums=(1,))
def estimate_conditional_entropy(images, gaussian_noise_sigma=None):
    """
    Estimate the conditional entropy H(Y | X) in nats per pixel, assuming either Gaussian or Poisson noise.

    Parameters
    ----------
    images : ndarray
        Clean images, shape (NxHxW) or (NxHxWxC).
    gaussian_noise_sigma : float, optional
        Standard deviation of the Gaussian noise. If None, Poisson noise is assumed.

    Returns
    -------
    conditional_entropy : float
        The estimated conditional entropy per pixel.
    """

    warnings.warn("This function is deprecated. Use GaussianNoiseModel or PoissonNoiseModel instead.")
    # vectorize
    images = images.reshape(-1, images.shape[-2] * images.shape[-1])
    n_pixels = images.shape[-1]
         
    # if np.any(images < 0):
    #     warnings.warn(f"{np.sum(images < 0) / images.size:.2%} of pixels are negative.")
    # images = np.where(images <= 0, 0, images) #always at least fraction of photon

    if gaussian_noise_sigma is None:
        # conditional entropy H(Y | x) for Poisson noise 
        gaussian_approx = 0.5 * (np.log(2 * np.pi * np.e) + np.log(images))
        gaussian_approx = np.where(images <= 0, 0, gaussian_approx)
        per_image_entropies = np.sum(gaussian_approx, axis=1) / n_pixels
        h_y_given_x = np.mean(per_image_entropies)

        # add small amount of gaussian noise (e.g. read noise)
        # read_noise_sigma = 1
        # h_y_given_x += 0.5 * np.log(2 * np.pi * np.e * read_noise_sigma**2)
        return h_y_given_x
    else:
        # conditional entropy H(Y | x) for Gaussian noise
        # only depends on the gaussian sigma
        return  0.5 * np.log(2 * np.pi * np.e * gaussian_noise_sigma**2)
    

def run_bootstrap(data, estimation_fn, num_bootstrap_samples=200, confidence_interval=90, seed=1234, return_median=True, 
                  upper_bound_confidence_interval=False,
                  verbose=False):
    """
    Run a bootstrap estimation procedure using a given estimation function on the provided data.

    Parameters
    ----------
    data : ndarray or dict of ndarrays
        The data to use for the bootstrap estimation. If a dictionary, each value must have the same number of samples.
    estimation_fn : function
        The function to use for estimating the desired quantity. It should accept the data as its input.
    num_bootstrap_samples : int, optional
        The number of bootstrap samples to generate (default is 200).
    confidence_interval : float, optional
        The confidence interval for the estimate, expressed as a percentage (default is 90%).
    seed : int, optional
        Random seed for generating bootstrap samples.
    return_median : bool, optional
        Whether to return the median (True) or mean (False) of the bootstrap estimates (default is True).
    upper_bound_confidence_interval : bool, optional
        If True, returns the upper-bound confidence interval (default is False).
    verbose : bool, optional
        If True, shows a progress bar.

    Returns
    -------
    estimate : float
        The median/mean estimate of the desired quantity across bootstrap samples.
    confidence_interval : list of float
        The lower and upper bounds of the confidence interval.
    """

    key = jax.random.PRNGKey(onp.random.randint(0, 1000000))
    N = data.shape[0] if not isinstance(data, dict) else data[list(data.keys())[0]].shape[0]
    results = []
    if verbose:
        iterator = tqdm(range(num_bootstrap_samples), desc="Running bootstraps")
    else:
        iterator = range(num_bootstrap_samples)
    for i in iterator:
        key, subkey = jax.random.split(key)
        if not isinstance(data, dict):
            # if verbose: 
            #     print('key', subkey, '\n')
            random_indices = jax.random.choice(subkey, np.arange(data.shape[0]), shape=(N,), replace=True)
            data_sample = data[random_indices, ...]
            results.append(estimation_fn(data_sample))
        else:
            data_samples = {}
            for k, v in data.items():
                key, subkey = jax.random.split(key)
                # if verbose: 
                #     print('key', subkey, '\n')
                random_indices = jax.random.choice(subkey, np.arange(v.shape[0]), shape=(N,), replace=True)
                data_samples[k] = v[random_indices, ...]
            results.append(estimation_fn(**data_samples))
        
    results = np.array(results)
    if not return_median:
        m = np.mean(results)
    elif upper_bound_confidence_interval:
        m = np.percentile(results, confidence_interval // 2)
    else:
        m = np.median(results)
    if upper_bound_confidence_interval:
        conf_int = [np.min(results), np.percentile(results, confidence_interval)]
    else:
        conf_int = [np.percentile(results, 50 - confidence_interval/2),
                np.percentile(results, 50 + confidence_interval/2)]
    return m, conf_int
        


def  estimate_task_specific_mutual_information(noisy_images, labels, test_set_fraction=0.2,
                                 patience=None, num_val_samples=None, batch_size=None, max_epochs=None, learning_rate=None, # generic params                                
                                 steps_per_epoch=None, num_hidden_channels=None, num_mixture_components=None, do_lr_decay=False, # pixelcnn params
                                 return_entropy_model=False,
                                 verbose=False,):
    """
    DEPRECATED: Use estimate_information() instead.

    Estimate the mutual information (in bits per pixel) between noisy images and labels using a PixelCNN entropy model.

    Parameters
    ----------
    noisy_images : ndarray
        Noisy images of shape (NxHxW) or (NxHxWxC).
    labels : ndarray
        Labels of shape (NxK) as one-hot vectors.
    test_set_fraction : float, optional
        Fraction of the data to be used as the test set for computing the entropy upper bound (default is 0.2).
    patience : int, optional
        How many iterations to wait for validation loss to improve (used in training). If None, the default is used.
    num_val_samples : int, optional
        Number of validation samples to use. If None, the default is used.
    batch_size : int, optional
        Batch size for training the PixelCNN model. If None, the default is used.
    max_epochs : int, optional
        Maximum number of epochs for training. If None, the default is used.
    learning_rate : float, optional
        Learning rate for training. If None, the default is used.
    steps_per_epoch : int, optional
        Number of steps per epoch for training the PixelCNN model.
    num_hidden_channels : int, optional
        Number of hidden channels in the PixelCNN model.
    num_mixture_components : int, optional
        Number of mixture components in the PixelCNN output.
    do_lr_decay : bool, optional
        Whether to decay the learning rate during training (default is False).
    return_entropy_model : bool, optional
        If True, returns the trained PixelCNN entropy model along with the mutual information (default is False).
    verbose : bool, optional
        If True, prints out the estimated values during the process.

    Returns
    -------
    mutual_info : float
        The estimated mutual information in bits per pixel.
    pixelcnn : PixelCNN model, optional
        If `return_entropy_model` is True, returns the trained PixelCNN model along with the mutual information.
    """

    warnings.warn("This function is deprecated. Use estimate_information() instead.")
    if np.any(noisy_images < 0):   
        warnings.warn(f"{np.sum(noisy_images < 0) / noisy_images.size:.2%} of pixels are negative.")
    if np.mean(noisy_images) < 20:
        warnings.warn(f"Mean pixel value is {np.mean(noisy_images):.2f}. More accurate results can probably be obtained"
                        "by setting estimate_conditional_from_model_samples=True")

    # duplicate training set and add labels of all 0s to the copy
    noisy_images = np.concatenate([noisy_images, noisy_images], axis=0)
    labels = np.concatenate([labels, np.zeros_like(labels)], axis=0)
    # shuffle the data
    shuffled_indices = onp.random.permutation(noisy_images.shape[0])
    noisy_images = noisy_images[shuffled_indices]
    labels = labels[shuffled_indices]

    training_set = noisy_images[:int(noisy_images.shape[0] * (1 - test_set_fraction))]
    test_set = noisy_images[-int(noisy_images.shape[0] * test_set_fraction):]
    training_set_labels = labels[:int(noisy_images.shape[0] * (1 - test_set_fraction))]
    test_set_labels = labels[-int(noisy_images.shape[0] * test_set_fraction):]


    arch_args = dict(num_hidden_channels=num_hidden_channels, num_mixture_components=num_mixture_components)
    arch_args = {k: v for k, v in arch_args.items() if v is not None}
    # collect all hyperparams that are not None
    hyperparams = {}
    for k, v in dict(patience=patience, num_val_samples=num_val_samples, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                            learning_rate=learning_rate, max_epochs=max_epochs, do_lr_decay=do_lr_decay).items():
            if v is not None:
                hyperparams[k] = v

    ### Estimate the entropy given the labels
    pixelcnn = PixelCNN(**arch_args)
    pixelcnn.fit(training_set, training_set_labels, verbose=verbose, **hyperparams)


    has_real_label_mask = np.sum(test_set_labels, axis=-1) > 0
    h_y_t = pixelcnn.compute_negative_log_likelihood(test_set[has_real_label_mask], test_set_labels[has_real_label_mask])
    h_y = pixelcnn.compute_negative_log_likelihood(test_set, np.zeros_like(test_set_labels))


    mutual_info = (h_y - h_y_t) / np.log(2)
    if verbose:
        print(f"Estimated H(Y|T) (Upper bound) = {h_y_t:.4f} differential entropy/pixel")
        print(f"Estimated H(Y) (Upper bound) = {h_y:.4f} differential entropy/pixel")
        print(f"Estimated I(Y;X) = {mutual_info:.4f} bits/pixel")
    if return_entropy_model:
        return mutual_info, pixelcnn
    return mutual_info

    
def  estimate_mutual_information(noisy_images, clean_images=None, entropy_model='gaussian', test_set_fraction=0.1,
                                  gaussian_noise_sigma=None, estimate_conditional_from_model_samples=False,
                                 patience=None, num_val_samples=None, batch_size=None, max_epochs=None, learning_rate=None, # generic params
                                 use_iterative_optimization=True, eigenvalue_floor=1e-3, gradient_clip=None, momentum=None, analytic_marginal_entropy=False,# gaussian params
                                 steps_per_epoch=None, num_hidden_channels=None, num_mixture_components=None, # pixelcnn params
                                 do_lr_decay=False, add_gaussian_training_noise=False, condition_vectors=None, # pixelcnn params
                                 return_entropy_model=False, verbose=False,):
    """
    DEPRECATED: Use estimate_information() instead.

    Estimate the mutual information (in bits per pixel) for a stack of noisy images using a probabilistic model (Gaussian or PixelCNN).
    Subtracts the conditional entropy assuming Poisson or Gaussian noise. Clean images can be used to estimate the conditional entropy,
    or it can be approximated from the noisy images themselves.

    Parameters
    ----------
    noisy_images : ndarray
        Stack of noisy images or image patches (NxHxW).
    clean_images : ndarray, optional
        Clean images corresponding to the noisy images. If None, the noisy images themselves are used.
    entropy_model : str, optional
        The model used for estimating the entropy of the noisy images. Options are:
        - 'gaussian': Uses a stationary Gaussian process model.
        - 'pixelcnn': Uses a PixelCNN model.
        - 'full_gaussian': Uses a full covariance matrix for the Gaussian process.
    test_set_fraction : float, optional
        Fraction of the data to be used as the test set for computing the entropy upper bound (default is 0.1).
    gaussian_noise_sigma : float, optional
        If provided, assumes that the noisy images arise from additive Gaussian noise with this standard deviation.
        Otherwise, Poisson noise is assumed.
    estimate_conditional_from_model_samples : bool, optional
        If True, estimates the conditional entropy from samples generated by a model fit to the data, rather than the data itself.
    patience : int, optional
        Number of iterations to wait for validation loss to improve (used for iterative optimization in the Gaussian process model).
    num_val_samples : int, optional
        Number of validation samples to use (default is None).
    batch_size : int, optional
        Batch size for training (default is None).
    max_epochs : int, optional
        Maximum number of epochs for training (default is None).
    learning_rate : float, optional
        Learning rate for training (default is None).
    use_iterative_optimization : bool, optional
        If True, performs iterative optimization to refine the stationary Gaussian process estimate (default is True).
    eigenvalue_floor : float, optional
        Sets the minimum allowed eigenvalue for the covariance matrix in the Gaussian process (default is 1e-3).
    gradient_clip : float, optional
        If using iterative optimization with a Gaussian model, clip the gradients to this value (default is None).
    momentum : float, optional
        Momentum for gradient descent in iterative optimization of the Gaussian model (default is None).
    analytic_marginal_entropy : bool, optional
        If True, uses the analytic entropy of the Gaussian fit for H(Y) instead of upper bounding it with the negative log likelihood.
    steps_per_epoch : int, optional
        Number of steps per epoch for training the PixelCNN model (if entropy_model is 'pixelcnn').
    num_hidden_channels : int, optional
        Number of hidden channels in the PixelCNN model (if entropy_model is 'pixelcnn').
    num_mixture_components : int, optional
        Number of mixture components in the PixelCNN output (if entropy_model is 'pixelcnn').
    do_lr_decay : bool, optional
        Whether to decay the learning rate during training for the PixelCNN model (default is False).
    add_gaussian_training_noise : bool, optional
        Whether to add Gaussian noise to the training data instead of uniform noise (default is False).
    condition_vectors : ndarray, optional
        Conditioning vectors to use for the PixelCNN model (if entropy_model is 'pixelcnn'), with shape (n_samples, d).
    return_entropy_model : bool, optional
        If True, returns the trained entropy model along with the mutual information (default is False).
    verbose : bool, optional
        If True, prints out the estimated values during the process.

    Returns
    -------
    mutual_info : float
        The estimated mutual information in bits per pixel.
    entropy_model : model, optional
        If `return_entropy_model` is True, returns the trained entropy model along with the mutual information.
    """

    warnings.warn("This function is deprecated. Use estimate_information() instead.")
    clean_images_if_available = clean_images if clean_images is not None else noisy_images
    if np.any(clean_images_if_available < 0):   
        warnings.warn(f"{np.sum(clean_images_if_available < 0) / clean_images_if_available.size:.2%} of pixels are negative.")
    if np.mean(clean_images_if_available) < 20 and not estimate_conditional_from_model_samples:
        warnings.warn(f"Mean pixel value is {np.mean(clean_images_if_available):.2f}. More accurate results can probably be obtained"
                        "by setting estimate_conditional_from_model_samples=True")

    ### Estimate the conditional entropy H(Y | X) from the clean images (or samples from a model fit to them)
    if estimate_conditional_from_model_samples:
        stationary_gp = StationaryGaussianProcess(clean_images_if_available, eigenvalue_floor=eigenvalue_floor)
        if use_iterative_optimization:
            hyperparams = {}
            # collect all hyperparams that are not None
            for k, v in dict(patience=patience, num_val_samples=num_val_samples, batch_size=batch_size,
                           gradient_clip=gradient_clip, learning_rate=learning_rate, momentum=momentum, max_epochs=max_epochs).items():
                if v is not None:
                    hyperparams[k] = v
            val_loss_history = stationary_gp.fit(clean_images_if_available, eigenvalue_floor=eigenvalue_floor, verbose=verbose, **hyperparams)  
        clean_images_if_available = stationary_gp.generate_samples(num_samples=clean_images_if_available.shape[0])
        
    h_y_given_x = estimate_conditional_entropy(clean_images_if_available, gaussian_noise_sigma=gaussian_noise_sigma,)

    ### Fit an entropy model to the noisy images
    # shuffle
    noisy_images = noisy_images[jax.random.permutation(jax.random.PRNGKey(onp.random.randint(10000)), np.arange(noisy_images.shape[0]))]
    training_set = noisy_images[:int(noisy_images.shape[0] * (1 - test_set_fraction))]
    test_set = noisy_images[-int(noisy_images.shape[0] * test_set_fraction):]
    if condition_vectors is not None:
        training_condition_vectors = condition_vectors[:int(noisy_images.shape[0] * (1 - test_set_fraction))]
        test_condition_vectors = condition_vectors[-int(noisy_images.shape[0] * test_set_fraction):]
    else:
        training_condition_vectors = None
        test_condition_vectors = None
    if entropy_model == 'gaussian':
        noisy_image_model = StationaryGaussianProcess(training_set, eigenvalue_floor=eigenvalue_floor)
        if use_iterative_optimization:
            hyperparams = {}
            # collect all hyperparams that are not None
            for k, v in dict(patience=patience, num_val_samples=num_val_samples, batch_size=batch_size,
                           gradient_clip=gradient_clip, learning_rate=learning_rate, momentum=momentum, max_epochs=max_epochs).items():
                if v is not None:
                    hyperparams[k] = v
            val_loss_history = noisy_image_model.fit(training_set, eigenvalue_floor=eigenvalue_floor, verbose=verbose, **hyperparams)        
    elif entropy_model == 'pixelcnn' or entropy_model == 'pixel_cnn':
        arch_args = dict(num_hidden_channels=num_hidden_channels, num_mixture_components=num_mixture_components)
        arch_args = {k: v for k, v in arch_args.items() if v is not None}
        noisy_image_model = PixelCNN(**arch_args)
        # collect all hyperparams that are not None
        hyperparams = {}
        for k, v in dict(patience=patience, num_val_samples=num_val_samples, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                             learning_rate=learning_rate, max_epochs=max_epochs, do_lr_decay=do_lr_decay, condition_vectors=training_condition_vectors).items():
                if v is not None:
                 hyperparams[k] = v
        noisy_image_model.fit(training_set, verbose=verbose, **hyperparams, add_gaussian_noise=add_gaussian_training_noise)
    elif entropy_model == 'full_gaussian':
        noisy_image_model = FullGaussianProcess(training_set, eigenvalue_floor=eigenvalue_floor)
    else:
        raise ValueError(f"Unrecognized entropy model {entropy_model}")
  
    if analytic_marginal_entropy:
        h_y = noisy_image_model.compute_analytic_entropy()
    else:
        ### Estimate the entropy of the noisy images using the upper bound provided by the entropy model negative log likelihood
        if condition_vectors is not None:
            h_y = noisy_image_model.compute_negative_log_likelihood(test_set, conditioning_vecs=test_condition_vectors, verbose=verbose)
        else:
            h_y = noisy_image_model.compute_negative_log_likelihood(test_set, verbose=verbose)

    mutual_info = (h_y - h_y_given_x) / np.log(2)
    if verbose:
        print(f"Estimated H(Y|X) = {h_y_given_x:.4f} differential entropy/pixel")
        if analytic_marginal_entropy:
            print(f"Estimated H(Y) = {h_y:.4f} differential entropy/pixel")
        else:
            print(f"Estimated H(Y) (Upper bound) = {h_y:.4f} differential entropy/pixel")
        print(f"Estimated I(Y;X) = {mutual_info:.4f} bits/pixel")
    if return_entropy_model:
        return mutual_info, noisy_image_model
    return mutual_info 

