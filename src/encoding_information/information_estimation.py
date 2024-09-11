"""
Functions for estimating entropy and mutual information
"""
from encoding_information.image_utils import *
from jax import jit
from jax.scipy.special import digamma, gammaln
from encoding_information.models.gaussian_process import StationaryGaussianProcess, FullGaussianProcess
from encoding_information.models.pixel_cnn import PixelCNN

from functools import partial
import jax.numpy as np
import warnings


def analytic_multivariate_gaussian_entropy(cov_matrix):
    """
    Numerically stable computation of the analytic entropy of a multivariate gaussian
    """
    d = cov_matrix.shape[0]
    entropy = 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * np.sum(np.log(np.linalg.eigvalsh(cov_matrix)))
    return entropy / d
 
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


@partial(jit, static_argnums=(1,))
def estimate_conditional_entropy(images, gaussian_noise_sigma=None):
    """
    Compute the conditional entropy H(Y | X) in "nats" 
    (differential entropy doesn't really have units...) per pixel,
    where Y is a random noisy realization of a random clean image X

    images : ndarray clean image HxW or images NxHxW
    gaussian_noise_sigma : float, if not None, assume gaussian noise with this sigma.
            otherwise assume poisson noise.
    """
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
    Runs a bootstrap estimation procedure on the given data using the provided estimation function.

    Parameters:
    -----------
    data : ndarray, shape (n_samples, ...) or a dictionary of ndarrays
        The data to be used for the bootstrap estimation. If a dictionary is provided, each value in the dictionary
        should be an ndarray with the same number of samples.
    estimation_fn : function
        The function to be used for estimating the desired quantity from the data. This function should take a single
        argument, which is the data to be used for the estimation.
    num_bootstrap_samples : int, optional (default=1000)
        The number of bootstrap samples to generate.
    confidence_interval : float, optional (default=90)
        The confidence interval to use for the estimation, expressed as a percentage.
    seed : int, optional (default=1234)
        The random seed to use for generating the bootstrap samples.
    return_median : bool, optional (default=True)
        Whether to return the median or mean estimate of the desired quantity across all bootstrap samples.
    upper_bound_confidence_interval : bool, optional (default=False)
        Whether to return a confidence interval on 0-(confidence_interval) or the regulare centered one 
    verbose : bool, optional (default=False)
        Print progress bar

    Returns:
    --------
    mean/median : float
        The median/mean estimate of the desired quantity across all bootstrap samples.
    conf_int : list of floats
        The lower and upper bounds of the confidence interval for the estimate, expressed as percentiles of the
        bootstrap sample distribution.
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
    Estimate the mutual information (in bits per pixel) between the noisy images and the labels using a PixelCNN entropy model.

    noisy_images : ndarray NxHxW array of images or image patches
    labels : ndarray NxK array of one-hot vectors of class labels
    test_set_fraction : float, fraction of the noisy data to use a test set for computing the entropy upper bound

    patience : int, How many iterations to wait for validation loss to improve. If None, use the default for the chosen model
    num_val_samples : int, How many samples to use for validation. If None, use the default for the chosen model
    batch_size : int, The batch size to use for training. If None, use the default for the chosen model
    max_epochs : int, The maximum number of epochs to train for. If None, use the default for the chosen model
    learning_rate : float, If None, use the default for the chosen model

    steps_per_epoch : int, (if entropy_model='pixelcnn') number of steps per epoch
    num_hidden_channels : int, (if entropy_model='pixelcnn') number of hidden channels in the PixelCNN
    num_mixture_components : int, (if entropy_model='pixelcnn') number of mixture components in the PixelCNN output
    do_lr_decay : bool, (if entropy_model='pixelcnn') whether to decay the learning rate during training

    return_entropy_model : bool, whether to return the noisy image entropy model
    verbose : bool, whether to print out the estimated values
    """
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
        print(f"Estimated H(Y|T) (Upper bound) = {h_y_t:.3f} differential entropy/pixel")
        print(f"Estimated H(Y) (Upper bound) = {h_y:.3f} differential entropy/pixel")
        print(f"Estimated I(Y;X) = {mutual_info:.3f} bits/pixel")
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
    Estimate the mutual information (in bits per pixel) of a stack of noisy images, by upper bounding the entropy of the noisy
    images using a probabilistic model (either a stationary Gaussian process or a PixelCNN) and subtracting the conditional entropy
    assuming Poisson distributed shot noise, or additive Gaussian noise. Uses clean_images to estimate the conditional entropy
    if provided. Otherwise, approximates the conditional entropy from the noisy images themselves.

    noisy_images : ndarray NxHxW array of images or image patches
    clean_images : ndarray NxHxW array of images or image patches
    entropy_model : str, which model to use for estimating the entropy of the noisy images. Either 'gaussian' 
            (meaning stationary gaussian process), 'pixelcnn', or 'full_gaussian' (meaning full covariance matrix)
    test_set_fraction : float, fraction of the noisy data to use a test set for computing the entropy upper bound

    gaussian_noise_sigma : float, if not None, assume noisy images arose from additive gaussian noise with this sigma.
                                     Otherwise assume poisson noise.
    estimate_conditional_from_model_samples : bool, whether to estimate the conditional entropy from a model fit to them
        rather than from the the data iteself.  

    patience : int, How many iterations to wait for validation loss to improve. If None, use the default for the chosen model
    num_val_samples : int, How many samples to use for validation. If None, use the default for the chosen model
    batch_size : int, The batch size to use for training. If None, use the default for the chosen model
    max_epochs : int, The maximum number of epochs to train for. If None, use the default for the chosen model
    learning_rate : float, If None, use the default for the chosen model

    use_iterative_optimization : bool, (if entropy_model='gaussian') whether to use iterative optimization to refine 
                                    the stationary Gaussian process estimate
    eigenvalue_floor : float, (if entropy_model='gaussian') make the eigenvalues of the covariance matrix at least this large
    gradient_clip : float, (if model='gaussian' and use_iterative_optimization=True) clip gradients to this value
    momentum : float, (if model='gaussian' and use_iterative_optimization=True) momentum for gradient descent
    analytic_marginal_entropy : bool, (if model='gaussian') use the analytic entropy of the Gaussian fit for H(Y) instead
                            of upper bounding it with the negative log likelihood of the Gaussian fit

    steps_per_epoch : int, (if entropy_model='pixelcnn') number of steps per epoch
    num_hidden_channels : int, (if entropy_model='pixelcnn') number of hidden channels in the PixelCNN
    num_mixture_components : int, (if entropy_model='pixelcnn') number of mixture components in the PixelCNN output
    do_lr_decay : bool, (if entropy_model='pixelcnn') whether to decay the learning rate during training
    add_gaussian_training_noise : bool, (if entropy_model='pixelcnn') whether to add gaussian noise to the training data instead of uniform noise
    condition_vectors : ndarray, (if entropy_model='pixelcnn') array of conditioning vectors to use for the PixelCNN. should be of shape (n_samples, d),
      where d is the dimensionality of the conditioning vector

    return_entropy_model : bool, whether to return the noisy image entropy model
    verbose : bool, whether to print out the estimated values
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
        print(f"Estimated H(Y|X) = {h_y_given_x:.3f} differential entropy/pixel")
        if analytic_marginal_entropy:
            print(f"Estimated H(Y) = {h_y:.3f} differential entropy/pixel")
        else:
            print(f"Estimated H(Y) (Upper bound) = {h_y:.3f} differential entropy/pixel")
        print(f"Estimated I(Y;X) = {mutual_info:.3f} bits/pixel")
    if return_entropy_model:
        return mutual_info, noisy_image_model
    return mutual_info 

