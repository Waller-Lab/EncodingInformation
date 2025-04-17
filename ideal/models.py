from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax
import numpy as onp
from tqdm import tqdm
import warnings
from enum import Enum, auto
from typing import List, Union, Tuple, Optional, Callable
from functools import partial


class MeasurementType(Enum):
    """
    Enum class to define different types of measurements.
    
    Attributes
    ----------
    HW : Enum
        Single-channel image measurement (Height, Width).
    HWC : Enum
        Multi-channel image measurement (Height, Width, Channels).
    D : Enum
        Vectorized data measurement.
    """
    HW = auto() # single channel image 
    HWC = auto() # multi-channel image
    D = auto() # vectorized data


class MeasurementModel(ABC):
    """
    Base class for probabilistic models of images and other measurement data.

    Parameters
    ----------
    measurement_types : Union[MeasurementType, List[MeasurementType]], optional
        The type(s) of measurements supported by this model. If None, all types are supported.
    measurement_dtype : type, optional
        The data type of the measurements, either float or complex.

    Attributes
    ----------
    measurement_types : list
        A list of supported measurement types.
    measurement_dtype : type
        The data type of the measurements.
    """
    
    def __init__(self, measurement_types: Union[MeasurementType, List[MeasurementType]] = None, measurement_dtype: type = float) -> None:
        """
        Initialize the model with the type of measurement and data type.

        Parameters
        ----------
        measurement_type : MeasurementType
            Type of measurement (MeasurementType.HW, MeasurementType.HWC, or MeasurementType.D).
            If None, then the model can accept any type of measurement.
        measurement_dtype : type
            Data type of the measurements (float or complex)
        """
        if measurement_types is None:
            self.measurement_types = None
        else:
            self.measurement_types = measurement_types if isinstance(measurement_types, list) or \
                        isinstance(measurement_types, tuple) else [measurement_types]
    
        if measurement_dtype not in (float, complex):
            raise ValueError("measurement_dtype must be either float or complex")
        self.measurement_dtype = measurement_dtype

    def _validate_data(self, data: Union[List, jnp.ndarray, onp.ndarray]):
        """
        Validate the input data, ensuring it matches the expected type and dtype.

        Parameters
        ----------
        data : Union[List, jnp.ndarray, onp.ndarray]
            Input data to validate.

        Raises
        ------
        ValueError
            If the data does not match the expected type or dtype.
        """
        if isinstance(data, list):
            data = jnp.array(data)

        if self.measurement_dtype == float and not jnp.issubdtype(data.dtype, jnp.floating):
            raise ValueError(f"Expected float dtype, but got {data.dtype}")
        elif self.measurement_dtype == complex and not jnp.issubdtype(data.dtype, jnp.complexfloating):
            raise ValueError(f"Expected complex dtype, but got {data.dtype}")
        
        # Check if data matches any of the valid measurement types
        if self.measurement_types is not None:
            valid = False
            for measurement_type in self.measurement_types:
                if  measurement_type.name == MeasurementType.HW.name and len(data.shape) == 3:
                    valid = True
                    break
                elif measurement_type.name == MeasurementType.HWC.name and len(data.shape) == 4:
                    valid = True
                    break
                elif measurement_type.name == MeasurementType.D.name and len(data.shape) == 2:
                    valid = True
                    break

            if not valid:
                raise ValueError(f"Data shape {data.shape} does not match any valid measurement type {self.measurement_types}")

    @abstractmethod
    def fit(self, train_images, learning_rate=1e-2, max_epochs=200, steps_per_epoch=100,  patience=10, 
            batch_size=64, num_val_samples=None, percent_samples_for_validation=0.1,
            data_seed=None, model_seed=None, verbose=True):
        """
        Train the model on the provided images.
        """
        pass
    
    @abstractmethod
    def compute_negative_log_likelihood(self, images, data_seed=123, average=True, verbose=True):
        """
        Compute the NLL of the images under the model
        """
        pass
    
    @abstractmethod
    def generate_samples(self, num_samples, sample_shape=None, verbose=True):
        """
        Generate samples from the model.
        """
        pass


@jax.jit
def _add_uniform_noise_fn(key: jnp.ndarray, images: jnp.ndarray) -> jnp.ndarray:
    """
    Add uniform noise to images using JAX.

    Parameters
    ----------
    key : jnp.ndarray
        JAX PRNG key for random number generation.
    images : jnp.ndarray
        The input images.

    Returns
    -------
    jnp.ndarray
        The images with added uniform noise.
    """
    return images + jax.random.uniform(key, shape=images.shape, minval=0, maxval=1)


def make_dataset_generators(data, batch_size, num_val_samples, add_uniform_noise=True, 
                            add_gaussian_noise=False, condition_vectors=None, seed=None):
    """
    Create JAX-based dataset generators for training and validation data.
    
    Parameters
    ----------
    data : jnp.ndarray
        Input data to create generators from.
    batch_size : int
        Size of each batch.
    num_val_samples : int
        Number of samples to use for validation.
    add_uniform_noise : bool, optional
        Whether to add uniform noise to the data.
    add_gaussian_noise : bool, optional
        Whether to add Gaussian noise to the data.
    condition_vectors : jnp.ndarray, optional
        Conditioning vectors associated with the data.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    tuple
        Tuple containing training and validation data generators.
    """
    if num_val_samples > data.shape[0]:
        raise ValueError("Number of validation samples must be less than the number of training samples")
    
    data = data.astype(jnp.float32)

    # Split images into train and validation
    val_images = data[:num_val_samples]
    train_images = data[num_val_samples:]
    
    # Create PRNG keys for noise addition
    if seed is not None:
        key = jax.random.PRNGKey(seed)
    else:
        key = jax.random.PRNGKey(0)
    
    # Add noise if requested
    if add_gaussian_noise and add_uniform_noise:
        raise ValueError("Cannot add both gaussian and uniform noise")
    
    def create_batches(images, batch_size, key, add_noise=True):
        num_samples = images.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        batches = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch = images[start_idx:end_idx]
            
            if add_noise:
                subkey = jax.random.fold_in(key, i)
                if add_gaussian_noise:
                    batch = batch + jax.random.normal(subkey, shape=batch.shape)
                elif add_uniform_noise:
                    batch = _add_uniform_noise_fn(subkey, batch)
            
            batches.append(batch)
        
        return batches
    
    # Create training and validation batches
    train_key, val_key = jax.random.split(key)
    train_batches = create_batches(train_images, batch_size, train_key, 
                                  add_noise=(add_uniform_noise or add_gaussian_noise))
    val_batches = create_batches(val_images, batch_size, val_key,
                                add_noise=(add_uniform_noise or add_gaussian_noise))
    
    # Create simple generators
    def train_generator():
        for batch in train_batches:
            yield batch
    
    def val_generator():
        for batch in val_batches:
            yield batch
    
    return train_generator(), val_generator


def match_to_generator_data(data, seed=None, add_uniform_noise=True):
    """
    Add noise to data to match the noise added during training.
    
    Parameters
    ----------
    data : jnp.ndarray
        Input data to add noise to.
    seed : int, optional
        Random seed for reproducibility.
    add_uniform_noise : bool, optional
        Whether to add uniform noise to the data.
        
    Returns
    -------
    jnp.ndarray
        Data with added noise.
    """
    if not add_uniform_noise:
        return data
    
    key = jax.random.PRNGKey(0 if seed is None else seed)
    return _add_uniform_noise_fn(key, data)


@jax.jit
def estimate_full_cov_mat(patches):
    """
    Take an NxWxH stack of patches, and compute the covariance matrix of the vectorized patches.
    
    Parameters
    ----------
    patches : jnp.ndarray
        Input patches to compute covariance matrix from.
        
    Returns
    -------
    jnp.ndarray
        Covariance matrix of the vectorized patches.
    """
    vectorized_patches = patches.reshape(patches.shape[0], -1).T
    # center on 0
    vectorized_patches = vectorized_patches - jnp.mean(vectorized_patches, axis=1, keepdims=True)
    return jnp.cov(vectorized_patches).reshape(vectorized_patches.shape[0], vectorized_patches.shape[0])


@jax.jit
def gaussian_log_likelihood_single(sample, mean_vec, cov_mat):
    """
    Compute the log likelihood of a single sample under a multivariate Gaussian.
    
    Parameters
    ----------
    sample : jnp.ndarray
        Input sample.
    mean_vec : jnp.ndarray
        Mean vector of the Gaussian.
    cov_mat : jnp.ndarray
        Covariance matrix of the Gaussian.
        
    Returns
    -------
    float
        Log likelihood of the sample.
    """
    sample_flat = sample.reshape(-1)
    diff = sample_flat - mean_vec
    
    # Use eigendecomposition directly instead of try/except
    eigvals = jnp.linalg.eigvalsh(cov_mat)
    log_det = jnp.sum(jnp.log(eigvals))
    
    # Use a more stable approach for the quadratic form
    # Add a small regularization to ensure numerical stability
    cov_mat_reg = cov_mat + jnp.eye(cov_mat.shape[0]) * 1e-8
    quad_form = diff @ jnp.linalg.solve(cov_mat_reg, diff)
    
    D = mean_vec.shape[0]
    return -0.5 * (D * jnp.log(2 * jnp.pi) + log_det + quad_form)


def gaussian_likelihood(cov_mat, mean_vec, batch):
    """
    Evaluate the log likelihood of a multivariate gaussian
    for a batch of samples.
    
    Parameters
    ----------
    cov_mat : jnp.ndarray
        Covariance matrix of the Gaussian.
    mean_vec : jnp.ndarray
        Mean vector of the Gaussian.
    batch : jnp.ndarray
        Batch of samples to evaluate.
        
    Returns
    -------
    jnp.ndarray
        Log likelihoods of the samples.
    """
    return jax.vmap(lambda x: gaussian_log_likelihood_single(x, mean_vec, cov_mat))(batch)


@jax.jit
def generate_multivariate_gaussian_samples(key, mean_vec, cov_mat, num_samples):
    """
    Generate samples from a multivariate Gaussian distribution.
    
    Parameters
    ----------
    key : jnp.ndarray
        JAX PRNG key for random number generation.
    mean_vec : jnp.ndarray
        Mean vector of the Gaussian.
    cov_mat : jnp.ndarray
        Covariance matrix of the Gaussian.
    num_samples : int
        Number of samples to generate.
        
    Returns
    -------
    jnp.ndarray
        Generated samples.
    """
    return jax.random.multivariate_normal(key, mean_vec, cov_mat, shape=(num_samples,))


class FullGaussianProcess(MeasurementModel):
    """
    Full (non-stationary) Gaussian process for arbitrary data.

    This model estimates a full covariance matrix from the data and uses it to generate or evaluate samples.
    """

    def __init__(self, data, eigenvalue_floor=1e-3, seed=None, verbose=False, add_uniform_noise=True):
        """
        Initialize the Full Gaussian Process model and estimate the mean and covariance matrix from the data.

        Parameters
        ----------
        data : ndarray
            Input data used to estimate the Gaussian process.
        eigenvalue_floor : float, optional
            Minimum eigenvalue to ensure the covariance matrix is positive definite (default is 1e-3).
        seed : int, optional
            Random seed for reproducibility.
        verbose : bool, optional
            Whether to print progress during computation (default is False).
        add_uniform_noise : bool, optional
            For discrete-valued data, add uniform noise (default True). For continuous-valued data, set to False.
        """
        super().__init__(measurement_types=None, measurement_dtype=float)
        self._validate_data(data)
        self._measurement_shape = data.shape[1:]
        self._add_uniform_noise = add_uniform_noise
        # vectorize
        data = data.reshape(data.shape[0], -1)

        # Create PRNG key
        key = jax.random.PRNGKey(0 if seed is None else seed)
        
        # Add noise if needed
        if add_uniform_noise:
            data = _add_uniform_noise_fn(key, data)
            
        # initialize parameters
        if verbose:
            print('computing full covariance matrix')
        self.cov_mat = estimate_full_cov_mat(data)
        
        # ensure positive definiteness - use JAX-friendly approach
        eigvals, eig_vecs = jnp.linalg.eigh(self.cov_mat)
        eigvals = jnp.maximum(eigvals, eigenvalue_floor)
        self.cov_mat = eig_vecs @ jnp.diag(eigvals) @ eig_vecs.T
        
        # Remove the problematic while loop - instead use a fixed higher floor if needed
        # This is a static approach that works with JAX tracing
        self.cov_mat = self.cov_mat + jnp.eye(self.cov_mat.shape[0]) * eigenvalue_floor

        if verbose:
            print('computing mean vector')
        self.mean_vec = jnp.mean(data, axis=0).flatten()
        

    def fit(self, *args, **kwargs):
        """
        Fit method is not needed for the Full Gaussian Process since the model is already fully estimated.

        Raises
        ------
        Warning
            This method raises a warning because fitting is not necessary.
        """
        warnings.warn('Gaussian process is already fit. No need to call fit method')


    def compute_negative_log_likelihood(self, data, data_seed=None, verbose=True, seed=None, average=True):
        """
        Compute the negative log-likelihood of the provided data under the Gaussian process.

        Parameters
        ----------
        data : ndarray
            Input data to evaluate.
        data_seed : int, optional
            Random seed for shuffling the data.
        verbose : bool, optional
            Whether to print progress (default is True).
        seed : int, optional, deprecated
            Deprecated argument for random seed, use `data_seed` instead.
        average : bool, optional
            Whether to average the negative log-likelihood over all samples (default is True).

        Returns
        -------
        float or jnp.ndarray
            The negative log-likelihood of the provided data.
        """
        if seed is not None:
            warnings.warn('seed argument is deprecated. Use data_seed instead')
            data_seed = seed

        self._validate_data(data)
        data = data.reshape(data.shape[0], -1)
        
        # Add noise if needed
        if self._add_uniform_noise:
            key = jax.random.PRNGKey(0 if data_seed is None else data_seed)
            data = _add_uniform_noise_fn(key, data)
        
        # Compute log likelihood
        log_likelihoods = gaussian_likelihood(self.cov_mat, self.mean_vec, data)
        
        # Return negative log likelihood
        if average:                        
            return -jnp.mean(log_likelihoods) / jnp.prod(jnp.array(data.shape[1:]))
        else:
            return -log_likelihoods / jnp.prod(jnp.array(data.shape[1:]))

        
    def generate_samples(self, num_samples, sample_shape=None, ensure_nonnegative=True, seed=None, verbose=True):
        """
        Generate new samples from the learned Gaussian process.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        sample_shape : tuple, optional
            Shape of the samples to generate (default is the shape of the training data).
        ensure_nonnegative : bool, optional
            Whether to ensure all values are non-negative (default is True).
        seed : int, optional
            Random seed for reproducibility.
        verbose : bool, optional
            Whether to print progress (default is True).

        Returns
        -------
        jnp.ndarray
            Generated samples.
        """
        if sample_shape is not None:
            # make sure sample shape is the same as the measurement shape
            assert sample_shape == self._measurement_shape, 'sample shape must be the same as the measurement shape'
        
        # Create PRNG key
        key = jax.random.PRNGKey(0 if seed is None else seed)
        
        # Generate samples
        samples = generate_multivariate_gaussian_samples(key, self.mean_vec, self.cov_mat, num_samples)
        
        samples = samples.reshape(num_samples, *self._measurement_shape)
        
        # Ensure non-negative if requested
        if ensure_nonnegative:
            samples = jnp.maximum(samples, 0.0)
            
        return samples


    @jax.jit
    def compute_analytic_entropy(self):
        """
        Compute the differential entropy per pixel of the Gaussian process.
        
        Returns
        -------
        float
            Differential entropy per pixel.
        """
        D = self.cov_mat.shape[0]
        # Use eigenvalues for numerical stability
        eigvals = jnp.linalg.eigvalsh(self.cov_mat)
        sum_log_evs = jnp.sum(jnp.log(eigvals))
        gaussian_entropy = 0.5 * (sum_log_evs + D * jnp.log(2 * jnp.pi * jnp.e)) / D
        return gaussian_entropy