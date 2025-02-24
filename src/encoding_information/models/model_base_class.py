from abc import ABC, abstractmethod
import tensorflow as tf
import jax.numpy as np
import jax
import numpy as onp
from tqdm import tqdm
import warnings
from enum import Enum, auto
from typing import List, Union

from abc import ABC, abstractmethod
import numpy as np
from functools import partial
from jax import jit


class MeasurementNoiseModel(ABC):
    """
    Abstract base class for noise models applied to measurement data.

    Methods
    -------
    estimate_conditional_entropy(*args)
        Abstract method for estimating conditional entropy, which must be implemented by derived classes.
    """

    @abstractmethod
    def estimate_conditional_entropy(self, *args):
        """
        Estimate the conditional entropy for a given noise model.

        This is an abstract method that must be implemented by subclasses.

        Parameters
        ----------
        *args : tuple
            Additional arguments needed for estimating conditional entropy.

        Returns
        -------
        float
            The estimated conditional entropy.
        """
        pass


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

    def _validate_data(self, data: Union[List, np.ndarray, onp.ndarray]):
        """
        Validate the input data, ensuring it matches the expected type and dtype.

        Parameters
        ----------
        data : Union[List, np.ndarray, onp.ndarray]
            Input data to validate.

        Raises
        ------
        ValueError
            If the data does not match the expected type or dtype.
        """
        if isinstance(data, list):
            data = np.array(data)

        if self.measurement_dtype == float and not np.issubdtype(data.dtype, np.floating):
            raise ValueError(f"Expected float dtype, but got {data.dtype}")
        elif self.measurement_dtype == complex and not np.issubdtype(data.dtype, np.complexfloating):
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

        Parameters
        ----------
        train_images : ndarray
            The training dataset consisting of images, with shape (N, H, W).
        learning_rate : float, optional
            The learning rate for optimization (default is 1e-2).
        max_epochs : int, optional
            Maximum number of training epochs (default is 200).
        steps_per_epoch : int, optional
            Number of steps per epoch (default is 100).
        patience : int, optional
            Number of epochs to wait for improvement before early stopping (default is 10).
        batch_size : int, optional
            The number of images in each batch (default is 64).
        num_val_samples : int, optional
            Number of validation samples to use, or None to compute automatically.
        percent_samples_for_validation : float, optional
            Fraction of samples to use for validation (default is 0.1).
        data_seed : int, optional
            Random seed for data shuffling.
        model_seed : int, optional
            Random seed for model initialization.
        verbose : bool, optional
            Whether to print training progress (default is True).

        Returns
        -------
        val_loss_history : list
            A list of validation losses at each epoch.
        """
        pass
    
    @abstractmethod
    def compute_negative_log_likelihood(self, images, data_seed=123, average=True, verbose=True):
        """
        Compute the NLL of the images under the model

        Parameters
        ----------
        images : ndarray
            Array of images, shape (N, H, W)
        verbose : bool, optional
            Whether to print progress
        data_seed : int, optional
            Random seed for shuffling images (and possibly adding noise)
        average : bool, optional
            Whether to average the NLL over all images

        Returns
        -------
        log_likelihood : ndarray
            Average negative log-likelihood of all images
        """
        pass
    
    @abstractmethod
    def generate_samples(self, num_samples, sample_shape=None, verbose=True):
        """
        Generate samples from the model.

        Parameters
        ----------
        num_samples : int
            The number of samples to generate.
        sample_shape : tuple, optional
            The shape of each sample. If None, the model will use the training image shape.
        verbose : bool, optional
            Whether to print progress (default is True).

        Returns
        -------
        samples : ndarray
            Generated samples from the model.
        """
        pass

def _add_gaussian_noise_fn(images, condition_vectors=None):
    """
    Add Gaussian noise to images.

    Parameters
    ----------
    images : ndarray
        The input images, shape (N, H, W).
    condition_vectors : ndarray, optional
        Conditioning vectors associated with the images.

    Returns
    -------
    noisy_images : ndarray
        The images with added Gaussian noise.
    condition_vectors : ndarray, optional
        If provided, returns the conditioning vectors.
    """

    noisy_images = images + tf.random.normal(shape=tf.shape(images), mean=0, stddev=1)
    if condition_vectors is not None:
        return noisy_images, condition_vectors
    else:
        return noisy_images

def _add_uniform_noise_fn(images, condition_vectors=None):
    """
    Add uniform noise to images.

    Parameters
    ----------
    images : ndarray
        The input images, shape (N, H, W).
    condition_vectors : ndarray, optional
        Conditioning vectors associated with the images.

    Returns
    -------
    noisy_images : ndarray
        The images with added uniform noise.
    condition_vectors : ndarray, optional
        If provided, returns the conditioning vectors.
    """

    noisy_images = images + tf.random.uniform(shape=tf.shape(images), minval=0, maxval=1)
    if condition_vectors is not None:
        return noisy_images, condition_vectors
    else:
        return noisy_images

def make_dataset_generators(data, batch_size, num_val_samples, add_uniform_noise=True, 
                            add_gaussian_noise=False, condition_vectors=None, seed=None):
    """
    Create TensorFlow dataset generators for training and validation data.

    Parameters
    ----------
    data : ndarray
        The input data, shape (N, H, W).
    batch_size : int
        The number of samples per batch.
    num_val_samples : int
        Number of validation samples.
    add_uniform_noise : bool, optional
        Whether to add uniform noise to the data (default is True).
    add_gaussian_noise : bool, optional
        Whether to add Gaussian noise to the data (default is False).
    condition_vectors : ndarray, optional
        Conditioning vectors associated with the images.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    train_ds : tf.data.Dataset
        TensorFlow dataset iterator for training data.
    val_ds : callable
        A function that returns a TensorFlow dataset iterator for validation data.
    """

    if seed is not None:
        tf.random.set_seed(seed)

    if num_val_samples > data.shape[0]:
        raise ValueError("Number of validation samples must be less than the number of training samples")
    

    data = data.astype(np.float32)


    # split images into train and validation
    val_images = data[:num_val_samples]
    train_images = data[num_val_samples:]

    if condition_vectors is not None:
        # Validate the shape of condition_vectors
        if condition_vectors.shape[0] != data.shape[0]:
            raise ValueError("Condition vectors and images must have the same number of samples")

        # Combine images and condition_vectors
        train_condition_vectors = condition_vectors[num_val_samples:]
        val_condition_vectors = condition_vectors[:num_val_samples]

        # Update TensorFlow datasets to include condition_vectors
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_condition_vectors))
        val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_condition_vectors))
    else:

        # make tensorflow datasets
        train_ds = tf.data.Dataset.from_tensor_slices(train_images)
        val_ds = tf.data.Dataset.from_tensor_slices(val_images)

    if add_gaussian_noise and add_uniform_noise:
        raise ValueError("Cannot add both gaussian and uniform noise")
    
    if add_gaussian_noise:
        train_ds = train_ds.map(_add_gaussian_noise_fn)
        val_ds = val_ds.map(_add_gaussian_noise_fn)

    if add_uniform_noise:
        train_ds = train_ds.map(_add_uniform_noise_fn)
        val_ds = val_ds.map(_add_uniform_noise_fn)

    train_ds = train_ds.repeat().shuffle(1024).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.shuffle(1024).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    return train_ds.as_numpy_iterator(), lambda : val_ds.as_numpy_iterator()


def _evaluate_nll(data_iterator, state, eval_step=None, batch_size=16, return_average=True, verbose=False):
    """
    Compute the negative log-likelihood (NLL) over batches of data.

    Parameters
    ----------
    data_iterator : Iterator or ndarray
        An iterator or array of input data batches.
    state : flax.linen.Module
        The current state of the model.
    eval_step : callable, optional
        The evaluation step to compute the NLL for a batch. Defaults to a compiled JAX function.
    batch_size : int, optional
        The number of samples per batch (default is 16).
    return_average : bool, optional
        Whether to return the average NLL over all data (default is True).
    verbose : bool, optional
        Whether to print progress (default is False).

    Returns
    -------
    nll : float or ndarray
        If return_average is True, returns the average NLL. Otherwise, returns the NLL for each batch.
    """

    if eval_step is None: # default eval step
        eval_step = jax.jit(lambda state, imgs: state.apply_fn(state.params, imgs))

    total_nll, count = 0, 0
    nlls = []
    if isinstance(data_iterator, np.ndarray) or isinstance(data_iterator, onp.ndarray):
        if not return_average:
            batch_size = 1
            print('return_average is False but batch_size is not 1. Setting batch_size to 1.')
        data_iterator = np.array_split(data_iterator, len(data_iterator) // batch_size + 1)  # split into batches of batch_size or less
    if verbose:
        data_iterator = tqdm(data_iterator, desc='Evaluating NLL')
    for batch in data_iterator:
        if isinstance(batch, tuple):
            images, condition_vector = batch
        else:
            images = batch
            condition_vector = None
        batch_nll_per_pixel = eval_step(state, images) if condition_vector is None else eval_step(state, images, condition_vector)
        if return_average:
            total_nll += images.shape[0] * batch_nll_per_pixel
            count += images.shape[0]
        else:
            nlls.append(batch_nll_per_pixel)
    if return_average:
        return (total_nll / count).item()
    else:
        return np.array(nlls)


def train_model(train_images, state, batch_size, num_val_samples, steps_per_epoch, num_epochs, patience,
                train_step, condition_vectors=None, add_gaussian_noise=False, add_uniform_noise=True, seed=None,
                verbose=True):
    """
    Train the model with early stopping based on validation performance.

    Parameters
    ----------
    train_images : ndarray
        The training dataset consisting of images, with shape (N, H, W).
    state : flax.linen.Module
        The model state.
    batch_size : int
        The number of samples per batch.
    num_val_samples : int
        The number of validation samples.
    steps_per_epoch : int
        Number of steps per epoch.
    num_epochs : int
        Maximum number of epochs for training.
    patience : int
        Number of epochs to wait before early stopping.
    train_step : callable
        The function to perform a single training step.
    condition_vectors : ndarray, optional
        Conditioning vectors associated with the images.
    add_gaussian_noise : bool, optional
        Whether to add Gaussian noise to the training data (default is False).
    add_uniform_noise : bool, optional
        Whether to add uniform noise to the training data (default is True).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Whether to print training progress (default is True).

    Returns
    -------
    best_params : flax.linen.Module.params
        The model parameters that achieved the best validation performance.
    val_loss_history : list
        A list of validation losses for each epoch.
    """ 

    if num_val_samples >= train_images.shape[0]:
        num_val_samples = int(train_images.shape[0] * 0.1)
        warnings.warn(f'Number of validation samples must be less than the number of training samples. Using {num_val_samples} validation samples instead.')
    if num_val_samples < 1:
        warnings.warn('Number of validation samples must be at least 1. Using 1 validation sample instead.')
        num_val_samples = 1
    train_ds_iterator, val_loader_maker_fn = make_dataset_generators(train_images,
                     batch_size=batch_size, num_val_samples=num_val_samples, condition_vectors=condition_vectors,
                     add_gaussian_noise=add_gaussian_noise, add_uniform_noise=add_uniform_noise, seed=seed
                     )

    if condition_vectors is not None:
        eval_step = jax.jit(lambda state, imgs, conditioning_vecs: state.apply_fn(state.params, imgs, conditioning_vecs))
    else: 
        eval_step = jax.jit(lambda state, imgs: state.apply_fn(state.params, imgs))

    best_params = state.params
    eval_nll = _evaluate_nll(val_loader_maker_fn(), state, eval_step=eval_step)
    if verbose:
        print(f'Initial validation NLL: {eval_nll:.2f}')
    
    best_eval = eval_nll
    best_eval_epoch = 0
    val_loss_history = [best_eval]
    for epoch_idx in tqdm(range(1, num_epochs+1), desc='training') if not verbose else range(1, num_epochs+1):
        avg_loss = 0
        iter = range(steps_per_epoch)
        for _ in iter if not verbose else tqdm(iter, desc=f'Epoch {epoch_idx}'):

            batch = next(train_ds_iterator)

            if condition_vectors is None:
                state, loss = train_step(state, batch)
            else:
                state, loss = train_step(state, batch[0], batch[1])

            avg_loss += loss / steps_per_epoch
        
        eval_nll = _evaluate_nll(val_loader_maker_fn(), state, eval_step=eval_step) 
        if np.isnan(eval_nll):
            warnings.warn('NaN encountered in validation loss. Stopping early.')
            break
        val_loss_history.append(eval_nll)
        if verbose:
            print(f'Epoch {epoch_idx}: validation NLL: {eval_nll:.2f}')
        if patience is None:
            best_params = state.params # no early stopping
        elif eval_nll <= best_eval:
            best_eval_epoch = epoch_idx
            best_eval = eval_nll
            best_params = state.params
            # if self.log_dir is not None:
            #     self.save_model(step=epoch_idx)  
        elif epoch_idx - best_eval_epoch >= patience:
            break   

    return best_params, val_loss_history

