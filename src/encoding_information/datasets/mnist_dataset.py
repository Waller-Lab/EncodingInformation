import tensorflow as tf
from encoding_information.datasets.dataset_base_class import MeasurementDatasetBase
import numpy as np

from encoding_information.image_utils import add_noise

class MNISTDataset(MeasurementDatasetBase):
    """
    Wrapper class for the MNIST dataset.

    This class wraps the MNIST dataset, providing an interface for retrieving measurements from the dataset, with optional
    noise and bias applied.
    """

    def __init__(self):
        """
        Initialize the MNIST dataset by downloading it if necessary.

        The dataset is loaded using TensorFlow's `keras.datasets.mnist` API. The training and test data are concatenated
        to create a single dataset.
        """
        # This downloads the dataset if it's not already downloaded
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # concatenate the training and test data
        self._image_data = np.concatenate([x_train, x_test], axis=0)
        self._label_data = np.concatenate([y_train, y_test], axis=0)

    def get_measurements(self, num_measurements, mean=None, bias=0, noise='Poisson', data_seed=None, noise_seed=None):
        """
        Retrieve a set of measurements from the MNIST dataset with optional noise and bias.

        Parameters
        ----------
        num_measurements : int
            Number of measurements to return.
        mean : float, optional
            Mean value to scale the measurements. If None, no scaling is applied (default is None).
        bias : float, optional
            Bias to add to the measurements (default is 0).
        noise : str, optional
            Type of noise to apply. Options are 'Poisson' or None (default is 'Poisson').
        data_seed : int, optional
            Seed for random selection of images from the dataset (default is None).
        noise_seed : int, optional
            Seed for noise generation (default is None).

        Returns
        -------
        np.ndarray
            Array of selected measurements with optional noise and bias applied.
        
        Raises
        ------
        Exception
            If the requested number of measurements exceeds the available dataset size, or if an unsupported noise 
            type is provided.
        """
        if noise not in ['Poisson', None]:
            raise Exception('Only Poisson noise is supported')

        data = self._image_data

        # Ensure enough data
        if num_measurements > data.shape[0]:
            raise Exception(f'Cannot load {num_measurements} measurements, only {data.shape[0]} available')

        # Select random images if data_seed is provided, otherwise select the first num_measurements
        if data_seed is not None:
            np.random.seed(data_seed)
            indices = np.random.choice(data.shape[0], size=num_measurements, replace=False)
        else:
            indices = np.arange(num_measurements)

        images = data[indices]

        # Rescale mean if provided
        if mean is not None:
            photons_per_pixel = images.mean()
            rescale_mean = mean - bias
            rescale_fraction = rescale_mean / photons_per_pixel
            images = images * rescale_fraction
        if bias is not None:
            images += bias

        if noise is None:
            return images
        if noise == 'Poisson':
            # Add Poisson noise 
            # and convert back to regular numpy array from jax array
            return np.array(add_noise(images, noise_seed))

    def get_shape(self):
        """
        Return the shape of the MNIST dataset images.

        Returns
        -------
        tuple
            Shape of the MNIST images (height, width).
        """
        return self._image_data.shape[1:]