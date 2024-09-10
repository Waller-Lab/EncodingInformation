import tensorflow as tf
from encoding_information.datasets.base_class import MeasurementDatasetBase
import numpy as np

from encoding_information.image_utils import add_noise

class MNISTDataset(MeasurementDatasetBase):
    """
    Wrapper for regular old MNIST dataset
    """

    def __init__(self):
        # This downloads the dataset if it's not already downloaded
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # concatenate the training and test data
        self._image_data = np.concatenate([x_train, x_test], axis=0)
        self._label_data = np.concatenate([y_train, y_test], axis=0)

    def get_measurements(self, num_measurements, mean=None, bias=0, noise='Poisson', data_seed=None, noise_seed=123456):
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
        return self._image_data.shape[1:]