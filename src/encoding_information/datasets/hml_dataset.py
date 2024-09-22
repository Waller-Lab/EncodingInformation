import h5py
import numpy as np
import os
import glob
from encoding_information.datasets.dataset_base_class import MeasurementDatasetBase
from encoding_information.image_utils import add_noise
from jax import random
import jax.numpy as jnp

class HyperspectralMetalensDataset(MeasurementDatasetBase):
    """
    Dataset of grayscale measurements captured with a metalens-based camera.  See Hazineh et al. for more details:
    
    Args:
        h5_dir (str): Directory containing the .h5 files.
        mean_value (float, optional): The mean value of the dataset. If None, it will be computed.
        center_crop (bool, optional): Whether to center crop 16 pixels from each side of the images.
    
    Attributes:
        h5_dir (str): Directory containing the .h5 files.
        patch_size (tuple): Tuple of patch height and width.
        mean_value (float): Mean value of the dataset.
        h5_files (list): List of .h5 file paths.
        num_images (int): Number of images in the dataset.
        image_shape (tuple): Shape of the images in the dataset.
        total_patches (int): Total number of patches.
    """
    def __init__(self, h5_dir, center_crop=None):
        self.h5_dir = h5_dir
        self.center_crop = center_crop

        # Get list of .h5 files
        self.h5_files = sorted(glob.glob(os.path.join(h5_dir, '*.h5')))
        self.num_images = len(self.h5_files)
        self.images = []

        if self.num_images == 0:
            raise ValueError(f"No .h5 files found in directory {h5_dir}")

        for h5_file in self.h5_files:
            with h5py.File(h5_file, 'r') as f:
                image = f['mgs'][:]
                # image shape is (height, width, 1) or (height, width)
                if image.ndim == 3 and image.shape[2] == 1:
                    image = image.squeeze(axis=2)

                if self.center_crop is not None:
                    image = self._center_crop(image, center_crop)

                self.images.append(image)
        self.images = np.array(self.images)



    def _center_crop(self, data, crop_size):
        """
        Center crop the data by crop_size pixels from each side.
        
        Args:
            data (np.ndarray): Image data to crop.
            crop_size (int): Number of pixels to crop from each side.
        
        Returns:
            np.ndarray: Cropped image.
        """
        return data[crop_size:-crop_size, crop_size:-crop_size]

    def get_measurements(self, num_measurements, mean=None, bias=0, data_seed=21, noise_seed=123456,
                         noise='Poisson'):
        """
        Get measurements with optional noise and bias.
        
        Args:
            num_measurements (int): Number of measurements to return.
            mean (float, optional): Mean value to scale the measurements.
            bias (float, optional): Bias to be added to the measurements.
            data_seed (int, optional): Seed for random data selection.
            noise_seed (int, optional): Seed for noise generation.
            noise (str, optional): Type of noise to apply. Defaults to 'Poisson'.
            kwargs: Additional parameters.
        
        Returns:
            np.ndarray: Measurements with optional noise and bias.
        
        Raises:
            ValueError: If unsupported noise type is provided or if `num_measurements`
                        exceeds the total available patches.
        """

        if data_seed is None:
            data_seed = np.random.randint(100000)
        key = random.PRNGKey(data_seed)
        indices = random.choice(key, self.images.shape[0], shape=(num_measurements,), replace=False)
        images = jnp.array(self.images[indices.tolist()])

        # Rescale mean if provided
        if mean is not None:
            photons_per_pixel = images.mean()
            rescale_mean = mean - bias
            rescale_fraction = rescale_mean / photons_per_pixel
            images = images * rescale_fraction
        if bias is not None:
            images += bias

        # Add noise if necessary
        if noise == 'Poisson':
            images = np.array(add_noise(images, noise_seed))
        elif noise == 'Gaussian':
            raise NotImplementedError('Gaussian noise not implemented yet')
        elif noise is None:
            pass
        else:
            raise ValueError(f'Noise type {noise} not recognized')
        
        return images

    def get_shape(self, **kwargs):
        """
        Return the shape of the dataset.
        
        Args:
            kwargs: Additional parameters.
        
        Returns:
            tuple: Shape of the dataset.
        """
        return self.image_shape
