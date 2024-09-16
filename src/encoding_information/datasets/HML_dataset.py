import h5py
import numpy as np
import os
import glob
from encoding_information.datasets.dataset_base_class import MeasurementDatasetBase
from encoding_information.image_utils import add_noise

class HML_Dataset(MeasurementDatasetBase):
    """
    Handles image datasets for the hyperspectral metalense diffusion imaging project.
    
    Args:
        h5_dir (str): Directory containing the .h5 files.
        patch_size (tuple): Tuple of patch height and width.
        mean_value (float, optional): The mean value of the dataset. If None, it will be computed.
    
    Attributes:
        h5_dir (str): Directory containing the .h5 files.
        patch_size (tuple): Tuple of patch height and width.
        mean_value (float): Mean value of the dataset.
        h5_files (list): List of .h5 file paths.
        num_images (int): Number of images in the dataset.
        image_shape (tuple): Shape of the images in the dataset.
        total_patches (int): Total number of patches.
    """
    def __init__(self, h5_dir, patch_size, mean_value=None):
        self.h5_dir = h5_dir
        self.patch_size = patch_size
        self.mean_value = mean_value

        # Get list of .h5 files
        self.h5_files = sorted(glob.glob(os.path.join(h5_dir, '*.h5')))
        self.num_images = len(self.h5_files)

        if self.num_images == 0:
            raise ValueError(f"No .h5 files found in directory {h5_dir}")

        # Open first file to get image shape
        with h5py.File(self.h5_files[0], 'r') as f:
            data = f['mgs'][:]
            # data shape is (height, width, 1) or (height, width)
            if data.ndim == 3 and data.shape[2] == 1:
                data = data.squeeze(axis=2)
            self.image_shape = data.shape  # Should be (height, width)

        self.img_h, self.img_w = self.image_shape
        # Compute total patches
        self.total_patches = self._compute_total_patches()

        # Compute mean value if not provided
        if self.mean_value is None:
            self.mean_value = self.compute_mean_value()

    def _compute_total_patches(self):
        """
        Compute the total number of possible patches across all images.
        
        Returns:
            int: Total number of patches.
        
        Raises:
            ValueError: If patch size is larger than the image size.
        """
        num_images = self.num_images
        img_h, img_w = self.image_shape
        patch_h, patch_w = self.patch_size
        patches_h = img_h - patch_h + 1
        patches_w = img_w - patch_w + 1

        if patches_h <= 0 or patches_w <= 0:
            raise ValueError("Patch size is larger than the image size.")

        return num_images * patches_h * patches_w

    def compute_mean_value(self):
        """
        Compute the mean pixel value across all images in the dataset.
        
        Returns:
            float: The mean pixel value of the dataset.
        """
        total_sum = 0.0
        total_pixels = 0

        for idx, h5_file in enumerate(self.h5_files):
            with h5py.File(h5_file, 'r') as f:
                data = f['mgs'][:]
                if data.ndim == 3 and data.shape[2] == 1:
                    data = data.squeeze(axis=2)
                total_sum += np.sum(data)
                total_pixels += data.size

            print(f"Processed image {idx + 1}/{self.num_images} for mean calculation.")

        mean_value = total_sum / total_pixels
        print(f"Computed mean pixel value: {mean_value}")
        return mean_value

    def get_measurements(self, num_measurements, mean=None, bias=0, data_seed=None, noise_seed=123456,
                         noise='Poisson', **kwargs):
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
        if noise not in ['Poisson', None]:
            raise ValueError('Only Poisson noise is supported')

        if num_measurements > self.total_patches:
            raise ValueError(f'Cannot load {num_measurements} patches, only {self.total_patches} available')

        if data_seed is not None:
            np.random.seed(data_seed)

        # Randomly select patch indices
        patch_indices = np.random.choice(self.total_patches, size=num_measurements, replace=False)

        # Map patch indices to image indices and patch positions
        num_images = self.num_images
        img_h = self.img_h
        img_w = self.img_w
        patch_h, patch_w = self.patch_size
        patches_h = img_h - patch_h + 1
        patches_w = img_w - patch_w + 1
        patches_per_image = patches_h * patches_w

        image_indices = patch_indices // patches_per_image
        patch_indices_in_image = patch_indices % patches_per_image

        # Collect patches
        patches = []
        for idx in range(num_measurements):
            img_idx = image_indices[idx]
            patch_idx_in_image = patch_indices_in_image[idx]
            row = patch_idx_in_image // patches_w
            col = patch_idx_in_image % patches_w

            # Open the .h5 file and extract the patch
            h5_file = self.h5_files[img_idx]
            with h5py.File(h5_file, 'r') as f:
                data = f['mgs'][:]
                if data.ndim == 3 and data.shape[2] == 1:
                    data = data.squeeze(axis=2)
                # Extract the patch
                patch = data[row:row + patch_h, col:col + patch_w]

                # Apply mean adjustment if needed
                if mean is not None:
                    scaling_factor = mean / self.mean_value
                    patch = patch * scaling_factor

                # Add bias
                if bias != 0:
                    patch += bias

                patches.append(patch)

        # Convert patches list to numpy array
        patches = np.array(patches)

        # Add noise if necessary
        if noise == 'Poisson':
            patches = np.array(add_noise(patches, noise_seed))

        return patches

    def get_shape(self, **kwargs):
        """
        Return the shape of the dataset.
        
        Args:
            kwargs: Additional parameters.
        
        Returns:
            tuple: Shape of the dataset.
        """
        return self.image_shape
