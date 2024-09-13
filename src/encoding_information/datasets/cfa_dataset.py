import dask.array as da
import numpy as np
import os
import sys
from encoding_information.datasets.dataset_base_class import MeasurementDatasetBase
from abc import ABC, abstractmethod
import zarr
import jax.numpy as jnp
test = jnp.array([1,2,3])
import torch
from torch.utils.data import Dataset, DataLoader
from numcodecs import Blosc
import glob
from encoding_information.image_utils import add_noise

class CFADataset(MeasurementDatasetBase):
    """
    Handles large image datasets using Zarr arrays with patch extraction and filtering.
    
    Args:
        zarr_path (str): Path to the Zarr store.
        patch_size (tuple): Tuple of patch height and width.
        filter_matrix (np.ndarray): The Bayer filter matrix to apply.
        mean_value (float, optional): The mean value of the dataset.
    
    Attributes:
        zarr_path (str): Path to the Zarr store.
        patch_size (tuple): Tuple of patch height and width.
        filter_matrix (np.ndarray): The Bayer filter matrix to apply.
        mean_value (float): Mean value of the dataset.
        zarr_store (zarr.core.Array): Zarr array.
        image_arrays (da.Array): Dask array from Zarr store.
        image_shape (tuple): Shape of the image dataset.
        total_patches (int): Total number of patches.
    """

    def __init__(self, zarr_path, patch_size, filter_matrix, mean_value=None):
        self.zarr_path = zarr_path
        self.patch_size = patch_size
        self.filter_matrix = filter_matrix
        self.mean_value = mean_value

        # Open Zarr store
        self.zarr_store = zarr.open(zarr_path, mode='r')

        # Create Dask array from Zarr store
        self.image_arrays = da.from_zarr(self.zarr_store)

        # Get image shape and compute total patches
        self.image_shape = self.zarr_store.shape  # (num_images, channels, height, width)
        self.total_patches = self._compute_total_patches()

    def _compute_total_patches(self):
        """
        Compute the total number of possible patches across all images.
        
        Returns:
            int: Total number of patches.
        
        Raises:
            ValueError: If patch size is larger than the image size.
        """
        num_images, _, img_h, img_w = self.image_shape
        patch_h, patch_w = self.patch_size
        patches_h = img_h - patch_h + 1
        patches_w = img_w - patch_w + 1

        if patches_h <= 0 or patches_w <= 0:
            raise ValueError("Patch size is larger than the image size.")

        return num_images * patches_h * patches_w

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
        num_images, _, img_h, img_w = self.image_shape
        patch_h, patch_w = self.patch_size
        patches_h = img_h - patch_h + 1
        patches_w = img_w - patch_w + 1
        patches_per_image = patches_h * patches_w

        image_indices = patch_indices // patches_per_image
        patch_indices_in_image = patch_indices % patches_per_image

        # Collect patches using Dask
        patches = []
        for idx in range(num_measurements):
            img_idx = image_indices[idx]
            patch_idx_in_image = patch_indices_in_image[idx]
            row = patch_idx_in_image // patches_w
            col = patch_idx_in_image % patches_w

            # Extract patch lazily
            patch = self.image_arrays[img_idx, :, row:row + patch_h, col:col + patch_w]

            # Apply mean adjustment if needed
            if mean is not None and self.mean_value is not None:
                scaling_factor = mean / self.mean_value
                patch = patch * scaling_factor

            # Ensure the 4th channel is the sum of the RGB channels
            patch = patch.compute()  # Compute the patch
            patch[3] = patch[0] + patch[1] + patch[2]

            # Apply the Bayer filter
            filtered_patch = self.apply_filter(patch, self.filter_matrix)

            # Add bias
            if bias != 0:
                filtered_patch += bias

            patches.append(filtered_patch)

        # Convert patches list to numpy array
        patches = np.array(patches)

        # Add noise if necessary
        if noise == 'Poisson':
            patches = np.array(add_noise(patches, noise_seed))

        return patches

    def apply_filter(self, patch, filter_matrix):
        """
        Apply the Bayer filter to the patch.
        
        Args:
            patch (np.ndarray): Patch to be filtered.
            filter_matrix (np.ndarray): Filter matrix to apply.
        
        Returns:
            np.ndarray: Filtered patch.
        """
        filter_h, filter_w = filter_matrix.shape
        patch_h, patch_w = self.patch_size

        # Tile the filter matrix to match the patch size
        tile_factor_h = patch_h // filter_h
        tile_factor_w = patch_w // filter_w
        tiled_filter = np.tile(filter_matrix, (tile_factor_h, tile_factor_w))

        # Apply the tiled filter using indexing
        filtered_patch = np.zeros((patch_h, patch_w), dtype=patch.dtype)
        for ch in range(4):
            mask = (tiled_filter == ch)
            filtered_patch[mask] = patch[ch][mask]

        return filtered_patch

    def get_shape(self, **kwargs):
        """
        Return the shape of the dataset.
        
        Args:
            kwargs: Additional parameters.
        
        Returns:
            tuple: Shape of the dataset.
        """
        return self.image_shape


# def preprocess_and_save_zarr(image_dir, zarr_path, chunk_size):
#     """
#     Preprocess the images and save them as a Zarr array with specified chunk size.

#     Args:
#         image_dir (str): Directory containing the .npy image files.
#         zarr_path (str): Path where the Zarr store will be saved.
#         chunk_size (tuple): Chunk size for Zarr array (num_images_chunk, channels_chunk, height_chunk, width_chunk).
#     """
#     # Get list of image files
#     image_files = sorted(glob.glob(os.path.join(image_dir, '*.npy')))
#     num_images = len(image_files)

#     # Load sample image to get shape
#     sample_image = np.load(image_files[0], mmap_mode='r')
#     channels, img_h, img_w = sample_image.shape

#     # Create empty Zarr array
#     compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
#     zarr_store = zarr.open(zarr_path, mode='w', shape=(num_images, channels, img_h, img_w),
#                            chunks=chunk_size, dtype=sample_image.dtype, compressor=compressor)

#     # Load images and store them in Zarr
#     for idx, image_file in enumerate(image_files):
#         image = np.load(image_file, mmap_mode='r')
#         zarr_store[idx] = image
#         print(f"Saved image {idx + 1}/{num_images} to Zarr store.")

#     print("Preprocessing complete.")