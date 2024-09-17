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
from tqdm import tqdm
from jax import jit
from jax import random

class ColorFilterArrayDataset(MeasurementDatasetBase):
    """
    Dataset of natural images with with various Bayer-like filters applied.

    Shi's Re-processing of Gehler's Raw Dataset
    https://www.cs.sfu.ca/~colour/data/shi_gehler/

    Lilong Shi and Brian Funt, "Re-processed Version of the Gehler Color Constancy Dataset of 568 Images,"

    Original source:

    Peter Gehler and Carsten Rother and Andrew Blake and Tom Minka and Toby Sharp, "Bayesian Color Constancy Revisited,"
    Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2008.
    and http://www.kyb.mpg.de/bs/people/pgehler/colour/index.html.
    
    """

    def __init__(self, zarr_path, tile_size=128):
        """
        Initialize the dataset.

        Args:
            zarr_path (str): Path to the Zarr store containing the dataset.
            tile_size (int): Size of the tiles to split the images into. This is done beceause the raw data is 
                (568, 4, 1359, 2041), but we often want a greater number of smaller images. So we split the images into
                non-overlapping tiles of size (tile_size, tile_size).
        """
        self.zarr_path = zarr_path
        self._tile_size = tile_size

        # Open Zarr store
        self._zarr_store = zarr.open(zarr_path, mode='r')

        # Create Dask array from Zarr store
        self._raw_data_dask_array = da.from_zarr(self._zarr_store)

        # split the raw data into non-overlapping tiles
        N, C, H, W = self._raw_data_dask_array.shape
        # Reshape the image arrays into tiles
        M = (H // tile_size) * (W // tile_size) * N
        # crop the images so that they are divisible by tile_size
        image_array = self._raw_data_dask_array[:, :, :H // tile_size * tile_size, :W // tile_size * tile_size]
        reshaped = image_array.reshape(N, C, H // tile_size, tile_size, W // tile_size, tile_size)
        
        # Transpose to bring the tiles in the desired order: M x tile_size x tile_size x C
        self._tiles = reshaped.transpose(0, 2, 4, 3, 5, 1).reshape(M, tile_size, tile_size, C)


    def get_measurements(self, num_measurements, mean=2000, bias=0, filter_matrix= np.array([[0, 1], [1, 2]]),
                         data_seed=None, noise_seed=None,
                       
                         noise='Poisson',):
        """
        Get a set of measurements from the dataset. Applys a Bayer-like filter to the measurements (i.e. RGB + White).
        The default filter is the one used in the Bayer filter: [[0, 1], [1, 2]] i.e. [[R, G], [G, B]]. Also rescales
        the mean value of the images to the desired value and optionally adds noise

        Args:
            num_measurements (int): Number of measurements to generate.
            mean (float): Mean value to scale the images by. In this context, mean refers to the number of photons 
                per pixel in the white channel. So any color filter that is applied will further reduce the number of
                photons per pixel in the color channels.

                This parameter defaults to a big number (2000) because we don't actually know how many photons were collected
                
            bias (float): Bias to add to the measurements.
            filter_matrix (np.ndarray): Filter matrix to apply to the measurements.
            data_seed (int): Seed for the random number generator.
            noise_seed (int): Seed for the noise generator.
            noise (str): Type of noise to add to the measurements.

        Return (num_measurements x H x W x 4) array of measurements, where the channels are R G B W

        """


        # select random tiles
        if data_seed is None:
            data_seed = np.random.randint(100000)
        key = random.PRNGKey(data_seed)
        indices = random.choice(key, self._tiles.shape[0], shape=(num_measurements,), replace=False)
        # Select number of tiles equal to the number of measurements we need and convert to a jax array
        tiles = jnp.array(self._tiles[indices.tolist()]) 

        ## Apply the filter to all tiles 
        filter_matrix = jnp.array(filter_matrix)
        # tile filter matrix to be the same size as the measurements
        filter_h, filter_w = filter_matrix.shape
        tile_h, tile_w = self._tile_size, self._tile_size
        # make sure tile is divisible by filter
        assert tile_h % filter_h == 0, 'Tile height must be divisible by filter height'
        assert tile_w % filter_w == 0, 'Tile width must be divisible by filter width'

        tiled_filter = jnp.tile(filter_matrix, (tile_h // filter_h, tile_w // filter_w))
        
        # Do advanced indexing to apply the filter to every tile
        batch_indices = jnp.arange(tiles.shape[0])[:, None, None]  # Shape: (N, 1, 1)
        mask = tiled_filter[None, :, :]  # Shape: (1, 256, 256)
        row_indices = jnp.arange(tile_h)[None, :, None]  # Shape: (1, H, 1)
        col_indices = jnp.arange(tile_w)[None, None, :]  # Shape: (1, 1, W)
        # apply the filter
        filtered_tiles = tiles[batch_indices, row_indices, col_indices, mask]

        # Rescale mean if provided
        if mean is not None:
            # TODO: could do this for mean over all the data, but for now just stick to the tiles used
            white_channel = tiles[..., -1]
            photons_per_pixel = white_channel.mean()
            rescale_mean = mean - bias
            rescale_fraction = rescale_mean / photons_per_pixel
            filtered_tiles = filtered_tiles * rescale_fraction
        if bias is not None:
            filtered_tiles += bias


        # Add noise if necessary
        if noise == 'Poisson':
            filtered_tiles = np.array(add_noise(filtered_tiles, noise_seed))
        elif noise == 'Gaussian':
            raise NotImplementedError('Gaussian noise not implemented yet')
        elif noise is None:
            pass # "clean" measurements
        else:
            raise ValueError(f'Noise type {noise} not recognized')

        return filtered_tiles


    def get_shape(self, tile_size=128):
        """
        Return the shape of the dataset (using the given tile size).

        Returns:
            tuple: Shape of the dataset.
        """
        return (self._tiles.shape[0], tile_size, tile_size)

