import numpy as np
from encoding_information.datasets.dataset_base_class import MeasurementDatasetBase
import jax.numpy as jnp
test = jnp.array([1,2,3])
from encoding_information.image_utils import add_noise
from jax import random
try:
    import dask.array as da
    import zarr
except ImportError:
    da = None
    zarr = None


class ColorFilterArrayDataset(MeasurementDatasetBase):
    """
    Dataset of natural images with various Bayer-like filters applied.

    This dataset is based on Shi's re-processing of Gehler's Raw Dataset, which consists of 568 images. The Bayer-like 
    filters simulate a color filter array, such as the one used in digital cameras.

    References:
    -----------
    Lilong Shi and Brian Funt, "Re-processed Version of the Gehler Color Constancy Dataset of 568 Images."
    
    Peter Gehler, Carsten Rother, Andrew Blake, Tom Minka, and Toby Sharp, "Bayesian Color Constancy Revisited," 
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2008.
    """

    def __init__(self, zarr_path, tile_size=128):
        """
        Initialize the dataset and split the images into non-overlapping tiles.

        Parameters
        ----------
        zarr_path : str
            Path to the Zarr store containing the dataset.
        tile_size : int
            Size of the tiles to split the images into. Images are divided into non-overlapping tiles of size 
            (tile_size, tile_size). Default is 128.
        """
        if da is None or zarr is None:
            raise ImportError("To use the ColorFilterArrayDataset class, install the required packages: "
                              "pip install encoding_information[dataset]")
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
        Get a set of measurements from the dataset by applying a Bayer-like filter.

        The default filter matrix simulates the pattern used in Bayer filters for RGB+White channels. The images are 
        rescaled to match a desired mean photon count and can be further corrupted by noise (Poisson or Gaussian).

        Parameters
        ----------
        num_measurements : int
            Number of measurements to generate.
        mean : float, optional
            Mean value to scale the images by, corresponding to the number of photons per pixel in the white channel.
            Default is 2000.
        bias : float, optional
            Bias to add to the measurements. Default is 0.
        filter_matrix : ndarray, optional
            Filter matrix to apply to the measurements. Default is [[0, 1], [1, 2]] (Bayer pattern).
        data_seed : int, optional
            Random seed for selecting tiles from the dataset.
        noise_seed : int, optional
            Random seed for noise generation.
        noise : str, optional
            Type of noise to add to the measurements. Options are 'Poisson' (default), 'Gaussian', or None.

        Returns
        -------
        filtered_tiles : ndarray
            Array of measurements with shape (num_measurements, H, W, 4), where the channels correspond to R, G, B, W.
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
        Return the shape of the dataset based on the given tile size.

        Parameters
        ----------
        tile_size : int, optional
            Size of the tiles in the dataset. Default is 128.

        Returns
        -------
        tuple
            Shape of the dataset (number of tiles, tile height, tile width).
        """
        return (self._tiles.shape[0], tile_size, tile_size)

