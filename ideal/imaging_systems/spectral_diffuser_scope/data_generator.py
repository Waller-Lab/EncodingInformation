import os
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
from tensorflow.keras.datasets import mnist

class SpectralDataGenerator:
    """
    A data generator class for creating hyperspectral image datasets from MNIST data.
    
    This class handles loading MNIST data, converting it to hyperspectral images using real spectra,
    and creating tiled mosaic arrangements suitable for training imaging systems.
    """
    
    def __init__(self, spectra_folder, subset_fraction=1.0, seed=42):
        """
        Initialize the data generator.
        
        Args:
            spectra_folder (str): Path to folder containing .npy spectral data files
            subset_fraction (float): Fraction of MNIST dataset to use (default: 1.0)
            seed (int): Random seed for reproducibility
        """
        self.spectra_folder = spectra_folder
        self.subset_fraction = subset_fraction
        self.seed = seed
        self.all_spectra = None
        self.all_spectra_np = None
        
        # Load the spectra
        self._load_spectra()
        
    def _load_spectra(self):
        """Load and preprocess all spectra from the spectra folder."""
        spectra_files = [f for f in os.listdir(self.spectra_folder) if f.endswith('.npy')]
        
        all_spectra = []
        for file in spectra_files:
            spectra_path = os.path.join(self.spectra_folder, file)
            spectra = jnp.load(spectra_path)[::4, 1]  # Downsample by factor of 4
            all_spectra.append(spectra / jnp.max(spectra))  # Normalize
            
        self.all_spectra = all_spectra
        self.all_spectra_np = np.array([np.array(s) for s in all_spectra])
        
    def load_mnist_data(self):
        """
        Load and preprocess MNIST dataset.
        
        Returns:
            tuple: (x_train, x_test) normalized and subset of MNIST data
        """
        (x_train, _), (x_test, _) = mnist.load_data()
        
        # Take subset if specified
        if self.subset_fraction < 1.0:
            train_size = int(len(x_train) * self.subset_fraction)
            test_size = int(len(x_test) * self.subset_fraction)
            x_train = x_train[:train_size]
            x_test = x_test[:test_size]
            
        # Normalize
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        return x_train, x_test
    
    def _hyperspectral_conversion(self, image):
        """
        Convert a 2D grayscale image to a hyperspectral image using a randomly selected spectrum.

        Args:
            image (np.ndarray): 2D grayscale image of shape (H, W)

        Returns:
            np.ndarray: Hyperspectral image of shape (C, H, W)
        """
        image = np.array(image)
        num_spectra = self.all_spectra_np.shape[0]
        idx = np.random.randint(0, num_spectra)
        selected_spectrum = self.all_spectra_np[idx]
        
        hyperspectral = (image[..., None] * selected_spectrum[None, None, :]).astype(np.float32)
        hyperspectral = np.transpose(hyperspectral, (2, 0, 1))
        
        return hyperspectral
    
    def _tile_mosaic_batch(self, batch, mosaic_rows, mosaic_cols):
        """
        Tiles a batch of hyperspectral images into a single mosaic.

        Args:
            batch (tf.Tensor): Batch of hyperspectral images with shape (B, C, H, W)
            mosaic_rows (int): Number of rows in the mosaic
            mosaic_cols (int): Number of columns in the mosaic

        Returns:
            tf.Tensor: Mosaic hyperspectral image of shape (C, mosaic_rows*H, mosaic_cols*W)
        """
        B, C, H, W = batch.shape
        total_required = mosaic_rows * mosaic_cols
        
        if B < total_required:
            pad_count = total_required - B
            pad_array = tf.zeros((pad_count, C, H, W), dtype=batch.dtype)
            batch = tf.concat([batch, pad_array], axis=0)
        elif B > total_required:
            batch = batch[:total_required]

        mosaic = tf.reshape(batch, (mosaic_rows, mosaic_cols, C, H, W))
        mosaic = tf.transpose(mosaic, perm=[2, 0, 3, 1, 4])
        mosaic = tf.reshape(mosaic, (C, mosaic_rows * H, mosaic_cols * W))
        return mosaic
    
    def create_dataset(self, x_data, mosaic_rows=4, mosaic_cols=4, batch_size=32):
        """
        Creates a TensorFlow dataset that converts grayscale images to hyperspectral mosaics.

        Args:
            x_data (np.ndarray): Array of grayscale images with shape (N, H, W)
            mosaic_rows (int): Number of rows in the mosaic grid
            mosaic_cols (int): Number of columns in the mosaic grid
            batch_size (int): Number of mosaics per batch

        Returns:
            tf.data.Dataset: Dataset yielding batches of mosaic hyperspectral images
        """
        def _convert_image(img):
            hyperspectral = tf.py_function(
                func=self._hyperspectral_conversion,
                inp=[img],
                Tout=tf.float32
            )
            hyperspectral.set_shape([None, None, None])
            return hyperspectral

        def _tile_batch(batch):
            mosaic = tf.py_function(
                func=lambda b: self._tile_mosaic_batch(b, mosaic_rows, mosaic_cols),
                inp=[batch],
                Tout=tf.float32
            )
            mosaic.set_shape([None, None, None])
            return mosaic

        dataset = tf.data.Dataset.from_tensor_slices(x_data)
        dataset = dataset.map(_convert_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(mosaic_rows * mosaic_cols)
        dataset = dataset.map(_tile_batch, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def create_sparse_dataset(self, x_data, sparsity_factor=2, scale=10.0, mosaic_rows=4, mosaic_cols=4, batch_size=32):
        """
        Creates a dataset with added sparsity by including zero-valued images.

        Args:
            x_data (np.ndarray): Input image data
            sparsity_factor (int): Factor by which to add zeros (e.g., 2 doubles the dataset with zeros)
            scale (float): Scaling factor for the non-zero images
            mosaic_rows (int): Number of rows in mosaic
            mosaic_cols (int): Number of columns in mosaic
            batch_size (int): Batch size for the dataset

        Returns:
            tf.data.Dataset: Dataset with added sparsity
        """
        num_samples = len(x_data)
        num_zeros = num_samples * sparsity_factor
        
        # Create zero arrays
        zeros = np.zeros((num_zeros, x_data.shape[1], x_data.shape[2]))
        
        # Concatenate with original data and scale
        x_combined = np.concatenate([x_data, zeros], axis=0) * scale
        
        # Shuffle the combined data
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(len(x_combined))
        x_combined = x_combined[indices]
        
        return self.create_dataset(x_combined, mosaic_rows, mosaic_cols, batch_size)
