import os
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist

class LenslessDataGenerator:
    """ A data generator class for creating lensless imaging datasets from existing image datasets. 
    
    This class handles loading the image dataset, converting it to photon counts, and tiling images for training. """
    # TODO future: change how tiling is done, right now it uses batching and selects. could use other random selection methods.

    def __init__(self, mean_photon_count, subset_fraction=1.0, seed=42):
        """ Initialize the data generator """
        self.subset_fraction = subset_fraction
        self.seed = seed
        self.mean_photon_count = mean_photon_count

    def load_mnist_data(self):
        """ 
        Load and preprocess MNIST dataset. Pads each image to be 32 x 32 
        
        Returns: 
        tuple: (x_train, x_test) converted to photon counts and subset of MNIST data. 
        """
        (x_train, _), (x_test, _) = mnist.load_data()

        # take subset if specified 
        if self.subset_fraction < 1.0:
            train_size = int(len(x_train) * self.subset_fraction)
            test_size = int(len(x_test) * self.subset_fraction)
            x_train = x_train[:train_size]
            x_test = x_test[:test_size]
        
        # pad images to be 32 x 32, NOTE that this can skew results compared to 28x28 un-padded MI estimates
        x_train = jnp.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'constant')
        x_test = jnp.pad(x_test, ((0, 0), (2, 2), (2, 2)), 'constant')

        # convert to photons
        x_train = x_train.astype('float32') 
        x_train = x_train / jnp.mean(x_train) 
        x_train = x_train * self.mean_photon_count
        x_test = x_test.astype('float32')
        x_test = x_test / jnp.mean(x_test)
        x_test = x_test * self.mean_photon_count

        return x_train, x_test 

    def load_fashion_mnist_data(self):
        """
        Load and preprocess Fashion MNIST dataset. 
        
        Returns:
        tuple: (x_train, x_test) converted to photon counts and subset of Fashion MNIST data. 
        """
        (x_train, _), (x_test, _) = fashion_mnist.load_data() 

        # take subset if specified:
        if self.subset_fraction < 1.0:
            train_size = int(len(x_train) * self.subset_fraction)
            test_size = int(len(x_test) * self.subset_fraction)
            x_train = x_train[:train_size]
            x_test = x_test[:test_size]
        
        # pad images to be 32 x 32, NOTE that this can skew results compared to 28x28 un-padded MI estimates
        x_train = jnp.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'constant')
        x_test = jnp.pad(x_test, ((0, 0), (2, 2), (2, 2)), 'constant')

        # convert to photons
        x_train = x_train.astype('float32')
        x_train = x_train / jnp.mean(x_train)
        x_train = x_train * self.mean_photon_count
        x_test = x_test.astype('float32')
        x_test = x_test / jnp.mean(x_test)
        x_test = x_test * self.mean_photon_count

        return x_train, x_test
    
    def load_cifar10_data(self):
        """ 
        Load and preprocess CIFAR10 dataset. 
        
        Returns:
        tuple: (x_train, x_test) converted to photon counts and subset of CIFAR10 data. 
        """
        (x_train, _), (x_test, _) = cifar10.load_data()

        # take subset if specified
        if self.subset_fraction < 1.0:
            train_size = int(len(x_train) * self.subset_fraction)
            test_size = int(len(x_test) * self.subset_fraction)
            x_train = x_train[:train_size]
            x_test = x_test[:test_size]

        # convert to grayscale 
        x_train = tf.image.rgb_to_grayscale(x_train).numpy()
        x_train = x_train.squeeze() 
        x_test = tf.image.rgb_to_grayscale(x_test).numpy()
        x_test = x_test.squeeze()

        # convert to photons
        x_train = x_train.astype('float32') 
        x_train = x_train / jnp.mean(x_train) 
        x_train = x_train * self.mean_photon_count
        x_test = x_test.astype('float32')
        x_test = x_test / jnp.mean(x_test)
        x_test = x_test * self.mean_photon_count

        return x_train, x_test 

    def _tile_image_batch(self, batch, tile_rows=3, tile_cols=3):
        """ 
        Tiles a batch of images into a single larger image using pure TensorFlow ops. 
        Args: 
            batch (tf.Tensor): Batch of grayscale images with shape (B, H, W) 
            tile_rows (int): Number of rows in the tiled image
            tile_cols (int): Number of columns in the tiled image
        Returns:
            tf.Tensor: Tiled images with shape (H*rows, W*cols) 
        """
        B, H, W = tf.shape(batch)[0], tf.shape(batch)[1], tf.shape(batch)[2]
        total_required = tile_rows * tile_cols

        # Ensure batch has exactly `tile_rows * tile_cols` elements
        batch = batch[:total_required]  # Truncate if too large
        batch = tf.cond(
            tf.shape(batch)[0] < total_required,
            lambda: tf.concat([batch, tf.zeros([total_required - tf.shape(batch)[0], H, W], dtype=batch.dtype)], axis=0),
            lambda: batch
        )

        # Reshape batch
        batch = tf.reshape(batch, [tile_rows, tile_cols, H, W])

        # Transpose to interleave tiles properly
        batch = tf.transpose(batch, perm=[0, 2, 1, 3])  # (rows, H, cols, W)

        # Reshape to final tiled image
        batch = tf.reshape(batch, [tile_rows * H, tile_cols * W])

        return batch





    def create_dataset(self, x_data, tile_rows=3, tile_cols=3, batch_size=32):
        """ Creates a TensorFlow dataset that makes tiled grayscale images. 
        Args: 
            x_data (np.ndarray): Array of grayscale images with shape (N, H, W) 
            batch_size: Number of images per batch 
        
        Returns:
            tf.data.Dataset: Dataset of tiled images with shape (N, H, W) 
        """

        def _tile_batch(batch):
            return self._tile_image_batch(batch, tile_rows, tile_cols)

        dataset = tf.data.Dataset.from_tensor_slices(x_data)
        dataset = dataset.batch(tile_rows * tile_cols)  # Ensure we have enough images for tiling
        dataset = dataset.map(_tile_batch, num_parallel_calls=tf.data.AUTOTUNE)  # Directly call function
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    
    # TODO future: data_generator base class if continuing to use tensorflow approach.