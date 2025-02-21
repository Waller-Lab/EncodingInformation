import numpy as np
import tensorflow as tf
from jax import random

import make_cell


class CellDataGenerator:
    """
    Data generator class for simulating cell images.
    
    """
    def __init__(self, energy=7e3, dx=2.29e-3, seed=42):
        """
        Initialize the data generator.
        
        Args:
            seed (int): Random seed for reproducibility
        """
        self.energy = energy
        self.dx = dx
        self.rng_key = random.PRNGKey(seed)

    def _generate_cells(self, batch_size: int = 10, height: int = 800, width: int = 800, channels: int = 1) -> np.ndarray:
        """
        Generate toy cell images for testing and optimization.
        
        Args:
            batch_size: Number of images to generate
            height: Image height 
            width: Image width
            channels: Number of channels
            
        Returns:
            images: Generated cell images of shape (B, H, W, C)
        """
        # Initialize output array
        images = np.zeros((batch_size, height, width, channels),dtype=np.complex64)
        
        # Generate cell images
        for b in range(batch_size):
            # Generate random cell thickness and water thickness
            cell_thickness = random.uniform(self.rng_key, (), minval=0.3, maxval=0.7)
            h2o_thickness = 0.5
            
            # Generate cell image
            cell_thickness_map = make_cell.make_single_cell(
                energy=self.energy,
                cell_thickness=cell_thickness,
                h2o_thickness=h2o_thickness,
                dx=self.dx,
                sz=height
            )
            
            images[b, ..., 0] = (cell_thickness_map)
            
        return images
        
    
    def create_dataset(self, batch_size=32):
        """
        Creates a dataset of simulated cell images.

        Args:
            batch_size (int): Number of cell images to generate per batch

        Returns:
            tf.data.Dataset: Dataset yielding batches of complex-valued cell transmission functions
        """
        def _generate_batch(batch_size):
            cells = tf.py_function(
                func=self._generate_cells,
                inp=[batch_size],
                Tout=tf.float32
            )
            cells.set_shape([None, None, None, None])
            return cells

        def _generate_batch_func(_):
            return _generate_batch(batch_size)

        dataset = tf.data.Dataset.from_tensors(0)
        dataset = dataset.map(_generate_batch_func)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset