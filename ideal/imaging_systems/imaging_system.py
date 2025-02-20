"""
Imaging System Base Class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Literal, Tuple
import jax.numpy as np
from jax import random, value_and_grad
import optax
from flax import struct



from src.encoding_information import (
    estimate_information,
    extract_patches,
    models
)

class ImagingSystemParams(struct.PyTreeNode):
    """Container for imaging system parameters."""
    fixed_params: Dict
    learnable_params: Dict

class ImagingSystem(ABC):
    """
    Abstract base class for imaging systems with information optimization.
    """
    
    def __init__(self, 
                 fixed_params: Optional[Dict] = None,
                 learnable_params: Optional[Dict] = None,
                 seed: int = 0):
        """Initialize imaging system."""
        self.params = ImagingSystemParams(
            fixed_params=fixed_params if fixed_params is not None else {},
            learnable_params=learnable_params if learnable_params is not None else {}
        )
        self.rng_key = random.PRNGKey(seed)

    @abstractmethod
    def forward_model(self, images: np.ndarray, learnable_params: Dict) -> np.ndarray:
        """
        Run forward model on batch of images.
        
        Args:
            images: Input images of shape (B, H, W, C)
            
        Returns:
            measurements: Output measurements of shape (B, H, W, C)
        """
        raise NotImplementedError("Subclasses must implement forward_model")
    
    @abstractmethod
    def reconstruct(self, measurements: np.ndarray) -> np.ndarray:
        """
        Reconstruct images from measurements.
        
        Args:
            measurements: Input measurements of shape (B, H, W, C)
            
        Returns:
            reconstructions: Reconstructed images of shape (B, H, W, C)
        """
        raise NotImplementedError("Subclasses must implement reconstruct")

    def toy_images(self, 
                  batch_size: int = 1,
                  height: int = 32,
                  width: int = 32,
                  channels: int = 1) -> np.ndarray:
        """
        Generate toy images for testing and optimization.
        
        Args:
            batch_size: Number of images to generate
            height: Image height
            width: Image width
            channels: Number of channels
            
        Returns:
            images: Generated images of shape (B, H, W, C)
        """
        # Split random key
        self.rng_key, _ = random.split(self.rng_key)
        
        # Generate random circles and squares
        images = np.zeros((batch_size, height, width, channels))
        
        for b in range(batch_size):
            # Generate random position and size
            self.rng_key, pos_key, size_key = random.split(self.rng_key, 3)
            pos = random.uniform(pos_key, (2,), minval=0.2, maxval=0.8)
            size = random.uniform(size_key, (), minval=0.1, maxval=0.3)
            
            # Create meshgrid
            y, x = np.meshgrid(np.linspace(0, 1, height), np.linspace(0, 1, width))
            
            # Randomly choose between circle and square
            self.rng_key, shape_key = random.split(self.rng_key)
            if random.uniform(shape_key) < 0.5:
                # Circle
                mask = ((x - pos[0])**2 + (y - pos[1])**2) < size**2
            else:
                # Square
                mask = (np.abs(x - pos[0]) < size) & (np.abs(y - pos[1]) < size)
            
            images = images.at[b, ..., 0].set(mask.astype(np.float32))
        
        return images
    
    def opt_step(self,
                     optimizer: Optional[optax.GradientTransformation] = None,
                     model: Literal['gaussian', 'pixel_cnn'] = 'pixel_cnn',
                     patch_size: int = 8,
                     batch_size: int = 4,
                     epochs: int = 10,
                     num_val_samples: int = 20,
                     steps_per_epoch: int = 10) -> Dict[str, np.ndarray]:
        """
        Optimize learnable parameters to maximize information content.
        
        Args:
            measurements: Training measurements of shape (B, H, W, C)
            optimizer: Optax optimizer (uses default if None)
            model: Model type for MI estimation ('gaussian' or 'pixel_cnn')
            n_iterations: Number of optimization iterations
            patch_size: Size of patches for MI estimation
            batch_size: Batch size for optimization
            
        Returns:
            Dictionary containing:
                'loss_history': Array of loss values
                'mi_history': Array of mutual information estimates
        """
        # Initialize optimizer if not provided
        if optimizer is not None:
            self.optimizer = optimizer
            self.opt_state = self.optimizer.init(self.params.learnable_params)
            
        # Initialize model for MI estimation
        if model == 'pixel_cnn':
            mi_model = models.MultiChannelPixelCNN()
        else:
            raise ValueError(f"Invalid model type: {model}")    
        
        def loss_fn(learned_params):
            images = self.toy_images(batch_size,height=800,width=800)
            measurements = self.forward_model(images,learned_params)
            patches = extract_patches(measurements, patch_size=patch_size, num_patches=100)
            mi_model.fit(patches,max_epochs=epochs,batch_size=batch_size,verbose=False,num_val_samples=num_val_samples,steps_per_epoch=steps_per_epoch,use_tfds=False)       
            mi = -estimate_information(mi_model, models.PoissonNoiseModel(), patches, patches, use_tfds=False)

            return mi

        # Compute loss and gradients
        loss, grads = value_and_grad(loss_fn, allow_int=True)(self.params.learnable_params)
        
        # Update parameters
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        for key in self.params.learnable_params:
            self.params.learnable_params[key] = optax.apply_updates(self.params.learnable_params[key], updates[key])
        learned_param = optax.apply_updates(self.params.learnable_params[key], updates[key])

        results = {'loss': loss, 'mi': -loss, 'learned_param': learned_param, 'grad': grads}
        return results
        
    def info_optimize(self,
                     optimizer: Optional[optax.GradientTransformation] = None,
                     model: Literal['gaussian', 'pixel_cnn'] = 'pixel_cnn',
                     n_iterations: int = 100,
                     patch_size: int = 8,
                     batch_size: int = 4,
                     epochs: int = 10,
                     num_val_samples: int = 20,
                     steps_per_epoch: int = 10) -> Dict[str, np.ndarray]:
        
        loss_history = np.zeros(n_iterations)
        mi_history = np.zeros(n_iterations)
        learned_param_history = []
        grad_history = []

        for i in range(n_iterations):
            result = self.opt_step(optimizer,model,patch_size,batch_size,epochs,num_val_samples,steps_per_epoch)
            loss_history = loss_history.at[i].set(result['loss'])
            mi_history = mi_history.at[i].set(result['mi'])
            learned_param_history.append(result['learned_param'])
            grad_history.append(result['grad'])

            if i % 1 == 0:
                print(f"Iteration {i}: MI = {-result['mi']:.3f}")
        
        results = {'loss_history': loss_history, 
                   'mi_history': mi_history, 
                   'learned_param_history': learned_param_history, 
                   'grad_history': grad_history}
        return results

    def add_learnable_param(self, name: str, param: np.ndarray):
        """Add a learnable parameter to the system."""
        self.params = self.params.replace(
            learnable_params={**self.params.learnable_params, name: param}
        )

    def get_param(self, name: str) -> np.ndarray:
        """Get parameter value by name."""
        if name in self.params.fixed_params:
            return self.params.fixed_params[name]
        elif name in self.params.learnable_params:
            return self.params.learnable_params[name]
        else:
            raise KeyError(f"Parameter {name} not found")

    def next_rng_key(self) -> random.PRNGKey:
        """Get next random key and update internal state."""
        self.rng_key, subkey = random.split(self.rng_key)
        return subkey