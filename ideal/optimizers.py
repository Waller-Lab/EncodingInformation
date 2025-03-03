import wandb
import equinox as eqx
import jax
import tensorflow as tf
from tqdm import tqdm
from typing import Optional, Callable
import jax.numpy as jnp
import matplotlib.pyplot as plt
from encoding_information.models import PixelCNN


class IDEALOptimizer:
    """Optimizer for IDEAL imaging system optimization.
    
    This class handles the optimization of imaging systems using various loss functions
    and provides automatic logging to Weights & Biases.
    
    Now assumes the input data is provided as a TensorFlow dataloader (tf.data.Dataset).
    """
    
    def __init__(
        self,
        imaging_system,
        optimizer,
        loss_fn: Callable,
        patch_size: int = 16,
        num_patches: int = 1024,
        patching_strategy: str = 'random',
        gaussian_sigma: Optional[float] = None,
        use_wandb: bool = True,
        project_name: str = 'ideal_development',
        run_name: str = 'run_1',
        wandb_config: dict = {}
    ):
        """Initialize the IDEAL optimizer.
        
        Args:
            imaging_system: The imaging system to optimize.
            optimizer: The optax optimizer to use.
            loss_fn: Loss function to optimize.
            patch_size: Size of patches to extract.
            num_patches: Number of patches to extract per iteration.
            patching_strategy: Strategy for extracting patches ('random', 'uniform_random', 'tiled', etc.).
            gaussian_sigma: Standard deviation of Gaussian noise (None for Poisson noise).
            use_wandb: Whether to log metrics to Weights & Biases.
            project_name: Name of W&B project for logging.
            run_name: Name of W&B run for logging.
        """
        self.imaging_system = imaging_system
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patching_strategy = patching_strategy
        self.gaussian_sigma = gaussian_sigma
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.run_name = run_name
        # Initialize optimizer state.
        self.opt_state = optimizer.init(eqx.filter([imaging_system], eqx.is_array))
        self.wandb_config = wandb_config
    
    # @eqx.filter_jit
    def step(
        self,
        imaging_system,
        opt_state,
        data: jnp.ndarray,
        key: jax.random.PRNGKey,
        **loss_kwargs
    ):
        """Perform one optimization step."""
        loss, grads = self.loss_fn(
            imaging_system, 
            data,
            key,
            num_patches=self.num_patches,
            patch_size=self.patch_size,
            strategy=self.patching_strategy,
            gaussian_sigma=self.gaussian_sigma,
            **loss_kwargs
        )
    
        # Update parameters.
        updates, new_opt_state = self.optimizer.update([grads], opt_state, [imaging_system])
        updates = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), updates)
        imaging_system = eqx.apply_updates(imaging_system, updates[0])  
    
        return imaging_system, new_opt_state, loss

    def _convert_batch(self, batch):
        """
        Convert a batch from the TensorFlow dataloader to a JAX array.
        
        Args:
            batch: A batch from the TensorFlow dataloader.
        
        Returns:
            The batch converted to a JAX array.
        """
        if isinstance(batch, (tuple, list)):
            # In case the batch is a tuple/list (e.g. (images, labels)), take the first element.
            batch = batch[0]
        if hasattr(batch, "numpy"):
            batch = batch.numpy()
        return jnp.array(batch)
    
    def optimize(
        self,
        data: tf.data.Dataset,
        num_steps: int,
        key: Optional[jax.random.PRNGKey] = None,
        log_every: int = 100,
        validate_every: int = 500,
        validation_data: Optional[tf.data.Dataset] = None,
        **loss_kwargs
    ):
        """Run optimization for a specified number of steps using a TensorFlow dataloader.
        
        Args:
            data: Training data as a TensorFlow dataloader (tf.data.Dataset).
            num_steps: Number of optimization steps.
            key: JAX random key (will be generated if None).
            log_every: How often to log metrics.
            validate_every: How often to run validation.
            validation_data: Optional validation dataloader (tf.data.Dataset).
            **loss_kwargs: Additional arguments passed to the loss function.
        
        Returns:
            The optimized imaging system.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
            
        # Create an iterator from the dataloader and get a sample batch for visualization.
        data_iter = iter(data)
        sample_batch = next(data_iter)
        sample_data = self._convert_batch(sample_batch)
        
        # If validation data is provided, convert one batch.
        if validation_data is not None:
            val_iter = iter(validation_data)
            val_batch = next(val_iter)
            validation_data = self._convert_batch(val_batch)
    
        if self.use_wandb:
            wandb.init(project=self.project_name, name=self.run_name, config=self.wandb_config)
            
        for iteration in tqdm(range(num_steps)):
            key, subkey = jax.random.split(key)
            
            # Get the next training batch, and restart the iterator if needed.
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data)
                batch = next(data_iter)
            batch = self._convert_batch(batch)
            
            # Execute one optimization step.
            self.imaging_system, self.opt_state, loss = self.step(
                self.imaging_system,
                self.opt_state,
                batch,
                subkey,
                **loss_kwargs
            )
            
            # Apply any constraints/normalization.
            self.imaging_system = self.imaging_system.normalize()
            
            if self.use_wandb:
                loss_value = float(loss)
                metrics = {
                    'train/loss': loss_value,
                    'train/iteration': iteration,
                }
                
                if iteration % log_every == 0:
                    try:
                        measurements = self.imaging_system(sample_data[0])
                        reconstructions = self.imaging_system.reconstruct(measurements)
                        try:
                            fig = self.imaging_system.display_measurement(measurements)
                            if fig is not None:
                                metrics['viz/measurement'] = wandb.Image(fig)
                                plt.close(fig)
                        except Exception:
                            pass
                        try:
                            fig = self.imaging_system.display_reconstruction(reconstructions)
                            if fig is not None:
                                metrics['viz/reconstruction'] = wandb.Image(fig)
                                plt.close(fig)
                        except Exception:
                            pass
                        try:
                            fig = self.imaging_system.display_object(sample_data[0])
                            if fig is not None:
                                metrics['viz/object'] = wandb.Image(fig)
                                plt.close(fig)
                        except Exception:
                            pass
                        try:
                            fig = self.imaging_system.display_optics()
                            if fig is not None:
                                metrics['viz/optics'] = wandb.Image(fig)
                                plt.close(fig)
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"Warning: Error during visualization: {str(e)}")
                    
                    wandb.log(metrics)
                else:
                    wandb.log(metrics)
                    
            if validation_data is not None and iteration % validate_every == 0:
                val_loss = self.validate(validation_data, subkey)
                if self.use_wandb:
                    wandb.log({
                        'val/loss': float(val_loss),
                        'val/iteration': iteration  
                    })
                    
        if self.use_wandb:
            wandb.finish()
            
        return self.imaging_system
    
    def validate(self, validation_data, key: jax.random.PRNGKey) -> float:
        """Run validation on a batch from the validation dataloader.
        
        Args:
            validation_data: A batch from the validation dataloader converted to a JAX array.
            key: JAX random key.
        
        Returns:
            Validation loss as a float.
        """
        val_loss, _ = self.loss_fn(
            self.imaging_system,
            validation_data, 
            key,
            num_patches=self.num_patches,
            patch_size=self.patch_size,
            strategy=self.patching_strategy,
            gaussian_sigma=self.gaussian_sigma
        )
        return val_loss 
    
def param_labels(model):
    print('Learnable parameters:')
    
    # Dictionary to store parameter labels
    labels = {}

    def traverse_fields(module, prefix=""):
        """Recursively traverse fields of the module."""
        for field_name, field_value in module.__dict__.items():
            full_field_name = f"{prefix}.{field_name}" if prefix else field_name

            # Check if the field is another Equinox module
            if isinstance(field_value, eqx.Module):
                traverse_fields(field_value, prefix=full_field_name)
            # Handle learnable parameters (arrays) in the module
            elif eqx.is_array(field_value) and eqx.is_inexact_array(field_value):
                labels[full_field_name] = full_field_name

    # Start traversal from the main model
    traverse_fields(model)

    # Create a tree with placeholders
    jax_labels = model

    # Update the fields in the tree structure
    for field_name in labels:
        # Split the field name for nested access
        field_parts = field_name.split(".")
        try:
            jax_labels = eqx.tree_at(
                lambda m, fp=field_parts: getattr_nested(m, fp),
                jax_labels,
                replace=labels[field_name]
            )
            print(field_name)

        except Exception as e:
            # print(f"Failed to update {field_name}: {e}")
            pass

    return [jax_labels]

def getattr_nested(obj, attrs):
    """Helper function to get nested attributes."""
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj
