import wandb
import equinox as eqx
import jax
import tensorflow as tf
from tqdm import tqdm
from typing import Optional, Callable
import jax.numpy as jnp
import matplotlib.pyplot as plt
from encoding_information.models import PixelCNN
import optax


class IDEALOptimizer:
    """Optimizer for IDEAL imaging system optimization.
    
    This class handles the optimization of imaging systems using various loss functions
    and provides automatic logging to Weights & Biases.
    
    Now assumes the input data is provided as a TensorFlow dataloader (tf.data.Dataset).
    """
    
    def __init__(
        self,
        imaging_system,
        learnable_parameters_w_lr: dict,
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
            learnable_parameters_w_lr: The parameters to optimize and their learning rates.
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
        self.learnable_parameters_w_lr = learnable_parameters_w_lr
        self.loss_fn = loss_fn
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patching_strategy = patching_strategy
        self.gaussian_sigma = gaussian_sigma
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.run_name = run_name
        # Initialize optimizer state.
        self.wandb_config = wandb_config
        self.optimizer, self.learnable_parameters, self.frozen_parameters, self.opt_state = setup_parameter_optimizer(self.imaging_system, self.learnable_parameters_w_lr)

    # @eqx.filter_jit
    def step(
        self,
        data: jnp.ndarray,
        key: jax.random.PRNGKey,
        **loss_kwargs
    ):
        """Perform one optimization step."""
        loss, grads = self.loss_fn(
            self.learnable_parameters,
            self.frozen_parameters,
            data,
            key,
            num_patches=self.num_patches,
            patch_size=self.patch_size,
            strategy=self.patching_strategy,
            gaussian_sigma=self.gaussian_sigma,
            **loss_kwargs
        )
    
        # Update parameters.
        updates, new_opt_state = self.optimizer.update([grads], self.opt_state, [self.learnable_parameters])
        updates = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), updates)
        self.learnable_parameters = eqx.apply_updates(self.learnable_parameters, updates[0])  
    
        return self.learnable_parameters, new_opt_state, loss

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
            self.learnable_parameters, self.opt_state, loss = self.step(
                batch,
                subkey,
                **loss_kwargs
            )
            
            # Apply any constraints/normalization.
            self.imaging_system = eqx.combine(self.learnable_parameters, self.frozen_parameters)
            self.imaging_system = self.imaging_system.normalize()

            # Resplit the parameters.
            self.learnable_parameters, self.frozen_parameters = split_model_params(self.imaging_system, self.learnable_parameters_w_lr.keys())
            
            if self.use_wandb:
                loss_value = float(loss)
                metrics = {
                    'train/loss': loss_value,
                    'train/iteration': iteration,
                }
                
                if iteration % log_every == 0:
                    try:
                        measurements = self.imaging_system(sample_data[0])
                        try:
                            fig = self.imaging_system.display_measurement(measurements)
                            metrics['viz/measurement'] = wandb.Image(fig) if fig is not None else None
                            if fig is not None:
                                plt.close(fig)
                        except Exception as e:
                            print(f"Warning: Failed to generate measurement visualization: {str(e)}")
                        try:
                            fig = self.imaging_system.display_object(sample_data[0])
                            metrics['viz/object'] = wandb.Image(fig) if fig is not None else None
                            if fig is not None:
                                plt.close(fig)
                        except Exception as e:
                            print(f"Warning: Failed to generate object visualization: {str(e)}")
                        try:
                            fig = self.imaging_system.display_optics()
                            metrics['viz/optics'] = wandb.Image(fig) if fig is not None else None
                            if fig is not None:
                                plt.close(fig)
                        except Exception as e:
                            print(f"Warning: Failed to generate optics visualization: {str(e)}")
                        # reconstructions = self.imaging_system.reconstruct(measurements)
                        try:
                            fig = self.imaging_system.display_reconstruction(measurements)
                            metrics['viz/reconstruction'] = wandb.Image(fig) if fig is not None else None
                            if fig is not None:
                                plt.close(fig)
                        except Exception as e:
                            print(f"Warning: Failed to generate reconstruction visualization: {str(e)}")
                    except Exception as e:
                        print(f"Warning: Error during visualization: {str(e)}")
                
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
    
def get_nested_attr(obj, attr_path):
    """Retrieve a nested attribute given a dot-separated path."""
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj

def split_model_params(model, trainable_names):
    trainable_params, frozen_params = eqx.partition(
            model,
            lambda x: any(get_nested_attr(model, path) is x for path in trainable_names)
        )
    return trainable_params, frozen_params

def setup_parameter_optimizer(model: eqx.Module, learnable_params: dict):
    """Splits model parameters into trainable and frozen sets, and initializes an optimizer.

    Args:
        model (eqx.Module): The model containing parameters.
        learnable_params (dict): A dictionary where keys are parameter paths (str) and values are learning rates (float).

    Returns:
        tuple: (optimizer, trainable_params, frozen_params, opt_state)
    """
    trainable_names = set(learnable_params.keys())
    trainable_params, frozen_params = split_model_params(model, trainable_names)

    labeled_model = [eqx.tree_at(
        lambda model: [get_nested_attr(model, path) for path in trainable_names],
        trainable_params,
        replace=[path for path in trainable_names]
    )]

    # Create optimizer
    optimizer = optax.multi_transform(
        {param: optax.adam(learning_rate=lr) for param, lr in learnable_params.items()},
        param_labels=labeled_model
    )

    # Initialize optimizer state
    opt_state = optimizer.init(eqx.filter([trainable_params], eqx.is_array))

    return optimizer, trainable_params, frozen_params, opt_state