"""Loss functions for IDEAL optimization."""
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional
from image_utils import (
    extract_patches, 
    add_noise, 
)
from models import IDEALFullGaussianProcess
from encoding_information.models.pixel_cnn import PixelCNN

class BaseLoss:
    """Base class for IDEAL loss functions."""
    
    def __call__(self, model, data, key, **kwargs):
        raise NotImplementedError

class PixelCNNLoss(BaseLoss):
    """Loss using PixelCNN model for entropy estimation."""
    
    def __init__(self, refit_every: Optional[int] = None, refit_patience: Optional[int] = 10, refit_learning_rate: Optional[float] = 1e-3, refit_steps_per_epoch: Optional[int] = 100, reinitialize_pixelcnn: Optional[bool] = True):

        """Initialize PixelCNNLoss.
        
        Args:
            refit_every: How often to refit the PixelCNN model (None means never refit)
            refit_patience: Patience for early stopping during refitting
            refit_learning_rate: Learning rate for refitting
            refit_steps_per_epoch: Number of steps per epoch during refitting
            reinitialize_pixelcnn: Whether to reinitialize the PixelCNN model at each refit, if True, it will use a fresh model each time. If False it will use the same model and only update the weights
        """
        self.pixel_cnn = PixelCNN()  # This model is trained separately
        # The loss function is jitted and returns both loss and gradients
        self._compute_loss = eqx.filter_jit(eqx.filter_value_and_grad(self._loss_fn))
        self.refit_every = refit_every
        self.refit_patience = refit_patience
        self.refit_learning_rate = refit_learning_rate
        self.refit_steps_per_epoch = refit_steps_per_epoch
        self.reinitialize_pixelcnn = reinitialize_pixelcnn
        self.step_counter = 0
        self._is_initialized = False

    def _loss_fn(self, model, data, key, num_patches, patch_size, strategy, gaussian_sigma, pixel_cnn):
        """
        Jitted loss function used for differentiable optimization.
        
        Args:
            model: The imaging system model.
            data: Input data.
            key: Random key.
            num_patches: Number of patches to extract.
            patch_size: Size of each patch.
            strategy: Patching strategy.
            gaussian_sigma: Gaussian noise sigma.
            pixel_cnn: The PixelCNN model used for entropy estimation.
            
        Returns:
            The loss value.
        """
        measurements = eqx.filter_vmap(model)(data).clip(1e-8)
        patches = extract_patches(
            measurements,
            key=key,
            num_patches=num_patches,
            patch_size=patch_size,
            strategy=strategy
        )
        patches = patches.reshape(patches.shape[0], -1)
        noisy_patches = jnp.maximum(1e-8, add_noise(patches, gaussian_sigma=gaussian_sigma))
        
        # When the PixelCNN has not been refitted at least once, return a dummy loss.
        if not self._is_initialized:
            return jnp.array(0.0)
            
        h_y = pixel_cnn._state.apply_fn(
            pixel_cnn._state.params, 
            noisy_patches.reshape(-1, patch_size, patch_size)[..., None]
        )
        h_y_given_x = estimate_conditional_entropy(noisy_patches, gaussian_noise_sigma=gaussian_sigma)
        mi = (h_y - h_y_given_x) / jnp.log(2)
        return -mi
    
    def __call__(self, model, data, key, **kwargs):
        """
        Compute the loss and its gradient in a jitted and differentiable way.
        
        Args:
            model: The imaging system model.
            data: Input data.
            key: Random key.
            **kwargs: Additional loss parameters (num_patches, patch_size, strategy, gaussian_sigma)
        
        Returns:
            A tuple of (loss, gradients).
        """
        # Check if refitting is needed
        if self.refit_every is not None and self.step_counter % self.refit_every == 0:
            self.refit_pixel_cnn(model, data, key, **kwargs)
        self.step_counter += 1
        
        return self._compute_loss(
            model, data, key,
            kwargs.get('num_patches'),
            kwargs.get('patch_size'),
            kwargs.get('strategy'),
            kwargs.get('gaussian_sigma'),
            self.pixel_cnn
        )
    
    
    def refit_pixel_cnn(self, model, data, key, **kwargs):
        """
        Refit the PixelCNN using patches extracted outside the jitted context.
        
        This method re-extracts the patches (thus avoiding caching a tracer),
        moves the noisy patches to host memory, reshapes them, and refits the PixelCNN.
        This update is non-differentiable and can safely be called from the training loop.
        
        Args:
            model: The imaging system model.
            data: Input data.
            key: Random key.
            **kwargs: Additional parameters (num_patches, patch_size, strategy, gaussian_sigma)
        """
        measurements = eqx.filter_vmap(model)(data).clip(1e-8)
        patches = extract_patches(
            measurements,
            key=key,
            num_patches=kwargs.get('num_patches'),
            patch_size=kwargs.get('patch_size'),
            strategy=kwargs.get('strategy')
        )
        patches = patches.reshape(patches.shape[0], -1)
        noisy_patches = jnp.maximum(1e-8, add_noise(patches, gaussian_sigma=kwargs.get('gaussian_sigma')))
        # Transfer to host memory to obtain a concrete array.
        patches_np = jax.device_get(noisy_patches)
        reshaped_patches = patches_np.reshape(-1, kwargs.get('patch_size'), kwargs.get('patch_size'))[..., None]
        
        # Refit the PixelCNN model (non-differentiable update)
        if self.reinitialize_pixelcnn or not self._is_initialized:
            self.pixel_cnn = PixelCNN()
        self.pixel_cnn.fit(reshaped_patches, patience=self.refit_patience, learning_rate=self.refit_learning_rate, steps_per_epoch=self.refit_steps_per_epoch, verbose=False)
        self._is_initialized = True
        self._compute_loss = eqx.filter_jit(eqx.filter_value_and_grad(self._loss_fn))

class GaussianEntropyLoss(BaseLoss):
    """Loss using analytic Gaussian entropy estimation."""
    
    def __init__(self):
        # Create a JIT-compiled function that returns both value and gradient
        self._compute_loss = eqx.filter_jit(eqx.filter_value_and_grad(self._loss_fn))
    
    @staticmethod
    def _loss_fn(model, data, key, num_patches, patch_size, strategy, gaussian_sigma):
        measurements = eqx.filter_vmap(model)(data).clip(1e-8)
        patches = extract_patches(measurements, key=key, num_patches=num_patches, 
                                patch_size=patch_size, strategy=strategy)
        patches = patches.reshape(patches.shape[0], -1)
        noisy_patches = jnp.maximum(1e-8, add_noise(
            patches, 
            gaussian_sigma=gaussian_sigma
        ))
        
        h_y_given_x = estimate_conditional_entropy(
            noisy_patches, 
            gaussian_noise_sigma=gaussian_sigma
        )
        
        # Compute empirical covariance matrix
        cov_matrix = jnp.cov(noisy_patches.T).clip(1e-8)
        h_y = analytic_multivariate_gaussian_entropy(cov_matrix)
        
        mi = (h_y - h_y_given_x) / jnp.log(2)
        return -mi
    
    def __call__(self, model, data, key, **kwargs):
        # This will return both the loss value and its gradient
        return self._compute_loss(
            model, data, key,
            kwargs.get('num_patches'),
            kwargs.get('patch_size'),
            kwargs.get('strategy'),
            kwargs.get('gaussian_sigma')
        )
    
class GaussianLoss(BaseLoss):
    """Loss using NLL of Gaussian model."""
    
    def __init__(self):
        self._compute_loss = eqx.filter_jit(eqx.filter_value_and_grad(self._loss_fn))
    
    @staticmethod
    def _loss_fn(model, data, key, num_patches, patch_size, strategy, gaussian_sigma):
        measurements = eqx.filter_vmap(model)(data).clip(1e-8)
        patches = extract_patches(measurements, key=key, num_patches=num_patches, patch_size=patch_size, strategy=strategy)
        patches = patches.reshape(patches.shape[0], -1)
        noisy_patches = jnp.maximum(1e-8, add_noise(
            patches, 
            gaussian_sigma=gaussian_sigma
        ))
        
        # Stop gradient before fitting Gaussian model
        noisy_patches_stopped = jax.lax.stop_gradient(noisy_patches)
        gaussian_model = IDEALFullGaussianProcess(
            noisy_patches_stopped, 
            eigenvalue_floor=1e-8
        )
        
        h_y = gaussian_model.compute_negative_log_likelihood(noisy_patches)
        h_y_given_x = estimate_conditional_entropy(
            noisy_patches, 
            gaussian_noise_sigma=gaussian_sigma
        )
        mi = (h_y - h_y_given_x) / jnp.log(2)
        return -mi
    
    def __call__(self, model, data, key, **kwargs):
        return self._compute_loss(
            model, data, key,
            kwargs.get('num_patches'),
            kwargs.get('patch_size'),
            kwargs.get('strategy'),
            kwargs.get('gaussian_sigma')
        )

# Entropy estimation functions

def estimate_conditional_entropy(images, gaussian_noise_sigma=None):\
     # vectorize
    images = images.reshape(-1, images.shape[-2] * images.shape[-1])
    n_pixels = images.shape[-1]

    if gaussian_noise_sigma is None:
        # conditional entropy H(Y | x) for Poisson noise
        # For large lambda, H(Poisson(λ)) ≈ (1/2)log(2πeλ)
        # For small lambda, we need the exact formula:
        # H(Poisson(λ)) = λ log(e) - λ log(λ) + log(Γ(λ+1))
        use_exact = images <= 10  # threshold for using exact formula
        
        # Approximate formula for large lambda
        gaussian_approx = 0.5 * jnp.log(2 * jnp.pi * jnp.e * images)
        
        # Exact formula for small lambda
        exact = (images * jnp.log(jnp.e) - 
                images * jnp.log(jnp.where(images <= 0, 1, images)) + 
                jax.scipy.special.gammaln(images + 1))
        
        # Combine based on threshold
        per_pixel_entropies = jnp.where(use_exact, exact, gaussian_approx)
        per_pixel_entropies = jnp.where(images <= 0, 0, per_pixel_entropies)
        
        per_image_entropies = jnp.sum(per_pixel_entropies, axis=1) / n_pixels
        h_y_given_x = jnp.mean(per_image_entropies)
    
    else:
        # conditional entropy H(Y | x) for Gaussian noise
        # only depends on the gaussian sigma
        h_y_given_x =  0.5 * jnp.log(2 * jnp.pi * jnp.e * gaussian_noise_sigma**2)

    return h_y_given_x

def analytic_multivariate_gaussian_entropy(cov_matrix):
    """
    Numerically stable computation of the analytic entropy of a multivariate gaussian
    """
    d = cov_matrix.shape[0]
    entropy = 0.5 * d * jnp.log(2 * jnp.pi * jnp.e) + 0.5 * jnp.sum(jnp.log(jnp.linalg.eigvalsh(cov_matrix).clip(1e-8)))
    return entropy / d

