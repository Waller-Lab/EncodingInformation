import jax
import jax.numpy as jnp
import warnings
from functools import partial
from encoding_information.models.model_base_class import MeasurementModel

class IDEALFullGaussianProcess(MeasurementModel):
    def __init__(self, data, eigenvalue_floor=1e-3, seed=None, verbose=False, add_uniform_noise=True):
        super().__init__(measurement_types=None, measurement_dtype=float)
        self._validate_data(data)
        self._measurement_shape = data.shape[1:]
        self._add_uniform_noise = add_uniform_noise
        
        # Convert data to JAX array and reshape
        data = jnp.array(data.reshape(data.shape[0], -1))
        
        # Add uniform noise if requested
        if add_uniform_noise:
            if seed is not None:
                key = jax.random.PRNGKey(seed)
            else:
                key = jax.random.PRNGKey(0)
            noise = jax.random.uniform(key, data.shape) * 1e-8
            data = data + noise
        
        if verbose:
            print('computing full covariance matrix')
            
        # Compute mean and centered data
        self.mean_vec = jnp.mean(data, axis=0)
        centered_data = data - self.mean_vec[None, :]
        
        # Compute covariance matrix using JAX
        self.cov_mat = (centered_data.T @ centered_data) / (data.shape[0] - 1)
        
        # Ensure positive definiteness using JAX operations
        eigvals, eig_vecs = jax.numpy.linalg.eigh(self.cov_mat)
        eigvals = jnp.maximum(eigvals, eigenvalue_floor)
        self.cov_mat = eig_vecs @ jnp.diag(eigvals) @ eig_vecs.T
        
        # Pre-compute Cholesky decomposition for likelihood calculations
        self._chol = jnp.linalg.cholesky(self.cov_mat)
        self._log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(self._chol)))

    @partial(jax.jit, static_argnums=(0,))
    def _compute_log_likelihood(self, data):
        """JIT-compiled log likelihood computation."""
        centered = data - self.mean_vec[None, :]
        solved = jax.scipy.linalg.solve_triangular(self._chol, centered.T, lower=True)
        mahalanobis = jnp.sum(solved**2, axis=0)
        d = self.mean_vec.shape[0]
        return -0.5 * (d * jnp.log(2 * jnp.pi) + self._log_det + mahalanobis)

    def compute_negative_log_likelihood(self, data, data_seed=None, verbose=True, seed=None, average=True):
        if seed is not None:
            warnings.warn('seed argument is deprecated. Use data_seed instead')
            data_seed = seed

        self._validate_data(data)
        data = jnp.array(data.reshape(data.shape[0], -1))
        
        # Add uniform noise if requested
        if self._add_uniform_noise:
            if data_seed is not None:
                key = jax.random.PRNGKey(data_seed)
            else:
                key = jax.random.PRNGKey(0)
            noise = jax.random.uniform(key, data.shape) * 1e-8
            data = data + noise
        
        log_likes = self._compute_log_likelihood(data)
        
        if average:
            return -jnp.mean(log_likes) / jnp.prod(jnp.array(data.shape[1:]))
        return -log_likes / jnp.prod(jnp.array(data.shape[1:]))
    
    def compute_analytic_entropy(self):
        d = self.mean_vec.shape[0]
        return 0.5 * jnp.sum(jnp.log(2 * jnp.pi * jnp.e * jnp.diag(self.cov_mat)))/d

    def generate_samples(self, num_samples, sample_shape=None, ensure_nonnegative=True, seed=None, verbose=True):
        """Generate samples from the Gaussian process."""
        if sample_shape is not None:
            assert sample_shape == self._measurement_shape, 'sample shape must match measurement shape'
        
        if seed is not None:
            key = jax.random.PRNGKey(seed)
        else:
            key = jax.random.PRNGKey(0)
            
        # Generate standard normal samples
        z = jax.random.normal(key, (num_samples, self.mean_vec.shape[0]))
        
        # Transform to match covariance
        samples = (self._chol @ z.T).T + self.mean_vec
        
        if ensure_nonnegative:
            samples = jnp.maximum(0, samples)
            
        # Reshape to original dimensions
        samples = samples.reshape(num_samples, *self._measurement_shape)
        return samples

    def fit(self, *args, **kwargs):
        warnings.warn('Gaussian process is already fit. No need to call fit method')
