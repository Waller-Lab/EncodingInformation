import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, Optional

class GaussianPSFLayer(eqx.Module):
    """A layer that models the Point Spread Function using Gaussian functions."""
    
    means: jnp.ndarray  # (N, 2) array of Gaussian centers
    covs: jnp.ndarray   # (N, 2, 2) array of covariance matrices
    weights: jnp.ndarray  # (N,) array of Gaussian weights
    object_size: int
    obj_padding: tuple
    psf_padding: tuple
    num_gaussians: int
    grid: jnp.ndarray  # Cached coordinate grid
    psf_shape: tuple # PSF shape

    def __init__(self, object_size: int, num_gaussians: int, psf_size: Tuple[int, int] = (100, 100), 
                 key: Optional[jax.random.PRNGKey] = None):
        super().__init__()
        key = jax.random.PRNGKey(0) if key is None else key
        self.num_gaussians = num_gaussians
        self.object_size = object_size
        self.psf_shape = psf_size
        
        k1, k2, k3 = jax.random.split(key, 3)
        self.means = jax.random.uniform(k1, (num_gaussians, 2), 
                                      minval=-psf_size[0]//3, 
                                      maxval=psf_size[0]//3)
        
        scales = jax.random.uniform(k2, (num_gaussians, 2), minval=1, maxval=5)
        thetas = jax.random.uniform(k3, (num_gaussians,), minval=0, maxval=2*jnp.pi)
        
        cos_t = jnp.cos(thetas)
        sin_t = jnp.sin(thetas)
        R = jnp.stack([jnp.stack([cos_t, -sin_t], axis=1),
                      jnp.stack([sin_t, cos_t], axis=1)], axis=1)
        
        S = jnp.zeros((num_gaussians, 2, 2))
        S = S.at[:, 0, 0].set(scales[:, 0])
        S = S.at[:, 1, 1].set(scales[:, 1])
        
        self.covs = jnp.matmul(jnp.matmul(R, S), R.transpose(0, 2, 1))
        self.weights = jnp.ones(num_gaussians) / num_gaussians
        
        self.obj_padding = ((0, 0), (psf_size[0] // 2, psf_size[0] // 2),
                           (psf_size[1] // 2, psf_size[1] // 2))
        self.psf_padding = ((object_size // 2, object_size // 2),
                           (object_size // 2, object_size // 2))
        
        y = jnp.linspace(-psf_size[0]//2, psf_size[0]//2, psf_size[0])
        x = jnp.linspace(-psf_size[1]//2, psf_size[1]//2, psf_size[1])
        X, Y = jnp.meshgrid(x, y)
        self.grid = jnp.stack([Y.flatten(), X.flatten()], axis=1)

    def compute_psf(self):
        """Compute the PSF from current parameters."""
        grid_expanded = self.grid[None, :, :]
        means_expanded = self.means[:, None, :]
        centered = grid_expanded - means_expanded
        covs_inv = jnp.linalg.inv(self.covs)
        
        quad_form = jnp.sum(
            centered * jnp.matmul(centered, covs_inv.transpose(0, 2, 1)),
            axis=-1
        )
        quad_form_clipped = jnp.clip(quad_form, a_min=-100, a_max=100)
        gaussians = jnp.exp(-0.5 * quad_form_clipped)
        gaussians = gaussians / (jnp.sum(gaussians, axis=1, keepdims=True) + 1e-10)
        weighted_sum = jnp.sum(self.weights[:, None] * gaussians, axis=0)
        psf = weighted_sum.reshape(self.psf_shape[0], self.psf_shape[1])
        return psf / (jnp.sum(psf) + 1e-10)

    def __call__(self, x: jnp.ndarray, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        psf = self.compute_psf()
        psf_pad = jnp.pad(psf, self.psf_padding)
        obj_pad = jnp.pad(x, self.obj_padding)
        
        psf_pad_f = jnp.fft.fft2(psf_pad)
        obj_pad_f = jnp.fft.fft2(obj_pad, axes=(1, 2))
        
        psf_pad_f = psf_pad_f[None, :, :]
        convolved_f = jnp.fft.ifft2(psf_pad_f * obj_pad_f, axes=(1, 2))
        convolved_f = jnp.fft.ifftshift(convolved_f, axes=(1, 2))
        
        return jnp.abs(self._crop2D(x.shape, convolved_f))

    def _crop2D(self, shape, x):
        """Crop the convolved image to original size."""
        _, h, w = shape
        H, W = x.shape[-2:]
        top = (H - h) // 2
        left = (W - w) // 2
        return x[:, top:top + h, left:left + w]

    def normalize_psf(self):
        """Normalize PSF parameters."""
        sym_covs = (self.covs + self.covs.transpose(0, 2, 1)) / 2
        min_eigenvalue = 1e-6
        eigvals, eigvecs = jnp.linalg.eigh(sym_covs)
        eigvals = jnp.clip(eigvals, min_eigenvalue, None)
        min_std = 1
        eigvals = jnp.clip(eigvals, min_std**2, None)
        
        new_covs = jnp.matmul(
            jnp.matmul(
                eigvecs,
                jnp.expand_dims(eigvals, -1) * jnp.eye(2)[None, :, :],
            ),
            eigvecs.transpose(0, 2, 1)
        )
        
        normalized_weights = self.weights.clip(0) / (jnp.sum(self.weights.clip(0)) + 1e-10)
        
        return eqx.tree_at(
            lambda l: (l.covs, l.weights),
            self,
            (new_covs, normalized_weights),
        )

class GaussianSensorLayer(eqx.Module):
    """A layer that models the spectral response using Gaussian functions."""
    
    means: jnp.ndarray
    stds: jnp.ndarray
    min_std: float = eqx.field(static=True)
    max_std: float = eqx.field(static=True)
    min_wave: float = eqx.field(static=True)
    max_wave: float = eqx.field(static=True)
    num_waves: int = eqx.field(static=True)
    sensor_size: int = eqx.field(static=True)
    super_pixel_size: int = eqx.field(static=True)
    wavelengths: jnp.ndarray = eqx.field(static=True)

    def __init__(self, min_wave: float, max_wave: float, num_waves: int, 
                 min_std: float, max_std: float, sensor_size: int, 
                 super_pixel_size: int, key: Optional[jax.random.PRNGKey] = None):
        key = jax.random.PRNGKey(0) if key is None else key
        self.means = jnp.linspace(min_wave*1.05, max_wave*.95, super_pixel_size**2)
        self.means = jax.random.permutation(key, self.means).reshape(super_pixel_size, super_pixel_size)
        self.min_std = min_std
        self.max_std = max_std
        self.min_wave = min_wave
        self.max_wave = max_wave
        self.num_waves = num_waves
        self.sensor_size = sensor_size
        self.super_pixel_size = super_pixel_size
        self.stds = jax.random.uniform(key, (super_pixel_size, super_pixel_size), minval=20, maxval=20)
        self.wavelengths = jnp.linspace(min_wave, max_wave, num_waves)

    def __call__(self, x: jnp.ndarray, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        wavelengths = self.wavelengths[:, None, None]
        sensor = jnp.exp(-0.5 * (wavelengths - self.means[None, :, :]) ** 2 / 
                        self.stds[None, :, :] ** 2)
        I = jnp.tile(sensor, (1, self.sensor_size//self.super_pixel_size, 
                             self.sensor_size//self.super_pixel_size))
        s = jnp.sum(I * x, axis=0)
        return s

    def update_means(self):
        """Update means within wavelength bounds."""
        new_means = self.means.clip(self.min_wave, self.max_wave)
        return eqx.tree_at(lambda layer: layer.means, self, new_means)

    def update_stds(self):
        """Update standard deviations within bounds."""
        new_stds = self.stds.clip(self.min_std)
        return eqx.tree_at(lambda layer: layer.stds, self, new_stds)
    
    def get_sensor(self) -> jnp.ndarray:
        """Get the current sensor response curves."""
        wavelengths = self.wavelengths[:, None, None]
        return jnp.exp(-0.5 * (wavelengths - self.means[None, :, :]) ** 2 / 
                      self.stds[None, :, :] ** 2)

