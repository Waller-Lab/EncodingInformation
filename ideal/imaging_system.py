"""
Imaging System Base Class
"""
from typing import Protocol, runtime_checkable
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import matplotlib.pyplot as plt

@runtime_checkable
class ImagingSystemProtocol(Protocol):
    """Protocol defining the interface for an imaging system."""
    # seed: int

    def forward_model(self, objects: jnp.ndarray) -> jnp.ndarray:
        """Simulates the forward model of the imaging system."""
        ...

    def reconstruct(self, measurements: jnp.ndarray) -> jnp.ndarray:
        """Reconstructs objects from measurements."""
        ...

    def toy_images(self, batch_size: int, height: int, width: int, channels: int) -> jnp.ndarray:
        """Generates toy images for testing the system."""
        ...

    def next_rng_key(self) -> jax.random.PRNGKey:
        """Generates the next random key."""
        ...

    def display_measurement(self, measurement: jnp.ndarray) -> None:
        """Displays the measurement."""
        return NotImplemented

    def display_reconstruction(self, reconstruction: jnp.ndarray) -> None:
        """Displays the reconstruction."""
        return NotImplemented

    def display_object(self, object: jnp.ndarray) -> None:
        """Displays the object."""
        return NotImplemented

    def display_optics(self) -> None:
        """Displays learned the optics."""
        return NotImplemented


class ImagingSystem(eqx.Module):
    """Abstract base class for an imaging system."""
    seed: int = eqx.field(static=True)
    rng_key: jax.random.PRNGKey = eqx.field(static=True)

    def __init__(self, seed: int = 0):
        """
        Initializes the imaging system with a given random seed.

        Args:
            seed: Seed for the random number generator.
        """
        self.seed = seed
        self.rng_key = random.PRNGKey(seed)

    def forward_model(self, objects: jnp.ndarray) -> jnp.ndarray:
        """
        Runs the forward model.

        Args:
            objects: Input objects of shape (H, W, C).

        Returns:
            measurements: Output measurements of shape (H, W, C).
        """
        raise NotImplementedError("Subclasses must implement forward_model.")

    def reconstruct(self, measurements: jnp.ndarray) -> jnp.ndarray:
        """
        Reconstructs objects from measurements.

        Args:
            measurements: Input measurements of shape (H, W, C).

        Returns:
            reconstructions: Reconstructed objects of shape (H, W, C).
        """
        raise NotImplementedError("Subclasses must implement reconstruct.")

    def toy_images(self, batch_size: int, height: int, width: int, channels: int) -> jnp.ndarray:
        """
        Generates toy images for testing the system.

        Args:
            batch_size: Number of images to generate.
            height: Height of each image.
            width: Width of each image.
            channels: Number of channels in each image.

        Returns:
            Toy images of shape (batch_size, height, width, channels).
        """
        key = self.next_rng_key()
        return random.uniform(key, shape=(batch_size, height, width, channels), minval=0, maxval=1)

    def next_rng_key(self) -> jax.random.PRNGKey:
        """
        Generates the next random key and updates the RNG state.

        Returns:
            A new random key.
        """
        rng_key, subkey = random.split(self.rng_key)
        object.__setattr__(self, 'rng_key', rng_key)
        return subkey
    
    def display_measurement(self, measurement: jnp.ndarray) -> plt.Figure:
        """
        Displays the measurement as a matplotlib figure.

        Args:
            measurement: Input measurement of shape (H, W, C).

        Returns:
            fig: Matplotlib figure showing the measurement.
        """
        fig, ax = plt.subplots()
        im = ax.imshow(measurement, cmap='inferno')
        plt.colorbar(im)
        ax.set_title('Measurement')
        return fig

    def display_reconstruction(self, reconstruction: jnp.ndarray) -> plt.Figure:
        """
        Displays the reconstruction as a matplotlib figure.

        Args:
            reconstruction: Input reconstruction of shape (H, W, C).

        Returns:
            fig: Matplotlib figure showing the reconstruction.
        """
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Reconstruction display not implemented for base class', 
                ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
    
    def display_object(self, object: jnp.ndarray) -> plt.Figure:
        """
        Displays the object as a matplotlib figure.

        Args:
            object: Input object of shape (H, W, C).

        Returns:
            fig: Matplotlib figure showing the object.
        """
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Object display not implemented for base class', 
                ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
    
    def display_optics(self) -> plt.Figure:
        """
        Displays the optical system configuration as a matplotlib figure.

        Returns:
            fig: Matplotlib figure showing the optical system.
        """
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Optics display not implemented for base class', 
                ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
