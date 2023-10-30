"""
PixelCNN in Jax/Flax. Adapted from :
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial12/Autoregressive_Image_Modeling.html

Mixture density output adapted from:
https://github.com/hardmaru/mdn_jax_tutorial/blob/master/mixture_density_networks_jax.ipynb
"""

## Standard libraries
import os

import numpy as onp
from typing import Any

## tqdm for progress bars
from tqdm import tqdm

## JAX
import jax
import jax.numpy as np
from jax import random
from jax.scipy.special import logsumexp 

from flax import linen as nn
from flax.training.train_state import TrainState
import optax

from encoding_information.models.image_distribution_models import ProbabilisticImageModel, train_model, evaluate_nll



class PreprocessLayer(nn.Module):
    mean: np.ndarray
    std: np.ndarray

    def __call__(self, x):
        return (x - self.mean) / (self.std + 1e-5)

class MaskedConvolution(nn.Module):
    c_out : int
    mask : np.ndarray
    dilation : int = 1

    @nn.compact
    def __call__(self, x):
        # Flax's convolution module already supports masking
        # The mask must be the same size as kernel
        # => extend over input and output feature channels
        if len(self.mask.shape) == 2:
            mask_ext = self.mask[...,None,None]
            mask_ext = np.tile(mask_ext, (1, 1, x.shape[-1], self.c_out))
        else:
            mask_ext = self.mask
        # Convolution with masking
        x = nn.Conv(features=self.c_out,
                    kernel_size=self.mask.shape[:2],
                    kernel_dilation=self.dilation,
                    mask=mask_ext)(x)
        return x
    

class VerticalStackConvolution(nn.Module):
    c_out : int
    kernel_size : int
    mask_center : bool = False
    dilation : int = 1

    def setup(self):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height, but for simplicity, we stick with masking here.
        mask = onp.ones((self.kernel_size, self.kernel_size), dtype=onp.float32)
        mask[self.kernel_size//2+1:,:] = 0
        # For the very first convolution, we will also mask the center row
        if self.mask_center:
            mask[self.kernel_size//2,:] = 0
        # Our convolution module
        self.conv = MaskedConvolution(c_out=self.c_out,
                                      mask=mask,
                                      dilation=self.dilation)

    def __call__(self, x):
        return self.conv(x)


class HorizontalStackConvolution(nn.Module):
    c_out : int
    kernel_size : int
    mask_center : bool = False
    dilation : int = 1

    def setup(self):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        mask = onp.ones((1, self.kernel_size), dtype=onp.float32)
        mask[0,self.kernel_size//2+1:] = 0
        # For the very first convolution, we will also mask the center pixel
        if self.mask_center:
            mask[0,self.kernel_size//2] = 0
        # Our convolution module
        self.conv = MaskedConvolution(c_out=self.c_out,
                                      mask=mask,
                                      dilation=self.dilation)

    def __call__(self, x):
        return self.conv(x)


class GatedMaskedConv(nn.Module):
    dilation : int = 1

    @nn.compact
    def __call__(self, v_stack, h_stack):
        c_in = v_stack.shape[-1]

        # Layers (depend on input shape)
        conv_vert = VerticalStackConvolution(c_out=2*c_in,
                                             kernel_size=3,
                                             mask_center=False,
                                             dilation=self.dilation)
        conv_horiz = HorizontalStackConvolution(c_out=2*c_in,
                                                kernel_size=3,
                                                mask_center=False,
                                                dilation=self.dilation)
        conv_vert_to_horiz = nn.Conv(2*c_in,
                                     kernel_size=(1, 1))
        conv_horiz_1x1 = nn.Conv(c_in,
                                 kernel_size=(1, 1))

        # Vertical stack (left)
        v_stack_feat = conv_vert(v_stack)
        v_val, v_gate = np.split(v_stack_feat, 2, axis=-1)
        v_stack_out = nn.tanh(v_val) * nn.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = np.split(h_stack_feat, 2, axis=-1)
        h_stack_feat = nn.tanh(h_val) * nn.sigmoid(h_gate)
        h_stack_out = conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out
    

class _PixelCNNFlaxImpl(nn.Module):
    num_hidden_channels : int = 64
    num_mixture_components : int = 40
    train_data_mean : float = None
    train_data_std : float = None
    train_data_min : float = None
    train_data_max : float = None
    sigma_min : float = 1

    def setup(self):
        if None in [self.train_data_mean, self.train_data_std, self.train_data_min, self.train_data_max]:
            raise Exception('Must pass in training data statistics constructor')

        if self.train_data_max.dtype != np.float32 or self.train_data_min.dtype != np.float32 or \
            self.train_data_mean.dtype != np.float32 or self.train_data_std.dtype != np.float32:
            raise Exception('Must pass in training data statistics as float32')

        self.normalize = PreprocessLayer(mean=self.train_data_mean, std=self.train_data_std)

        if not isinstance(self.num_hidden_channels, int):
            raise ValueError("num_hidden_channels must be an integer")
        # Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalStackConvolution(self.num_hidden_channels, kernel_size=3, mask_center=True)
        self.conv_hstack = HorizontalStackConvolution(self.num_hidden_channels, kernel_size=3, mask_center=True)
        # Convolution block of PixelCNN. We use dilation instead of downscaling
        self.conv_layers = [
            GatedMaskedConv(),
            GatedMaskedConv(dilation=2),
            GatedMaskedConv(),
            GatedMaskedConv(dilation=4),
            GatedMaskedConv(),
            GatedMaskedConv(dilation=2),
            GatedMaskedConv(),
        ]
        # Output classification convolution (1x1)
        self.conv_out = nn.Conv(self.num_hidden_channels, kernel_size=(1, 1))
        

        # parameters for mixture density 
        def my_bias_init(rng, shape, dtype):
            return random.uniform(rng, shape, dtype=dtype,
                                  minval=self.train_data_min, maxval=self.train_data_max + 1)
        
        self.mu_dense = nn.Dense(self.num_mixture_components, bias_init=my_bias_init)
        self.sigma_dense = nn.Dense(self.num_mixture_components)
        self.mix_logit_dense = nn.Dense(self.num_mixture_components)

    def __call__(self, x):
        """
        Do forward pass output the parameters of the gaussian mixture output
        """
        # add trailing channel dimension if necessary
        if x.ndim == 3:
            x = x[..., np.newaxis]

        return self.forward_pass(x)
    
    def compute_nll(self, mu, sigma, mix_logit, x):
        # numerically efficient implementation of mixture density, slightly modified
        # see https://github.com/hardmaru/mdn_jax_tutorial/blob/master/mixture_density_networks_jax.ipynb
        # compute per-pixel negative log-likelihood
        nll = - logsumexp(mix_logit - logsumexp(mix_logit, axis=-1, keepdims=True) + self.lognormal(x, mu, sigma), axis=-1)
        return nll

    def compute_loss(self, mu, sigma, mix_logit, x):
        """ 
        Compute average negative log likelihood per pixel averaged over batch and pixels
        """
        return self.compute_nll(mu, sigma, mix_logit, x).mean()

    def lognormal(self, y, mean, sigma):
        logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
        return -0.5 * ((y - mean) / sigma) ** 2 - np.log(sigma) - logSqrtTwoPI


    def forward_pass(self, x):
        """
        Forward image through model and return parameters of mixture density 
        Inputs:
            x - Image BxHxWx1 (multiple channels not supported yet)
        """
        # check shape
        if x.ndim != 4:
            raise ValueError("Input image must have shape BxHxWx1")

        # rescale to 0-1ish
        x = self.normalize(x)
        # Initial convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        # 1x1 classification convolution
        # Apply ELU before 1x1 convolution for non-linearity on residual connection
        out = self.conv_out(nn.elu(h_stack))

        # must be positive and within data range
        # make +1 because uniform noise is added to data
        mu = np.clip(self.mu_dense(out), 0, self.train_data_max + 1) 

        sigma = nn.activation.softplus( self.sigma_dense(out) )  # must be positive
         # avoid having tiny components that overly concentrate mass, and don't need components larger than data standard deviation
        sigma = np.clip(sigma, self.sigma_min, self.train_data_std )

        mix_logit = self.mix_logit_dense(out)

        return mu, sigma, mix_logit





class PixelCNN(ProbabilisticImageModel):
    """
    Translation layer between the PixelCNNFlaxImpl and the probabilistic image model API

    """

    def __init__(self, num_hidden_channels=64, num_mixture_components=40):
        self.num_hidden_channels = num_hidden_channels
        self.num_mixture_components = num_mixture_components
        self._flax_model = None

    def fit(self, train_images, learning_rate=1e-2, max_epochs=200, steps_per_epoch=100,  patience=10, 
            sigma_min=1, batch_size=64, num_val_samples=1000,  seed=0, verbose=True):
        train_images = train_images.astype(np.float32)

        # add trailing channel dimension if necessary
        if train_images.ndim == 3:
            train_images = train_images[..., np.newaxis]

        self.image_shape = train_images.shape[1:3]

        if self._flax_model is None:
            self._flax_model = _PixelCNNFlaxImpl(num_hidden_channels=self.num_hidden_channels, num_mixture_components=self.num_mixture_components,
                                    train_data_mean=np.mean(train_images), train_data_std=np.std(train_images),
                                    train_data_min=np.min(train_images), train_data_max=np.max(train_images), sigma_min=sigma_min)
            initial_params = self._flax_model.init(jax.random.PRNGKey(seed), train_images[:3]) # pass in an intial batch
            
            self._optimizer = optax.adam(learning_rate)

            def apply_fn(params, x):
                output = self._flax_model.apply(params, x)
                return self._flax_model.compute_loss(*output, x)

            self._state = TrainState.create(apply_fn=apply_fn, params=initial_params, tx=self._optimizer)        
        
        best_params, val_loss_history = train_model(train_images=train_images, state=self._state, batch_size=batch_size, num_val_samples=int(num_val_samples),
                                                    steps_per_epoch=steps_per_epoch, num_epochs=max_epochs, patience=patience, verbose=verbose)
        self._state = self._state.replace(params=best_params)
        return val_loss_history


    def compute_negative_log_likelihood(self, data, verbose=True):
        if data.ndim == 3:
            # add a trailing channel dimension if necessary
            data = data[..., np.newaxis]
        elif data.ndim == 2:
            # add trailing channel and batch dimensions
            data = data[np.newaxis, ..., np.newaxis]

        # TODO: extend to images sizes different the default
        # check if data shape is different than image shape
        if data.shape[1:3] != self.image_shape:
            raise ValueError("Data shape is different than image shape of trained model. This is not yet supported"
                             "Expected {}, got {}".format(self.image_shape, data.shape[1:3]))

        return evaluate_nll(data, self._state, verbose=verbose)

    def generate_samples(self, num_samples, sample_shape=None, ensure_nonnegative=True, seed=None, verbose=True):
        if seed is None:
            seed = 123
        key = jax.random.PRNGKey(seed)
        if sample_shape is None:
            sample_shape = self.image_shape
        if type(sample_shape) == int:
            sample_shape = (sample_shape, sample_shape)

        sampled_images = onp.zeros((num_samples, *sample_shape))
        for i in tqdm(onp.arange(sample_shape[0]), desc='Generating PixelCNN samples') if verbose else np.arange(sample_shape[0]):
            for j in onp.arange(sample_shape[1]):
                i_limits = max(0, i - self.image_shape[0] + 1), max(self.image_shape[0], i+1)
                j_limits = max(0, j - self.image_shape[1] + 1), max(self.image_shape[1], j+1)

                conditioning_images = sampled_images[:, i_limits[0]:i_limits[1], j_limits[0]:j_limits[1]]
                i_in_cropped_image = i - i_limits[0]
                j_in_cropped_image = j - j_limits[0]

                assert conditioning_images.shape[1:] == self.image_shape

                key, key2 = jax.random.split(key)
                mu, sigma, mix_logit = self._flax_model.apply(self._state.params, conditioning_images)
                # only sampling one pixel at a time
                # make onp arrays for range checking
                mu = onp.array(mu)[:, i_in_cropped_image, j_in_cropped_image, :]
                sigma = onp.array(sigma)[:, i_in_cropped_image, j_in_cropped_image, :]
                mix_logit = onp.array(mix_logit)[:, i_in_cropped_image, j_in_cropped_image, :]

                # mix_probs = np.exp(mix_logit - logsumexp(mix_logit, axis=-1, keepdims=True))
                component_indices = jax.random.categorical(key, mix_logit, axis=-1)
                # draw categorical sample
                sample_mus = mu[np.arange(num_samples), component_indices]
                sample_sigmas = sigma[np.arange(num_samples), component_indices]
                sample = jax.random.normal(key2, shape=sample_mus.shape) * sample_sigmas + sample_mus
                sampled_images[:, i, j] = sample

        if ensure_nonnegative:
            sampled_images = np.where(sampled_images < 0, 0, sampled_images)
        return sampled_images
   
