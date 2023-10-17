"""
PixelCNN in Jax/Flax. Adapted from :
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial12/Autoregressive_Image_Modeling.html

Mixture density output adapted from:
https://github.com/hardmaru/mdn_jax_tutorial/blob/master/mixture_density_networks_jax.ipynb
"""

## Standard libraries
import os

import numpy as np
from typing import Any

## tqdm for progress bars
from tqdm import tqdm

## JAX
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp 

from flax import linen as nn
from flax.training import train_state, checkpoints
import optax


class PreprocessLayer(nn.Module):
    mean: jnp.ndarray
    std: jnp.ndarray

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
            mask_ext = jnp.tile(mask_ext, (1, 1, x.shape[-1], self.c_out))
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
        mask = np.ones((self.kernel_size, self.kernel_size), dtype=np.float32)
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
        mask = np.ones((1, self.kernel_size), dtype=np.float32)
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


inp_img = np.zeros((1, 11, 11, 1), dtype=np.float32)

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
        v_val, v_gate = jnp.split(v_stack_feat, 2, axis=-1)
        v_stack_out = nn.tanh(v_val) * nn.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = jnp.split(h_stack_feat, 2, axis=-1)
        h_stack_feat = nn.tanh(h_val) * nn.sigmoid(h_gate)
        h_stack_out = conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out
    

class PixelCNNFlaxImpl(nn.Module):
    c_hidden : int
    train_data_max : int
    train_data_std : int
    num_mixture_components : int = 40
    preprocess_mean : float = 0
    preprocess_std : float = 1


    def setup(self):
        self.normalize = PreprocessLayer(mean=self.preprocess_mean, std=self.preprocess_std)

        # Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalStackConvolution(self.c_hidden, kernel_size=3, mask_center=True)
        self.conv_hstack = HorizontalStackConvolution(self.c_hidden, kernel_size=3, mask_center=True)
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
        self.conv_out = nn.Conv(self.c_hidden, kernel_size=(1, 1))
        

        # # parameters for dense layers leading to mixture density
        # self.dense_layers_mu = [
        #     nn.Dense(256),
        #     # nn.Dense(512),
        # ]
        # self.dense_layers_sigma = [
        #     nn.Dense(256),
        #     # nn.Dense(512),
        # ]
        # self.dense_layers_mix_logit = [
        #     nn.Dense(256),
        #     # nn.Dense(512),
        # ]

        # parameters for mixture density 
        self.mu_dense = nn.Dense(self.num_mixture_components)
        self.sigma_dense = nn.Dense(self.num_mixture_components)
        self.mix_logit_dense = nn.Dense(self.num_mixture_components)

        # self.mu = self.param('mu', nn.initializers.uniform(255), (28, 28, self.num_mixture_components))
        # self.sigma = self.param('sigma', nn.initializers.constant(50), (28, 28, self.num_mixture_components))
        # self.mix_logit = self.param('mix_logit', nn.initializers.constant(1), (28, 28, self.num_mixture_components))

            

    def __call__(self, x):
        """
        Do forward pass and complute the negative log likelihood of the given image(s)
        """
        # add trailing channel dimension if necessary
        if x.ndim == 3:
            x = x[..., np.newaxis]

        # Forward pass with nll calculation
        mu, sigma, mix_logit = self.forward_pass(x)
        
        # numerically efficient implementation of mixture density, slightly modified
        # see https://github.com/hardmaru/mdn_jax_tutorial/blob/master/mixture_density_networks_jax.ipynb
        # compute per-pixel negative log-likelihood
        nll = - logsumexp(mix_logit - logsumexp(mix_logit, axis=3, keepdims=True) + self.lognormal(x, mu, sigma), axis=3)
        

        # image_nll = (0.1 * first_column).sum((1,)) + (other_columns).sum((1,2))
        
        # add up negative log-likelihoods for all pixels in each image
        image_nll = jnp.sum(nll, axis=(1,2))



        # y = np.arange(0, self.train_data_max).reshape(-1, 1, 1, 1, 1)
        # mu, sigma, log_mix = self.forward_pass(x_normalized)
        # nll = - logsumexp(log_mix - logsumexp(log_mix, axis=-1, keepdims=True) + self.lognormal(y, mu, sigma), axis=-1)

        regularizer = 0

        # # regularize the standard deviation of the mixture components
        # lam = 1e4
        # regularizer += lam * ((sigma / self.train_data_std) ** 2).mean()

        # # regularize the mixture components to be uniform
        # lam = 1e16
        # regularizer += (logsumexp(mix_logit) - 1 / self.num_mixture_components * mix_logit.sum(axis=3)).mean()


        image_nll += regularizer


        return image_nll.mean()
    

    def lognormal(self, y, mean, sigma):
        logSqrtTwoPI = jnp.log(jnp.sqrt(2.0 * jnp.pi))
        return -0.5 * ((y - mean) / sigma) ** 2 - jnp.log(sigma) - logSqrtTwoPI


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

        out_mu = out
        out_sigma = out
        out_mix_logit = out


        # for i in range(len(self.dense_layers_mu)):
        #     out_mu = nn.elu(self.dense_layers_mu[i](out_mu))
        #     out_sigma = nn.elu(self.dense_layers_sigma[i](out_sigma))
        #     out_mix_logit = nn.elu(self.dense_layers_mix_logit[i](out_mix_logit))


        mu = jnp.clip(self.mu_dense(out_mu), 0, self.train_data_max) # must be positive and within data range
        mu = jnp.clip(self.mu_dense(out_mu), 0, self.train_data_max) # must be positive and within data range

        sigma = nn.activation.softplus( self.sigma_dense(out_sigma) )  # must be positive
        sigma = jnp.clip(sigma, 1, self.train_data_std ) # avoid having tiny components that overly concentrate mass

        mix_logit = self.mix_logit_dense(out_mix_logit)
        # mix = jnp.exp(mix_logit - logsumexp(mix_logit, axis=-1, keepdims=True))
        # mix = jnp.clip(mix, 0, 1 / self.num_mixture_components)
        # mix_logit = jnp.log(mix + 1e-8)

        # Repeat tensors by batch dimension size
        batch_size = x.shape[0]
        # mu = jnp.tile(self.mu, (batch_size, 1, 1, 1))
        # sigma = jnp.tile(self.sigma, (batch_size, 1, 1, 1))
        # mix_logit = jnp.tile(self.mix_logit, (batch_size, 1, 1, 1))

        return mu, sigma, mix_logit

    def sample(self, img_shape, rng, img=None):
        """
        Sampling function for the autoregressive model.
        Inputs:
            img_shape - Shape of the image to generate (B,C,H,W)
            img (optional) - If given, this tensor will be used as
                             a starting image. The pixels to fill
                             should be -1 in the input tensor.
        """
        # Create empty image
        if img is None:
            img = jnp.zeros(img_shape, dtype=jnp.int32) - 1
        # We jit a prediction step. One could jit the whole loop, but this
        # is expensive to compile and only worth for a lot of sampling calls.
        # TODO: had to remove this jit due to flax error and not sure why
        # get_logits = jax.jit(lambda inp: self.pred_logits(inp))
        get_logits = lambda inp: self.forward_pass(inp)
 
        # Generation loop
        for h in tqdm(range(img_shape[1]), leave=False):
            for w in range(img_shape[2]):
                for c in range(img_shape[3]):
                    # Skip if not to be filled (-1)
                    if (img[:,h,w,c] != -1).all().item():
                        continue
                    # For efficiency, we only have to input the upper part of the image
                    # as all other parts will be skipped by the masked convolutions anyways
                    logits = get_logits(img)
                    logits = logits[:,h,w,c,:]
                    rng, pix_rng = random.split(rng)
                    img = img.at[:,h,w,c].set(random.categorical(pix_rng, logits, axis=-1))
        return img


class TrainerModule:

    def __init__(self,                 
                 c_hidden : int,
                 exmp_imgs : Any,
                 num_mixture_components : int = 40,
                 seed : int = 42,
                 checkpoint_path : str = None,):
        """
        Module for summarizing all training functionalities for the PixelCNN.
        """
        super().__init__()
        self.seed = seed
        self.model_name = 'PixelCNN'
        # Create empty model. Note: no parameters yet
        self.model = PixelCNNFlaxImpl(c_hidden=c_hidden, num_mixture_components=num_mixture_components,
                                      train_data_max=exmp_imgs.max(), train_data_std=exmp_imgs.std(),
                                      preprocess_mean=exmp_imgs.mean(), preprocess_std=exmp_imgs.std())
        # Prepare logging
        if checkpoint_path is not None:
            self.log_dir = os.path.join(checkpoint_path, self.model_name)
        else: 
            self.log_dir = None
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(exmp_imgs)

    def create_functions(self):
        # Training function
        def train_step(state, imgs):
            loss_fn = lambda params: state.apply_fn(params, imgs)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss
        # Eval function
        def eval_step(state, imgs):
            loss = state.apply_fn(state.params, imgs)
            return loss
        # jit for efficiency
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)
        # no jit for debugging
        # self.train_step = train_step
        # self.eval_step = eval_step

    def init_model(self, exmp_imgs):
        # Initialize model
        init_rng = random.PRNGKey(self.seed)
        params = self.model.init(init_rng, exmp_imgs)
        _, rng1, rng2 = random.split(init_rng, num=3)
        # Initialize parameters for mixture density so that likelihood is reasonable for the given data, 
        # but also with some random variation so it trains different components
        
        # randomly in the data range
        params['params']['mu_dense']['bias'] = jnp.ones_like(params['params']['mu_dense']['bias']) * random.uniform(rng1, params['params']['mu_dense']['bias'].shape, minval=exmp_imgs.min(), maxval=exmp_imgs.max())
        # params['params']['mu_dense']['bias'] = jnp.ones_like(params['params']['mu_dense']['bias']) * exmp_imgs.mean()
        params['params']['mu_dense']['kernel'] *= (exmp_imgs.mean() /  params['params']['mu_dense']['kernel'].size)
        # TODO this is heuristically chosen

        data_range = exmp_imgs.max() - exmp_imgs.min()
        params['params']['sigma_dense']['bias'] = jnp.ones_like(
                            params['params']['sigma_dense']['bias']) * (
                                    random.uniform(rng2, params['params']['sigma_dense']['bias'].shape, minval=1, maxval=5))
        params['params']['sigma_dense']['kernel'] = params['params']['sigma_dense']['kernel'] * 0
                                            
        # all equal
        params['params']['mix_logit_dense']['kernel'] = np.ones_like(params['params']['mix_logit_dense']['kernel']) 
                                            
        self.state = train_state.TrainState(step=0,
                                            apply_fn=self.model.apply,
                                            params=params,
                                            tx=None,
                                            opt_state=None)

    def init_optimizer(self, num_steps_per_epoch, learning_rate=1e-2):
        # Initialize learning rate schedule and optimizer
        # lr_schedule = optax.exponential_decay(
        #     init_value=self.learning_rate,
        #     transition_steps=num_steps_per_epoch,
        #     decay_rate=0.99
        # )
        optimizer = optax.adam(learning_rate)
        # Initialize training state
        self.state = train_state.TrainState.create(apply_fn=self.state.apply_fn,
                                                   params=self.state.params,
                                                   tx=optimizer)

    def train_model(self, train_loader, val_loader_maker_fn, steps_per_epoch, patience=15,
                   num_epochs=200, learning_rate=1e-2, verbose=True):
        # Train model for defined number of epochs
        # We first need to create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(steps_per_epoch, learning_rate=learning_rate)
        best_params = self.state.params
        eval_nll = self.eval_model(val_loader_maker_fn())
        print(f'Initial validation NLL: {eval_nll:.2f}')
        # eval_nll = 0
        # Track best eval nll
        best_eval = eval_nll
        best_eval_epoch = 0
        val_loss_history = [best_eval]
        for epoch_idx in range(1, num_epochs+1):
            self.train_epoch(train_loader, steps_per_epoch, epoch=epoch_idx)
            if epoch_idx % 1 == 0:
                eval_nll = self.eval_model(val_loader_maker_fn())
                val_loss_history.append(eval_nll)
                print(f'Epoch {epoch_idx}: validation NLL: {eval_nll:.2f}')
                if eval_nll <= best_eval:
                    best_eval_epoch = epoch_idx
                    best_eval = eval_nll
                    best_params = self.state.params
                    if self.log_dir is not None:
                        self.save_model(step=epoch_idx)  
                elif epoch_idx - best_eval_epoch >= patience:
                    break   

        self.state = self.state.replace(params=best_params)  # Replace the parameters with the best found    
        return self.model.bind(self.state.params), val_loss_history       

    def train_epoch(self, train_loader, steps_per_epoch, epoch):
        # Train model for one epoch, and log avg NLL
        avg_loss = 0
        for i in tqdm(range(steps_per_epoch), desc=f'Epoch{epoch}', leave=False):
            batch = next(train_loader)
            self.state, loss = self.train_step(self.state, batch)
            avg_loss += loss / steps_per_epoch

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg nll
        avg_nll, count = 0, 0
        for batch in data_loader:
            nll = self.eval_step(self.state, batch)
            avg_nll += nll * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_nll = (avg_nll / count).item()
        return eval_nll

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
                                    target=self.state.params,
                                    step=step,
                                    overwrite=True)

    def load_model(self, checkpoint_path, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        else:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(checkpoint_path, f'{self.model_name}.ckpt'), target=None)
        self.state = train_state.TrainState.create(apply_fn=self.state.apply_fn,
                                                   params=state_dict,
                                                   tx=self.state.tx if self.state.tx else optax.sgd(0.1)   # Default optimizer
                                                  )

    def checkpoint_exists(self, checkpoint_path):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.isfile(os.path.join(checkpoint_path, f'{self.model_name}.ckpt'))


def train_pixel_cnn(train_loader, val_loader_maker_fn, steps_per_epoch,
                    c_hidden=64, num_mixture_components=40,
                     patience=15, max_epochs=150,learning_rate=1e-2, verbose=True):
    # Create a trainer module with specified hyperparameters
    trainer = TrainerModule(exmp_imgs=next(train_loader), c_hidden=c_hidden, num_mixture_components=num_mixture_components,)
    return trainer.train_model(train_loader, val_loader_maker_fn, steps_per_epoch, patience=patience, num_epochs=max_epochs, 
                               learning_rate=learning_rate, verbose=verbose)
   
