"""
PixelCNN in Jax/Flax. Adapted from :
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial12/Autoregressive_Image_Modeling.html
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
# Seeding for random operations

from flax import linen as nn
from flax.training import train_state, checkpoints
import optax


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
    

class PixelCNN(nn.Module):
    c_in : int
    c_hidden : int

    def setup(self):
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
            GatedMaskedConv()
        ]
        # Output classification convolution (1x1)
        self.conv_out = nn.Conv(self.c_in * 256, kernel_size=(1, 1))

    def __call__(self, x):
        # Forward pass with bpd likelihood calculation
        logits = self.pred_logits(x)
        labels = x.astype(jnp.int32)
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        bpd = nll.mean() * np.log2(np.exp(1))
        return bpd

    def pred_logits(self, x):
        """
        Forward image through model and return logits for each pixel.
        Inputs:
            x - Image tensor with integer values between 0 and 255.
        """
        # Scale input from 0 to 255 back to -1 to 1
        x = (x / 255.0) * 2 - 1

        # Initial convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        # 1x1 classification convolution
        # Apply ELU before 1x1 convolution for non-linearity on residual connection
        out = self.conv_out(nn.elu(h_stack))

        # Output dimensions: [Batch, Height, Width, Channels, Classes]
        out = out.reshape(out.shape[0], out.shape[1], out.shape[2], out.shape[3]//256, 256)
        return out

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
        get_logits = lambda inp: self.pred_logits(inp)
 
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
                 c_in : int,
                 c_hidden : int,
                 exmp_imgs : Any,
                 lr : float = 1e-3,
                 seed : int = 42,
                 checkpoint_path : str = None,):
        """
        Module for summarizing all training functionalities for the PixelCNN.
        """
        super().__init__()
        self.lr = lr
        self.seed = seed
        self.model_name = 'PixelCNN'
        # Create empty model. Note: no parameters yet
        self.model = PixelCNN(c_in=c_in, c_hidden=c_hidden)
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

    def init_model(self, exmp_imgs):
        # Initialize model
        init_rng = random.PRNGKey(self.seed)
        params = self.model.init(init_rng, exmp_imgs)
        self.state = train_state.TrainState(step=0,
                                            apply_fn=self.model.apply,
                                            params=params,
                                            tx=None,
                                            opt_state=None)

    def init_optimizer(self, num_steps_per_epoch):
        # Initialize learning rate schedule and optimizer
        lr_schedule = optax.exponential_decay(
            init_value=self.lr,
            transition_steps=num_steps_per_epoch,
            decay_rate=0.99
        )
        optimizer = optax.adam(lr_schedule)
        # Initialize training state
        self.state = train_state.TrainState.create(apply_fn=self.state.apply_fn,
                                                   params=self.state.params,
                                                   tx=optimizer)

    def train_model(self, train_loader, val_loader_maker_fn, steps_per_epoch, patience=15,
                   num_epochs=200):
        # Train model for defined number of epochs
        # We first need to create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(steps_per_epoch)
        best_params = None
        eval_bpd = self.eval_model(val_loader_maker_fn())
        print(f'Initial validation bpd: {eval_bpd:.2f}')
        # Track best eval bpd score.
        best_eval = eval_bpd
        best_eval_epoch = 0
        val_loss_history = [best_eval]
        for epoch_idx in range(1, num_epochs+1):
            self.train_epoch(train_loader, steps_per_epoch, epoch=epoch_idx)
            if epoch_idx % 1 == 0:
                eval_bpd = self.eval_model(val_loader_maker_fn())
                val_loss_history.append(eval_bpd)
                print(f'Epoch {epoch_idx}: validation bpd: {eval_bpd:.2f}')
                if eval_bpd <= best_eval:
                    best_eval_epoch = epoch_idx
                    best_eval = eval_bpd
                    best_params = self.state.params
                    if self.log_dir is not None:
                        self.save_model(step=epoch_idx)  
                elif epoch_idx - best_eval_epoch >= patience:
                    break   

        self.state = self.state.replace(params=best_params)  # Replace the parameters with the best found    
        return self.model.bind(self.state.params), val_loss_history       

    def train_epoch(self, train_loader, steps_per_epoch, epoch):
        # Train model for one epoch, and log avg bpd
        avg_loss = 0
        for i in tqdm(range(steps_per_epoch), desc=f'Epoch{epoch}', leave=False):
            batch = next(train_loader)
            self.state, loss = self.train_step(self.state, batch)
            avg_loss += loss / steps_per_epoch

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg bpd
        avg_bpd, count = 0, 0
        for batch in data_loader:
            bpd = self.eval_step(self.state, batch)
            avg_bpd += bpd * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_bpd = (avg_bpd / count).item()
        return eval_bpd

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


def train_pixel_cnn(train_loader, val_loader_maker_fn, steps_per_epoch, patience=15, max_epochs=150, **model_args):
    # Create a trainer module with specified hyperparameters
    trainer = TrainerModule(exmp_imgs=next(train_loader), **model_args)
    return trainer.train_model(train_loader, val_loader_maker_fn, steps_per_epoch, patience=patience, num_epochs=max_epochs)
   
