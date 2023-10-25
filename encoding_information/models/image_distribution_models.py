from abc import ABC, abstractmethod
import tensorflow as tf
import jax.numpy as np
import jax
import numpy as onp
from tqdm import tqdm


class ProbabilisticImageModel(ABC):
    """
    Base class for different probabilistic models images
    """

    @abstractmethod
    def fit(self, train_images, learning_rate=1e-2, max_epochs=200, steps_per_epoch=100,  patience=10, 
            batch_size=64, num_val_samples=1000, seed=0, verbose=True):
        """
        Fit the model to the images

        Parameters
        ----------
        train_images : ndarray
            Array of images, shape (N, H, W)
        learning_rate : float, optional
            Learning rate
        max_epochs : int, optional
            Maximum number of epochs to train for
        steps_per_epoch : int, optional
            Number of steps per epoch
        patience : int, optional
            Number of epochs to wait before early stopping
        batch_size : int, optional
            Batch size
        num_val_samples : int, optional
            Number of validation samples to use
        seed : int, optional
            Random seed to initialize the model
        verbose : bool, optional
            Whether to print training progress

        Returns
        -------
        val_loss_history : list
            List of validation losses at each epoch
        """
        pass
    
    @abstractmethod
    def compute_negative_log_likelihood(self, images, verbose=True):
        """
        Compute the NLL of the images under the model

        Parameters
        ----------
        images : ndarray
            Array of images, shape (N, H, W)
        verbose : bool, optional
            Whether to print progress

        Returns
        -------
        log_likelihood : ndarray
            Average negative log-likelihood of all images
        """
        pass
    
    @abstractmethod
    def generate_samples(self, num_samples, sample_size=None, verbose=True):
        """
        Generate samples from the model

        Parameters
        ----------
        num_samples : int
            Number of samples to generate
        sample_size : tuple, optional
            Size of each sample. If None, the size of the training images used
        verbose : bool, optional
            Whether to print progress
        """
        pass



def add_uniform_noise_fn(images):
    """
    Add uniform noise to images
    """
    return images + tf.random.uniform(shape=tf.shape(images), minval=0, maxval=1)

def _make_dataset_generators(images, batch_size, num_val_samples, add_uniform_noise=True):
    """
    Use tensorflow datasets to make fast data pipelines
    """
    
    # add trailing channel dimension if necessary
    if images.ndim == 3:
        images = images[..., np.newaxis]

    # add trailing channel dimension if necessary
    if images.ndim == 3:
        images = images[..., np.newaxis]
    elif images.shape[-1] != 1:
        raise ValueError("Only supports single-channel images currently")                

    images = images.astype(np.float32)
    # split images into train and validation
    val_images = images[:num_val_samples]
    train_images = images[num_val_samples:]

    # make tensorflow datasets
    train_ds = tf.data.Dataset.from_tensor_slices(train_images)
    val_ds = tf.data.Dataset.from_tensor_slices(val_images)

    if add_uniform_noise:
        train_ds = train_ds.map(add_uniform_noise_fn)
        val_ds = val_ds.map(add_uniform_noise_fn)

    train_ds = train_ds.repeat().shuffle(1024).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.shuffle(1024).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    return train_ds.as_numpy_iterator(), lambda : val_ds.as_numpy_iterator()


def evaluate_nll(data_iterator, state, add_uniform_noise=True, seed=0, batch_size=32, verbose=True):
    """
    Compute negative log likelihood over many batches

    batch_size only comes into play if data_iterator is a numpy array
    """
    key = jax.random.PRNGKey(seed)
    total_nll, count = 0, 0
    if isinstance(data_iterator, np.ndarray) or isinstance(data_iterator, onp.ndarray):
        data_iterator = np.array_split(data_iterator, batch_size)  # split into 32 batches
    if verbose:
        data_iterator = tqdm(data_iterator, desc='Computing loss')
    for batch in data_iterator:
        if add_uniform_noise:
            batch = batch + jax.random.uniform(key=key, minval=0, maxval=1, shape=batch.shape)
            key = jax.random.split(key)[0]
        batch_nll_per_pixel = _eval_step(state, batch)
        total_nll += batch_nll_per_pixel * batch[0].shape[0]
        count += batch[0].shape[0]
    nll = (total_nll / count).item()
    return nll

@jax.jit
def _train_step(state, imgs):
    """
    A standard gradient descent training step
    """
    loss_fn = lambda params: state.apply_fn(params, imgs)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def _eval_step(state, imgs):
    loss = state.apply_fn(state.params, imgs)
    return loss


def train_model(train_images, state, batch_size, num_val_samples,
                 steps_per_epoch, num_epochs, patience,
                 train_step=None,  verbose=True):
    """
    Training loop with early stopping. Returns a callable with 
    """
    if num_val_samples > train_images.shape[0]:
        raise ValueError("Number of validation samples must be less than the number of training samples")
    train_ds_iterator, val_loader_maker_fn = _make_dataset_generators(train_images, batch_size=batch_size, num_val_samples=num_val_samples)

    if train_step is None:
        train_step = _train_step
    best_params = state.params
    # uniform noise already added in the dataset generators
    eval_nll = evaluate_nll(val_loader_maker_fn(), state, add_uniform_noise=False, verbose=verbose)
    if verbose:
        print(f'Initial validation NLL: {eval_nll:.2f}')
    
    best_eval = eval_nll
    best_eval_epoch = 0
    val_loss_history = [best_eval]
    for epoch_idx in range(1, num_epochs+1):
        avg_loss = 0
        iter = range(steps_per_epoch)
        for _ in iter if not verbose else tqdm(iter, desc=f'Epoch {epoch_idx}'):
            batch = next(train_ds_iterator)
            state, loss = train_step(state, batch)

            avg_loss += loss / steps_per_epoch
        
        # uniform noise already added in the dataset generators
        eval_nll = evaluate_nll(val_loader_maker_fn(), state, add_uniform_noise=False, verbose=verbose) 
        val_loss_history.append(eval_nll)
        if verbose:
            print(f'Epoch {epoch_idx}: validation NLL: {eval_nll:.2f}')
        if patience is None:
            best_params = state.params # no early stopping
        elif eval_nll <= best_eval:
            best_eval_epoch = epoch_idx
            best_eval = eval_nll
            best_params = state.params
            # if self.log_dir is not None:
            #     self.save_model(step=epoch_idx)  
        elif epoch_idx - best_eval_epoch >= patience:
            break   

    return best_params, val_loss_history

