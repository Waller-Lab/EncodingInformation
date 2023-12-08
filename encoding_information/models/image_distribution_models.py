from abc import ABC, abstractmethod
import tensorflow as tf
import jax.numpy as np
import jax
import numpy as onp
from tqdm import tqdm
import warnings


class ProbabilisticImageModel(ABC):
    """
    Base class for different probabilistic models images
    """

    @abstractmethod
    def fit(self, train_images, learning_rate=1e-2, max_epochs=200, steps_per_epoch=100,  patience=10, 
            batch_size=64, num_val_samples=None, percent_samples_for_validation=0.1,
            seed=0, verbose=True):
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
            Number of validation samples to use. If None, use percent_samples_for_validation
        percent_samples_for_validation : float, optional
            Percentage of samples to use for validation
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
    def generate_samples(self, num_samples, sample_shape=None, verbose=True):
        """
        Generate samples from the model

        Parameters
        ----------
        num_samples : int
            Number of samples to generate
        sample_shape : tuple, optional
            shape of each sample. If None, the shape of the training images used
        verbose : bool, optional
            Whether to print progress
        """
        pass



def add_uniform_noise_fn(images, condition_vectors=None):
    """
    Add uniform noise to images
    """
    noisy_images = images + tf.random.uniform(shape=tf.shape(images), minval=0, maxval=1)
    if condition_vectors is not None:
        return noisy_images, condition_vectors
    else:
        return noisy_images

def make_dataset_generators(images, batch_size, num_val_samples, add_uniform_noise=True, condition_vectors=None):
    """
    Use tensorflow datasets to make fast data pipelines
    """
    if num_val_samples > images.shape[0]:
        raise ValueError("Number of validation samples must be less than the number of training samples")
    
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

    if condition_vectors is not None:
        # Validate the shape of condition_vectors
        if condition_vectors.shape[0] != images.shape[0]:
            raise ValueError("Condition vectors and images must have the same number of samples")

        # Combine images and condition_vectors
        train_condition_vectors = condition_vectors[num_val_samples:]
        val_condition_vectors = condition_vectors[:num_val_samples]

        # Update TensorFlow datasets to include condition_vectors
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_condition_vectors))
        val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_condition_vectors))
    else:

        # make tensorflow datasets
        train_ds = tf.data.Dataset.from_tensor_slices(train_images)
        val_ds = tf.data.Dataset.from_tensor_slices(val_images)

    if add_uniform_noise:
        train_ds = train_ds.map(add_uniform_noise_fn)
        val_ds = val_ds.map(add_uniform_noise_fn)

    train_ds = train_ds.repeat().shuffle(1024).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.shuffle(1024).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    return train_ds.as_numpy_iterator(), lambda : val_ds.as_numpy_iterator()


def evaluate_nll(data_iterator, state, eval_step=None, add_uniform_noise=True, seed=0, batch_size=16, verbose=True):
    """
    Compute negative log likelihood over many batches

    batch_size only comes into play if data_iterator is a numpy array
    """
    if eval_step is None: # default eval step
        eval_step = jax.jit(lambda state, imgs: state.apply_fn(state.params, imgs))

    key = jax.random.PRNGKey(seed)
    total_nll, count = 0, 0
    if isinstance(data_iterator, np.ndarray) or isinstance(data_iterator, onp.ndarray):
        data_iterator = np.array_split(data_iterator, len(data_iterator) // batch_size + 1)  # split into batches of batch_size or less
        data_iterator = tqdm(data_iterator, desc='Computing loss')
    for batch in data_iterator:
        if isinstance(batch, tuple):
            images, condition_vector = batch
        else:
            images = batch
            condition_vector = None
        if add_uniform_noise:
            images = images + jax.random.uniform(key=key, minval=0, maxval=1, shape=images.shape)
            key = jax.random.split(key)[0]
        batch_nll_per_pixel = eval_step(state, images) if condition_vector is None else eval_step(state, images, condition_vector)
        total_nll += images.shape[0] * batch_nll_per_pixel
        count += images.shape[0]
    # compute average nll per pixel
    nll = (total_nll / count).item()
    return nll


def train_model(train_images, state, batch_size, num_val_samples,
                 steps_per_epoch, num_epochs, patience, train_step, condition_vectors=None,
                  verbose=True):
    """
    Training loop with early stopping. Returns a callable with 
    """
    if num_val_samples >= train_images.shape[0]:
        num_val_samples = int(train_images.shape[0] * 0.1)
        warnings.warn(f'Number of validation samples must be less than the number of training samples. Using {num_val_samples} validation samples instead.')
    train_ds_iterator, val_loader_maker_fn = make_dataset_generators(train_images,
                     batch_size=batch_size, num_val_samples=num_val_samples, condition_vectors=condition_vectors)

    if condition_vectors is not None:
        eval_step = jax.jit(lambda state, imgs, conditioning_vecs: state.apply_fn(state.params, imgs, conditioning_vecs))
    else: 
        eval_step = jax.jit(lambda state, imgs: state.apply_fn(state.params, imgs))

    best_params = state.params
    # uniform noise already added in the dataset generators
    eval_nll = evaluate_nll(val_loader_maker_fn(), state, eval_step=eval_step, add_uniform_noise=False, verbose=verbose)
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
            if condition_vectors is None:
                state, loss = train_step(state, batch)
            else:
                state, loss = train_step(state, batch[0], batch[1])

            avg_loss += loss / steps_per_epoch
        
        # uniform noise already added in the dataset generators
        eval_nll = evaluate_nll(val_loader_maker_fn(), state, eval_step=eval_step, add_uniform_noise=False, verbose=verbose) 
        if np.isnan(eval_nll):
            warnings.warn('NaN encountered in validation loss. Stopping early.')
            break
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

