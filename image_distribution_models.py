from abc import ABC, abstractmethod
from gaussian_process import *
from pixel_cnn import train_pixel_cnn
import tensorflow as tf

class ProbabilisticImageModel(ABC):
    """
    Base class for different probabilistic models of image patches
    """

    @abstractmethod
    def fit(self, images):
        """
        Fit the model to the images
        """
        pass
    
    @abstractmethod
    def compute_likelihood(self, images):
        """
        Compute the likelihood of the images under the model

        Parameters
        ----------
        images : ndarray
            Array of images, shape (N, H, W)

        Returns
        -------
        log_likelihood : ndarray
            Average log-likelihood of all images
        """
        pass
    
    @abstractmethod
    def generate_samples(self, num_samples, sample_size=None, ):
        """
        Generate samples from the model

        Parameters
        ----------
        num_samples : int
            Number of samples to generate
        sample_size : tuple, optional
            Size of each sample. If None, the size of the training images used
        """
        pass
    



class StationaryGaussianProcess(ProbabilisticImageModel):

    def __init__(self):
        self.mean_vec = None
        self.cov_mat = None

    def fit(self, images, optimize=True):
        self.mean_vec, self.cov_mat = estimate_stationary_cov_mat(images, eigenvalue_floor=1e-3, optimize=optimize, return_mean=True)

    def compute_likelihood(self, images):
        if self.mean_vec is None or self.cov_mat is None:
            raise ValueError("Model not fitted yet")
        lls = compute_stationary_log_likelihood(images, self.mean_vec, self.cov_mat)
        return lls.mean()
        
    def generate_samples(self, num_samples, sample_size=None, ensure_nonnegative=True):
        if self.mean_vec is None or self.cov_mat is None:
            raise ValueError("Model not fitted yet")
        samples = generate_stationary_gaussian_process_samples(self.mean_vec, self.cov_mat, 
                                                     num_samples, sample_size, ensure_nonnegative=ensure_nonnegative)
        return samples


class PixelCNN(ProbabilisticImageModel):

    def __init__(self):
        pass

    def fit(self, images, max_epochs=200, steps_per_epoch=100, val_iter_maker=None, patience=10, 
            batch_size=64, num_val_samples=1000, num_hidden_channels=64, learning_rate=1e-2, num_mixture_components=40,
            verbose=True):
        
        # add trailing channel dimension if necessary
        if images.ndim == 3:
            images = images[..., np.newaxis]

        def get_datasets(images, batch_size):
            """
            Use tensorflow datasets to make fast data pipelines
            """
            # add trailing channel dimension if necessary
            if images.ndim == 3:
                images = images[..., np.newaxis]
            elif images.shape[-1] != 1:
                raise ValueError("PixelCNN only supports single-channel images currently")
                

            # split images into train and validation
            val_images = images[:num_val_samples]
            train_images = images[num_val_samples:]

            # make tensorflow datasets
            train_ds = tf.data.Dataset.from_tensor_slices(train_images)
            val_ds = tf.data.Dataset.from_tensor_slices(val_images)

            train_ds = train_ds.repeat().shuffle(1024).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            val_ds = val_ds.shuffle(1024).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        
            return train_ds.as_numpy_iterator(), lambda : val_ds.as_numpy_iterator()

        train_iter, val_iter_maker = get_datasets(images, batch_size)
        self._pixel_cnn_flax, self.validation_loss_history = train_pixel_cnn(train_iter, val_iter_maker, steps_per_epoch,
                                            c_hidden=num_hidden_channels, num_mixture_components=num_mixture_components,
                                        patience=patience, max_epochs=max_epochs, learning_rate=learning_rate, verbose=verbose)


    def compute_likelihood(self, data):
        pass

    def generate_samples(self, num_samples, sample_size=None, ensure_nonnegative=True):
        pass
