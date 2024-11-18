from encoding_information.datasets.dataset_base_class import MeasurementDatasetBase
from encoding_information.image_utils import add_noise
import numpy as np 
import tensorflow as tf
from tqdm import tqdm
import scipy



class LenslessDataset(MeasurementDatasetBase):
    """ 
    Initialize dataset of lensless imaging measurements. Measurements are modeled by convolving natural images with various 
    lensless imaging phase masks. 

    This class is based on the CIFAR-10 dataset of natural images. It provides an interface for retrieving measurements
    from the dataset with optional phase mask, noise, and bias applied. 
    """

    def __init__(self, grayscale=True):
        """
        Initialize the dataset by loading the CIFAR-10 dataset. 
        
        The dataset is loaded using TensorFlow's `keras.datasets.cifar10` API. The training and test data are concatenated 
        to create a single dataset. Images are converted to grayscale by default. 
        """
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        self._image_data = np.concatenate([x_train, x_test], axis=0).astype(np.float32)
        self._label_data = np.concatenate([y_train, y_test], axis=0)
        if grayscale:
            assert len(self._image_data.shape) == 4, "If converting to grayscale input data must be 4D"
            self._image_data = tf.image.rgb_to_grayscale(self._image_data).numpy().squeeze()

    
    def get_measurements(self, num_measurements, phase_mask=None, tiling='random', mean=None, bias=0, noise='Poisson', data_seed=None, noise_seed=None):
        """
        Retrieve a set of lensless measurements from the CIFAR-10 dataset with optional phase mask, noise, and bias. 
        
        Parameters
        ----------
        num_measurements : int 
            Number of measurements to return. 
        phase_mask : np.ndarray, optional
            Phase mask point spread function to convolve with the images. If None, no phase mask is applied. Default is None.
        tiling : str, optional 
            Tiling pattern to apply to the images. Default is 'random'.
        mean : float, optional
            Mean value (in photon counts) to scale the images by, if there is bias the mean value will be shifted accordingly. 
            If None, then no scaling is applied.
        bias : float, optional 
            Bias to add to the measurements. Default is 0.
        noise : str, optional
            Type of noise to apply. 'Poisson' is supported, no noise added if None. Default is 'Poisson'. 
        data_seed : int, optional
            Seed for random selection of images from the dataset. Default is None.
        noise_seed : int, optional
            Seed for noise generation. Default is None.

        Returns 
        -------
        np.ndarray
            Array of selected measurements with optional phase mask, noise, and bias applied. 

        Raises
        ------
        Exception
            If the requested number of measurements exceeds the available dataset size, if an unsupported tiling type is provided, or
            if an unsupported noise type is provided.
        """

        data = self._image_data 
        image_dim = data.shape[-1]

        # Make sure there is enough data 
        if num_measurements > data.shape[0]:
            raise Exception(f'Cannot load {num_measurements} measurements, only {data.shape[0]} available')
        
        # Make sure the tiling type for images is supported 
        if tiling not in ['random', 'repeated']:
            raise Exception('Only random and repeated tiling is supported')
        
        # Make sure the phase mask is the same size as the images 
        if phase_mask is not None:
            assert phase_mask.shape == data.shape[1:], "Phase mask must be the same size as the images"
        
        # rescale the mean of images if provided. bias will shift the mean value for measurements in this implementation 
        if mean is not None:
            data /= np.mean(data) 
            data *= mean 
        
        # tile the images into 9x9 grids to simulate infinite extent
        tile_seed = data_seed if data_seed is not None else 0 
        if tiling == 'random':
            data, _ = generate_random_tiled_data(data, self._label_data, tile_seed) 
        elif tiling == 'repeated':
            data, _ = generate_repeated_tiled_data(data, self._label_data)

        # select random images if data_seed is provided, otherwise select the first num_measurements from the tiled images
        if data_seed is not None:
            np.random.seed(data_seed)
            indices = np.random.choice(data.shape[0], size=num_measurements, replace=False)
        else:
            indices = np.arange(num_measurements)
        images = data[indices]
        # convolve images with phase mask to produce measurements 
        if phase_mask is None:
            start_idx = image_dim // 2
            end_idx = image_dim // 2 - 1
            measurements = images[:, start_idx:-end_idx, start_idx:-end_idx] 
        else:
            measurements = convolved_dataset(phase_mask, images)

        # add small bias to measurements 
        measurements += bias 

        # add noise to measurements if necessary
        if noise is None:
            pass
        elif noise == 'Poisson':
            # add poisson noise and convert back to numpy array from jax array 
            measurements = np.array(add_noise(measurements, noise_seed))
        elif noise == 'Gaussian':
            raise NotImplementedError('Gaussian noise not implemented yet')
        else:
            raise ValueError(f'Noise type {noise} not recognized')
        
        return measurements


    def get_shape(self):
        """ 
        Return the shape of the initial CIFAR-10 dataset. 

        Returns
        -------
        tuple
            Shape of the CIFAR-10 images (num_images, height, width).
        """
        return self._image_data.shape
    

   
"""
These helper functions are copied in from the lensless_imager subfolder
"""
def tile_9_images(data_set):
    # takes 9 images and forms a tiled image
    assert len(data_set) == 9
    return np.block([[data_set[0], data_set[1], data_set[2]],[data_set[3], data_set[4], data_set[5]],[data_set[6], data_set[7], data_set[8]]])

def generate_random_tiled_data(x_set, y_set, seed_value=-1):
    # takes a set of images and labels and returns a set of tiled images and corresponding labels
    # the size of the output should be 3x the size of the input
    vert_shape = x_set.shape[1] * 3
    horiz_shape = x_set.shape[2] * 3
    random_data = np.zeros((x_set.shape[0], vert_shape, horiz_shape)) # for mnist this was 84 x 84
    random_labels = np.zeros((y_set.shape[0], 1))
    if seed_value==-1:
        np.random.seed()
    else: 
        np.random.seed(seed_value)
    for i in range(x_set.shape[0]):
        img_items = np.random.choice(x_set.shape[0], size=9, replace=True)
        data_set = x_set[img_items]
        random_labels[i] = y_set[img_items[4]]
        random_data[i] = tile_9_images(data_set)
    return random_data, random_labels

def generate_repeated_tiled_data(x_set, y_set):
    # takes set of images and labels and returns a set of repeated tiled images and corresponding labels, no randomness
    # the size of the output is 3x the size of the input, this essentially is a wrapper for np.tile
    repeated_data = np.tile(x_set, (1, 3, 3))
    repeated_labels = y_set # the labels are just what they were
    return repeated_data, repeated_labels

def convolved_dataset(psf, random_tiled_data):
    # takes a psf and a set of tiled images and returns a set of convolved images, convolved image size is 2n + 1? same size as the random data when it's cropped
    # tile size is two images worth plus one extra index value
    vert_shape = psf.shape[0] * 2 + 1
    horiz_shape = psf.shape[1] * 2 + 1
    psf_dataset = np.zeros((random_tiled_data.shape[0], vert_shape, horiz_shape)) # 57 x 57 for the case of mnist 28x28 images, 65 x 65 for the cifar 32 x 32 images
    for i in tqdm(range(random_tiled_data.shape[0])):
        psf_dataset[i] = scipy.signal.fftconvolve(psf, random_tiled_data[i], mode='valid')
    return psf_dataset