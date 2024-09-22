

# images = load_bsccm_images(bsccm, channel_name, num_images=num_images, 
#                             edge_crop=edge_crop, convert_units_to_photons=True, median_filter=False, verbose=False, batch=1)
# patches = extract_patches(images, patch_size=patch_size, num_patches=num_patches, verbose=False, seed=patches_seed)


from abc import ABC, abstractmethod


class MeasurementDatasetBase(ABC):
    


    @abstractmethod
    def get_measurements(self, num_measurements, mean=None, bias=0, data_seed=None, noise_seed=123456,
                               noise='Poisson', **kwargs):
        """
        Return a numpy array of noisy_measurements in the dataset. 
        
        Datasets may have different shapes depending on the type of data they contain. e.g. if you're requesting 
        different channels. e.g. (num_images, height, width) or (num_images, vector_length, channels)
        Additional kwargs can be used to specify dataset-specific parameters. e.g. channel names, etc

        :param num_measurements: Number of measurements to return
        :param mean: Mean of the data. If provided, the data will be rescaled to have this mean. Otherwise, the mean of
                        the data is dataset-dependent
        :param bias: Bias to add to the data. If there is noise, then no pixel values will be lower than this value
        :param data_seed: Seed for selecting random data
        :param noise_seed: Seed for adding noise
        :param noise: Type of noise to add. E.g. None or 'Poisson' or 'Gaussian'. Not all datasets support all types of noise
        
        :return: numpy array of noisy measurements
        """
        ...


    @abstractmethod
    def get_shape(self, **kwargs):
        """
        Return the shape of the dataset. e.g. (num_images, height, width, channels) or (num_images, vector_length, channels)
        
        Additional kwargs can be used to specify dataset-specific parameters. e.g. channel names, etc
        """
        ...

    




