"""
Callback functions for use during training in keras
"""
import pandas as pd
from cleanplots import *
import numpy as np
from pathlib import Path
import matplotlib.gridspec as gridspec
import os
import shutil
from led_array.models.visualization import plot_density_network_output, plot_input_recon_sample_montage
import time
import yaml

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers



def get_marker_index(target_row):
    return np.flatnonzero(np.logical_not(np.isnan(target_row)))[0]            
            


class ElapsedTimeCallback(tfk.callbacks.Callback):

    def __init__(self, config, config_file_path, already_elapsed_time=0):
        super().__init__()
        self.already_elapsed_time = already_elapsed_time
        self.start_time = time.time()
        self.config = config
        self.config_file_path = config_file_path


    def on_epoch_end(self, epoch, logs=None):
        self.config['training']['elapsed'] = time.time() - self.start_time + self.already_elapsed_time 
        print('saving elapsed time: ' + str(self.config['training']['elapsed'] / 60 ** 2) + 'h')
        #Resave config file
        with open(self.config_file_path, 'w') as file:
            documents = yaml.dump(self.config, file)

class DensityNetworkVisualizeCallback(tfk.callbacks.Callback):
    """
    Callback for visualizing probabilistic marker prediction during training.
    Shows image, distribution, true value
    """
    
    def __init__(self, validation_dataset, markers, logging_dir=None, num_images=16, test_size=0, display_range=None):
        super().__init__()
        
        if not os.path.exists(logging_dir + 'images'):
            os.mkdir(logging_dir + 'images')
            os.mkdir(logging_dir + 'images/png/')
            os.mkdir(logging_dir + 'images/pdf/')
        self.logging_dir = logging_dir

            
        self.display_range = display_range
        self.markers = markers
        
        ### visualize examples from model
        self.num_images = num_images
        # Read and store images that will be used for live display
        self.images = {m: [] for m in markers}
        self.targets = {m: [] for m in markers} 
        for image, target in validation_dataset:
            marker_index = get_marker_index(target)
            marker = markers[marker_index]
            if len(self.images[marker]) == num_images:
                continue            
            self.images[marker].append(image)
            self.targets[marker].append(target)
            if np.sum([len(self.images[m]) for m in markers]) == self.num_images * len(markers):
                break
        self.images = np.stack([val for sublist in [self.images[m] for m in markers] for val in sublist])
        self.targets = np.stack([val for sublist in [self.targets[m] for m in markers] for val in sublist])
        
        self.image_fig = ipympl_fig(figsize=(12, len(markers) * num_images * 0.3))

        plt.show()      
        
    def make_initial_plot(self, model):
        plot_density_network_output(model, self.images, self.targets, self.markers, self.image_fig, 
                display_range=self.display_range, verbose=False)

        self.image_fig.canvas.draw()

        # Save stuff to logging dir
        if hasattr(self, 'logging_dir'):
            self.image_fig.savefig(self.logging_dir + 'images/png/denisty_0.png')
            self.image_fig.savefig(self.logging_dir + 'images/pdf/denisty_0.pdf', transparent=True)
        # clear figure
        self.image_fig.clear()
        
    def on_epoch_end(self, epoch, logs=None):        
    
        plot_density_network_output(self.model, self.images, self.targets, self.markers, self.image_fig, 
                    display_range=self.display_range, verbose=False)

        self.image_fig.canvas.draw()

        # Save stuff to logging dir
        self.image_fig.savefig(self.logging_dir + 'images/png/denisty_{}.png'.format(epoch+1))
        self.image_fig.savefig(self.logging_dir + 'images/pdf/denisty_{}.pdf'.format(epoch+1), transparent=True)

        # clear figure
        self.image_fig.clear()
            
            
class VAEVisualizeCallback(tfk.callbacks.Callback):
    """
    Callback for visualizing data during training. Shows input/recon/sampled images
    """
    
    def __init__(self, validation_dataset, logging_dir=None,num_images=8, test_size=0):
        super().__init__()
        
        os.mkdir(logging_dir)
        os.mkdir(logging_dir + 'images/')
        self.logging_dir = logging_dir
        
  
        ### visualize examples from model
        self.num_images = num_images
        # Read and store images that will be used for live display
        self.images = []
        for image, image in validation_dataset:
            self.images.append(image)
            if len(self.images) == self.num_images:
                break
        self.images = np.stack(self.images)
        
        self.image_fig = ipympl_fig(figsize=(8, 4))

        plt.show()        

    def on_epoch_end(self, epoch, logs=None):        
        # Update image plots
        params = self.model.encoder_model(self.images)
        dist = self.model.z_sampler_model(params)
        reconstructions = self.model.decoder_model(dist.mean()).mean().numpy().squeeze()

        #TODO: could move this all into the visualization function
        
        # Generate new samples by sampling from prior
        np.random.seed(12345)
        prior_sample = np.random.normal(0, 1, size=(self.num_images, self.model.decoder_model.input_shape[1]))
        generated_samples = self.model.decoder_model(prior_sample).mean().numpy().squeeze()
        
        plot_input_recon_sample_montage(np.stack(self.images).squeeze(), reconstructions, 
                                        generated_samples, fig=self.image_fig)
        self.image_fig.canvas.draw()
        
        # Save stuff to logging dir
        if hasattr(self, 'logging_dir'):
            self.image_fig.savefig(self.logging_dir + 'images/vae_img_recon_sample_{}.png'.format(epoch))
