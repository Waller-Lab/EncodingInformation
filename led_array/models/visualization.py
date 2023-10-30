"""
Functions for visualizing the outputs of trained/training models and other stuff
"""

import pandas as pd
from cleanplots import *
import numpy as np
from pathlib import Path
import matplotlib.gridspec as gridspec
import os
import shutil
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

class OverlayedHistograms:
    """
    Convenience class for plotting multiple histograms on the same axes (with equal bin sizes)
    """
    def __init__(self, ax=None, bins=50, log=True, logx=True):
        self.ax = ax
        self.bins = bins
        self.all_values = []
        self.labels = []
        self.log = log
        self.logx = logx

    def add(self, values, label=None):
        self.all_values.append(values)
        self.labels.append(label)
    
    def get_hist_counts(self, eigenvalues):
        bins = self.get_bins()
        counts, _ = np.histogram(eigenvalues, bins=bins)
        return counts
    
    def get_bins(self):
        min_value = np.array([np.min(e) for e in self.all_values]).min()
        max_value = np.array([np.max(e) for e in self.all_values]).max()
        if self.logx:
            bins = np.logspace(np.log(min_value), np.log(max_value), self.bins, base=np.e)
        else:
            bins = np.linspace(min_value, max_value, self.bins)
        return bins

    def plot(self, zorder=None, **kwargs):
        bins = self.get_bins()

        for eigenvalues, label in zip(self.all_values, self.labels):
            _ = self.ax.hist(eigenvalues, bins=bins, log=self.log, label=label, alpha=0.5, 
                             zorder=zorder[label] if zorder is not None else 1, **kwargs)
                             

        if self.logx:
            self.ax.set(xscale='log')



def plot_density_network_output(model, images, targets, markers,  
                                fig=None, fig_size=(8, 16), legend=False, log_likelihood=False,
                               display_range=None, show_gridlines=True, num_cols=4, display_channel_index=0, model_point=None,
                               verbose=True):
    """
    run images through the network and show their true values along with ground truth target
    """

    if model_point is not None:
        point_estimates = model_point(images)
    else:
        point_estimates = None

    predictions = model(images)

    
    density = {}
    for marker in predictions.keys():
        x_range = display_range[marker]
        domain = np.linspace(x_range[0], x_range[1], 1000)
        
        x = np.stack(images.shape[0] * [domain], axis=1)
        if predictions[marker].event_shape == 1:
            #replicate along sample dimension and add [1] event dimension
            x = x.reshape(*x.shape, 1)
        
        if log_likelihood:
            density[marker] = predictions[marker].log_prob(x)
        else:
            density[marker] = predictions[marker].tensor_distribution.prob(x)
            
    
    num_images = predictions[markers[0]].shape[0]
    outputs = []
    if fig is None:
        fig = ipympl_fig(figsize=fig_size)
    gs1 = gridspec.GridSpec(num_images // num_cols, 2 * num_cols)
    gs1.update(wspace=0.25, hspace=0.25,
              left=0.02, right=.98, top=.98, bottom=0.02) # set the spacing between axes. 

    # if verbose use tqdm, otherwise just use range
    if verbose:
        iter = tqdm(enumerate(zip(images, targets)))
    else:
        iter = enumerate(zip(images, targets))
    for index, (image, target_row) in iter:
        img_col = (index % num_cols) * 2
        plot_col = (index % num_cols) * 2 + 1 
        img_ax = plt.subplot(gs1[index // num_cols, img_col])
        plot_ax = plt.subplot(gs1[index // num_cols, plot_col])
        
        marker_index = np.flatnonzero(np.logical_not(np.isnan(target_row)))[0]
        marker = markers[marker_index]
        target = target_row[marker_index]
        
        if point_estimates is not None:
            point_estimate = point_estimates[marker][index]
        
        sample_density_for_marker = density[marker][:, index]

        plot_image_and_scalar_distribution(img_ax, plot_ax, image, domain, sample_density_for_marker, target, 
                                           display_channel_index=display_channel_index, 
                                           point_estimate=None, annotation=marker, 
                                           legend=legend and index == 0, show_gridlines=show_gridlines)
        
        names = ['True value','Predicted']
        plt.gcf().legend(names, loc='upper right')


        
def plot_image_and_scalar_distribution(img_ax, plot_ax, image, domain, density, target, 
                                       display_channel_index=0, point_estimate=None, annotation=None, legend=True,
                                      show_gridlines=False, show_x_axis=True):
    
    domain = np.exp(domain)
    if point_estimate is not None:
        point_estimate = np.exp(point_estimate)
    target = np.exp(target)

    img_ax.imshow(image[..., display_channel_index], cmap='inferno')
    img_ax.set_axis_off()

    # orange line for point estimate
    if point_estimate is not None:
        plot_ax.plot([target, target], [0, np.max(density)], point_estimate, color='tab:orange', linewidth=2)


    plot_ax.fill_between(domain, density, 0, color='royalblue', alpha=0.6)
    plot_ax.set_ylim([0, np.max(density) * 1.1])


    plot_ax.plot([target, target], [0, np.max(density)], color='black', linewidth=2)
    #         plot_ax.set_yticks([], []) 
    #         if index < images.shape[0] - 4:
    #             plot_ax.set_xticks([]) 
    plot_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    if annotation is not None:
        plot_ax.annotate(annotation, (0.4, 0.85), xycoords='axes fraction', bbox=dict(boxstyle="round", fc="w"))

    # plot_ax.set_xticks(np.arange(-10, 10))
    plot_ax.set_xscale("log")
    plot_ax.set_xlim(np.min(domain), np.max(domain))
    if show_gridlines:
        plot_ax.grid(True)
    if not show_x_axis:
        plot_ax.set_xticks([])
    plot_ax.set(yticks=[], ylabel='Probability')

    if legend:
        names = ['True value','Predicted']
        plt.gcf().legend(names, loc='upper right')


def plot_input_recon_sample_montage(input_img, recon_img, sample_img, fig=None):
    """
    Plot input image, reconstructed image, and smapled image in 3 rows
    """
    num_images = input_img.shape[0]
    if fig is None:
        fig = ipympl_fig(figsize=size)
    gs1 = gridspec.GridSpec(3, num_images)
    gs1.update(wspace=0.0, hspace=0.) # set the spacing between axes. 
    for row_index, images in enumerate([input_img, recon_img, sample_img]):
        for i, img in enumerate(images):
            index = i + row_index * num_images
            ax = plt.subplot(gs1[index])
            ax.imshow(img)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            if i == 0:
                ax.set_ylabel(['Original data', 'Reconstructed', 'Prior samples'][row_index])
    return fig