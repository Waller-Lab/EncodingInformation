# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Import necessary modules

# %%
import os
import sys
sys.path.append('/home/your_username/eht_stuff')
sys.path.append('/home/your_username/eht_stuff/ehtplot')
import ehtim as eh
import numpy as np
import matplotlib.pyplot as plt
import ehtplot as ep
import ehtplot.color
import argparse

from ehtim.imaging.imager_utils import imager_func

from skimage import metrics


# %% [markdown]
# ## Imaging Parameters

# include argparse 
parser = argparse.ArgumentParser(description='Generate black hole measurements for array index')
parser.add_argument('--array_idx', type=int, required=True, help='index of the array to use, 0 through 69')
parser.add_argument('--image_idx_start', type=int, required=False, default=0, help='index of the image to start at')
args = parser.parse_args() 

array_idx = args.array_idx # set based on command line argument
print(array_idx, " is the array index")
image_idx_start = args.image_idx_start

# %%
prior_fwhm  = 50.*eh.RADPERUAS  # Gaussian prior FWHM (radians)

ttype     = 'nfft'               # Type of Fourier transform ('direct', 'nfft', or 'fast') # this remains as nfft
npix      = 64                   # Number of pixels across the reconstructed image - our images have 64 x 64 pixels
fov       = 128*eh.RADPERUAS     # Field of view of the reconstructed image - our fov is 128 x radperuas
maxit     = 100                  # Maximum number of convergence iterations for imaging
regtype   = 'tv'                 # Regularization type after first round
stop      = 1e-6                 # Imager stopping criterion

flux      = 1.0                  # Total flux of the image in Janskys (Jy) I assume it is known

# %% [markdown]
# ## Define paths for specific recon

# %%
image_idx_end = 2000

inpath = 'observations_combination_{}/'.format(array_idx) # TODO change this to the path you want to save to
outpath = inpath + 'rml_recons/'
# Make output directories
if not os.path.exists(os.path.dirname(outpath)):
    os.makedirs(os.path.dirname(outpath))

# loop the recons 

for image_idx in range(image_idx_start, image_idx_end):
    # get the paths ready
    inobs   = os.path.join(inpath, 'uvfits/{}.uvfits'.format(image_idx)) #'tutorial1_data.uvfits')
    gt_image_path = os.path.join('sources_06_07/black_hole_{}.npy'.format(image_idx))
    save_path = os.path.join(outpath, '{}.npy'.format(image_idx))
    # load the gt image
    gt_image = np.load(gt_image_path)
    # Load the uvfits file
    obs = eh.obsdata.load_uvfits(inobs)
    obs_sc = obs.copy()
    # get the system resolution
    res = obs.res() 

    # make prior object
    prior = eh.image.make_square(obs_sc, npix, fov)
    prior = prior.add_gauss(flux, (prior_fwhm, prior_fwhm, 0, 0, 0))
    #print('prior total flux: {:.2f} Jy'.format(prior.total_flux()))

    # default parameters for reconstruction, matches EHT people's code implementation. 100 iterations per round, simple then two rounds of TV reg
    outim = imager_func(obs, prior, prior, flux, d1='vis', s1='simple', alpha_s1=1, show_updates=False, maxit=maxit)
    # round 2
    outim = outim.blur_circ(res)
    outim = imager_func(obs, outim, outim, flux, d1='vis', s1=regtype, alpha_s1=1, show_updates=False, maxit=maxit)
    # round 3
    outim = outim.blur_circ(res / 2.0)
    outim = imager_func(obs, outim, outim, flux, d1='vis', s1=regtype, alpha_s1=1, show_updates=False, maxit=maxit)
    #fig = outim.display(cfun='afmhot_10us')
    # save the reconned image
    out_image_np = outim.imvec.reshape((64, 64))
    np.save(save_path, out_image_np)
            

# %%
