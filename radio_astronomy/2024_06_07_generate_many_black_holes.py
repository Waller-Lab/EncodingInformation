# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Go through the GRF to full image pipeline for each black hole, save as .npy files, 2024/06/07

# %%
import ehtim as eh # installed using https://github.com/achael/eht-imaging
# #%matplotlib ipympl
import pynoisy
import sys 
sys.path.append('/home/your_username/eht_stuff/ehtplot')
import ehtplot.color # installed using https://github.com/liamedeiros/ehtplot (only necessary if you want cfun="afmhot_10us", the colormap used in EHT papers)
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## Generate a GRF and make a matching output - same radius as the original M87 Image
#
# - Switch to a deterministic seed as possible.
#
# - If you keep running the solver with the original seed, then each time you call the solver for a single sample, it'll output the constant/same PDE solution.

# %%
timepoints = 64
seed_val = 5 # setting a seed
num_samples = 100000

# %%
flux = 1            # total flux of the image in Janskys (Jy). M87 has a flux of around 1, Sgr A* has a flux of around 2
object_name = 'M87' # the source you are observing 

# the right ascension (ra) in fractional hours and source declination (dec) in fractional degrees    
if object_name == "M87":
    ra = 12.513728717168174 
    dec = 12.39112323919932 
elif object_name == "SgrA":
    ra = 17.761121055553343 
    dec = -29.00784305556  

# the field of view of the image in radians. 
# This depends on the image you use In this tutorial, ...
if object_name == 'M87':
    fov = 128.0 * eh.RADPERUAS  # chosen to ring that is 40 uas for the example 64x64 image
elif object_name == 'SgrA':
    fov = 160.0 * eh.RADPERUAS  # chosen to ring that is 50 uas for the example 64x64 image

# %%
"""
Define diffusion and advection fields and plot the velocity and diffusion tensor principle axis
"""
diffusion = pynoisy.diffusion.general_xy(nx=64, ny=64)
advection = pynoisy.advection.general_xy(nx=64, ny=64)
# #%matplotlib notebook
#fig, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(8,3.5))
#diffusion.utils_visualization.major_axis(ax=ax[0])
#advection.utils_visualization.velocity_quiver(ax=ax[1])

# %%
"""
Run inoisy which solves an anisotropic spatio-temporal diffusion PDE. 
Generate two GRFs, the solver has a seed instantiated, from that seed, samples are drawn.
"""
solver = pynoisy.forward.HGRFSolver(advection, diffusion, nt=timepoints, seed=seed_val)
alpha = 2.0

for idx in range(num_samples):
    seed_use = idx 
    grfs = solver.run(seed=seed_use, verbose=1) # when you run the solver for a sample, each time you call it, it gives the same thing according to the base seed, unless you change the seed in the call
    # verbose=1 will just put the iterations and final residual, verbose=2 will put the iterations and the residual at each iteration

    envelope = pynoisy.envelope.ring(nx=64, ny=64, inner_radius=0.22).utils_image.set_fov(grfs.utils_image.fov)
    movie = pynoisy.forward.modulate(envelope, grfs, alpha)
    synthetic_data = movie.data

    synthetic_image = synthetic_data[0]
    image_size = synthetic_image.shape[0]
    #output_filename = 'fits/black_hole_{}.fits'.format(idx)
    output_raw_image = 'sources_06_07/black_hole_{}.npy'.format(idx)
    # normalize image to match total desired flux 
    synthetic_image = (synthetic_image/np.sum(synthetic_image) ) * flux
    # create the image object
    #synthetic_im = eh.image.Image(synthetic_image, psize=fov / float(image_size), ra=ra, dec=dec, source=object_name)
    # save the image as a fit
    #synthetic_im.save_fits(output_filename)
    # save the raw image as a numpy array, it's normalized by the flux but that can obviously be renormalized as needed.
    np.save(output_raw_image, synthetic_image)
