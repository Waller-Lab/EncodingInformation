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
# ## Generate a synthetic black hole with the same radius as the existing M87 data, and go through the GRF to image pipeline for each image one by one, 2024/09/20
# - Also save the uvfits file
# - Also save the dirty image

# %%
import sys 
sys.path.append('/home/your_username/eht_stuff')
sys.path.append('/home/your_username/eht_stuff/ehtplot')
import ehtim as eh # installed using https://github.com/achael/eht-imaging
# #%matplotlib ipympl
#import pynoisy # have an import error but also it works as is without this.
#import ehtplot.color #installed using https://github.com/liamedeiros/ehtplot (only necessary if you want cfun="afmhot_10us", the colormap used in EHT papers)
import matplotlib.pyplot as plt
import numpy as np
import argparse 

# %% [markdown]
# # Load Synthetic Image and Set Observation Input parameters
# - loads from a numpy file

# include argparse 
parser = argparse.ArgumentParser(description='Generate black hole measurements for array index')
parser.add_argument('--array_idx', type=int, required=True, help='index of the array to use, 0 through 69')
parser.add_argument('--image_idx_start', type=int, required=False, default=0, help='index of the image to start at')
args = parser.parse_args() 

image_idx_start = args.image_idx_start
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
array_idx = args.array_idx # set based on command line argument
print(array_idx, " is the array index")
array_filename = 'all-4-eht-combos/combination_{}.txt'.format(array_idx) #'ehtim-tutorial/arrays/EHT2017.txt'   # telescope array parameters filename # TODO change this to the correct array you want to image with

# observation parameters
tstart = 0.0         # start time of observations in hours
tstop = 24.0         # end time of observations in hours
timetype = 'UTC'     # how to interpret tstart and tstop, either 'GMST' or 'UTC'
tint = 60.0          # scan integration time in seconds
tadv = 120.0         # scan advance to new measurement time in seconds (must be larger than tint)
mjd = 57850          # Modified Julian Date of observation (night of April 6-7, 2017)
rf = 230000000000.0  # the observation frequency in Hz
bw = 4000000000.0    # the observation bandwidth in Hz

# measurement noise parameters
add_th_noise=True   # add thermal noise to the measurements. False if you don't want to add any noise
ampcal=True          # true if you want the amplitudes to be calibrated, false if not (due to atmospheric gain error)
phasecal=True        # true if you want the phases to be calibrated, false if not (due to atmospheric phase error)

# computational parameters
# ttype = 'direct'     # how to commpute the DTFT measurements. 'direct' formms a DTFT matrix. Note if you install nfft, ttype='nfft' is much faster for large images or when you have lots of measurements
ttype = 'nfft' # using nfft to make it fast

# %%
array = eh.array.load_txt(array_filename)

# %%
print(array.tarr)

# %% [markdown]
# # Make directory structure

# %%
num_images_to_observe = 60000 # TODO change to the number of images you want to observe
loading_path = 'sources_06_07/'


# %%
saving_path = 'observations_combination_{}/'.format(array_idx) # TODO change this to the path you want to save to

clean_path = saving_path + 'cleans_s/cleans/' 
vis_path = saving_path + 'visibilities_s/visibilities/'
dirty_path = saving_path + 'dirty_images/'
sigma_path = saving_path + 'sigmas/'
uv_path = saving_path + 'uvfits/'

uv_coords = saving_path + 'uv.npy'
# make the directory if it doesn't exist, along with the nesting structure
import os
if not os.path.exists(saving_path):
    os.makedirs(saving_path)
    os.makedirs(uv_path)
    os.makedirs(sigma_path)
    os.makedirs(clean_path)
    os.makedirs(vis_path)
    os.makedirs(dirty_path)

# I have a path for the .txt file containing the telescope. I want to copy that file to the saving path directory too, so I can keep track of the telescope used for the observation
# I also want to rename the file to just telescope.txt
import shutil
shutil.copy(array_filename, saving_path)
shutil.move(saving_path + array_filename.split('/')[-1], saving_path + 'telescope_array.txt')

# also add a data loading template notebook to check data progress
data_loading_notebook_path = '/home/your_username/black_holes/data_progress_template.ipynb'
shutil.copy(data_loading_notebook_path, saving_path)

# %%
#one-time setup operation
np_filename = loading_path + 'black_hole_0.npy'
synthetic_image = np.load(np_filename)
image_size = synthetic_image.shape[0]
synthetic_image = (synthetic_image / np.sum(synthetic_image)) * flux 
im = eh.image.Image(synthetic_image, psize=fov / float(image_size), ra=ra, dec=dec, source=object_name)

# print the field of view (fov) in micro-arcseconds
print('field of view:')
print(im.psize*im.xdim / eh.RADPERUAS)
print(im.fovx() / eh.RADPERUAS)
# print the image size
print('image size:')
print(image_size)


# %%
def get_dirty_image(A_vis, vis):
    # for now, let's assume that the images are always square (should be true), then the output dirty_image can be reshaped into a square image
    dirty_image = A_vis.conj().T @ vis 
    # this will be complex, so take the absolute value to get the magnitude of it 
    dirty_image = np.abs(dirty_image)
    image_dim = int(np.sqrt(dirty_image.shape[0]))
    dirty_image = dirty_image.reshape(image_dim, image_dim)
    return dirty_image


# %% [markdown]
# ## This does the actual imaging and data saving part

# %%
start_index = image_idx_start # TODO change this if things crash
for image_idx in range(start_index, num_images_to_observe):
    # load the numpy file
    np_filename = loading_path + 'black_hole_{}.npy'.format(image_idx)
    synthetic_image = np.load(np_filename)
    # convert to the .fits format to work with it
    synthetic_image = (synthetic_image / np.sum(synthetic_image)) * flux
    im = eh.image.Image(synthetic_image, psize=fov / float(image_size), ra=ra, dec=dec, source=object_name)
    # observe the source with the telescope array using the parameters specified above
    obs = im.observe(array, ttype=ttype, mjd=mjd, timetype=timetype, 
                     tstart=tstart, tstop=tstop, tint=tint, tadv=tadv, 
                     bw=bw, add_th_noise=add_th_noise, ampcal=ampcal, 
                     phasecal=phasecal)
    # see the telescope uv coverage
    #obs.plotall('u','v', conj=True) # uv-coverage. plot the (u,v) frequencies sampled by the telescope array. Since the image we observe is real (u,v) and (-u,-v) should be complex conjugates of one another. therefore we also plot the conjugate
    vis, sigma_vis, A_vis = eh.imaging.imager_utils.chisqdata_vis(obs, im, mask=[])
    #multiplying the image with the forward matrix A gives you the ideal visibilities with no noise
    out = np.matmul(A_vis, im.imvec)
    # get the dirty image from the visibilities 
    dirty_image = get_dirty_image(A_vis, vis)

    #plot to show the overlap in the noisy measured data from obs and the clean visibilities generated through matrix multiplication
    # plt.subplot(1,2,1)
    # plt.scatter(np.real(vis), np.imag(vis), label="noisy")
    # plt.scatter(np.real(out), np.imag(out), label="clean")
    # plt.xlabel("real")
    # plt.ylabel("imag")
    # plt.legend()
    # plt.subplot(1,2,2)
    # plt.scatter(np.abs(vis), np.angle(vis), label='noisy')
    # plt.scatter(np.abs(out), np.angle(out), label='clean')
    # plt.xlabel("magnitude")
    # plt.ylabel("phase")
    # plt.legend()
    # plt.figure()
    # plt.imshow(dirty_image, cmap='afmhot')

    # save the frequency array only once since it's the same each time. 
    if image_idx == 0:
        u = obs.data['u']         # u coordinate of data points
        v = obs.data['v']         # v coordinate of data points
        # concatenate u and v into two columns
        uv = np.column_stack((u, v))
        print("uv sample shape is: ", uv.shape)
        np.save(uv_coords, uv)
        plt.figure()
        plt.scatter(u, v)
        plt.xlim(-1e10, 1e10)
        plt.ylim(-1e10, 1e10)
        plt.savefig(saving_path + 'uv_coords.png')
        plt.close()
    # save the uvfits so you can do recon later
    obs.save_uvfits(uv_path + '{}.uvfits'.format(image_idx))    
    # save dirty image
    np.save(dirty_path + '{}.npy'.format(image_idx), dirty_image)
    # save sigma values - technically they're the same for every image so not necessary each time. but it's ok for now as is. TODO could change this in the future. 
    np.save(sigma_path + '{}.npy'.format(image_idx), sigma_vis) 
    # save visibilities with noise
    np.save(vis_path + '{}.npy'.format(image_idx), vis)
    # save clean measurements (out)
    np.save(clean_path + '{}.npy'.format(image_idx), out)
