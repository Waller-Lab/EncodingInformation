# 5/13 at 11:30pm
import os
import gc
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from cleanplots import *
import matplotlib.pyplot as plt
#from bsccm import BSCCM
import scipy 
import jax.numpy as np
import numpy as onp
import time
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/lkabuli_waller/workspace/microscoBayes/')
#from analysis.entropy import * 
#from data.bsccm_util import *
from leyla_fns import *
import skimage
from skimage.transform import resize

phlat_psf = skimage.io.imread('phlat_psf.png')
phlat_psf = phlat_psf[900:2900, 1500:3500, 1]
phlat_psf = resize(phlat_psf, (200, 200), anti_aliasing=True)
phlat_region = phlat_psf[10:38, 20:48]
phlat_region /= np.sum(phlat_region)

diffuser_psf = skimage.io.imread('diffuser_psf.png')
diffuser_psf = diffuser_psf[:,:,1]
diffuser_resize = diffuser_psf[200:500, 250:550]
diffuser_resize = resize(diffuser_resize, (100, 100), anti_aliasing=True)  #resize(diffuser_psf, (28, 28))
diffuser_region = diffuser_resize[:28, :28]
diffuser_region /=  np.sum(diffuser_region)

psf = np.zeros((28, 28))
psf[20,20] = 1
psf[15, 10] = 1
psf[5, 13] = 1
psf[23, 6] = 1
psf = scipy.ndimage.gaussian_filter(psf, sigma=1)
psf /= np.sum(psf)

import tensorflow as tf
import tensorflow.keras as tfk 

(x_train, y_train), (x_test, y_test) = tfk.datasets.mnist.load_data()
data = np.concatenate((x_train, x_test), axis=0) # make one big glob of data
data = data.astype(np.float32)
for i in range(data.shape[0]):
    data[i] /= np.max(data[i])
    data[i] *= 1000 # convert to photons with arbitrary maximum value of 1000 photons
labels = np.concatenate((y_train, y_test), axis=0) # make one big glob of labels. 

# set training, testing, validation fractions
training_fraction = 0.8
testing_fraction = 0.1
validation_fraction = 0.1

# set seed values for reproducibility
seed_values = np.arange(1, 11)

#for noise in [0, 25, 50, 75, 100, 125, 150, 200]:
#    print("testing psf_4 at noise {}".format(noise))
#    test_system(noise, 'psf_4', 'simple', seed_values, data, labels, training_fraction, testing_fraction, diffuser_region, phlat_region, psf)

#for noise in [0, 25, 50, 75, 100, 125, 150, 200]:
#    print("testing phlat at noise {}".format(noise))
#    test_system(noise, 'phlat', 'simple', seed_values, data, labels, training_fraction, testing_fraction, diffuser_region, phlat_region, psf)

#for noise in [0, 25, 50, 75, 100, 125, 150, 200]:
#    print("testing diffuser at noise {}".format(noise))
#    test_system(noise, 'diffuser', 'simple', seed_values, data, labels, training_fraction, testing_fraction, diffuser_region, phlat_region, psf)

#for noise in [0, 25, 50, 75, 100, 125, 150, 200]:
#    print("testing unconvolved at noise {}".format(noise))
#    test_system(noise, 'uc', 'simple', seed_values, data, labels, training_fraction, testing_fraction, diffuser_region, phlat_region, psf)

#for noise in [2, 5, 10, 25, 50, 60, 80, 100]:
#    print("testing phlat PSF at noise {}, ".format(noise))
#    test_system(noise, 'phlat', 'simple', seed_values, data, labels, training_fraction, testing_fraction, diffuser_region, phlat_region, psf, 'poisson')
#for noise in [2, 5, 10, 25, 50, 60, 80, 100]:
#    print("testing diffuser PSF at noise {}, ".format(noise))
#    test_system(noise, 'diffuser', 'simple', seed_values, data, labels, training_fraction, testing_fraction, diffuser_region, phlat_region, psf, 'poisson')
#for noise in [2, 5, 10, 25, 50, 60, 80, 100]:
#    print("testing unconvolved PSF at noise {}, ".format(noise))
#    test_system(noise, 'uc', 'simple', seed_values, data, labels, training_fraction, testing_fraction, diffuser_region, phlat_region, psf, 'poisson')
# redo poisson noise for rml in simple and cnn networks
for noise in [2, 5, 10, 25, 50, 60, 80, 100]:
    print("testing RML PSF at noise {}, ".format(noise))
    test_system(noise, 'psf_4', 'simple', seed_values, data, labels, training_fraction, testing_fraction, diffuser_region, phlat_region, psf, 'poisson')
for noise in [2, 5, 10, 25, 50, 60, 80, 100]:
    print("testing RML PSF at noise {}, ".format(noise))
    test_system(noise, 'psf_4', 'cnn', seed_values, data, labels, training_fraction, testing_fraction, diffuser_region, phlat_region, psf, 'poisson')
for noise in [0, 25, 50, 75, 100, 125, 150, 200]:
    print("testing RML PSF at noise {}, ".format(noise))
    test_system(noise, 'psf_4', 'simple', seed_values, data, labels, training_fraction, testing_fraction, diffuser_region, phlat_region, psf, 'gaussian')
for noise in [0, 25, 50, 75, 100, 125, 150, 200]:
    print("testing RML PSF at noise {}, ".format(noise))
    test_system(noise, 'psf_4', 'cnn', seed_values, data, labels, training_fraction, testing_fraction, diffuser_region, phlat_region, psf, 'gaussian')
