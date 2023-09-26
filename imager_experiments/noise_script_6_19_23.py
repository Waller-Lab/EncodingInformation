# 5/13 at 11:30pm
import os
import gc
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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
from leyla_fns import *
import skimage
from skimage.transform import resize

# load the 4 psfs
phlat_psf = load_rml_psf()
diffuser_psf = load_diffuser_psf()
psf_4 = load_4_psf()
rml_psf = load_rml_psf()

# make the mnist datasets
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

# loop each of the systems
for noise in [0, 25, 50, 75, 100, 125, 150, 200]:
    print("testing rml_psf for gaussian at noise {}".format(noise))
    # test gaussian on simple rml model
    test_system(noise, 'rml', 'simple', seed_values, data, labels, training_fraction, testing_fraction, diffuser_psf, phlat_psf, psf_4, 'gaussian', rml_psf)
    # gaussian on cnn
    test_system(noise, 'rml', 'cnn', seed_values, data, labels, training_fraction, testing_fraction, diffuser_psf, phlat_psf, psf_4, 'gaussian', rml_psf)
for noise in [2, 5, 10, 25, 50, 60, 80, 100]:
    print("testing rml_psf for poisson at noise {}".format(noise))
    # poisson on simple
    test_system(noise, 'rml', 'simple', seed_values, data, labels, training_fraction, testing_fraction, diffuser_psf, phlat_psf, psf_4, 'poisson', rml_psf)
    # poisson on cnn
    test_system(noise, 'rml', 'cnn', seed_values, data, labels, training_fraction, testing_fraction, diffuser_psf, phlat_psf, psf_4, 'poisson', rml_psf)

