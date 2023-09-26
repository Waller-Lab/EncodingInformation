import os
import gc
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

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

# set psf name
psf_name = 'uc'
seed_value = 1 # change if want 

seed_value = int(seed_value)
training, testing, validation = permute_data(data, labels, seed_value, training_fraction, testing_fraction)
x_train, y_train = training
x_test, y_test = testing
x_validation, y_validation = validation

random_test_data, random_test_labels = generate_random_tiled_data(x_test, y_test, seed_value)
random_train_data, random_train_labels = generate_random_tiled_data(x_train, y_train, seed_value)
random_valid_data, random_valid_labels = generate_random_tiled_data(x_validation, y_validation, seed_value)

if psf_name == 'uc':
    test_data = random_test_data[:, 14:-13, 14:-13]
    train_data = random_train_data[:, 14:-13, 14:-13]
    valid_data = random_valid_data[:, 14:-13, 14:-13]
if psf_name == 'psf_4':
    test_data = convolved_dataset(psf, random_test_data)
    train_data = convolved_dataset(psf, random_train_data)
    valid_data = convolved_dataset(psf, random_valid_data)
if psf_name == 'diffuser':
    test_data = convolved_dataset(diffuser_region, random_test_data)
    train_data = convolved_dataset(diffuser_region, random_train_data)
    valid_data = convolved_dataset(diffuser_region, random_valid_data)  
if psf_name == 'phlat':
    test_data = convolved_dataset(phlat_region, random_test_data)
    train_data = convolved_dataset(phlat_region, random_train_data)
    valid_data = convolved_dataset(phlat_region, random_valid_data)

