.. currentmodule:: encoding_information.image_utils

Image Processing Functions
==========================

This module contains functions for processing images, including extracting patches, adding synthetic noise, and normalizing image stacks.

Functions
---------

.. autofunction:: _extract_random_patches
   :noindex:

.. autofunction:: extract_patches

.. autofunction:: add_noise

.. autofunction:: normalize_image_stack


Usage Examples
--------------

#### Extracting Patches

.. code-block:: python

   from image_processing import extract_patches
   import numpy as np

   data = np.random.rand(100, 64, 64)  # Example dataset of 100 images, each 64x64
   patches = extract_patches(data, num_patches=500, patch_size=16, strategy='random')

#### Adding Noise to Images

.. code-block:: python

   from image_processing import add_noise
   import numpy as np

   images = np.random.rand(100, 64, 64)  # Example dataset of 100 images, each 64x64
   noisy_images = add_noise(images, gaussian_sigma=0.1)

#### Normalizing an Image Stack

.. code-block:: python

   from image_processing import normalize_image_stack
   import numpy as np

   stack = np.random.rand(100, 64, 64)  # Example image stack
   normalized_stack = normalize_image_stack(stack)

