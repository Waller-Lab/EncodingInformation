.. currentmodule:: encoding_information.information_estimation

Information Estimation Functions
================================

This module contains functions for estimating entropy and mutual information between variables using various probabilistic models.

Usage Examples
--------------

#### Estimating Mutual Information

.. code-block:: python

   from information_estimation import estimate_information
   from models import PixelCNN, GaussianNoiseModel
   import numpy as np

   measurement_model = PixelCNN(...)
   noise_model = GaussianNoiseModel(...)
   train_set = np.random.randn(100, 32, 32)  # Example training data
   test_set = np.random.randn(100, 32, 32)  # Example test data

   mutual_info = estimate_information(measurement_model, noise_model, train_set, test_set)

#### Running Bootstrapped Estimations

.. code-block:: python

   from information_estimation import run_bootstrap
   import numpy as np

   data = np.random.randn(100, 32, 32)
   def estimation_fn(data_sample):
       return np.mean(data_sample)

   median, conf_int = run_bootstrap(data, estimation_fn, num_bootstrap_samples=200, confidence_interval=90)


Functions
---------

.. autofunction:: estimate_information

.. autofunction:: analytic_multivariate_gaussian_entropy

.. autofunction:: nearest_neighbors_entropy_estimate

.. autofunction:: estimate_conditional_entropy

.. autofunction:: run_bootstrap

.. autofunction:: estimate_task_specific_mutual_information

.. autofunction:: estimate_mutual_information

