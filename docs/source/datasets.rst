.. currentmodule:: encoding_information.datasets
Datasets Documentation
=======================

This documentation focuses on key datasets and utility functions used in the project.

.. autoclass:: encoding_information.datasets.bsccm_utils.BSCCMDataset
   :members: __init__, get_shape, get_measurements
   :show-inheritance:

.. autoclass:: encoding_information.datasets.cfa_dataset.ColorFilterArrayDataset
   :members: __init__, get_measurements, get_shape
   :show-inheritance:

.. autoclass:: encoding_information.datasets.hml_dataset.HyperspectralMetalensDataset
   :members: __init__, get_measurements, get_shape, _center_crop
   :show-inheritance: 

.. autoclass:: encoding_information.datasets.mnist_dataset.MNISTDataset
   :members: __init__, get_measurements, get_shape
   :show-inheritance:
