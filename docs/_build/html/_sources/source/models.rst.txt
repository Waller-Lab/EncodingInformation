Models Documentation
=====================

This documentation provides an overview of key models, focusing on essential input parameters and high-level functionality.

.. Modules
.. -------

.. - Conditional Entropy Models
.. - Gaussian Process Model
.. - Base Model Class
.. - Pixel CNN Model

.. ---

Conditional Entropy Models
---------------------------

.. autoclass:: encoding_information.models.conditional_entropy_models.AnalyticGaussianNoiseModel
   :members: __init__, estimate_conditional_entropy
   :show-inheritance:

.. autoclass:: encoding_information.models.conditional_entropy_models.PoissonNoiseModel
   :members: estimate_conditional_entropy
   :show-inheritance:


Gaussian Process Models
-----------------------

.. autoclass:: encoding_information.models.gaussian_process.StationaryGaussianProcess
   :members: __init__, fit, compute_negative_log_likelihood, generate_samples
   :show-inheritance:

.. autoclass:: encoding_information.models.gaussian_process.FullGaussianProcess
   :members: __init__, compute_negative_log_likelihood, generate_samples
   :show-inheritance:


.. Base Model Class
.. ----------------

.. .. autoclass:: encoding_information.models.model_base_class.MeasurementModel
..    :members: __init__, fit, compute_negative_log_likelihood, generate_samples
..    :show-inheritance:

.. .. autoclass:: encoding_information.models.model_base_class.MeasurementNoiseModel
..    :members: estimate_conditional_entropy
..    :show-inheritance:

Pixel CNN Model
----------------

.. autoclass:: encoding_information.models.pixel_cnn.PixelCNN
   :members: __init__, fit, compute_negative_log_likelihood, generate_samples
   :show-inheritance:
