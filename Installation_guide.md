# Mutual Information Estimation Installation Guide

The minimal installation requires jax and tensorflow. Installing jax after other deep learning packages seems to work best due to difficulties with CUDA and CuDNN installations. In recent installs (jax >= 0.5), installing jax first has seemed to work fine.

Installation varies according to system specifications. This is an incomplete guide, providing directions for our system, running Ubuntu 20.04.6 x86_64, CUDA 12.3, CUDNN >=8.9.

# Instructions for 2025 
## Python 3.11 instructions 
This is a simple installation that uses pip. The same process can be used for python 3.10, both versions of python work fine. 

1. Use conda for environment management 

`conda create -n infotheory2025 python=3.11`

2. Activate your environment

`conda activate infotheory2025`

3. Install jax using the pip install command corresponding to your CUDA version.

`pip install "jax[cuda12]"`

4. Install tensorflow with the wheel corresponding to your system configuration from https://www.tensorflow.org/install/pip#package_location. For example, for python 3.11 with GPU support: 

`pip install https://storage.googleapis.com/tensorflow/versions/2.18.0/tensorflow-2.18.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`

5. The other packages that are necessary are `tqdm` and `scikit-image`. `cleanplots` and `gpustat` are helpful. All of these packages can be pip installed. 

`pip install tqdm` 

`pip install scikit-image`

6. For design with IDEAL, the equinox package and Weights and Biases are necessary. These do not interfere with the base installation from step 1-5. 

`pip install equinox`

`pip install wandb` 

This installation runs smoothly but does include warnings with cuDNN for tensorflow. These do not affect performance. Specifically:

`E0000 00:00:1741215469.150672 3893437 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered`




# Old instructions from 2024 
## Python 3.10 instructions

1. Use conda for environment management

`conda create -n infotheory python=3.10`

2. Install pytorch if needed, including CUDA version (here, 12.1). Conda installation works better than pip.

`conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia`

3. Install tensorflow, this seems to work more reliably than just `pip install tensorflow` (python 3.10 version).

`pip install https://storage.googleapis.com/tensorflow/versions/2.16.1/tensorflow-2.16.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`


4. Install jax. Manually installing jaxlib and then jax used to work but does not anymore. This installed jax 0.4.29.

`pip install -U "jax[cuda12]"`

This installs a fairly new version of jax, which doesn't support the outdated ml-dtypes version (0.3.2) that tensorflow needs, instead using the newest ml-dtypes version (0.4.0). Tensorflow runs with "Out of Range" warnings, but seems to work fine within the estimation framework. Downgrading ml-dtypes to work with tensorflow is incompatible with jax. In the future, tensorflow dependencies will be removed. It may be possible to downgrade the jax version but attempts to do so resulted in failed installs with no CuDNN found.

5. Install flax

`pip install flax`


## Python 3.11 instructions
Same as python 3.10, seems to work fine but hasn't been rigorously tested. Instead of the tensorflow command used above, find the one for the relevant python version from https://www.tensorflow.org/install/pip. 

`pip install https://storage.googleapis.com/tensorflow/versions/2.16.1/tensorflow-2.16.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`

