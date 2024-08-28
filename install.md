# Mutual Information Estimation Installation Guide

The minimal installation requires jax and tensorflow. Installing jax after other deep learning packages seems to work best due to difficulties with CUDA and CuDNN installations.

Installation varies according to system specifications. This is an incomplete guide, providing directions for our system, running Ubuntu 20.04.6 x86_64, CUDA 12.3, CUDNN >=8.9.


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



## Contributing

1. Make a fork and clone the fork
2. `git remote add upstream https://github.com/Waller-Lab/EncodingInformation.git`
3. `git config pull.rebase false`
4. Use pull requests to contribute
5. `git pull upstream main` to pull latest updates from the main repo


