[![doc](https://readthedocs.org/projects/encodinginformation/badge/?version=latest)(https://encodinginformation.readthedocs.io/en/latest/)]
[![License](https://img.shields.io/pypi/l/encoding_information.svg)](https://github.com/EncodingInformation/EncodingInformation/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/encoding_information.svg)](https://pypi.org/project/encoding-information)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/encoding_information.svg)](https://pypistats.org/packages/encoding_information)


Code and experiments from the paper [Information-driven design of imaging systems](https://waller-lab.github.io/EncodingInformationWebsite/). 

For detailed usage, see the [documentation](https://encodinginformation.readthedocs.io/en/latest/).

## Installation guide

`pip install encoding_information`

There may be more setup required to get the correct versions of Jax/Flax, see:

https://github.com/Waller-Lab/EncodingInformation/blob/main/Installation_guide.md


## Quick Start

```python
from encoding_information.models import PixelCNN, PoissonNoiseModel
from encoding_information import estimate_information, extract_patches

# Load measurement data (N x H x W numpy array of images) 
measurements = load_measurements()  

# Split into training/test sets and extract patches
# Breaking large images into patches increases computational efficiency
# Test set is used to evaluate information estimates
patches = extract_patches(measurements[:-200], patch_size=16)
test_patches = extract_patches(measurements[-200:], patch_size=16) 

# Initialize and fit model to training data
model = PixelCNN()  # Also supports FullGaussianProcess, StationaryGaussianProcess
noise_model = PoissonNoiseModel()
model.fit(patches)

# Estimate information content with confidence bounds
# Error bars are calculated based on test set size
info, lower_bound, upper_bound = estimate_information(
   model, 
   noise_model,
   patches,
   test_patches,
   confidence_interval=0.95
)

print(f"Information: {info:.2f} ± {(upper_bound-lower_bound)/2:.2f} bits/pixel")
```

We provide three models with different tradeoffs:

- **PixelCNN**: Most accurate estimates but slowest
- **FullGaussianProcess**: Fastest
- **StationaryGaussianProcess**: Intermediate speed; Best performance on limited data

For highest accuracy, train multiple models and select the one giving the lowest information estimate, as each provides an upper bound on the true information content.
   

## Documentation

https://encodinginformation.readthedocs.io/en/latest/


## Contributing

1. Make a fork and clone the fork
2. `git remote add upstream https://github.com/Waller-Lab/EncodingInformation.git`
3. `git config pull.rebase false`
4. Use pull requests to contribute
5. `git pull upstream main` to pull latest updates from the main repo


