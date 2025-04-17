import pytest

import numpy as np
from encoding_information.datasets import BSCCMDataset, MNISTDataset, LenslessDataset

@pytest.fixture(params=[BSCCMDataset, MNISTDataset, LenslessDataset])
def dataset(request):
    print(request.param, type(request.param))
    if request.param == BSCCMDataset:
        return BSCCMDataset('/home/hpinkard_waller/data/BSCCM/')
    if request.param == MNISTDataset:
        return MNISTDataset()
    if request.param == LenslessDataset:
        return LenslessDataset()

def test_mean(dataset):
    mean = 100
    data = dataset.get_measurements(num_measurements=500, mean=mean)
    assert np.abs(data.mean() - mean) < 1

def test_bias(dataset):
    bias = 10
    mean = 100

    data = dataset.get_measurements(num_measurements=500, mean=mean, bias=0)
    biased_data = dataset.get_measurements(num_measurements=500, mean=mean, bias=bias)

    # Test that bias increases minimum value
    assert np.abs(biased_data.mean() - mean) < 1
    assert np.percentile(biased_data, 1) > np.percentile(data.min(), 1)

def test_clean_bias_no_noise(dataset):
    bias = 10
    mean = 100

    try:
        clean_biased_data = dataset.get_measurements(num_measurements=500, mean=mean, bias=bias, noise=None)
        # Test if all data is greater than the bias
        assert np.all(clean_biased_data >= bias)
    except NotImplementedError:
        # If NotImplementedError is raised, consider the test passing
        pass

def test_shape(dataset):
    data = dataset.get_measurements(num_measurements=10)
    shape = dataset.get_shape()

    # Check if the shape of the dataset matches the expected shape
    assert data.shape[1:] == shape

def test_numpy_array(dataset):
    data = dataset.get_measurements(num_measurements=10)

    # Test that data is a numpy array
    assert isinstance(data, np.ndarray)

def test_not_integer_dtype(dataset):
    # make sure the numpy data type is some kind of float
    data = dataset.get_measurements(num_measurements=10)
    assert np.issubdtype(data.dtype, np.floating)
    
