"""
encoding_information package

Information estimators
"""
from ._version import __version__, version_info

from .information_estimation import estimate_information
from .image_utils import extract_patches, add_noise

