# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Somehow this fixes the docs when built on readthedocs?
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


project = 'EncodingInformation'
copyright = '2024, Henry Pinkard'
author = 'Henry Pinkard'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import sphinx_rtd_theme


extensions = [
    'sphinx_rtd_theme',
    'sphinx_togglebutton',
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',  # for better type hint support
]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_theme_options = {
#     'collapse_navigation': True,
#     'navigation_depth': 6,
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
