# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..')) # Add project root to path

import shapleyx # Import the package to get version info


# -- Project information -----------------------------------------------------

project = 'My Python Package'
copyright = '2025, Your Name' # Placeholder - User should update
author = 'Your Name' # Placeholder - User should update

# The full version, including alpha/beta/rc tags
release = shapleyx.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', # Include documentation from docstrings
    'sphinx.ext.napoleon', # Support NumPy and Google style docstrings
    'sphinx.ext.viewcode', # Add links to source code
    'sphinx.ext.intersphinx', # Link to other projects' documentation
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Intersphinx mapping
intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
#
html_theme = 'furo' # Use the Furo theme

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']