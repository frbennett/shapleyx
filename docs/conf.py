# Configuration file for Sphinx documentation
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'ShapleyX'
copyright = '2025, Frederick Bennett'
author = 'Frederick Bennett'
release = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'
html_static_path = ['_static']