.. My Python Package documentation master file, created by
   Sphinx quickstart on Wed Apr  2 10:16:00 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to My Python Package's documentation!
=============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   readme
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. include:: ../README.md
   :parser: myst_parser.sphinx_

.. note::
   For the ``include`` directive above to work with Markdown, you might need to install ``myst-parser`` (`pip install myst-parser`) and add `'myst_parser'` to the `extensions` list in your `conf.py`. I haven't added it by default to keep dependencies minimal, but it's a common way to include READMEs. Alternatively, convert your README to reStructuredText.

API Reference
=============

.. automodule:: shapleyx
   :members: