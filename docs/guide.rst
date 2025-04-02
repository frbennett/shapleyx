User Guide
==========

Getting Started
---------------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install shapleyx

Basic Usage
-----------

Importing the package
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import shapleyx

Example Analysis
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Sample code demonstrating basic functionality
   from shapleyx import ShapleyX
   
   # Initialize analyzer
   analyzer = ShapleyX()
   
   # Run analysis
   results = analyzer.analyze(data)

Advanced Features
----------------

Custom Configuration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Advanced configuration example
   analyzer = ShapleyX(
       method='rshdmr',
       parameters={'order': 2}
   )

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

- Problem: Import errors
  Solution: Verify installation and dependencies

- Problem: Analysis errors
  Solution: Check input data format