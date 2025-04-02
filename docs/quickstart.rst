Quick Start Guide
================

This guide provides a quick introduction to using shapleyx for global sensitivity analysis.

Basic Usage
-----------

1. Import the package:

.. code-block:: python

    from shapleyx import rshdmr

2. Initialize the analyzer with your data:

.. code-block:: python

    analyzer = rshdmr(data_file='input_data.csv', polys=[10, 5], method='ard')

3. Run the complete analysis pipeline:

.. code-block:: python

    sobol_indices, shapley_effects, total_index = analyzer.run_all()

4. View the results:

.. code-block:: python

    print("Sobol Indices:")
    print(sobol_indices)
    
    print("\nShapley Effects:")
    print(shapley_effects)
    
    print("\nTotal Index:")
    print(total_index)

5. Make predictions with new data:

.. code-block:: python

    predictions = analyzer.predict(new_data)

Key Parameters
--------------

- ``data_file``: Path to CSV file or pandas DataFrame containing input data
- ``polys``: List of polynomial orders for Legendre expansion
- ``method``: Regression method ('ard', 'omp', etc.)
- ``n_iter``: Number of iterations for regression
- ``resampling``: Whether to perform bootstrap resampling
- ``CI``: Confidence interval percentage for resampling

For more detailed usage, see the :doc:`user_guides/index`.