Examples
=======

This section provides practical examples of using shapleyx for sensitivity analysis.

Basic Example
-------------

.. code-block:: python

    from shapleyx import rshdmr
    
    # Initialize analyzer
    analyzer = rshdmr(data_file='input_data.csv', 
                     polys=[10, 5],
                     method='ard',
                     resampling=True,
                     CI=95.0)
    
    # Run complete analysis
    sobol, shapley, total = analyzer.run_all()
    
    # Print results
    print("Sobol Indices:")
    print(sobol)
    print("\nShapley Effects:")
    print(shapley)

Advanced Example
---------------

Using custom polynomial orders and parallel processing:

.. code-block:: python

    analyzer = rshdmr(data_file='complex_data.csv',
                     polys=[15, 10, 5],  # Higher order polynomials
                     method='omp',
                     n_jobs=-1,  # Use all available cores
                     n_iter=500)  # More iterations
    
    results = analyzer.run_all()

PAWN Analysis Example
--------------------

.. code-block:: python

    # Get PAWN indices
    pawn_results = analyzer.get_pawn(S=20)
    
    # Get PAWNx indices with more control
    pawnx_results = analyzer.get_pawnx(
        num_unconditioned=1000,
        num_conditioned=100,
        num_ks_samples=50,
        alpha=0.01
    )

For more complete examples, see the Jupyter notebooks in the ``Examples/`` directory of the project.