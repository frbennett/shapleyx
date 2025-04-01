---
title: "API Reference"
layout: default
classes: wide
mathjax: true
---

# API Reference

## Main Class: `rshdmr`

The core class for performing sensitivity analysis using RS-HDMR.

### Initialization
```python
rshdmr(data_file, polys=[10, 5], n_jobs=-1, test_size=0.25, limit=2.0, 
       k_best=1, p_average=2, n_iter=300, verbose=False, method='ard',
       starting_iter=5, resampling=True, CI=95.0, number_of_resamples=1000,
       cv_tol=0.05)
```

**Parameters**:
- `data_file`: Input data (DataFrame or file path)
- `polys`: List of polynomial orders for expansion (default [10, 5])
- `n_jobs`: Number of parallel jobs (default -1 for all cores)
- `test_size`: Fraction of data for testing (default 0.25)
- `limit`: Threshold for coefficient selection (default 2.0)
- `k_best`: Number of best models to average (default 1)
- `p_average`: Power for model averaging (default 2)
- `n_iter`: Number of iterations (default 300)
- `verbose`: Verbosity flag (default False)
- `method`: Regression method ('ard', 'ard_cv', 'omp', 'lasso') (default 'ard')
- `starting_iter`: Starting iteration for cross-validation (default 5)
- `resampling`: Enable bootstrap resampling (default True)
- `CI`: Confidence interval percentage (default 95.0)
- `number_of_resamples`: Number of bootstrap samples (default 1000)
- `cv_tol`: Cross-validation tolerance (default 0.05)

### Key Methods

#### `run_all()`
Executes the complete analysis pipeline:
1. Data transformation
2. Legendre polynomial expansion
3. Regression analysis
4. Sensitivity index calculation
5. Resampling (if enabled)

**Returns**:
- `sobol_indices`: DataFrame of Sobol sensitivity indices
- `shapley_effects`: DataFrame of Shapley effects
- `total_index`: DataFrame of total sensitivity indices

#### `predict(X)`
Predicts outputs for new input data using the trained model.

**Parameters**:
- `X`: Input data (DataFrame or array-like)

**Returns**:
- Array of predicted values

#### `get_pawn(S=10)`
Computes PAWN sensitivity indices.

**Parameters**:
- `S`: Number of slides (default 10)

**Returns**:
- Dictionary of PAWN indices

#### `get_pawnx(num_unconditioned, num_conditioned, num_ks_samples, alpha=0.05)`
Advanced PAWN analysis with Kolmogorov-Smirnov test.

**Parameters**:
- `num_unconditioned`: Number of unconditioned samples
- `num_conditioned`: Number of conditioned samples
- `num_ks_samples`: Number of KS test samples
- `alpha`: Significance level (default 0.05)

**Returns**:
- DataFrame of PAWN indices with statistics

## Utility Modules

### `legendre`
Handles Legendre polynomial expansion.

Key functions:
- `legendre_expand(X_T, polys)`: Performs polynomial expansion
- `shift_legendre(n, x)`: Computes shifted Legendre polynomial

### `regression`
Regression analysis methods.

Key classes:
- `regression(X_T_L, Y, method, n_iter, verbose, cv_tol, starting_iter)`
  - `run_regression()`: Executes regression analysis

### `stats`
Statistical evaluation and visualization.

Key functions:
- `stats(Y, y_pred, coef_)`: Computes model statistics
- `plot_hdmr(Y, y_pred)`: Creates HDMR visualization plot

### `indicies`
Sensitivity index calculations.

Key functions:
- `eval_indices(X_T_L, Y, coef_, evs)`: Computes sensitivity indices
- `eval_shapley(columns)`: Calculates Shapley effects
- `eval_total_index(columns)`: Computes total sensitivity indices

### `pawn`
PAWN sensitivity analysis implementation.

Key classes:
- `pawnx(X, Y, ranges, non_zero_coefficients)`
  - `get_pawnx()`: Computes advanced PAWN indices
- `DeltaX(X, Y, ranges, non_zero_coefficients)`
  - `get_deltax()`: Computes delta indices

## Example Usage

```python
from shapleyx import rshdmr
import pandas as pd

# Load data
data = pd.read_csv('input_data.csv')

# Initialize analyzer
analyzer = rshdmr(data, polys=[10,5], method='ard_cv')

# Run analysis
sobol, shapley, total = analyzer.run_all()

# Make predictions
new_data = pd.DataFrame(...)
predictions = analyzer.predict(new_data)

# Get PAWN indices
pawn_results = analyzer.get_pawnx(1000, 500, 100)