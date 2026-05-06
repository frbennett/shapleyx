# Quickstart Guide

This guide walks through a basic sensitivity analysis using ShapleyX.

## Loading Data

Prepare your data in CSV format with columns for input parameters and
one column for output (named `'Y'`):

```python
import pandas as pd
data = pd.read_csv('input_data.csv')
```

## Running Analysis

```python
from shapleyx import rshdmr

# Initialise the analyser
analyzer = rshdmr(
    data,                       # DataFrame or path to CSV
    polys=[10, 5],              # up to 10th-degree univariate, 5th-degree bivariate
    n_iter=300,                 # ARD iterations
    method='ard_cv',            # ARD with Bayesian cross-validation
    cv_method='bayesian',
    cv_tol=0.01,
    resampling=True,            # bootstrap confidence intervals
    number_of_resamples=500,
)

# Run the complete pipeline
sobol, shapley, total = analyzer.run_all()
```

## Viewing Results

```python
# Shapley effects (scaled to sum to 1)
print(shapley[['label', 'scaled effect', 'lower', 'upper']])

# Sobol indices to arbitrary order
print(sobol[['derived_labels', 'index', 'lower', 'upper']])

# Total sensitivity indices
print(total)
```

## Plotting

```python
# Predicted vs actual
analyzer.run_plot_hdmr()
```

## Monte Carlo Shapley for Correlated Inputs

```python
# With a correlation matrix (uses Gaussian copula)
import numpy as np
corr = np.array([
    [1.0, 0.0, 0.8],
    [0.0, 1.0, 0.0],
    [0.8, 0.0, 1.0],
])
mc = analyzer.get_mc_shapley(corr=corr, N=5000, method='exhaustive', B=500)

# With a custom distribution
from shapleyx.utilities.mc_shapley import MultivariateNormal
joint = MultivariateNormal(mean=[0, 0, 0], cov=[[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])
mc = analyzer.get_mc_shapley(joint=joint, N=5000, B=500)

# Coalition truncation (auto-detected from polys)
mc = analyzer.get_mc_shapley(N=5000, method='exhaustive')
# df now includes sobol_first, sobol_total alongside Shapley effects
```

## Moment-Free Measures

```python
pawn  = analyzer.get_pawnx(1000, 500, 100)   # PAWN (density-based)
delta = analyzer.get_deltax(1000, 500)       # Delta (moment-independent)
h_idx = analyzer.get_hx(1000, 500)           # H-index (distribution-based)
```

## Next Steps

- [Tutorials](../tutorials/basic-usage.md) — detailed walkthroughs
- [MC Shapley How-to](../how-to-guides/mc-shapley.md) — correlated inputs guide
- [Example Notebooks](https://github.com/frbennett/shapleyx/tree/main/Examples) — full case studies
