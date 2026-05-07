# Common Tasks

## Handling Large Datasets

```python
# Process data in chunks
analyzer = rshdmr(
    data_file='large_data.csv',
    chunksize=10000,  # Process 10,000 rows at a time
    polys=[5, 3],     # Lower polynomial orders for large datasets
    method='ard'
)
```

## Customizing Polynomial Orders

```python
# Set different polynomial orders for each input
analyzer = rshdmr(
    data_file='data.csv',
    polys=[10, 5, 8, 3, 6],  # Specific orders for each parameter
    method='ard'
)
```

## Saving and Loading Results

```python
# Save results to file
import pickle
with open('sensitivity_results.pkl', 'wb') as f:
    pickle.dump({
        'sobol': sobol_indices,
        'shapley': shapley_effects,
        'total': total_index
    }, f)

# Load results
with open('sensitivity_results.pkl', 'rb') as f:
    results = pickle.load(f)
```

## Comparing Different Methods

```python
# Compare ARD and OMP methods
analyzer_ard = rshdmr(data_file='data.csv', method='ard')
analyzer_omp = rshdmr(data_file='data.csv', method='omp')

ard_results = analyzer_ard.run_all()
omp_results = analyzer_omp.run_all()
```

## Streaming OMP for High-Dimensional Problems

For problems with many input variables ($d \\ge 12$) or high-degree
expansions, the standard OMP path materialises a dense design matrix
that can exceed available RAM.  The streaming OMP path avoids this by
computing basis columns on the fly from compact *primitive* Legendre
terms, using **50–100× less memory** while running **4–10× faster**
(when Numba is installed).

### Basic Streaming OMP

```python
analyzer = rshdmr(
    data_file='data.csv',
    polys=[8, 6, 2],            # 3rd-order expansion
    method='omp_stream',        # streaming OMP
    n_iter=80,                  # max coefficients
    verbose=True,               # show per-iteration R² progress
)
sobol, shapley, total = analyzer.run_all()
```

### Streaming OMP with Cross-Validation

```python
analyzer = rshdmr(
    data_file='data.csv',
    polys=[8, 6, 2],
    method='omp_cv_stream',     # streaming OMP + CV
    n_iter=100,                 # max sparsity to consider
    n_jobs=4,                   # parallel CV fold evaluation
    verbose=True,               # show CV scores per sparsity level
)
sobol, shapley, total = analyzer.run_all()
```

### When to Use Streaming

| Scenario | Recommendation |
|----------|---------------|
| $d \\le 10$, moderate `polys` | `method='ard_cv'` — ARD gives best feature selection |
| $d \\le 10$, quick exploration | `method='omp'` — fastest, no CV overhead |
| $d \\ge 12$, any `polys` | `method='omp_stream'` — memory-efficient |
| High-dimensional, need CV | `method='omp_cv_stream'` with `n_jobs=4` |
| A/B comparison | Run both dense and streaming on a small problem — they agree |

### Requirements

- **Numba** (recommended): `pip install numba` — provides 4–10× speedup
  via JIT-compiled correlation scans.  Without Numba the streaming path
  still works but is slower than dense OMP.
- **joblib** (optional): `pip install joblib` — required for parallel CV
  fold evaluation (`n_jobs > 1`).

See the [Streaming Basis Expansion](../explanation/streaming-omp.md)
explanation and [Streaming OMP](../reference/streaming-omp.md) API
reference for benchmarks and architecture details.

## Computing Shapley Effects for Correlated Inputs

When inputs are correlated, use the MC Shapley method instead of the default coefficient-based approach:

```python
import numpy as np
from shapleyx.utilities.mc_shapley import GaussianCopulaUniform

# Define correlation structure
corr = np.array([
    [1.0, 0.5, 0.0],
    [0.5, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])

# Using the surrogate model with a correlation matrix
mc_results = analyzer.get_mc_shapley(corr=corr, N=5000, B=500)
print(mc_results[['variable', 'effect', 'lower', 'upper']])

# Using a user-defined function with a custom distribution
joint = GaussianCopulaUniform(
    lows=[0.0, -1.0, 0.5],
    highs=[1.0, 1.0, 2.0],
    corr=corr,
)

def my_model(x):
    return np.sin(x[0]) + 7 * np.sin(x[1])**2 + 0.1 * x[2]**4 * np.sin(x[0])

mc_user = analyzer.get_mc_shapley(joint=joint, f=my_model, N=5000, B=500)
```

See the [MC Shapley guide](mc-shapley.md) for full details.

## Computing Additional Sensitivity Indices

The trained surrogate supports moment-free sensitivity methods:

```python
# PAWN (distribution-based) sensitivity
pawn_results = analyzer.get_pawn(S=10)

# PAWN with surrogate model
pawnx_results = analyzer.get_pawnx(1000, 500, 100, alpha=0.05)

# Delta moment-free indices
delta_indices = analyzer.get_deltax(1000, 500)

# H-index (KL divergence based)
h_indices = analyzer.get_hx(1000, 500)

# Owen-Shapley interaction values
interactions = analyzer.get_interactions(order=1)
```

## Troubleshooting Common Issues

### Memory Errors
- Reduce polynomial orders
- Use smaller chunksize
- Filter less important parameters

### Convergence Issues
- Check data quality
- Try different polynomial orders
- Normalize input data