# Streaming OMP Reference

The `shapleyx.utilities.streaming` module provides a **memory-efficient
regression pipeline** for RS-HDMR basis expansions that are too large to
materialise in RAM.  Instead of building the full $(n \times p)$ design
matrix, the pipeline stores only the *primitive* Legendre terms
($n \times d\cdot\max(\text{polys})$) plus compact integer *recipes*
describing how to combine them into each basis function.  Columns are
computed on demand via Numba-compiled kernels.

**New `method` strings that activate the streaming path:**

| Method string | Description |
|---------------|-------------|
| `'omp_stream'` | Streaming Orthogonal Matching Pursuit — computes basis columns on the fly, never builds the dense matrix. |
| `'omp_cv_stream'` | Streaming OMP with K-fold cross-validation and retrospective sparsity selection. Supports `n_jobs` parallelism per fold. |

These are drop-in replacements for `'omp'` and `'omp_cv'` — the
downstream API (`run_all()`, `predict()`, `get_pawnx()`, etc.) is
identical.

---

## Core Classes

### `FeatureRecipes`

```python
class FeatureRecipes:
    feature_names: list[str]     # E.g. "X0_3*X1_2"
    n_features: int              # Number of candidate basis terms
    prim_indices: np.ndarray     # (n_features, max_factors) int32
    n_factors: np.ndarray        # (n_features,) int32
    max_factors: int
```

Compact integer representation of every candidate basis function.
For a term like $\phi_i(\mathbf{x}) = L_3(x_0) \cdot L_2(x_1)$,
`prim_indices[i]` stores the flat indices into the primitives array
so that the hot inner loop is `prod *= primitives[j, idx]`.

---

### `LazyBasisMatrix`

```python
class LazyBasisMatrix:
    primitives: np.ndarray       # (n_samples, n_prim) float64
    recipes: FeatureRecipes
    columns: list[str]           # column labels
    shape: tuple[int, int]       # (n_samples, n_features)
    n_features: int
    n_samples: int
```

A Legendre basis matrix that **never materialises** the full design
matrix.  The `primitives` array contains pre-computed shifted Legendre
values for every `(variable, degree)` pair:

$$\text{prim\_idx}(v, d) = v \cdot \max(\text{polys}) + (d - 1)$$

**Key methods:**

- `dot_with_residual(residual)` — Compute $X^\top r$ (the correlation of every feature with the residual vector).  This is the OMP hot path.  For 16K features and 2K samples, a single scan takes ~40 ms (Numba, parallelised via `prange`).

`active_submatrix(indices)` — Build the design submatrix for a small set of features (used by the OMP least-squares step).

`column(idx)` / `column_batch(indices)` — Compute individual columns on demand.

---

### `StreamingOMP`

```python
class StreamingOMP:
    lazy_basis: LazyBasisMatrix
    n_nonzero_coefs: int          # max features to select
    fit_intercept: bool            # default True (matches sklearn OMP)
    tol: float                     # early stopping tolerance

    coef_: np.ndarray              # (n_features,) sparse coefficients
    intercept_: float              # fitted intercept
    active_: np.ndarray            # indices of selected features
    n_nonzero_coefs_: int          # actual number selected
    n_iter_: int                   # iterations performed
```

OMP that operates on a `LazyBasisMatrix`.  At each iteration:

1. Scan all features via `dot_with_residual` to find the one most
   correlated with the residual.
2. Solve the least-squares problem on the (small) active submatrix.
3. Update the residual.

When `fit_intercept=True` (default), the algorithm centres $y$ for the
correlation scan and solves the augmented problem
$[\,\mathbf{1} \mid X_{\text{active}}\,]\,\mathbf{w} \approx \mathbf{y}$,
matching sklearn's `OrthogonalMatchingPursuit`.

**Methods:** `fit(y)`, `predict(lazy_basis)`

---

### `StreamingOMPCV`

```python
class StreamingOMPCV:
    lazy_basis: LazyBasisMatrix
    cv: int                        # number of CV folds (default 10)
    max_iter: int                  # max non-zero coefs to consider
    fit_intercept: bool
    scoring: str                   # 'r2' or 'mse'
    random_state: int
    n_jobs: int                    # parallel fold evaluation (default 1)

    coef_: np.ndarray
    intercept_: float
    active_: np.ndarray
    n_nonzero_coefs_: int          # optimal sparsity level
    best_cv_score_: float
    cv_scores_: list               # [(n_nz, mean_score, std_score), ...]
```

Cross-validated OMP with retrospective sparsity selection.  Evaluates
the full sparsity path (1 … `max_iter`) via K-fold CV and selects the
level with the best mean score.

**Parallel CV:** When `n_jobs > 1` and `joblib` is installed, folds are
evaluated in parallel.  Each fold computes the full sparsity path
independently; scores are then aggregated.  On Linux, the shared
primitives array is accessed read-only across forked workers
(zero-copy).  The warm-up JIT compilation is shared via Numba's
disk cache.

**Methods:** `fit(y)`, `predict(lazy_basis)`

---

## Memory Scaling

For $d$ inputs, `polys=[8,6,2]`, and $n$ samples:

| Component | Formula | Size at $d{=}20$, $n{=}10\,000$ |
|-----------|---------|-------------------------------|
| Primitives | $d \cdot 8 \cdot n \cdot 8$ B | **12.8 MB** |
| Feature recipes | $\sum \binom{d}{k} \cdot p_k \cdot k \cdot 4$ B | **~1 MB** |
| Full design matrix (dense) | $\sum \binom{d}{k} \cdot p^k \cdot n \cdot 8$ B | **~1.5 GB** |
| Active submatrix (sparse) | $\approx 100 \cdot n \cdot 8$ B | **~8 MB** |

The streaming path uses **~22 MB** instead of **~1.5 GB** — a 70×
reduction.  The gap widens with higher-dimensional expansions.

---

## Benchmarks

Sobol G-function, $d{=}20$, `polys=[8,6,2]`, $n{=}2\,000$, 80 non-zero
coefficients.  Numba 0.65.1, Intel i7 (8 cores):

| Method | Time | Memory | R² |
|--------|------|--------|----|
| Dense OMP (`method='omp'`) | 49.5 s | 258 MB | 0.9803 |
| Streaming OMP (`method='omp_stream'`) | **4.5 s** | 3 MB | 0.9802 |
| Streaming OMP-CV 10-fold, seq | 51.0 s | 3 MB | 0.8271 (CV) |
| Streaming OMP-CV 10-fold, 4 jobs | **35.3 s** | 3 MB | 0.8271 (CV) |

Streaming OMP is **4.4× faster** than dense OMP — the primitives fit in
L3 cache while the dense matrix is RAM-bandwidth-bound.  CV
parallelisation adds 1.4× on top (or more for larger $n$).

---

## How to Use

```python
from shapleyx import rshdmr

# Streaming OMP — drop-in for method='omp'
model = rshdmr(df, polys=[8, 6, 2], method='omp_stream', n_iter=80)
sobol, shapley, total = model.run_all()

# Streaming OMP with CV and 4-fold-parallel evaluation
model = rshdmr(df, polys=[8, 6, 2], method='omp_cv_stream',
               n_iter=100, n_jobs=4)
sobol, shapley, total = model.run_all()
```
