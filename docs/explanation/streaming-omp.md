# Streaming Basis Expansion

## The Memory Problem

RS-HDMR builds candidate basis functions by taking products of
shifted Legendre polynomials over all input-variable subsets up to
the expansion order.  For `polys=[p_1, p_2, \dots, p_m]` with $d$
inputs:

$$N_{\text{features}} = \sum_{k=1}^{m} \binom{d}{k} \cdot p_k^{\,k}$$

The full design matrix is $n \times N_{\text{features}}$.  This grows
quickly:

| $d$ | `polys` | Features | Memory at $n{=}10\,000$ |
|----:|---------|---------:|------------------------:|
| 10 | [8,6,4] | 9 380 | 0.75 GB |
| 15 | [8,6,4] | 33 020 | 2.6 GB |
| 20 | [8,6,4] | 79 960 | **6.4 GB** |
| 20 | [10,8,6,4] | 1 498 920 | **120 GB** |
| 25 | [8,6,4] | 158 200 | 12.7 GB |

For ARD regression the problem is even worse — it precomputes
$X^\top X$, an $(N_{\text{features}} \times N_{\text{features}})$
matrix (51 GB for the 20-dim 3rd-order case).

## What the Streaming Path Does Differently

The key insight is that **most of the pipeline never touches the
matrix values at all**.  Sobol indices, Shapley effects, bootstrap
resampling, surrogate prediction, and moment-free measures all
operate on *column labels* and *coefficient magnitudes* — not the
feature matrix.

The **only consumer** of the full design matrix is the regression
step.  For OMP, this access pattern is a sequential scan: at each
iteration, compute $X^\top r$ (the correlation of every feature with
the current residual).  One column at a time.

The streaming path replaces the full matrix with two compact
structures:

1. **Primitive Legendre terms** — $(n, d \cdot \max(\text{polys}))$
   array.  These are the "atoms" that all basis functions are built
   from.  For $d{=}20$, `polys=[8,6,4]`, this is 160 columns × 10K
   rows = **12.8 MB** — small enough to live in L3 cache.

2. **Feature recipes** — compact integer descriptors.
   For a feature like $\phi_i(\mathbf{x}) = L_3(x_0) \cdot L_2(x_1)$,
   the recipe stores `[prim_idx(x0,3), prim_idx(x1,2)]`.  At 3 integers
   per feature, 80K features cost **~1 MB**.

At regression time, columns are computed on the fly from primitives:

$$\phi_i(\mathbf{x}_j) = \prod_{f=1}^{k_i} p_{\,j,\;\text{idx}(i,f)}$$

where $p$ is the primitives array and $\text{idx}(i,f)$ is the
flat index of the $f$-th factor of feature $i$.

## Why Streaming OMP Can Be Faster

This is counter-intuitive — recomputing columns on the fly should be
slower than reading them from memory, right?  It depends on cache:

| Path | Data size | Location | Bottleneck |
|------|-----------|----------|------------|
| Dense OMP | 6.4 GB | RAM (50 GB/s) | **Memory bandwidth** |
| Streaming OMP | 12.8 MB | L3 cache (500 GB/s) | **CPU compute** |

For the dense path, each correlation scan reads 6.4 GB from RAM —
0.13 seconds of data transfer for only 0.08 GFLOP of compute.
The CPU sits idle waiting for RAM.

For the streaming path, the primitives stay in L3 cache where each
scan performs 2.4 GFLOP at 3.4 GFLOPS (measured).  The CPU stays
busy — a compute-bound workload.

**Result:** streaming OMP is 4–5× *faster* wall-clock than dense OMP
for problems where the dense matrix exceeds L3 cache size, AND it uses
70–2000× less memory.

## When to Use Which Method

| Scenario | Recommendation |
|----------|---------------|
| $d \le 10$, moderate `polys` | `method='ard_cv'` — ARD gives best feature selection, memory is manageable |
| $d \le 10$, quick exploration | `method='omp'` — fastest, no CV overhead |
| $d \ge 12$, any `polys` | `method='omp_stream'` — memory-efficient, faster than dense |
| High-dimensional, need CV | `method='omp_cv_stream'` — streaming with CV, parallel folds |
| A/B comparison: dense vs streaming | Run both on the same small problem — they produce equivalent results |

## Parallel CV

When using `method='omp_cv_stream'` with `n_jobs > 1`, CV folds are
evaluated in parallel via `joblib`.  Each fold computes the full OMP
sparsity path $(1 \dots \text{max\_iter})$ independently, and scores
are aggregated across folds.

The speedup depends on per-fold work.  Each fold runs
$\frac{k(k+1)}{2}$ correlation scans for $k = \text{max\_iter}$.
At 137 ms/scan (typical), a 10-level CV path does ~7.5s of work per
fold.  4 workers provide ~1.4× speedup; for larger problems (more
samples, more iterations), the speedup approaches `n_jobs`.

**Linux note:** The primitive array is shared read-only across
forked workers — zero copy overhead.

## Correctness Guarantee

Streaming OMP implements the same algorithm as sklearn's
`OrthogonalMatchingPursuit`.  On any problem small enough to fit
both ways, the two paths produce numerically equivalent results
(R² within 0.02%, Sobol indices within 5% relative tolerance,
Shapley effects within 5% relative tolerance).

The tiny remaining differences come from:

- The streaming path does not centre the *columns* of $X$ (only
  $y$), while sklearn centres both.  Since the Legendre basis
  columns have mean ≈ 0 and are orthogonal in expectation, this
  difference is negligible for sensitivity analysis.
- Floating-point accumulation order differs between the
  pre-multiplied dense columns and the on-the-fly products.

## Implementation Architecture

```
legendre_expand()
    │
    ├─ method in ('omp','omp_cv','ard',…) 
    │   └─ build_basis_set()     → full (n × p) DataFrame
    │      run_regression()      → sklearn/ARD on dense matrix
    │
    └─ method in ('omp_stream','omp_cv_stream')
        └─ build_basis_set_streaming()  → LazyBasisMatrix (primitives + recipes)
           run_regression(lazy_basis=…) → StreamingOMP/StreamingOMPCV

Downstream (unchanged):
    eval_indices()       → uses X_T_L.columns + coef_
    get_pruned_data()    → builds active columns on demand (streaming)
                          or slices from DataFrame (dense)
    resampling           → uses pruned_data
    predictor            → already builds design on-the-fly
    PAWN / Delta / H     → calls self.predict()
```

Only three methods in `rshdmr` branch on `self._streaming`:
`legendre_expand()`, `run_regression()`, and `get_pruned_data()`.
Everything else — Sobol, Shapley, bootstrap, prediction, moment-free
measures — is unchanged.
