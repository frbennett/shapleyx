# Performance Optimisations

ShapleyX v0.3 includes a suite of performance optimisations that
collectively accelerate Monte Carlo Shapley effect estimation by
**10–30×** when using an RS-HDMR surrogate model.  This document
describes each optimisation, the motivation behind it, and the
measured impact.

---

## Overview of the Hot Path

The MC Shapley algorithm spends virtually all of its time evaluating
the model function $f(\mathbf{x})$ inside nested Monte Carlo loops.
For $d$ input dimensions and $N$ samples per subset, each run of the
**exhaustive** method requires

$$
(2^d - 2) \times 2N + N
$$

function evaluations.  With $d=8$ and $N=3000$ this is
$\approx 1.5$&nbsp;million calls to `predict()`.

Every `predict()` call on the surrogate model must:

1. Normalise the input to $[0,1]^d$
2. Evaluate shifted Legendre polynomials for each active basis term
3. Compute the dot product of the design vector with the RVM
   coefficients

The optimisations target steps&nbsp;2 and&nbsp;3, plus the
surrounding Monte Carlo infrastructure.

---

## 1. Surrogate Model Prediction (predictor.py)

### 1.1 Bypass `sklearn` predict overhead

**Change:** Replaced `self.ridgereg.predict(design)` with the direct
computation `design @ self._coef + self._intercept`.

**Why:** `sklearn.linear_model.Ridge.predict()` validates and
type-checks its input on every call.  The actual computation is a
single dot product.  Skipping the validation eliminates Python-level
dispatch overhead for each of the millions of predictions made during
MC Shapley.

**Impact:** ~1.1&times; speedup on the `predict` path.

### 1.2 Numba-compiled Legendre polynomials

**Change:** Replaced `scipy.special.eval_sh_legendre` with a
recurrence-based shifted Legendre evaluation compiled with
`@njit(cache=True)`.

**Why:** `eval_sh_legendre` is a general-purpose Fortran routine
reached through SciPy's Python wrapper.  For the low-degree
polynomials used in RS-HDMR (typically $n \le 10$), the recurrence

$$
\begin{aligned}
\hat{P}_0(x) &= 1, \\
\hat{P}_1(x) &= 2x - 1, \\
\hat{P}_{k+1}(x) &= \frac{(2k+1)(2x-1)\hat{P}_k(x) - k\hat{P}_{k-1}(x)}{k+1},
\end{aligned}
$$

requires only a handful of floating-point operations and can be fully
inlined by Numba's LLVM compiler.  The normalisation factor
$\sqrt{2n+1}$ is applied at the end.

**Impact:** 2–3&times; speedup on Legendre evaluation within
single-sample predictions (Python↔SciPy boundary crossing
eliminated).

### 1.3 Compiled prediction pipeline

**Change:** The entire surrogate prediction — normalise, build design
matrix, dot product — is compiled into a single `@njit` function
`_predict_numba(X_arr, mins, maxs, term_cols, term_degs,
term_n_factors, coef, intercept)`.

**Why:** When the MC Shapley loop evaluates $f(x)$ on single samples
(or small batches), the Python interpreter executes the same sequence
of NumPy operations millions of times.  Compiling that sequence
eliminates all interpreter overhead: type checks, attribute look-ups,
function-call dispatch.

**The `term_cols` / `term_degs` representation.**  The surrogate's
basis expansion is a product of Legendre polynomials for specific
input columns:

$$
\psi_i(\mathbf{x}) = \prod_{f=1}^{n_i} \hat{P}_{d_{i,f}}(x_{c_{i,f}})
$$

During `fit()`, the string-based labels (e.g., `"X1_3*X2_5"`) are
converted to padded integer arrays:

- `term_cols[i, f]` — column index of factor $f$ in term $i$
- `term_degs[i, f]` — degree of that factor
- `term_n_factors[i]` — number of factors in term $i$

These are the only data the compiled function needs; no Python
strings or dictionaries cross the JIT boundary.

### 1.4 Batch-size routing

**Change:** `predict()` now routes to the compiled path when
$n_\text{rows} \le 1000$ and to the original NumPy/SciPy path for
larger batches.

**Why:** For large batches ($n_\text{rows} \gg 1000$), NumPy's
column-vectorised Legendre evaluation (each `eval_sh_legendre` call
processes a full column array inside pre-compiled C) beats Numba's
per-term loop.  For small batches — the MC Shapley regime — the
compiled path wins by eliminating Python overhead.  The threshold of
1000 rows sits safely below the crossover.

**Benchmark (120-term surrogate, d=8):**

| Batch size | Original | Numba | Winner |
|---|---:|---:|---|
| 1 row (MC Shapley per-sample) | 1,571&nbsp;calls/s | 9,339&nbsp;calls/s | **Numba (5.9×)** |
| 50,000 rows | 308K&nbsp;samples/s | 161K&nbsp;samples/s | Original (1.9×) |

---

## 2. Monte Carlo Sampling (mc_shapley.py)

### 2.1 Batched conditional sampling

**Change:** Added `sample_conditional_batch(u_indices, fixed_X)` to
both `MultivariateNormal` and `GaussianCopulaUniform`.  Uses a single
Cholesky decomposition of the conditional covariance matrix plus one
$(N, |v|)$ standard-normal draw, transformed in one matrix multiply —
instead of $N$ separate `multivariate_normal` calls.

The common conditional-distribution parameters
($\Sigma_{uu}^{-1}\Sigma_{uv}$, $\Sigma_{vv} - \Sigma_{vu}\Sigma_{uu}^{-1}\Sigma_{uv}$)
are computed once per subset and shared across all $N$ draws.

**Impact:** Eliminates the Python `for`-loop of $N$ iterations inside
each conditional-sampling step.  Combined with the batched predict
path, the entire Side&nbsp;B of the MC loop becomes two function
calls (one batch draw, one batch predict).

*Note on `TruncatedMultivariateNormal`.*  The truncated-normal class
also implements `sample_conditional_batch`, but uses **vectorised
Gibbs sampling** rather than a single Cholesky draw — the conditional
distribution of a truncated MVN does not have a closed-form sampler.
Each Gibbs iteration sweeps over all $|v|$ variables, drawing from
univariate truncated normals conditioned on the current values of the
other variables.  The per-iteration cost is comparable to the
Cholesky-based batch draw for small $|v|$, but $n_\text{iter}$
iterations are needed (default 5 for conditional draws).  This adds a
constant factor of $n_\text{iter}$ compared to the untruncated case.

### 2.2 Batched data-collection functions

**Before:**

```python
# Side B — conditional draws
Y2 = np.zeros(N)
for i in range(N):
    x_cond = joint.sample_conditional(u_list, X[i, u_list])
    Y2[i] = f(x_cond)       # single-sample predict
```

**After:**

```python
# Side B — conditional draws (batched)
X_cond = joint.sample_conditional_batch(u_list, X[:, u_list])
Y2 = predict_batch(X_cond)  # single batch predict
```

Both `collect_shapley_data` (exhaustive method) and
`compute_subset_data` (permutation method) now batch both the
unconditional and conditional sides.  When `predict_batch` is
available (always the case for the surrogate model), the
Python-level per-sample loop disappears entirely from the
data-collection phase.

**Impact:** 3–5&times; speedup on the MC sampling phase beyond the
predict-compilation gains.

### 2.3 Benched summary (d=8, N=2000, analytical G-function)

| Method | Before optimisation | After optimisation |
|---|---|---|
| Exhaustive (255 subsets, no bootstrap) | ~3.5s | **0.8s** |
| Permutation (500 perms, no bootstrap) | ~4s | **0.9s** |

*Measured with the analytical G-function to isolate MC infrastructure
from surrogate-model cost.*

### 2.4 Coalition truncation (`k_max`)

**Change:** Added `k_max` parameter to `collect_shapley_data()` and
`shapley_from_data()`.  When `k_max < d`, only subsets up to size
$k_{\max}$ are evaluated (plus the full set for total variance).

**Why:** The standard exhaustive method evaluates $2^d - 1$ subsets.
For RS-HDMR surrogate models with bounded interaction order (e.g.,
`polys=[10,5]` has only second-order terms), coalitions larger than the
expansion order contribute nothing that isn't already captured by their
sub-coalitions.  Evaluating only up to the surrogate's interaction order
is **exact** and reduces the subset count from $O(2^d)$ to $O(d^{k_{\max}})$.

**Auto-detection:** `model.get_mc_shapley()` sets $k_{\max} =
\text{len}(\text{polys})$ automatically, requiring no user configuration.

**Subset reduction examples:**

| $d$ | Full ($2^d-1$) | $k_{\max}=2$ | Factor |
|---|---:|---:|---:|
| 6 | 63 | 22 | 2.9× |
| 8 | 255 | 37 | 6.9× |
| 10 | 1,023 | 56 | 18× |
| 12 | 4,095 | 79 | 52× |
| 15 | 32,767 | 121 | 270× |

**Impact:** For a second-order RS-HDMR model ($k_{\max}=2$) at $d=10$,
the MC sampling phase evaluates 56 subsets instead of 1,023 — an 18×
reduction — with zero loss of accuracy.  At $d=15$ the reduction is 270×.

---

## 3. Shapley Computation and Bootstrap

### 3.1 Precomputed Shapley weights

**Change:** The Shapley weight

$$
w(k, d) = \frac{k!\,(d-k-1)!}{d!}
$$

depends only on the subset size $k$ and the dimensionality $d$.
Previously, `math.factorial` was called for every $(i, u)$ pair
inside the nested Shapley accumulation loop — $d \times 2^d$
invocations per bootstrap iteration.  For $d=18$ and $B=100$, this
would be $18 \times 262{,}144 \times 100 \approx 472$&nbsp;million
factorial calls.

The weights are now computed once per dimensionality and stored in a
module-level cache (`_shapley_weight_cache`).  The array has $d+1$
entries; index $d$ (the full set) is a zero-valued guard.

### 3.2 Compiled bootstrap (exhaustive method)

**Change:** A new `@njit` function `_bootstrap_iter_numba` performs
$B$ bootstrap iterations of *resample* → *compute v(u)* → *compute
Shapley* entirely inside compiled code.

**Data conversion.**  The Python `dict`-of-tuples produced by the
data-collection phase is converted once to flat `float64` arrays via
`_data_to_arrays`:

- `Y_full` — $(N,)$ outputs for the full set
- `pair_Y1`, `pair_Y2` — $(n_\text{pairs}, N)$ paired outputs
- `subset_sizes` — $(2^d,)$ array of $|u|$ per canonical subset
- `union_lookup` — $(2^d, d)$ precomputed index of $u \cup \{i\}$
  (or $-1$ if $i \in u$)

The compiled loop then does:

```
for b in range(B):
    idx = random_choice(N)                # bootstrap resample
    v[full] = var(Y_full[idx])
    for each pair subset s:
        v[s] = cov(Y1_s[idx], Y2_s[idx])
    for each subset s (including empty):
        for each variable i:
            if i not in s:
                sh[i] += w(|s|) * (v[s ∪ {i}] - v[s])
    boot_effects[b] = sh / v[full]
```

The empty set ($s = 0$, $v[0] = 0$) is included so that the
$\emptyset \to \{i\}$ contribution — weight $1/d$, value $v(\{i\})$ —
is correctly accumulated.

**Impact:** The bootstrap phase of the exhaustive method runs in
compiled code from start to finish.  For $d=8$, $B=200$, this phase
takes **0.9s** (including JIT compilation of the first call).

---

## Summary of Speedup Factors

| # | Optimisation | Module | Approx. Speedup | Applies To |
|---|---|---|---|---|
| 1 | Bypass sklearn predict overhead | `predictor.py` | 1.1× | Surrogate predict |
| 2 | Numba Legendre polynomials | `predictor.py` | 2–3× | Per-sample predict |
| 3 | Compiled predict pipeline | `predictor.py` | 5–6× | MC Shapley hot path |
| 4 | Batch-size routing | `predictor.py` | — | Prevents regression on large batches |
| 5 | Batched conditional sampling | `mc_shapley.py` | 3–5× | Both MC methods |
| 6 | Batched data collection | `mc_shapley.py` | 2–3× | Both MC methods |
| 7 | Precomputed Shapley weights | `mc_shapley.py` | 2–5× | Bootstrap ($d \ge 10$) |
| 8 | Compiled bootstrap | `mc_shapley.py` | 3–10× | Exhaustive bootstrap |

The combined effect for a typical workflow (RS-HDMR surrogate,
$d=8$, $N=3000$, $B=100$) is a **10–30× reduction** in wall-clock time
compared to the unoptimised v0.2 code path.

---

## Compatibility and Fallbacks

All optimisations are backwards-compatible and degrade gracefully:

- **Numba unavailable:** The compiled Legendre, predict, and bootstrap
  paths are replaced with pure-Python equivalents.  The sklearn bypass
  and batched sampling are always active (pure NumPy).

- **Numba version:** Requires Numba ≥&nbsp;0.60 with NumPy &lt;&nbsp;2.4.
  The `@njit(cache=True)` decorator caches compiled functions on disk
  so that JIT compilation cost is paid only once per Python session.

- **Prediction routing:** `predict(use_numba=True)` is the default;
  pass `use_numba=False` to force the original SciPy-based path.
