# Monte Carlo Shapley Effects

This guide explains how to estimate Shapley effects using the Monte Carlo approach, which correctly handles correlated inputs — a scenario where standard variance-based methods break down.

## Overview

The MC Shapley method estimates Shapley effects via conditional Monte Carlo sampling. It supports:

- **Two computation methods**: exhaustive enumeration and random permutations
- **Bootstrap confidence intervals** for all estimates
- **Three distribution families**: `GaussianCopulaUniform`, `MultivariateNormal`, and `TruncatedMultivariateNormal`
- **Flexible model functions**: use the trained surrogate or any Python callable

## Basic Usage with the Surrogate Model

After training an RS-HDMR surrogate, call `get_mc_shapley()` directly:

```python
from shapleyx import rshdmr

# Train the RS-HDMR surrogate
model = rshdmr(data_file='data.csv', polys=[10, 5], method='ard_cv')
sobol, shapley, total = model.run_all()

# Compute MC Shapley effects (independent inputs, surrogate model)
mc_results = model.get_mc_shapley(N=5000, method='exhaustive', B=500)

print(mc_results)
#   variable    effect  shapley_value  total_variance     lower     upper
# 0       X1  0.435622       5.982737         13.7330  0.408571  0.466487
# 1       X2  0.424561       5.830838         13.7330  0.396925  0.453246
# 2       X3  0.139817       1.920069         13.7330  0.114170  0.166073
```

## Correlated Inputs

Provide a correlation matrix via the `corr` parameter to model dependence:

```python
import numpy as np

corr = np.array([
    [1.0, 0.0, 0.8],
    [0.0, 1.0, 0.0],
    [0.8, 0.0, 1.0],
])

mc_corr = model.get_mc_shapley(corr=corr, N=5000, B=500)
```

This constructs a `GaussianCopulaUniform` distribution using the training data's min–max ranges and the supplied correlation matrix.

## User-Defined Model Functions

You are not limited to the surrogate model. Pass any callable `f(x)` that takes a 1D numpy array and returns a scalar:

```python
def my_model(x):
    return np.sin(x[0]) + 7 * np.sin(x[1])**2 + 0.1 * x[2]**4 * np.sin(x[0])

mc_user = model.get_mc_shapley(f=my_model, N=5000, B=500)
```

## Custom Distributions

For full control over the input distribution, provide a distribution object via the `joint` parameter.

### Gaussian Copula with Uniform Marginals

```python
from shapleyx.utilities.mc_shapley import GaussianCopulaUniform

joint = GaussianCopulaUniform(
    lows=[0.0, -1.0, 0.5],
    highs=[1.0, 1.0, 2.0],
    corr=np.eye(3),
)

mc = model.get_mc_shapley(joint=joint, f=my_model, N=5000)
```

### Multivariate Normal

```python
from shapleyx.utilities.mc_shapley import MultivariateNormal

joint = MultivariateNormal(
    mean=[0.0, 0.0, 0.0],
    cov=[[1.0, 0.5, 0.0],
         [0.5, 1.0, 0.0],
         [0.0, 0.0, 1.0]],
)

mc = model.get_mc_shapley(joint=joint, f=my_model, N=5000)
```

### Truncated Multivariate Normal

```python
from shapleyx.utilities.mc_shapley import TruncatedMultivariateNormal

joint = TruncatedMultivariateNormal(
    mean=[0.0, 0.0, 0.0],
    cov=[[1.0, 0.5, 0.0],
         [0.5, 1.0, 0.0],
         [0.0, 0.0, 1.0]],
    lower=[-1.0, -1.0, -1.0],
    upper=[1.0,  1.0,  1.0],
)

mc = model.get_mc_shapley(joint=joint, f=my_model, N=5000)
```

Use `-np.inf` and `np.inf` for unbounded dimensions.  The class uses
**Gibbs sampling** for both joint and conditional draws; tune
`joint_burn_in` (default 30) and `cond_burn_in` (default 5) to
balance speed versus convergence.

## Choosing a Computation Method

| Method | When to Use |
|---|---|
| `'exhaustive'` | Small to moderate $d$ (≤ 8). Enumerates all $2^d-1$ subsets. Exact. |
| `'permutation'` | Larger $d$. Uses random permutations with lazy caching. Approximate but scalable. |

```python
# Exhaustive (default)
mc = model.get_mc_shapley(N=5000, method='exhaustive')

# Permutation
mc = model.get_mc_shapley(N=5000, method='permutation', n_perm=2000)
```

## Coalition Truncation (`k_max`)

The exhaustive method evaluates all $2^d - 1$ non-empty subsets, which becomes
prohibitively expensive for $d > 8$.  However, many models — especially those
built from RS-HDMR surrogates — have **no interactions above a known order**.
For example, an RS-HDMR model trained with `polys=[10, 5]` has only
first-order (main effects) and second-order (pairwise) interactions; third-order
and higher terms are absent by construction.

The **`k_max` parameter** limits coalition evaluation to subsets of size
$\leq k_{\max}$ (plus the full set, always needed for total variance).  This is
**exact** when the model truly has no interactions above order $k_{\max}$,
dramatically reducing the number of subsets:

| $d$ | All subsets ($2^d-1$) | $k_{\max}=2$ | $k_{\max}=3$ |
|---|---:|---:|---:|
| 6 | 63 | 22 | 42 |
| 8 | 255 | 37 | 93 |
| 10 | 1,023 | 56 | 176 |
| 15 | 32,767 | 121 | 576 |
| 20 | 1,048,575 | 211 | 1,351 |

### Auto-Detection

When using `model.get_mc_shapley()`, `k_max` is **auto-detected** from the
surrogate model's `polys` parameter — `len(polys)` gives the highest interaction
order in the Legendre expansion.  No manual configuration is needed:

```python
# polys=[10, 5] → k_max=2 auto-detected (exact for this surrogate)
mc = model.get_mc_shapley(N=5000, method='exhaustive')
```

### Explicit Control

Override the auto-detected value or use `k_max` with the low-level API:

```python
# Explicit: limit to pairwise coalitions
mc = model.get_mc_shapley(N=5000, method='exhaustive', k_max=2)

# Full enumeration (backward-compatible default)
mc = model.get_mc_shapley(N=5000, method='exhaustive', k_max=None)

# Low-level API
from shapleyx.utilities.mc_shapley import MCShapley
mc = MCShapley(f=my_model, joint=my_dist)
df = mc.compute(N=5000, method='exhaustive', k_max=2)
```

### Approximation Behaviour

When `k_max < d-1`, some coalitions are not evaluated.  The estimator uses
a **conservative fallback**: missing $v(u)$ values are approximated as the
total variance $v(\text{full})$, meaning missing high-order interactions are
assumed to contribute proportionally.  For models with sparse high-order
structure (the intended use case), this approximation is excellent.

**Note on Sobol indices:** When $k_{\max} < d-1$, the complement sets
$\{-i\}$ needed for total-order Sobol indices $T_i$ are not evaluated,
so `sobol_total` is returned as `NaN`.  First-order Sobol indices $S_i$
remain available (they only require singleton subsets).

## Bootstrap Confidence Intervals

Set `B > 0` to enable bootstrap CIs. The returned DataFrame gains `'lower'` and `'upper'` columns:

```python
mc = model.get_mc_shapley(N=5000, B=500, alpha=0.05)
print(f"X1 effect: {mc.loc[0, 'effect']:.4f} "
      f"[{mc.loc[0, 'lower']:.4f}, {mc.loc[0, 'upper']:.4f}]")
```

## Sobol Indices from the Same Data

The Owen & Prieur covariance formulation $v(u) = \text{Cov}[f(\mathbf{X}), f(\mathbf{X}_u)]$ equals
the **closed Sobol index** $\mathbb{V}[\mathbb{E}(f(\mathbf{X}) \mid \mathbf{X}_u)]$.  This means
**first-order ($S_i$) and total-order ($T_i$) Sobol indices** can be extracted from the same
Monte Carlo data at zero additional cost:

$$S_i = \frac{v(\{i\})}{v(\text{full})}, \qquad
T_i = 1 - \frac{v(\text{all}\setminus\{i\})}{v(\text{full})}$$

When using `method='exhaustive'`, the returned DataFrame includes:

| Column | Description |
|---|---|
| `sobol_first` | First-order Sobol index $S_i$ |
| `sobol_total` | Total-order Sobol index $T_i$ |
| `sobol_first_lower`, `sobol_first_upper` | Bootstrap CIs for $S_i$ (if `B > 0`) |
| `sobol_total_lower`, `sobol_total_upper` | Bootstrap CIs for $T_i$ (if `B > 0`) |

```python
mc = model.get_mc_shapley(N=5000, method='exhaustive', B=500)
print(mc[['variable', 'effect', 'sobol_first', 'sobol_total']])
```

For the permutation method, Sobol columns are returned as `NaN` — the lazy subset
evaluation may not include all singleton and $(d-1)$-variable subsets needed.

**Note on correlated inputs:** Under correlation, the standard inequality
$S_i \leq T_i$ no longer holds.  See the [MC Shapley + Sobol](../tutorials/mc_shapley_sobol/)
tutorial for a detailed explanation.

## Performance Tips

- **Start with small N**: Use `N=1000–2000` for quick exploration, increase for final results.
- **Prefer the permutation method** for $d > 6$ to avoid exponential subset explosion.
- **The `B` bootstrap iterations** are independent of the $2^d$ scaling — they re-use cached data.
- **Random state**: Fix `random_state` for reproducible outputs.

## See Also

- [MC Shapley Reference](../reference/mc_shapley.md) — complete API documentation
- [Theory: Shapley Effects with Correlated Inputs](../explanation/theory.md#shapley-effects-with-correlated-inputs)
- [Examples: MC Shapley Notebook](https://github.com/frbennett/shapleyx/blob/main/Examples/mc_shapley.ipynb)
