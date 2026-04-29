# Monte Carlo Shapley Effects

This guide explains how to estimate Shapley effects using the Monte Carlo approach, which correctly handles correlated inputs — a scenario where standard variance-based methods break down.

## Overview

The MC Shapley method estimates Shapley effects via conditional Monte Carlo sampling. It supports:

- **Two computation methods**: exhaustive enumeration and random permutations
- **Bootstrap confidence intervals** for all estimates
- **Two distribution families**: `GaussianCopulaUniform` and `MultivariateNormal`
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

## Bootstrap Confidence Intervals

Set `B > 0` to enable bootstrap CIs. The returned DataFrame gains `'lower'` and `'upper'` columns:

```python
mc = model.get_mc_shapley(N=5000, B=500, alpha=0.05)
print(f"X1 effect: {mc.loc[0, 'effect']:.4f} "
      f"[{mc.loc[0, 'lower']:.4f}, {mc.loc[0, 'upper']:.4f}]")
```

## Performance Tips

- **Start with small N**: Use `N=1000–2000` for quick exploration, increase for final results.
- **Prefer the permutation method** for $d > 6$ to avoid exponential subset explosion.
- **The `B` bootstrap iterations** are independent of the $2^d$ scaling — they re-use cached data.
- **Random state**: Fix `random_state` for reproducible outputs.

## See Also

- [MC Shapley Reference](../reference/mc_shapley.md) — complete API documentation
- [Theory: Shapley Effects with Correlated Inputs](../explanation/theory.md#shapley-effects-with-correlated-inputs)
- [Examples: MC Shapley Notebook](https://github.com/frbennett/shapleyx/blob/main/Examples/mc_shapley.ipynb)
