# Arbitrary Marginal Distributions

## Motivation

`GaussianCopulaUniform` assumes every input follows a **uniform** distribution
bounded by `[lows[i], highs[i]]`. In many real-world applications, calibrated
parameters cluster around preferred values — e.g., UZTWM might have a
Beta(2, 5) shape within [30, 125], or REXP might follow a LogNormal distribution.

`GaussianCopulaArbitrary` lets you specify **any marginal distribution** per
variable while preserving the correlation structure through the Gaussian copula.

## Usage

```python
from shapleyx.utilities.mc_shapley import GaussianCopulaArbitrary
from scipy.stats import beta, lognorm

joint = GaussianCopulaArbitrary(
    marginals={
        'UZTWM':  beta(a=2, b=5),              # scipy distribution
        'UZFWM':  'uniform',                     # string shortcut
        'REXP':   lognorm(s=0.5),                # unbounded → handled by CDF
        'PFREE':  None,                          # → uniform (backward compat)
        'custom': lambda u: 10 + 10 * u,         # callable PPF
        'empirical': (my_cdf, my_ppf),           # explicit (cdf, ppf) tuple
    },
    corr=corr_matrix,
)

# Returns samples in PHYSICAL space (matching GaussianCopulaUniform convention)
X = joint.sample_joint(1000)  # shape (1000, d)
```

## Supported Marginal Specifications

| Specification | Example | Use case |
|--------------|---------|----------|
| `'uniform'` | `'uniform'` | Default — Uniform(0, 1) |
| `None` | `None` | Same as `'uniform'` (backward compat) |
| `scipy.stats` distribution | `beta(2, 5)` | Parametric distributions |
| `callable(u) → float` | `lambda u: 10 + 10*u` | Custom PPF, no CDF needed for sampling |
| `(cdf, ppf)` tuple | `(my_cdf, my_ppf)` | Full specification — required for conditional sampling |

## How It Works

1. **Latent space:** $Z \sim \mathcal{N}(0, R)$ — same as `GaussianCopulaUniform`
2. **Uniform scores:** $U_j = \Phi(Z_j)$ — uniform [0, 1]
3. **Physical space:** $X_j = F_j^{-1}(U_j)$ — applied per marginal

The surrogate model always receives physical-space samples (matching the
`GaussianCopulaUniform` convention). The physical → [0, 1] → physical
round-trip is handled transparently in `sample_conditional_batch()`.

## Integration with MC Shapley

Works directly — same interface as `GaussianCopulaUniform`:

```python
from shapleyx.utilities.mc_shapley import shapley_effects

effects, sh, tv = shapley_effects(
    my_model, joint, N=10000, method='exhaustive'
)
```

Or via the RS-HDMR pipeline:

```python
analyzer.get_mc_shapley(joint=joint, N=5000)
```

## Custom Marginals from MCMC Posterior

```python
import numpy as np

def empirical_ppf(samples):
    """Build a PPF from posterior samples via linear interpolation."""
    sorted_samples = np.sort(samples)
    quantiles = np.linspace(0, 1, len(sorted_samples))
    def ppf(u):
        return np.interp(u, quantiles, sorted_samples)
    def cdf(x):
        return np.interp(x, sorted_samples, quantiles)
    return cdf, ppf

# Build copula directly from MCMC chains
posterior_UZTWM = np.load('posterior_samples/UZTWM.npy')
cdf_uztwm, ppf_uztwm = empirical_ppf(posterior_UZTWM)

joint = GaussianCopulaArbitrary(
    marginals={'UZTWM': (cdf_uztwm, ppf_uztwm), ...},
    corr=np.corrcoef(posterior_samples.T),  # also from data
)
```
