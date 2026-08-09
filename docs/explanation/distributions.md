# Distribution Classes

The Monte Carlo Shapley estimator requires **conditional sampling** from the
joint input distribution — drawing from $P(\mathbf{X}_{-u} \mid \mathbf{X}_u = \mathbf{x}_u)$
for arbitrary subsets $u$.  ShapleyX defines a **distribution class interface**
that any input model can implement to plug into the MC Shapley machinery.

This page describes the built-in classes, the interface contract, and several
custom classes developed for the example case studies.

---

## Interface Contract

A distribution class must provide four attributes and methods:

| Requirement | Signature | Description |
|---|---|---|
| `self.d` | `int` | Number of input dimensions |
| `sample_joint(n)` | `(n,)` → `(n, d)` array | Draw $n$ i.i.d. samples from the full joint distribution |
| `sample_conditional(u, x)` | `(list, 1D array)` → `(d,)` array | Draw *one* sample conditioned on $\mathbf{X}_u = \mathbf{x}$ |
| `sample_conditional_batch(u, X)` | `(list, 2D array)` → `(N, d)` array | Draw $N$ conditional samples, vectorised — used by the MC loops for efficiency |

Where:

- `u` is a list of variable indices (e.g., `[0, 2]`)
- `x` is a 1D array of fixed values for those variables
- `X` is a 2D array of shape `(N, len(u))` — $N$ different conditioning points

The `sample_conditional` method can delegate to `sample_conditional_batch`:

```python
def sample_conditional(self, u, x):
    X = self.sample_conditional_batch(u, np.atleast_2d(x))
    return X[0]
```

---

## Built-in Classes

### `MultivariateNormal`

**Jointly normal** inputs with analytical conditional distributions.

```python
from shapleyx.utilities.mc_shapley import MultivariateNormal

joint = MultivariateNormal(
    mean=[0.0, 0.0, 0.0],
    cov=[[1.0, 0.5, 0.0],
         [0.5, 1.0, 0.0],
         [0.0, 0.0, 1.0]],
)
```

| Method | Implementation |
|---|---|
| `sample_joint` | `np.random.multivariate_normal` |
| `sample_conditional_batch` | Closed-form: $\boldsymbol{\mu}_{v\mid u} = \boldsymbol{\mu}_v + \boldsymbol{\Sigma}_{vu}\boldsymbol{\Sigma}_{uu}^{-1}(\mathbf{x}_u - \boldsymbol{\mu}_u)$, Cholesky of conditional covariance |

**Fastest** of the built-in classes — uses a single Cholesky decomposition per subset.

### `GaussianCopulaUniform`

**Uniform marginals** $[a_i, b_i]$ with dependence induced by a latent multivariate normal.

```python
from shapleyx.utilities.mc_shapley import GaussianCopulaUniform

joint = GaussianCopulaUniform(
    lows=[-np.pi, -np.pi, -np.pi],
    highs=[np.pi, np.pi, np.pi],
    corr=np.eye(3),
)
```

| Method | Implementation |
|---|---|
| `sample_joint` | Draw latent normal → `norm.cdf` → scale to $[a_i, b_i]$ |
| `sample_conditional_batch` | Map fixed variables to latent normal via `norm.ppf`, condition with `MultivariateNormal` formulas, map back via `norm.cdf` |

Used in the Iooss & Prieur correlation sweep tutorials.

### `TruncatedMultivariateNormal`

**Jointly normal** with per-dimension truncation bounds $[a_i, b_i]$.

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
```
| Method | Implementation |
|---|---|
| `sample_joint` | Vectorised Gibbs sampling — iterates over variables, each drawn from a univariate truncated normal conditioned on current values of all others |
| `sample_conditional_batch` | Gibbs on the $\lvert v\rvert$-dimensional truncated conditional, started at the well-centred untruncated conditional mean |

Tune `joint_burn_in` (default 30) and `cond_burn_in` (default 5) to balance
speed and convergence.  Use `-np.inf` / `np.inf` for unbounded dimensions.

---

## Built-in Mixed-Marginal Copulas

`GaussianCopulaMixed(marginals, corr)` and `TCopulaMixed(marginals, corr, nu)`
are the package's mixed-marginal copula classes: arbitrary per-variable
marginals under a Gaussian or Student-t copula, with a **declarative marginal
spec registry**:

```python
from shapleyx.utilities.mc_shapley import GaussianCopulaMixed, TCopulaMixed

marginals = {
    'FX': ('lognormal', {'mean': 556.8, 'cv': 0.08}),   # paper-Table-1 style
    'FY': ('lognormal', {'mu': 6.31, 'sigma': 0.0799}), # underlying-normal style
    'E':  ('normal', 200e9, 1.2e10),                    # (mean, std)
    'lX': ('uniform', 0.025, 0.100),                    # (lo, hi)
    'lY': ('truncnorm', 0.04, 0.16, 0.0987, 0.00987),   # (lo, hi, mean, std)
    'L':  ('beta', 2.0, 5.0, 2.0, 7.0),                 # (a, b, lo, hi)
    'Q':  weibull_min(c=2),                             # scipy passthrough
}

joint_g = GaussianCopulaMixed(marginals, corr)          # Gaussian copula
joint_t = TCopulaMixed(marginals, corr, nu=5.0)         # Student-t copula
```

**Registered families:** `normal`, `lognormal`, `uniform`, `truncnorm`,
`beta`, `gamma`, `exponential`, `weibull`, `gumbel`, `frechet`, `pareto` —
plus the full `GaussianCopulaArbitrary` passthrough contract (scipy
distributions, `(cdf, ppf)` tuples, callable PPFs, `None`/`'uniform'`).

**Lognormal parameterisation** is deliberately restricted to *named* forms —
exactly one of `{'mean', 'cv'}` or `{'mu', 'sigma'}` — to prevent the classic
parameterisation bug; a bare positional tuple raises.

**`TCopulaMixed`** applies the t-copula to the *correlated block* only
(uncorrelated variables stay independent — a plain block-diagonal
$t_\nu(0,R)$ does not factorise).  Conditional sampling is **exact** via the
multivariate-t conditioning result $X_f \mid X_c \sim t_{\nu+k}(\mu_c,
\frac{\nu+Q}{\nu+k}\Sigma_c)$ (no MCMC, Rosenblatt-validated).  RQMC/
deterministic sampling is not implemented for the t kernel yet — use
`method='exhaustive'`.

`GaussianCopulaMixed` delegates latent sampling to `MultivariateNormal`, so
its RNG stream is identical to the hand-rolled class historically defined in
`Examples/cantilever_beam.ipynb` — published numbers reproduce exactly.

Full worked example: `docs/tutorials/cantilever_beam_copulas.ipynb`.

---

## Custom Classes from Case Studies

Several notebooks define **ad-hoc distribution classes** to handle
challenging input models.  They illustrate the flexibility of the interface —
and are now largely **superseded by the built-in classes above**.

### `GaussianCopulaMixed` — Cantilever Beam (historical)

**Mixed Normal + LogNormal marginals** with a Gaussian copula for correlation.
Defined inline in `Examples/cantilever_beam.ipynb`; the package now ships an
identical-stream implementation as `shapleyx.utilities.mc_shapley.GaussianCopulaMixed`.

```python
class GaussianCopulaMixed:
    def __init__(self, marginals, latent_corr):
        self.d = len(marginals)
        self._mvn = MultivariateNormal(mean=np.zeros(d), cov=latent_corr)
        # ...

    def _to_latent(self, x, params):   # original → N(0,1)
    def _from_latent(self, z, params): # N(0,1) → original
    def sample_joint(self, n):         # latent draw → _to_original
    def sample_conditional_batch(...): # map fixed to latent → condition →
                                        # map back
```

**Key pattern:** the class wraps a `MultivariateNormal` in latent space and
provides two static methods (`_to_latent`, `_from_latent`) that are dispatched
per column based on the marginal type.  Adding a new marginal type (e.g.,
Weibull, Beta) requires only a new branch in these two methods.

### `GaussianCopulaFull` — Borehole Function

Extends the mixed-copula pattern to **Uniform marginals** in addition to
Normal and LogNormal.  Defined in `Examples/borehole.ipynb`.

The only change from `GaussianCopulaMixed` is an extra branch in the
transform methods:

```python
# Uniform[a, b]: map via Φ⁻¹((x-a)/(b-a))
if params[0] == 'uniform':
    _, a, b = params
    u = np.clip((x - a) / (b - a), 1e-15, 1 - 1e-15)
    return norm.ppf(u)
```

---

## Writing Your Own Distribution Class

Follow these steps to create a custom distribution:

### Step 1: Choose a strategy

| If your distribution is... | Consider... |
|---|---|
| A known multivariate family | Wrapping `MultivariateNormal` in latent space (Gaussian copula) |
| Defined by physical constraints | Rejection sampling on top of a base distribution |
| Available only via a black-box simulator | Pre-generating a large sample and using nearest-neighbour approximation (see Demange-Chryst 2022, Appendix I) |

### Step 2: Implement the interface

```python
class MyDistribution:
    def __init__(self, ...):
        self.d = ...  # number of dimensions

    def sample_joint(self, n):
        """Return (n, d) array of joint samples."""
        ...

    def sample_conditional(self, u_indices, fixed_x):
        """Return (d,) array — single conditional sample."""
        X = self.sample_conditional_batch(
            u_indices, np.atleast_2d(np.asarray(fixed_x, dtype=float))
        )
        return X[0]

    def sample_conditional_batch(self, u_indices, fixed_X):
        """Return (N, d) array — N conditional samples."""
        ...
```

### Step 3: Test your class

Verify that the class produces correct correlations and respects
the conditional structure:

```python
joint = MyDistribution(...)

# Check marginals
X = joint.sample_joint(10000)
print("Means:", X.mean(axis=0))

# Check conditional: E[X_i | X_j = some_value] should differ from E[X_i]
x_cond = joint.sample_conditional([0], [some_value])

# Check batch: should be faster than a Python loop
X_batch = joint.sample_conditional_batch([0], np.array([[v1], [v2], [v3]]))
```

### Step 4: Plug into ShapleyX

```python
from shapleyx.utilities.mc_shapley import shapley_effects

effects, sh, var = shapley_effects(my_model, joint, N=5000, method='exhaustive')
```

Your class is now a first-class citizen of the MC Shapley pipeline — it
works with the exhaustive method, permutation method, bootstrap, batch
prediction, and progress bars.

---

## Case Studies

| Report | Distribution | Key Features |
|---|---|---|
| [Cantilever Beam](../tutorials/cantilever_beam.ipynb) | `GaussianCopulaMixed` | 6 inputs, LogNormal + Normal marginals, correlated dimensional parameters, analytical and surrogate target Shapley effects validated against Demange-Chryst (2022) [[PDF report](../case_study_reports/cantilever_case_study.pdf)] |
| [Borehole Function](../tutorials/borehole/) | `GaussianCopulaFull` | 8 inputs, Normal + LogNormal + Uniform marginals, optional geological correlations, Sobol + Shapley from single run |
| [Truncated Normal](../tutorials/mc_shapley_truncated_normal/) | `TruncatedMultivariateNormal` (built-in) | 3 inputs, per-dimension truncation, multiple truncation schemes, RS-HDMR surrogate comparison |
| [Iooss & Prieur Correlation Sweep](../tutorials/iooss_prieur_ishigami_correlation/) | `GaussianCopulaUniform` (built-in) | 3 inputs, Shapley + $S_i$ + $T_i$ vs correlation $\rho$, exhaustive and permutation methods |
| [Wing Weight Function](../tutorials/wing_weight/) | `GaussianCopulaUniform` (built-in) | 10 inputs, all Uniform spanning 5 orders of magnitude, independent and correlated analyses, validated against OpenTURNS reference [[PDF report](../case_study_reports/wing_weight_case_study.pdf)] |
| [Fire Spread Model](../tutorials/fire_spread/) | `GaussianCopulaFire` | 10 inputs, mixed LogNormal + Normal + scaled LogNormal with truncation, target Shapley effects without IS, demonstrates MC limits for rare events ($p_f \approx 10^{-4}$) [[PDF report](../case_study_reports/fire_spread_case_study.pdf)] |
| [SAC-SMA Model](../tutorials/sac_sma_shapleyx_example.ipynb) | — | High-dimensional conceptual rainfall-runoff model for flood forecasting, operational hydrologic parameters, RS-HDMR with ARD [[PDF report](../case_study_reports/sac_sma_case_study.pdf)] |

---

## See Also

- [MC Shapley How-to Guide](../how-to-guides/mc-shapley.md) — usage instructions
- [Theory: Shapley Effects with Correlated Inputs](theory.md#shapley-effects-with-correlated-inputs)
- [MC Shapley Reference](../reference/mc_shapley.md) — API documentation
- [Owen & Prieur (2017)](https://arxiv.org/abs/1704.06942) — the covariance formulation
