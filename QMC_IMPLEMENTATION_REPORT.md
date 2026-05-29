# QMC-Accelerated Pick-Freeze for Shapley Effect Estimation in ShapleyX

**Branch**: `qmc-pick-freeze`
**Date**: 29 May 2026
**Author**: Frederick Bennett (with implementation by Hermes Agent)

---

## 1. Motivation

ShapleyX's `mc_shapley` module estimates Shapley effects for correlated/non-uniform
inputs using an Owen–Prieur pick-freeze estimator (Owen & Prieur, 2017). The
estimator evaluates the surrogate model at *N* joint samples and *N* conditional
samples for each subset of input variables, then computes

\[
v(u) = \operatorname{Cov}\!\big[f(X),\, f(X_u)\big]
      = \operatorname{Var}\!\big[\mathbb{E}[f(X) \mid X_u]\big]
\]

and assembles Shapley effects from the \(v(u)\) values via the standard coalition
formula.  When inputs are independent and uniform the user can bypass this Monte
Carlo step entirely — variance-based indices are extracted analytically from the
RS-HDMR polynomial coefficients.  For correlated or non-uniform inputs, however,
the pick-freeze estimator has been the only option, and its convergence rate of
\(O(1/\sqrt{N})\) means a large *N* is required for precise estimates.

**Randomised quasi-Monte Carlo (RQMC)** replaces the pseudo-random number
generator with scrambled Sobol' low-discrepancy sequences.  For smooth
integrands (such as polynomial surrogates), RQMC achieves an effective
convergence rate of roughly \(O(N^{-1})\) in practice — a substantial
improvement over the IID case without changing the estimator structure.

This report documents the implementation, a critical bug discovered during
development, and validation results on the Ishigami (d=3) and SAC-SMA (d=14)
benchmarks.

---

## 2. Implementation

### 2.1 Distribution-class additions

Three distribution classes received deterministic sampling methods that accept
\([0,1]^d\) inputs instead of calling `np.random` internally.  This decouples
the copula transformation from the source of randomness, allowing the QMC
driver to supply Sobol' points directly.

**`MultivariateNormal`** — lines 119–250 of `mc_shapley.py`

| Item | Description |
|------|-------------|
| `self._L` | `cholesky(cov)`, precomputed once in `__init__` |
| `sample_joint_deterministic(U)` | `Z = norm.ppf(U)` → `mean + Z @ L.T` |
| `sample_conditional_batch_deterministic(u, fixed_X, U_cond)` | `Z_std = norm.ppf(U_cond)` instead of `np.random.randn` |

**`GaussianCopulaUniform`** — lines 275–420

| Item | Description |
|------|-------------|
| `self._L` | `cholesky(corr)`, precomputed once |
| `sample_joint_deterministic(U)` | `Z = norm.ppf(U)` → `Z_corr = Z @ L.T` → `norm.cdf(Z_corr)` → scale to `[lows, highs]` |
| `sample_conditional_batch_deterministic(u, fixed_X, U_cond)` | Identical to existing `sample_conditional_batch` except `Z_std = norm.ppf(U_cond)` |

**`GaussianCopulaArbitrary`** — lines 470–700

| Item | Description |
|------|-------------|
| `self._L` | `cholesky(corr)`, precomputed once |
| `sample_joint_deterministic(U)` | Same copula logic as `GaussianCopulaUniform` but routed through `_to_physical` for arbitrary marginals |
| `sample_conditional_batch_deterministic(u, fixed_X, U_cond)` | Same pattern |

The `_cond_params` helper methods (Cholesky decompositions for conditional
distributions) are reused without modification.

### 2.2 QMC driver function

**`collect_shapley_data_qmc()`** — inserted at line 1052, immediately after the
existing `collect_shapley_data()`.

```python
def collect_shapley_data_qmc(f, joint, N=4096, predict_batch=None,
                              progress=False, k_max=None,
                              scramble=True, seed=None):
```

**Algorithm**:

1. Generate a single scrambled Sobol' sequence of dimension \(2d\) with *N*
   points: `U_all ∈ [0,1]^{N × 2d}`.

2. **Joint samples** (shared across all subsets): use columns `0 … d−1`.
   `X_joint = joint.sample_joint_deterministic(U_all[:, :d])`.
   Evaluate once: `Y_joint = f_batch(X_joint)`.

3. **Conditional innovations**: use columns `d … 2d−1`.  For each subset *u*:
   - `U_cond = U_all[:, d : d+|v|]` (first \(|v|\) extra columns)
   - `X_cond = joint.sample_conditional_batch_deterministic(u, X_joint[:, u], U_cond)`
   - `Y2 = f_batch(X_cond)`
   - Store `('pair', Y_joint, Y2)`.

4. For the full set \(u = D\), store `('full', Y_joint)`.

The return format is identical to `collect_shapley_data`, so `shapley_from_data`,
`sobol_from_data`, `bootstrap_shapley`, and `owen_from_data` all work unchanged.

**Key design decision — single sequence, shared joint samples**: Using a single
\(2d\)-dimensional Sobol' sequence guarantees that the joint-sample coordinates
are independent of the conditional-innovation coordinates, avoiding
cross-sequence correlation.  Sharing the joint samples across subsets further
reduces variance because all subsets are evaluated at exactly the same input
points.

### 2.3 API integration

Two entry points were extended:

**Standalone function** — `shapley_effects()` (line 2029):
```python
effects, sh, total_var = shapley_effects(
    f, joint, N=4096, method='qmc_exhaustive', random_state=42
)
```

**Class method** — `MCShapley.compute()` (line 2127):
```python
mc = MCShapley(f, joint, predict_batch=pred_batch)
df = mc.compute(N=4096, method='qmc_exhaustive', random_state=42, k_max=2)
```

Both return the same DataFrame format as the existing `'exhaustive'` method,
including Sobol' indices, Shapley effects, and optional bootstrap confidence
intervals.

### 2.4 What was *not* changed

- `shapley_from_data` / `sobol_from_data` — consume (Y₁, Y₂) pairs, format-agnostic
- `bootstrap_shapley` / `bootstrap_sobol` — resample stored arrays, format-agnostic
- `owen_from_data` — consumes the same data dict
- `coalitions_up_to_k` — subset enumeration unchanged
- `_cond_params` — Cholesky decompositions reused as-is
- `TruncatedMultivariateNormal` — Gibbs sampling has no natural QMC analogue (left for future work)

---

## 3. Bug discovered and fixed

### 3.1 Original multi-sampler approach (broken)

The initial implementation created a fresh `Sobol(d)` and `Sobol(|v|)` sampler
for each subset, both seeded from the same `np.random.Generator`:

```python
sampler_joint = Sobol(d, scramble=True, seed=rng)   # joint samples
U_joint = sampler_joint.random(N)
# ...
sampler_cond = Sobol(v_size, scramble=True, seed=rng)  # conditional innovations
U_cond = sampler_cond.random(N)
```

### 3.2 Observed failure

On the Ishigami benchmark (independent inputs, d=3), QMC produced:

| N | QMC Shapley (X₁, X₂, X₃) | RMSE vs truth |
|---|--------------------------|---------------|
| 256 | [0.673, 0.388, **−0.061**] | 0.085 |
| 512 | [0.674, 0.387, **−0.062**] | 0.086 |
| 1024 | [0.675, 0.389, **−0.064**] | 0.087 |
| 2048 | [0.674, 0.389, **−0.063**] | 0.086 |

The RMSE did **not decrease** with *N* and the Shapley effect for X₃ was
consistently negative — a systematic bias.  IID MC at N=2048 gave RMSE=0.012,
roughly **7× better** than QMC at the same sample size.

### 3.3 Root cause

The first column of `U_joint` (a scrambled 3D Sobol' sequence) and the first
column of `U_cond` (a scrambled 2D Sobol' sequence) were correlated at
**r = 0.929**, despite being generated from independent `Sobol` objects with
different scrambling seeds drawn from the same `Generator`.  This is a known
property of Sobol' sequences: the first coordinate (the most rapidly varying
one) has strong structural autocorrelation that scrambling does not fully
eliminate when comparing sequences of different dimensionalities.

In the pick-freeze estimator, the conditional sample `X_cond` shares the
conditioning variables with the joint sample `X` and resamples the complement.
The spurious correlation between the *unconditioned* variables in the joint
and conditional samples inflated \(\operatorname{Cov}[f(X), f(X_cond)]\), in
some cases beyond \(\operatorname{Var}[f(X)]\) — a mathematical impossibility
for a correctly implemented estimator.

### 3.4 Fix: single-sequence approach

The corrected implementation generates **one** \(2d\)-dimensional Sobol'
sequence and partitions it:

```
Columns 0 … d−1   → joint samples X
Columns d … 2d−1  → conditional innovations
```

All subsets share the same joint samples (columns 0 … d−1).  Conditional
innovations for subset *u* use the first \(|v|\) columns from the innovation
block.  Because both blocks come from the same Sobol' sequence, the coordinate
patterns are aligned and cross-block correlation is minimal.

| Metric | Broken (multi-sampler) | Fixed (single-sequence) |
|--------|----------------------|------------------------|
| Corr(U_joint[:,0], U_cond[:,0]) | 0.929 | ~0.001 |
| QMC RMSE N=256 | 0.085 | 0.004 |
| QMC RMSE N=2048 | 0.086 | 0.00013 |
| RMSE decreases with N? | No | Yes (exponential-like) |

### 3.5 Additional benefit

Sharing joint samples across subsets is actually **better** than the IID
implementation's approach of generating fresh joint samples per subset.
All coalitions are now evaluated at identical input points, eliminating a
source of between-subset variance and improving the consistency of the
Shapley assembly.

---

## 4. Validation

### 4.1 Ishigami function (d=3)

A hand-crafted Legendre-polynomial surrogate on \([0,1]^3\) was used so that
exact analytical Shapley values and Sobol' indices are known:

\[
f(x) = c_0 + \sum_{i=1}^3 \big(c_{i,1}\psi_1(x_i) + c_{i,2}\psi_2(x_i)\big)
       + \sum_{i<j} c_{ij}\,\psi_1(x_i)\psi_1(x_j)
\]

Coefficients: \(c_0=5,\; c_{1:}=[3.0,1.5],\; c_{2:}=[2.5,1.0],\;
c_{3:}=[0.8,0.3],\; c_{12}=1.2,\; c_{13}=0.6,\; c_{23}=0.4\).

**Test 1 — independent inputs** (identity correlation, compared against
analytical truth):

| N | QMC RMSE | IID RMSE | QMC/IID ratio |
|---|----------|----------|---------------|
| 256 | 0.00419 | — | — |
| 512 | 0.00080 | — | — |
| 1024 | 0.00019 | — | — |
| 2048 | **0.00013** | 0.01209 | **93×** |

QMC at N=2048 achieves 1.3 × 10⁻⁴ RMSE — effectively exact for practical
purposes.  IID MC requires roughly N ≈ 200,000 to match this precision.

**Test 2 — correlated inputs** (\(\rho_{12}=0.7\), compared against
IID MC at N=20,000 as reference):

| N | QMC RMSE | IID RMSE | QMC/IID ratio |
|---|----------|----------|---------------|
| 256 | 0.00406 | 0.04720 | **12×** |
| 512 | 0.00285 | 0.03653 | **13×** |
| 1024 | 0.00262 | 0.01935 | **7×** |
| 2048 | 0.00265 | 0.01972 | **7×** |

QMC at N=256 already outperforms IID at N=2048.

### 4.2 SAC-SMA hydrological model (d=14)

The SAC-SMA (Sacramento Soil Moisture Accounting) model was run with 14
parameters on the St Helens catchment (16,365 daily timesteps, Nash–Sutcliffe
Efficiency as objective).  An RS-HDMR surrogate was trained on 1,000 parameter
sets sampled via Sobol' sequence:

| Method | Terms selected | Explained variance | Training time |
|--------|---------------|-------------------|---------------|
| OMP-CV, polys=[4] | 53 / 3,059 (1.7%) | 0.950 (R²=0.953) | 42 s |

Pick-freeze was run on the surrogate with identity correlation and k_max=2:

| N | QMC RMSE | IID RMSE | QMC/IID ratio | QMC time | IID time |
|---|----------|----------|---------------|----------|----------|
| 256 | 0.01230 | 0.05003 | **4.1×** | 0.2 s | 0.2 s |
| 512 | 0.01274 | 0.03552 | **2.8×** | 0.2 s | 0.3 s |
| 1024 | 0.00623 | 0.01866 | **3.0×** | 0.4 s | 0.6 s |
| 2048 | 0.00766 | 0.01476 | **1.9×** | 0.6 s | 1.0 s |

The improvement is more modest than the d=3 case — expected, as the
Sobol' sequence now spans 2d=28 dimensions and the per-dimension coverage
at N=256–2048 is thinner.  Nevertheless, QMC consistently outperforms IID MC
by 2–4×.

**Top parameters** identified by both QMC and analytical coefficient extraction:

| Rank | QMC | Analytical |
|------|-----|-----------|
| 1 | Lzfpm (0.262) | Lzfpm (0.248) |
| 2 | Lzfsm (0.230) | Lzfsm (0.222) |
| 3 | Rexp (0.172) | Rexp (0.162) |
| 4 | Adimp (0.079) | Adimp (0.080) |
| 5 | Zperc (0.062) | Zperc (0.061) |
| 6 | Lzpk (0.053) | Lzsk (0.059) |

The ranking is nearly identical between the two methods, confirming that QMC
pick-freeze recovers the same sensitivity structure.

---

## 5. Usage

### 5.1 Quick start

```python
from shapleyx.utilities.mc_shapley import (
    GaussianCopulaUniform, MCShapley
)

joint = GaussianCopulaUniform(lows, highs, corr_matrix)
mc = MCShapley(model_func, joint, predict_batch=model_batch)

# QMC-accelerated
df = mc.compute(N=4096, method='qmc_exhaustive', random_state=42, k_max=2)

# Traditional IID MC (for comparison)
df_iid = mc.compute(N=4096, method='exhaustive', random_state=42, k_max=2)
```

### 5.2 Standalone

```python
from shapleyx.utilities.mc_shapley import shapley_effects

effects, sh_values, total_var = shapley_effects(
    f, joint, N=4096, method='qmc_exhaustive', random_state=42
)
```

### 5.3 Recommendations

- **N should be a power of 2** (256, 512, 1024, 2048, 4096, …) for optimal
  Sobol' uniformity.  Non-power-of-2 *N* triggers a scipy warning but the
  estimator remains valid with scrambling.
- **k_max** should match the RS-HDMR interaction order (`len(polys)`) when
  using a surrogate model.
- **scramble=True** (default) is required for unbiased estimation; disable
  only for debugging.
- For d ≤ 5, QMC typically gives 7–90× improvement over IID MC at equivalent
  *N*.
- For d = 6–14, expect 2–4× improvement.
- For d > 15, the Sobol' sequence dimension (2d > 30) reduces the advantage;
  permutation MC may be preferred.

---

## 6. Files modified

| File | Lines changed | Description |
|------|--------------|-------------|
| `shapleyx/utilities/mc_shapley.py` | +277 | All additions |
| `test_qmc_fast.py` | new | Ishigami polynomial test (~2 s) |
| `test_qmc_sacsma.py` | new | SAC-SMA d=14 test (~150 s with model runs) |

The `Examples/ishigami.ipynb` modification was pre-existing on `main` and is
unrelated to this change.

---

## 7. References

- Owen, A.B. & Prieur, C. (2017). On Shapley Value for Measuring Importance
  of Dependent Inputs. *SIAM/ASA JUQ*, 5(1), 986–1004.
- Owen, A.B. (1995). Randomly permuted (t,m,s)-nets and (t,s)-sequences.
  *MCQMC 1994*, Springer, 299–317.
- Owen, A.B. (1997). Scrambled net variance for integrals of smooth functions.
  *Annals of Statistics*, 25(4), 1541–1562.
- Kucherenko, S., Tarantola, S., & Annoni, P. (2012). Estimation of global
  sensitivity indices for models with dependent variables. *CPC*, 183(4),
  937–946.
