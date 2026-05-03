# Example Notebooks

This directory contains Jupyter notebooks demonstrating the capabilities of
ShapleyX.  Each notebook is self-contained and can be run independently.

## Inventory

| # | Notebook | $d$ | Description | Key Features |
|---|---|---|---|---|
| 1 | [`ishigami.ipynb`](ishigami.ipynb) | 3 | **Basic RS-HDMR workflow.**  Sobol, Shapley, total indices, and moment-free measures (PAWN, Delta, H-index) for the Ishigami benchmark function. | First-time users start here.  Covers `rshdmr.run_all()`. |
| 2 | [`ishigami_new_legendre.ipynb`](ishigami_new_legendre.ipynb) | 3 | **Legendre expansion strategies.**  Compares PCE (`polys=[10]`), HDMR-2 (`polys=[10,5]`), and HDMR-3 (`polys=[8,6,4]`) on the Ishigami function. | Basis-set size analysis; strategy selection guide. |
| 3 | [`mc_shapley.ipynb`](mc_shapley.ipynb) | 3 | **MC Shapley for correlated inputs.**  Four usage patterns: surrogate with independent/correlated inputs, user-defined function, and multivariate normal distribution. | Primary MC Shapley tutorial.  Covers `get_mc_shapley()`. |
| 4 | [`mc_shapley_sobol.ipynb`](mc_shapley_sobol.ipynb) | 3 | **Shapley + Sobol from a single run.**  Demonstrates that `MCShapley.compute()` returns `sobol_first` and `sobol_total` columns alongside Shapley effects.  Explains the $S_i > T_i$ phenomenon under correlation. | $v(u) = \text{Cov}[f(\mathbf{X}), f(\mathbf{X}_u)] = \mathbb{V}[\mathbb{E}(f \mid \mathbf{X}_u)]$ |
| 5 | [`mc_shapley_truncated_normal.ipynb`](mc_shapley_truncated_normal.ipynb) | 3 | **TruncatedMultivariateNormal distribution.**  Four truncation schemes (independent, correlated, tight, asymmetric) for the Ishigami function.  Includes RS-HDMR surrogate comparison. | Gibbs sampling, `joint_burn_in` / `cond_burn_in` tuning. |
| 6 | [`mc_shapley_kmax.ipynb`](mc_shapley_kmax.ipynb) | 6 | **Coalition truncation ($k_{\max}$).**  Demonstrates subset reduction (63→42→22→7) for the Owen product function.  Compares full enumeration vs auto-detected vs explicit $k_{\max}$. | $k_{\max}$ auto-detection from `polys`; scaling plot. |
| 7 | [`mc_shapley_kmax_gfunction.ipynb`](mc_shapley_kmax_gfunction.ipynb) | 10 | **Coalition truncation at scale.**  Sobol' G-function at $d=10$ — full (1,023 subsets) vs $k_{\max}=3$ (176) vs $k_{\max}=2$ (56).  Analytical Sobol' indices for validation. | 18× reduction at $k_{\max}=2$; timing comparison. |
| 8 | [`iooss_prieur_ishigami_correlation.ipynb`](iooss_prieur_ishigami_correlation.ipynb) | 3 | **Correlation sweep (Iooss & Prieur 2019).**  Shapley effects and Sobol indices as a function of $\rho$ between $X_1$ and $X_3$.  Exhaustive vs permutation comparison. | 96% CI bands; 3-panel $S_i$/Shapley/$T_i$ plot. |
| 9 | [`owen_product_function.ipynb`](owen_product_function.ipynb) | 6 | **Owen product function.**  High-order interaction benchmark with analytical Sobol' indices.  RS-HDMR surrogate vs exact Shapley effects. | Validation against closed-form values. |
| 10 | [`cantilever_beam.ipynb`](cantilever_beam.ipynb) | 6 | **Cantilever beam (Demange-Chryst 2022, Ex. 4.2).**  Mixed LogNormal + Normal marginals with correlations.  Variance-based and target Shapley effects.  RS-HDMR surrogate comparison. | `GaussianCopulaMixed` custom distribution class. |
| 11 | [`borehole.ipynb`](borehole.ipynb) | 8 | **Borehole function (Harper & Gupta 1983).**  Normal + LogNormal + Uniform marginals.  MC Shapley + Sobol from single run.  Correlated vs independent surrogate comparison; literature validation (Saltelli 2004). | `GaussianCopulaFull` custom distribution class. |
| 12 | [`fire_spread.ipynb`](fire_spread.ipynb) | 10 | **Rothermel fire spread model (Demange-Chryst 2022, Ex. 4.3).**  Scaled LogNormal marginals, physical constraints via rejection sampling, Gaussian copula with correlation.  Permutation method at $d=10$. | `ConstrainedGaussianCopula` custom distribution class. |

## Learning Path

| Step | Notebook(s) | What you'll learn |
|---|---|---|
| 1 | `ishigami.ipynb` | Train an RS-HDMR surrogate, extract Sobol & Shapley indices |
| 2 | `mc_shapley.ipynb` | MC Shapley for correlated inputs via `get_mc_shapley()` |
| 3 | `mc_shapley_sobol.ipynb` | Shapley + Sobol indices from a single data collection |
| 4 | `mc_shapley_kmax.ipynb` | Speed up exhaustive MC Shapley with coalition truncation |
| 5 | `iooss_prieur_ishigami_correlation.ipynb` | How Shapley effects evolve with correlation |
| 6 | `borehole.ipynb` | Real-world benchmark with mixed distributions and literature validation |

## Distribution Classes Used

| Notebook | Distribution Class | Key Feature |
|---|---|---|
| Most notebooks | `MultivariateNormal` / `GaussianCopulaUniform` | Built-in |
| `mc_shapley_truncated_normal.ipynb` | `TruncatedMultivariateNormal` | Built-in (Gibbs sampling) |
| `cantilever_beam.ipynb` | `GaussianCopulaMixed` | Custom (Normal + LogNormal) |
| `borehole.ipynb` | `GaussianCopulaFull` | Custom (Normal + LogNormal + Uniform) |
| `fire_spread.ipynb` | `ConstrainedGaussianCopula` | Custom (rejection sampling) |
