# ShapleyX — Global Sensitivity Analysis with RS-HDMR

[![PyPI](https://img.shields.io/pypi/v/shapleyx)](https://pypi.org/project/shapleyx/)
[![Python](https://img.shields.io/pypi/pyversions/shapleyx)](https://pypi.org/project/shapleyx/)
[![License](https://img.shields.io/github/license/frbennett/shapleyx)](https://github.com/frbennett/shapleyx/blob/main/LICENSE)

ShapleyX is a Python package for global sensitivity analysis using
Random Sampling High-Dimensional Model Representation (RS-HDMR).  It
builds sparse polynomial surrogate models via Automatic Relevance
Determination (ARD) regression and extracts Sobol indices, Shapley
effects, and moment-free sensitivity measures — with full support for
**correlated inputs** through Monte Carlo Shapley estimation.

## Installation

```bash
pip install shapleyx
```

To upgrade:

```bash
pip install --upgrade shapleyx
```

### From GitHub (development version)

```bash
pip install https://github.com/frbennett/shapleyx/archive/main.zip
```

## Features

- **Variance-based indices**:
  - Sobol sensitivity indices to arbitrary order
  - Shapley effects (independent inputs, via coefficient decomposition)
  - **Monte Carlo Shapley effects** for correlated inputs (Owen & Prieur 2017)
  - Owen-Shapley interaction indices
  - Total sensitivity indices

- **Moment-free measures**:
  - PAWN (density-based)
  - Delta index (moment-independent)
  - H-index (distribution-based)

- **Distribution classes for correlated inputs**:
  - `GaussianCopulaUniform` — uniform marginals with latent normal dependence
  - `MultivariateNormal` — jointly normal with analytical conditional sampling
  - `TruncatedMultivariateNormal` — per-dimension truncation bounds with Gibbs sampling

- **Computation methods**:
  - Exhaustive subset enumeration (exact, for $d \le 8$)
  - Random permutation method (scalable, for larger $d$)

- **Infrastructure**:
  - Legendre polynomial expansion on $[0,1]^d$
  - Automatic Relevance Determination (ARD) with Bayesian cross-validation
  - Bootstrap resampling for confidence intervals
  - Progress bars (via tqdm, optional) and Numba-accelerated bootstrap (optional)

## Dependencies

| Required | Optional |
|---|---|
| `numpy` | `numba` (compiled bootstrap & Legendre evals) |
| `scipy` | `tqdm` (progress bars) |
| `pandas` | |
| `matplotlib` | |
| `scikit-learn` | |

## Documentation

Full documentation is available at:
- [Documentation Home](https://frbennett.github.io/shapleyx/)
- [Quick Start Guide](https://frbennett.github.io/shapleyx/getting-started/quickstart/)
- [MC Shapley How-to](https://frbennett.github.io/shapleyx/how-to-guides/mc-shapley/)
- [Theory Background](https://frbennett.github.io/shapleyx/explanation/theory/)
- [API Reference](https://frbennett.github.io/shapleyx/reference/api/)

## Examples

Jupyter notebooks demonstrating usage:

| Notebook | Description |
|---|---|
| [Ishigami Function](https://github.com/frbennett/shapleyx/blob/main/Examples/ishigami.ipynb) | Basic RS-HDMR workflow with Sobol & Shapley |
| [MC Shapley](https://github.com/frbennett/shapleyx/blob/main/Examples/mc_shapley.ipynb) | Monte Carlo Shapley for correlated inputs |
| [Truncated Normal](https://github.com/frbennett/shapleyx/blob/main/Examples/mc_shapley_truncated_normal.ipynb) | `TruncatedMultivariateNormal` with the Ishigami function |
| [Owen Product Function](https://github.com/frbennett/shapleyx/blob/main/Examples/owen_product_function.ipynb) | Higher-dimensional example workflow |

## License

This project is licensed under the MIT License — see the [LICENSE](https://github.com/frbennett/shapleyx/blob/main/LICENSE) file for details.
