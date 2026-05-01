# ShapleyX — Global Sensitivity Analysis with RS-HDMR

ShapleyX is a Python package for global sensitivity analysis using
Random Sampling High-Dimensional Model Representation (RS-HDMR).  It
builds sparse polynomial surrogate models via Automatic Relevance
Determination (ARD) regression and extracts Sobol indices, Shapley
effects, and moment-free sensitivity measures — with full support for
**correlated inputs** through Monte Carlo Shapley estimation.

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

## Installation

### From GitHub (latest version)
```bash
pip install https://github.com/frbennett/shapleyx/archive/main.zip
```

To upgrade:
```bash
pip uninstall -y shapleyx
pip install https://github.com/frbennett/shapleyx/archive/main.zip
```

### Development Installation
```bash
git clone https://github.com/frbennett/shapleyx.git
cd shapleyx
python setup.py develop
```

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
- [Document Home](https://frbennett.github.io/shapleyx/)
- [Quick Start Guide](https://frbennett.github.io/shapleyx/getting-started/quickstart/)
- [MC Shapley How-to](https://frbennett.github.io/shapleyx/how-to-guides/mc-shapley/)
- [Theory Background](https://frbennett.github.io/shapleyx/explanation/theory/)
- [API Reference](https://frbennett.github.io/shapleyx/reference/api/)

## Examples

Jupyter notebooks demonstrating usage:

| Notebook | Description |
|---|---|
| [Ishigami Function](Examples/ishigami.ipynb) | Basic RS-HDMR workflow with Sobol & Shapley |
| [MC Shapley](Examples/mc_shapley.ipynb) | Monte Carlo Shapley for correlated inputs |
| [Truncated Normal](Examples/mc_shapley_truncated_normal.ipynb) | `TruncatedMultivariateNormal` with the Ishigami function |
| [Owen Product Function](Examples/owen_product_function.ipynb) | Higher-dimensional example workflow |

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
