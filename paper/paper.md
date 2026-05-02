---
title: 'ShapleyX: A Python package for global sensitivity analysis with RS-HDMR'
tags:
  - sensitivity analysis
  - Python
  - Shapley effects
  - Sobol indices
  - uncertainty quantification
  - HDMR
  - correlated inputs
authors:
  - name: Frederick Bennett
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Department of Environment, Science and Innovation, Queensland Government
    index: 1
date: 2 May 2026
bibliography: paper.bib
---

# Summary

ShapleyX is a Python package for global sensitivity analysis (GSA) using
Random Sampling High-Dimensional Model Representation (RS-HDMR).  It builds
sparse polynomial surrogate models via Automatic Relevance Determination (ARD)
regression with Bayesian cross-validation and extracts a comprehensive suite of
sensitivity indices: Sobol indices to arbitrary order, Shapley effects (for
both independent and correlated inputs), Owen-Shapley interaction indices, and
moment-free measures (PAWN, Delta, H-index).

# Statement of Need

Global sensitivity analysis is essential for understanding which input
parameters drive the behaviour of complex computational models.  Traditional
variance-based methods such as Sobol indices [@Sobol2001; @Saltelli2010a]
require independent inputs and produce up to $2^d - 1$ indices for a
$d$-dimensional model, making interpretation challenging.  Shapley effects
[@Owen2017] address both limitations: they provide a single importance measure
per input that fairly distributes interaction variance, and they remain valid
under input correlation — a common scenario in engineering, environmental, and
financial models.

However, estimating Shapley effects for correlated inputs requires conditional
Monte Carlo sampling from the joint input distribution, which is computationally
demanding and requires distribution-specific sampling algorithms.  Existing
packages either lack Shapley effect estimation entirely (e.g., SALib
[@Herman2017]) or do not support correlated inputs (e.g., UQLab's
coefficient-based extraction).

ShapleyX fills this gap by providing:

- **Monte Carlo Shapley effects** for correlated inputs via the Owen--Prieur
  covariance formulation [@Owen2017], with two estimation methods (exhaustive
  subset enumeration and scalable random permutations) and bootstrap confidence
  intervals.
- **Built-in distribution classes** for Gaussian copula models, multivariate
  normal, and truncated multivariate normal inputs, with a documented interface
  for user-defined distributions.
- **First-order and total-order Sobol indices** extracted from the same Monte
  Carlo data at no additional computational cost, via the identity
  $v(u) = \text{Cov}[f(\mathbf{X}), f(\mathbf{X}_u)] = \mathbb{V}[\mathbb{E}(f(\mathbf{X}) \mid \mathbf{X}_u)]$.
- **RS-HDMR surrogate modelling** with sparse Legendre polynomial expansion
  and ARD regression, enabling accurate sensitivity analysis from a few hundred
  model evaluations.
- **Moment-free measures** (PAWN, Delta, H-index) for distribution-based
  sensitivity analysis that does not rely on variance decomposition.

# Methodology

## RS-HDMR Surrogate Construction

ShapleyX constructs a surrogate model $\hat{f}(\mathbf{x})$ using a sparse
polynomial chaos expansion with shifted Legendre polynomials (orthonormal on
$[0,1]^d$).  The basis set is defined by the `polys` parameter: e.g.,
`polys=[10, 5]` includes up to 10th-degree univariate terms and 5th-degree
bivariate interactions.  ARD regression [@Tipping2001] prunes irrelevant basis
terms via Bayesian evidence maximisation, yielding a sparse representation with
typically 10–100 active terms from thousands of candidates.

Under independent inputs, sensitivity indices are extracted directly from the
squared coefficients grouped by variable labels [@Sudret2008].  Bootstrap
resampling provides confidence intervals.

## Monte Carlo Shapley Effects

For correlated inputs, the coefficient-based extraction breaks down because
the variance of interaction terms no longer factorises.  ShapleyX implements
the Owen--Prieur [@Owen2017] estimator: for each subset $u$ of input variables,

$$v(u) = \text{Cov}[f(\mathbf{X}), f(\mathbf{X}_u)]$$

where $\mathbf{X}_u$ shares the background variables with $\mathbf{X}$ but
re-samples the complement $\mathbf{X}_{-u}$ from the conditional distribution
$P(\mathbf{X}_{-u} \mid \mathbf{X}_u)$.  The Shapley effect for variable $i$
is then:

$$\phi_i = \sum_{u \subseteq D \setminus \{i\}} \frac{|u|!\,(d-|u|-1)!}{d!}\,[v(u \cup \{i\}) - v(u)]$$

Two computation methods are provided:

- **Exhaustive**: evaluates all $2^d - 1$ non-empty subsets.  Exact but scales
  exponentially — suitable for $d \leq 8$.
- **Permutation**: evaluates only subsets encountered in random permutations
  with lazy caching [@Plischke2013].  Scales as $O(n_{\text{perm}} \cdot d \cdot N)$,
  suitable for $d > 8$.

## Sobol Indices as a By-Product

A key computational insight is that the $v(u)$ values are exactly the closed
Sobol indices: $v(u) = \mathbb{V}[\mathbb{E}(f(\mathbf{X}) \mid \mathbf{X}_u)]$.
First-order ($S_i$) and total-order ($T_i$) Sobol indices are therefore
obtained without additional model evaluations:

$$S_i = \frac{v(\{i\})}{v(\text{full})}, \qquad
T_i = 1 - \frac{v(\text{all}\setminus\{i\})}{v(\text{full})}$$

# Distribution Classes

ShapleyX defines a simple distribution class interface for plugging arbitrary
input models into the MC Shapley pipeline.  A distribution class must provide
`sample_joint(n)` (draw $n$ i.i.d. samples) and `sample_conditional_batch(u, X)`
(draw $N$ conditional samples).  Built-in classes include:

| Class | Description |
|---|---|
| `MultivariateNormal` | Jointly normal — closed-form conditional sampling via Cholesky decomposition |
| `GaussianCopulaUniform` | Uniform marginals with latent normal dependence |
| `TruncatedMultivariateNormal` | Per-dimension truncation bounds with Gibbs sampling |

The interface is designed to be extensible: the accompanying documentation and
example notebooks demonstrate custom classes for mixed Normal/LogNormal/Uniform
marginals, rejection-sampled physical constraints, and $6.9 \times \text{LogNormal}$
wind-speed distributions.

# Case Studies

The package includes several realistic case studies demonstrating the
methodology:

| Example | Dimension | Features |
|---|---|---|
| Ishigami function | 3 | Standard benchmark, correlation sweep (Iooss & Prieur 2019) |
| Cantilever beam | 6 | Mixed LogNormal + Normal marginals, target Shapley effects (Demange-Chryst 2022) |
| Borehole function | 8 | Normal + LogNormal + Uniform marginals, RS-HDMR surrogate, Sobol + Shapley |
| Rothermel fire spread | 10 | Scaled marginals, physical constraints, permutation method |
| Owen product function | 6 | High-order interactions, analytical Sobol validation |

The Ishigami and borehole examples include comparisons with published reference
values from Saltelli [-@Saltelli2004] and Iooss & Prieur [-@Iooss2019].

# Acknowledgements

The author acknowledges the foundational work of Owen & Prieur (2017) on
Shapley effects for correlated inputs, and Demange-Chryst et al. (2022) on
target Shapley estimation.

# References
