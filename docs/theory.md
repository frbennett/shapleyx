---
title: "Theory Background"
layout: single
classes: wide
mathjax: true
---

# Theoretical Background

## RS-HDMR Methodology

Random Sampling High-Dimensional Model Representation (RS-HDMR) is a global sensitivity analysis framework that combines:

1. **Random Sampling**: Efficient exploration of input space
2. **HDMR**: Decomposition of model output into summands of increasing dimensionality
3. **GMDH**: Group Method of Data Handling for parameter selection
4. **Regression**: Parameter refinement through linear regression

The general HDMR expansion is given by:

$$
f(\mathbf{x}) = f_0 + \sum_{i=1}^n f_i(x_i) + \sum_{1\leq i<j\leq n} f_{ij}(x_i,x_j) + \cdots + f_{1,2,\ldots,n}(x_1,x_2,\ldots,x_n)
$$

where:
- $f_0$ is the constant term
- $f_i(x_i)$ are first-order component functions
- $f_{ij}(x_i,x_j)$ are second-order component functions
- Higher-order terms capture interactions

## Legendre Polynomial Expansion

The method uses shifted Legendre polynomials as basis functions:

$$
P_n^*(x) = \sqrt{2n+1} \cdot P_n(2x-1)
$$

where $P_n$ is the standard Legendre polynomial. These form an orthonormal basis on [0,1].

## Sensitivity Indices

### Sobol Indices
First-order Sobol index for variable $x_i$:

$$
S_i = \frac{\text{Var}_{x_i}(E_{\sim x_i}[f|x_i])}{\text{Var}(f)}
$$

Total effect index:

$$
S_{Ti} = 1 - \frac{\text{Var}_{\sim x_i}(E_{x_i}[f|\sim x_i])}{\text{Var}(f)}
$$

### Shapley Effects
Shapley value for variable $x_i$:

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} [v(S \cup \{i\}) - v(S)]
$$

where $v(S)$ is the variance contribution of subset $S$.

### PAWN Indices
PAWN sensitivity index based on Kolmogorov-Smirnov statistic:

$$
P_i = 1 - \max_t |F_{Y|X_i}(t) - F_Y(t)|
$$

## Algorithm Overview

1. **Input Transformation**: Map inputs to unit hypercube
2. **Basis Expansion**: Create Legendre polynomial terms
3. **Regression**: Fit model using ARD/OMP/Lasso
4. **Index Calculation**: Compute sensitivity indices
5. **Resampling**: Estimate confidence intervals via bootstrap

## Mathematical Foundations

### Orthogonal Polynomials
The method leverages the orthogonality property:

$$
\int_0^1 P_n^*(x)P_m^*(x)dx = \delta_{nm}
$$

### Sparse Regression
Automatic Relevance Determination (ARD) prior:

$$
p(\mathbf{w}|\boldsymbol{\alpha}) = \prod_{i=1}^N \mathcal{N}(w_i|0,\alpha_i^{-1})
$$

### Model Selection
GMDH uses iterative polynomial network construction with:

1. Generation of candidate models
2. Evaluation using external criteria
3. Selection of best-performing models
4. Iterative refinement

## References

1. Sobol, I. M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates.
2. Owen, A. B. (2014). Sobol' indices and Shapley value.
3. Pianosi, F., et al. (2016). Sensitivity analysis of environmental models: A systematic review with practical workflow.
4. Saltelli, A., et al. (2008). Global Sensitivity Analysis: The Primer.
