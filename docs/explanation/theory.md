# Theoretical Background

## Shapley Values in Sensitivity Analysis

Shapley values originate from cooperative game theory and provide a principled way to:

1. Fairly distribute the "payout" (output variance) among "players" (input parameters)
2. Account for all possible interaction effects
3. Provide unique attribution under certain axioms (efficiency, symmetry, dummy, additivity)

## Shapley Effects — Mathematical Formulation

For a model output $Y = f(X_1, X_2, \ldots, X_d)$, the Shapley effect $\phi_i$ for parameter $X_i$ is:

$$
\phi_i = \sum_{S \subseteq D \setminus \{i\}} \frac{|S|!\,(d-|S|-1)!}{d!}\,
\Big[\text{Var}\big(E[Y \mid X_{S \cup \{i\}}]\big) - \text{Var}\big(E[Y \mid X_S]\big)\Big]
$$

where:

- $D = \{1, 2, \ldots, d\}$ is the set of all input parameters
- $S$ is a subset of parameters excluding $i$
- $X_S$ represents the parameters in subset $S$
- The weight $\frac{|S|!\,(d-|S|-1)!}{d!}$ ensures each coalition size is equally represented

## Relationship to Sobol Indices

Shapley effects generalise Sobol indices by combining all order effects involving a parameter:

- A Sobol index $S_u$ quantifies the contribution of subset $u$ **alone**
- A Shapley effect $\phi_i$ redistributes every interaction term equally among its participating variables

This yields a complete decomposition where:

- $\sum_{i=1}^d \phi_i = \text{Var}(Y)$ (efficiency)
- Each $\phi_i \geq 0$ (non-negativity under independent inputs)

## Advantages of Shapley Effects

1. **Complete Attribution**: Accounts for all interactions, unlike first-order Sobol indices
2. **Additivity**: Effects sum to total output variance
3. **Interpretability**: Direct measure of importance in variance units
4. **Single number per input**: Avoids the proliferation of $2^d - 1$ Sobol indices

## Implementation via RS-HDMR (Independent Inputs)

ShapleyX computes Shapley effects from a sparse polynomial surrogate model built by:

- **Polynomial chaos expansion** using shifted Legendre polynomials (orthonormal on $[0, 1]$)
- **Automatic Relevance Determination (ARD)** for sparse Bayesian regression — prunes irrelevant basis terms
- **Coefficient-based index extraction**: squared coefficients are grouped by variable labels and distributed according to the Shapley weights

Under independent inputs, the variance contributed by a basis term $\psi_u(\mathbf{x})$ is simply its squared coefficient. The Shapley effect for variable $i$ is then the sum of all coefficient-squared terms divided equally among the variables appearing in each term's label.

## Shapley Effects with Correlated Inputs

When inputs are correlated, the coefficient-based decomposition used by RS-HDMR breaks down because:

- The variance of a product term no longer factorises into independent contributions
- Conditional expectations $E[Y \mid X_S]$ depend on the joint distribution in a non-trivial way

The **Monte Carlo Shapley** method addresses this by directly estimating the conditional variance terms from samples of the joint distribution.

### The Covariance-Based Formulation

Owen & Prieur (2017) showed that the Shapley value function can be written as:

$$
v(u) = \text{Cov}\big[f(\mathbf{X}),\, f(\mathbf{X}_u)\big]
$$

where $\mathbf{X}_u$ is a random vector constructed by:

1. Drawing a joint sample $\mathbf{x} \sim P_{\mathbf{X}}$
2. Fixing the coordinates in $u$ to $\mathbf{x}_u$
3. Drawing the remaining coordinates from the conditional distribution $P(\mathbf{X}_{-u} \mid \mathbf{X}_u = \mathbf{x}_u)$

The MC Shapley algorithm estimates $v(u)$ for each subset by Monte Carlo, then assembles the Shapley effects via the standard coalition formula.

### Distribution Classes

ShapleyX provides two distribution families with built-in conditional sampling:

| Class | Description |
|---|---|
| `GaussianCopulaUniform` | Uniform marginals $[a_i, b_i]$ with dependence induced by a latent multivariate normal. Correlation matrix $\mathbf{R}$ controls dependence. |
| `MultivariateNormal` | Jointly normal with mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$. Conditional distributions are analytically tractable. |

### Computation Methods

| Method | Description | Complexity |
|---|---|---|
| `'exhaustive'` | Enumerates all $2^d - 1$ non-empty subsets. Each subset requires $N$ Monte Carlo iterations. | $O(2^d \cdot N)$ |
| `'permutation'` | Evaluates only subsets encountered in random permutations, with lazy caching. | $O(n_{\text{perm}} \cdot d \cdot N)$ |

Bootstrap confidence intervals are available for both methods by resampling the stored output arrays.

---

**Reference**: Owen, A. B., & Prieur, C. (2017). On Shapley value for measuring importance of dependent inputs. *SIAM/ASA Journal on Uncertainty Quantification*, 5(1), 986–1004.
