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

### MC Sobol Indices as a By-Product

The same $v(u)$ values equal the **closed Sobol indices**:

$$v(u) = \text{Cov}[f(\mathbf{X}), f(\mathbf{X}_u)] = \mathbb{V}[\mathbb{E}(f(\mathbf{X}) \mid \mathbf{X}_u)]$$

This is a direct consequence of the Owen & Prieur (2017) formulation — see the
[proof in their paper](https://arxiv.org/abs/1704.06942).  From $v(u)$ we obtain
both first-order and total-order Sobol indices without additional model evaluations:

$$S_i = \frac{v(\{i\})}{v(\text{full})}, \qquad
T_i = 1 - \frac{v(\text{all}\setminus\{i\})}{v(\text{full})}$$

where $v(\text{full}) = \mathbb{V}(f(\mathbf{X}))$ is the total output variance.

The `MCShapley.compute()` method returns `sobol_first` and `sobol_total` columns
alongside the Shapley effects, with bootstrap confidence intervals available
via `bootstrap_sobol()`.  This gives practitioners three complementary perspectives
from a single data collection: $S_i$ (main effect), Shapley (fair attribution
including interactions), and $T_i$ (total effect).

**Important:** Under correlated inputs, the standard inequality $S_i \leq T_i$ can
break down — correlation inflates $S_i$ (shared information through dependence) and
deflates $T_i$ (other variables proxy for $X_i$).  The Shapley effect remains
axiomatically interpretable regardless of the correlation structure.

### Distribution Classes

ShapleyX provides two distribution families with built-in conditional sampling:

| Class | Description |
|---|---|
| `GaussianCopulaUniform` | Uniform marginals $[a_i, b_i]$ with dependence induced by a latent multivariate normal. Correlation matrix $\mathbf{R}$ controls dependence. |
| `MultivariateNormal` | Jointly normal with mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$. Conditional distributions are analytically tractable. |
| `TruncatedMultivariateNormal` | Jointly normal with per-dimension truncation bounds $[a_i, b_i]$. Both joint and conditional sampling use Gibbs sampling, making it suitable for any hyper-rectangular truncation region. |

### Computation Methods

| Method | Description | Complexity |
|---|---|---|
| `'exhaustive'` | Enumerates all $2^d - 1$ non-empty subsets. Each subset requires $N$ Monte Carlo iterations. | $O(2^d \cdot N)$ |
| `'permutation'` | Evaluates only subsets encountered in random permutations, with lazy caching. | $O(n_{\text{perm}} \cdot d \cdot N)$ |
| `'exhaustive'` with `k_max` | Enumerates only subsets up to size $k_{\max}$ (plus the full set). Exact when interactions are bounded at order $k_{\max}$. | $O(d^{k_{\max}} \cdot N)$ |

Bootstrap confidence intervals are available for all methods by resampling the stored output arrays.

### Coalition Truncation

The exhaustive method's $O(2^d)$ complexity limits its use to $d \leq 8$ in
practice.  However, many models — particularly RS-HDMR surrogates — have
**bounded interaction order**.  An RS-HDMR model built with `polys=[10, 5]`
contains only first-order (main effects) and second-order (pairwise)
interactions; all higher-order terms are identically zero.

When the model's HDMR decomposition has no interactions above order $k$, the
Shapley value for variable $i$ simplifies: coalitions larger than $k$ contribute
nothing beyond what is already captured by their sub-coalitions of size $\leq k$.
Formally, for any $u$ with $|u| > k$:

$$v(u) = \sum_{w \subseteq u, |w| \leq k} \sigma_w^2$$

and the Shapley difference $v(u \cup \{i\}) - v(u)$ only involves terms where
$i \in w$ and $|w| \leq k$.  The Shapley weight for coalitions of size $|u| > k$
distributes these contributions identically to how they are distributed through
the sub-coalitions already evaluated.

This justifies setting `k_max` to the highest interaction order of the
surrogate model.  ShapleyX **auto-detects** this from the `polys` parameter
($k_{\max} = \text{len}(\text{polys})$) when using `model.get_mc_shapley()`.
For general models where the interaction order is unknown, `k_max` provides a
controlled approximation that converges to the exact result as $k_{\max} \to d$.

---

**Reference**: Owen, A. B., & Prieur, C. (2017). On Shapley value for measuring importance of dependent inputs. *SIAM/ASA Journal on Uncertainty Quantification*, 5(1), 986–1004.
