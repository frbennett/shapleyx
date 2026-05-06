# RS-HDMR Construction and the Coefficient-to-Index Mapping

This page explains how ShapleyX constructs a Random Sampling – High Dimensional Model
Representation (RS-HDMR) surrogate and how the fitted Legendre coefficients directly
yield Sobol indices and Shapley effects without additional Monte Carlo sampling.

---

## The RS-HDMR Pipeline

ShapleyX follows a four-stage pipeline from raw data to sensitivity indices:

```
Raw data  →  [0,1]ᵈ scaling  →  Legendre basis  →  ARD regression  →  Indices
```

### Stage 1: Scaling to the Unit Hypercube

All input variables are min-max scaled to $[0,1]^d$. This is required because the
shifted Legendre polynomials $\phi_n(x) = \sqrt{2n+1} \cdot P_n^*(x)$ (where $P_n^*$
is the standard shifted Legendre polynomial) are orthonormal on $[0,1]$:

$$\int_0^1 \phi_m(x)\,\phi_n(x)\,dx = \delta_{mn}$$

Without this scaling, the orthonormality property — and therefore the clean
coefficient-to-variance relationship — would not hold.

### Stage 2: Basis Function Construction

The `polys` parameter controls which basis functions are generated. There are two modes:

**RS-HDMR mode** (`len(polys) > 1`): Each list entry sets the maximum Legendre degree
for one interaction order. For `polys=[8, 6, 4]`:

| HDMR Order | Max Degree per Variable | Example Terms | Terms per Subset |
|-----------|------------------------|---------------|-----------------|
| 1st | 8 | $x_i^1, x_i^2, \ldots, x_i^8$ | 8 |
| 2nd | 6 | $x_i^a x_j^b$ with $a,b \in \{1,\ldots,6\}$ | $6^2 = 36$ |
| 3rd | 4 | $x_i^a x_j^b x_k^c$ with $a,b,c \in \{1,\ldots,4\}$ | $4^3 = 64$ |

The degree constraints are **per-variable** (not total-degree). For second-order terms
with `polys=[8, 6, 4]`, the maximum combined degree is $6 + 6 = 12$, not $6$.

**Full PCE mode** (`len(polys) == 1`): A polynomial chaos expansion with a
total-degree constraint. `polys=[10]` generates all multi-indices where the sum of
degrees across all inputs is $\le 10$.

!!! note "Why per-order degree decay?"

    Main effects are often complex and nonlinear (justifying high-degree polynomials),
    while higher-order interactions tend to be simpler. RS-HDMR's per-order degree
    decay spends representational power where it's most valuable.

The basis terms are named by concatenating variable names with their degrees.
For a bivariate interaction between $X_1$ (degree 3) and $X_4$ (degree 2), the
feature name is `X1_3*X4_2`. For a univariate term $X_2$ (degree 5), it's `X2_5`.

### Stage 3: Sparse Regression

The design matrix $X_{T,L}$ (samples $\times$ basis functions) is fitted to the
model output $Y$ using either:

- **ARD** (`method='ard'`): Automatic Relevance Determination via Sparse Bayesian
  Learning. Uses evidence maximisation to prune irrelevant terms.
- **OMP** (`method='omp'`): Orthogonal Matching Pursuit. A greedier alternative
  that can be faster on very large basis sets.

The result is a sparse coefficient vector $\beta$, where most entries are exactly
zero. The surrogate prediction is:

$$\hat{Y} = X_{T,L}\,\beta$$

### Stage 4: Fit Quality

The explained variance score ($R^2$) is reported. For a good surrogate fit, this
should be close to 1.0. If it isn't, consider increasing the `polys` values or the
number of training samples.

---

## From Coefficients to Sobol Indices

Because the shifted Legendre basis is orthonormal on $[0,1]^d$, the variance
contributed by each basis term is simply the square of its coefficient. This is
the central insight that makes the mapping from coefficients to indices trivial.

### Deriving Variable Labels

Each basis term's feature name is parsed to determine which variables participate:

- `X1_3*X4_2` → variables $X_1$ and $X_4$ → derived label `X1_X4`
- `X2_5` → variable $X_2$ only → derived label `X2`

!!! note "Column naming"

    Labels are **not** alphabetically sorted. They follow the order in which
    variables appear in the initial `combinations()` call from `itertools`,
    which is lexicographic by input position.

### First-Order (Closed) Sobol Indices

For each derived label (variable subset), sum the squared coefficients of all
basis terms that map to that label. Then normalise:

$$S_u = \frac{\sum_{\text{terms} \in u} \beta_{\text{term}}^2}{\sum_{\text{all terms}} \beta_{\text{term}}^2} \times R^2$$

where $R^2$ is the explained variance score. The multiplication by $R^2$ ensures
that indices represent fractions of **total** output variance — any unexplained
variance is not attributed to any input.

- $S_{\{i\}}$ is the first-order Sobol index for variable $i$ alone
- $S_{\{i,j\}}$ captures the pure $i$-$j$ interaction plus any higher-order
  effects that happen to be labeled with that subset

### Total Sobol Indices

For each variable $i$, the total index sums the closed indices of **all** subsets
that contain $i$:

$$S_i^T = \sum_{u \ni i} S_u$$

Under independent inputs, this equals the classical total Sobol index: the fraction
of variance that would remain unexplained if all inputs *except* $i$ were known.

---

## From Coefficients to Shapley Effects

Shapley effects address a limitation of classical Sobol indices: they are not
interpretable under correlated inputs. ShapleyX computes surrogate-based Shapley
effects directly from the fitted coefficients, without further Monte Carlo sampling.

### The Shapley Allocation Rule

A basis term involving $k$ variables contributes variance equal to its squared
coefficient. The Shapley allocation divides this variance **equally** among the
$k$ participating variables:

$$\phi_i = \sum_{\text{terms containing } i} \frac{\beta_{\text{term}}^2}{k_{\text{term}}}$$

where $k_{\text{term}}$ is the number of variables in that term.

### Normalisation

The raw effects are normalised to sum to 1:

$$\tilde{\phi}_i = \frac{\phi_i}{\sum_{j=1}^d \phi_j}$$

This ensures $\sum_i \tilde{\phi}_i = 1$, satisfying the Shapley efficiency axiom.

### Why This Works

Under independent inputs, the orthonormal Legendre basis yields an exact
variance decomposition. Each basis term's variance contribution is unambiguously
$\beta_{\text{term}}^2$, and the equal division among participants is the unique
allocation satisfying the Shapley axioms (efficiency, symmetry, dummy, additivity).

Under correlated inputs, the surrogate coefficients absorb the correlation structure
during fitting (they are estimated from jointly sampled training data). The Shapley
allocation then distributes the shared variance fairly — a variable that co-varies
with others receives a larger share because it appears in more interaction terms
with non-zero coefficients.

---

## Complete Example

```python
from shapleyx import rshdmr

# Train surrogate and compute all indices
analyzer = rshdmr(
    data_file='my_model_samples.csv',
    polys=[8, 6, 4],
    method='ard',
    resampling=True,
    number_of_resamples=500
)

sobol, shapley, total = analyzer.run_all()

# Inspect results
print(sobol.head())       # First-order (closed) Sobol indices
print(shapley)            # Shapley effects with 95% CIs
print(total)              # Total Sobol indices
```

If you want only the surrogate (no Shapley MC sampling), the fitted coefficients
are available as `analyzer.coef_` and the derived labels in
`analyzer.non_zero_coefficients`.

---

## Summary of the Mapping

| Coefficient Property | Sensitivity Index |
|---------------------|-------------------|
| $\beta_{\text{term}}^2$ | Variance contributed by that term |
| Group by derived label, sum $\beta^2$ | Closed Sobol index $S_u$ |
| Sum all $S_u$ containing variable $i$ | Total Sobol index $S_i^T$ |
| For each term, divide $\beta^2$ by $k$, allocate to each variable | Shapley effect $\phi_i$ |

The entire mapping from surrogate coefficients to sensitivity indices requires
only group-by and sum operations on the fitted coefficient vector — no additional
model evaluations. This is what makes surrogate-based sensitivity analysis
computationally trivial once the surrogate is trained.

---

## Further Reading

- [Theory of Shapley Effects](theory.md) — the Shapley axioms and proofs
- [ARD Theory](ard_theory.md) — sparse Bayesian learning details
- Li, Wang & Rabitz (2002) — the original RS-HDMR paper
- Owen & Prieur (2017) — Shapley effects for dependent inputs
- [Wing Weight Case Study](../case_study_reports/wing_weight_case_study.pdf) —
  a worked example with surrogate validation
