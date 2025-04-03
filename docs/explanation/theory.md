# Theoretical Background

## Shapley Values in Sensitivity Analysis

Shapley values originate from cooperative game theory and provide a principled way to:

1. Fairly distribute the "payout" (output variance) among "players" (input parameters)
2. Account for all possible interaction effects
3. Provide unique attribution under certain axioms

## Mathematical Formulation
For a model output \( Y = f(X_1, X_2, \dots, X_d)\) , the Shapley effect \(\phi_i\) for parameter \(X_i\) is:

$$
\phi_i = \sum_{S \subseteq D \setminus \{i\}} \frac{|S|!(d-|S|-1)!}{d!} \left[\text{Var}\big(E[Y|X_S \cup \{i\}]\big) - \text{Var}\big(E[Y|X_S]\big)\right]
$$

where:
- \(D\) is the set of all parameters
- \(S\) is a subset of parameters excluding \(i\)
- \(X_S\) represents the parameters in subset \(S\)
For a model output $$ Y = f(X_1, X_2, \ldots, X_d) $$, the Shapley effect $$\phi_i$$ for parameter $$X_i$$ is:

$$
\phi_i = \sum_{S \subseteq D \setminus \{i\}} \frac{|S|!(d-|S|-1)!}{d!} [\text{Var}(E[Y|X_S \cup \{i\}]) - \text{Var}(E[Y|X_S])]
$$

where:
- $D$ is the set of all parameters
- $S$ is a subset of parameters excluding $i$
- $X_S$ represents the parameters in subset $S$

## Relationship to Sobol Indices

Shapley effects generalize Sobol indices by:
- Combining all order effects involving a parameter
- Providing a complete decomposition where:
  - $\sum_{i=1}^d \phi_i = \text{Var}(Y)$
  - Each $\phi_i \geq 0$

## Advantages

1. **Complete Attribution**: Accounts for all interactions
2. **Additivity**: Effects sum to total variance
3. **Interpretability**: Direct measure of importance
4. **Robustness**: Works well with correlated inputs

## Implementation in ShapleyX

The package uses:
- Polynomial chaos expansions for efficient computation
- Automatic Relevance Determination (ARD) for robust estimation
- Legendre polynomials for orthogonal basis functions