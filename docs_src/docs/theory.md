# Theory Background

## RS-HDMR Methodology

Random Sampling High-Dimensional Model Representation (RS-HDMR) decomposes a model output $f(x)$ into:

$$
f(x) = f_0 + \sum_{i=1}^n f_i(x_i) + \sum_{i<j}^n f_{ij}(x_i,x_j) + \cdots + f_{1,2,\ldots,n}(x_1,x_2,\ldots,x_n)
$$

where:
- $f_0$ is the constant term
- $f_i(x_i)$ are first-order component functions
- $f_{ij}(x_i,x_j)$ are second-order interactions
- Higher-order terms represent more complex interactions

## Shapley Values

Shapley effects $\phi_i$ provide a variance-based sensitivity measure:

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} [V(f_{S \cup \{i\}}) - V(f_S)]
$$

where:
- $N$ is the set of all parameters
- $S$ is a subset of parameters
- $V$ is the variance operator

## GMDH Parameter Selection

The Group Method of Data Handling (GMDH) is used for:
- Automatic model structure detection
- Parameter selection
- Complexity control through:
  - Layer-wise growth
  - Selection thresholds
  - Validation metrics