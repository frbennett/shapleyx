# Owen (Grouped) Shapley Effects

## Motivation

Standard Shapley effects decompose output variance across **individual input variables**.
But in practice, parameters often form natural **subsystems** — e.g., in a hydrology model:

- Soil zone parameters (UZTWM, UZFWM, LZTWM, LZFPM, LZFSM)
- Routing parameters (PFREE, REXP)
- Impervious-surface parameters (ADIMP, PCTIM)
- Groundwater parameters (LZPK, LZSK, UZK)

Owen effects (Owen 2014) extend Shapley effects to a **two-level hierarchy**:

| Level | Question | Example |
|-------|----------|---------|
| **Outer (Group)** | Which *subsystem* matters most? | "Soil zone vs. routing vs. impervious" |
| **Inner (Individual)** | Within a subsystem, which *parameter* dominates? | "LZFPM vs. UZTWM within soil zone" |

## Why Not Just Sum Individual Shapley Effects?

The Shapley value is additive in the *game* (Shᵢ(v+w) = Shᵢ(v) + Shᵢ(w)),
but **not** additive in the *players*. Grouping changes the game itself.

Consider three variables with groups A={1,2}, B={3}. The group Shapley treats
A as an atomic block — it never considers orderings where 1 and 2 are separated
by variable 3. The individual Shapley considers all 6 permutations, including
(1, 3, 2) where variables 1 and 2 are separated by 3.

$$\Phi_A \neq \phi_1 + \phi_2$$

The difference is especially large when groups have **strong internal correlation**
(e.g., UZTWM↔LZTWM at ρ=−0.84).

## Mathematical Definition

Given a partition $P = \{G_1, \dots, G_K\}$ of the $d$ variables, and the
value function $v(S) = \text{Var}[\mathbb{E}[f(X) \mid X_S]]$:

### Outer (Group) Effect

The outer Shapley effect for group $G_k$ is the standard Shapley value in
a $K$-player game over groups:

$$\Phi_{G_k}^{\text{outer}} = \frac{1}{K!} \sum_{\pi \in \Pi_K} \Big[v(\text{predecessors}_\pi(G_k) \cup G_k) - v(\text{predecessors}_\pi(G_k))\Big]$$

where $\text{predecessors}_\pi(G_k)$ is the union of groups appearing before
$G_k$ in permutation $\pi$.

### Inner (Individual) Effect

The inner Owen effect for variable $i$ in group $G_k$ is:

$$\Phi_i^{\text{inner}} = \frac{1}{|G_k|!\, (K-1)!} \sum_{R \subseteq P \setminus \{G_k\}} \sum_{T \subseteq G_k \setminus \{i\}} |T|!\,(|G_k|-|T|-1)!\,|R|!\,(K-|R|-1)! \cdot \Big[v(Q \cup T \cup \{i\}) - v(Q \cup T)\Big]$$

where $Q = \bigcup_{G \in R} G$ is the union of the groups in $R$.

This double summation iterates over:
1. All subsets $R$ of *other* groups (between-group predecessors)
2. All subsets $T$ of the *same* group, excluding $i$ (within-group predecessors)

### Decomposition Property

$$\sum_{k=1}^{K} \Phi_{G_k}^{\text{outer}} = \sum_{i=1}^{d} \Phi_i^{\text{inner}} = \text{Var}[f]$$

Both levels independently sum to the total output variance.

## Implementation in ShapleyX

### API

```python
from shapleyx.utilities.mc_shapley import owen_from_data, owen_effects

# After collecting pick-freeze data (same as standard Shapley):
data = collect_shapley_data(model_func, joint, N=5000, k_max=2)

groups = {
    'soil':  ['UZTWM', 'UZFWM', 'LZTWM', 'LZFPM'],
    'routing': ['PFREE', 'REXP'],
}

ge, ie, tv = owen_from_data(data, groups, var_names)

# ge = {'soil': 0.45, 'routing': 0.55}
# ie = {'UZTWM': 0.12, 'UZFWM': 0.08, 'LZTWM': 0.15, 'LZFPM': 0.10,
#       'PFREE': 0.30, 'REXP': 0.25}
# tv = 1.00
```

### Key Properties

1. **Reuses existing data** — `owen_from_data()` operates on the same `data`
   dict produced by `collect_shapley_data()`. No additional model evaluations.

2. **Reduces to standard Shapley** — when all groups are size 1, the Owen
   decomposition recovers the standard Shapley effects identically.

3. **Supports coalition truncation** — `k_max` limits the subset enumeration
   for computational efficiency, exactly as in standard Shapley.

4. **Works with correlated inputs** — the `v(S)` values come from the
   pick-freeze estimator, which handles arbitrary joint distributions
   via `GaussianCopulaUniform` or `GaussianCopulaArbitrary`.

### Computational Cost

The Owen computation is $O(K \cdot 2^K + \sum_k |G_k| \cdot 2^{|G_k|} \cdot 2^{K-1})$
— more expensive than standard Shapley's $O(d \cdot 2^d)$, but since no new
model evaluations are needed, it's still negligible compared to data collection.

## References

- Owen, A. B. (2014). Sobol' indices and Shapley value. *SIAM/ASA Journal on
  Uncertainty Quantification*, 2(1), 245-251.
- Owen, A. B., & Prieur, C. (2017). On Shapley value for measuring importance
  of dependent inputs. *SIAM/ASA Journal on Uncertainty Quantification*,
  5(1), 986-1004.
