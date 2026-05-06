# ARD Reference

The `RegressionARD` class implements **Automatic Relevance Determination**
(ARD) via the Sparse Bayesian Learning (SBL) algorithm [^1] [^2].  When used
with ``cv=True``, the model internally runs K-fold cross-validation at each
ARD iteration and selects the best-performing model state via **retrospective
selection**.

## Cross-Validation Methods

Three scoring methods are available through the ``cv_method`` parameter:

| Method         | Description |
|----------------|-------------|
| ``'bayesian'`` | Predictive log-likelihood CV with **per-fold centering** — the training data within each fold is centered independently using only that fold's statistics, and the validation data is transformed with the training-fold statistics.  This eliminates the data-leakage that would result from centering on the full dataset before splitting.  The score is $\log p(y_{\text{val}} \mid X_{\text{val}}, \mathcal{D}_{\text{train}})$, i.e. the log predictive density of the held-out data under the ARD posterior fitted on the training fold.  **This is the recommended method.** |
| ``'predictive'`` | Alias for ``'bayesian'``. |
| ``'ridge'`` | Legacy Ridge-regression CV (``sklearn.linear_model.Ridge``). Evaluates the current active feature set using a non-sparse L2-penalised model, which is statistically inconsistent with the ARD framework.  Retained for backward compatibility only. |

All methods use 10-fold CV with a fixed random seed (42) for reproducibility.
The best model iteration is chosen retrospectively after all ARD iterations
complete, avoiding the instability of early-stopping heuristics.

## Per-Fold Centering (v0.5.2+)

Prior to v0.5.2, CV was performed on data that had been centered using the
**full dataset** mean before splitting.  This leaked information from the
validation fold into the centering transformation, producing systematically
optimistic CV scores.  The current implementation centres each fold
independently:

$$
\begin{aligned}
\bar{\mathbf{x}}_{\text{train}} &= \frac{1}{n_{\text{train}}}
\sum_{i \in \text{train}} \mathbf{x}_i, \qquad
\bar{y}_{\text{train}} = \frac{1}{n_{\text{train}}}
\sum_{i \in \text{train}} y_i \\[4pt]
\mathbf{X}_{\text{train}}^{(c)} &= \mathbf{X}_{\text{train}} -
\bar{\mathbf{x}}_{\text{train}}, \qquad
\mathbf{y}_{\text{train}}^{(c)} = \mathbf{y}_{\text{train}} -
\bar{y}_{\text{train}} \\[4pt]
\mathbf{X}_{\text{val}}^{(c)} &= \mathbf{X}_{\text{val}} -
\bar{\mathbf{x}}_{\text{train}}
\end{aligned}
$$

The posterior $\mathbf{m}, \mathbf{S}$ is computed from
$\mathbf{X}_{\text{train}}^{(c)}$ and predictions are converted back to
original scale via $\hat{\mathbf{y}}_{\text{val}} =
\mathbf{X}_{\text{val}}^{(c)} \mathbf{m} + \bar{y}_{\text{train}}$.

[^1]: Tipping, M. E., & Faul, A. C. (2003). Fast marginal likelihood
  maximisation for sparse Bayesian models. *AISTATS*.
[^2]: Tipping, M. E., & Faul, A. C. (2001). Analysis of sparse Bayesian
  learning. *NeurIPS*.

::: shapleyx.utilities.ARD
    options:
      show_root_heading: true
      show_source: true

