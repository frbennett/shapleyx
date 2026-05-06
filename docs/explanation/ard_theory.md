The provided code implements a **Sparse Bayesian Learning (SBL)** algorithm, specifically a variant of the **Relevance Vector Machine (RVM)**, which is a Bayesian approach to sparse linear regression and classification. Below is a detailed mathematical explanation of the key components:

---

### **1. Overview of Sparse Bayesian Learning (SBL)**
SBL is a Bayesian framework for learning sparse models by introducing **precision parameters (α)** for each weight in the model. These parameters are estimated from the data, and many are driven to infinity (effectively setting the corresponding weights to zero), leading to a sparse solution.

#### **Key Mathematical Components:**
- **Prior Distribution**: Gaussian prior over weights with precision parameters **α**.
- **Likelihood**: Gaussian likelihood of the data given the weights.
- **Posterior Distribution**: Gaussian distribution over weights, combining prior and likelihood.
- **Marginal Likelihood (Evidence)**: Used to optimize the precision parameters **α**.

---

### **2. Posterior Distribution Calculation**
The function `_posterior_dist` computes the mean (`Mn`) and covariance (`Sn`) of the posterior distribution of the weights.

#### **Mathematical Formulation:**
- **Posterior Precision Matrix (Inverse of Covariance)**:
  $$
  \mathbf{S}^{-1} = \beta \mathbf{X}^T \mathbf{X} + \mathbf{A}
  $$
  where:
  - $\beta$ is the noise precision (inverse variance).
  - $\mathbf{A} = \text{diag}(\alpha_1, \alpha_2, \dots, \alpha_n)$ is the diagonal matrix of weight precisions.
  - $\mathbf{X}$ is the design matrix.

- **Posterior Mean**:
  $$
  \mathbf{m} = \beta \mathbf{S} \mathbf{X}^T \mathbf{y}
  $$

- **Cholesky Decomposition**:
  The code uses Cholesky decomposition for numerical stability:
  $$
  \mathbf{S}^{-1} = \mathbf{R}^T \mathbf{R}
  $$
  where $\mathbf{R}$ is an upper triangular matrix.

---

### **3. Sparsity and Quality Parameters**
The function `_sparsity_quality` computes the **sparsity (s)** and **quality (q)** parameters for each feature, which are used to determine whether a feature should be added, removed, or updated.

#### **Mathematical Formulation:**
- **Sparsity (s)** and **Quality (q)**:
  $$
  s_i = \beta \mathbf{x}_i^T \mathbf{x}_i - \beta^2 \mathbf{x}_i^T \mathbf{X} \mathbf{S} \mathbf{X}^T \mathbf{x}_i
  $$
  $$
  q_i = \beta \mathbf{x}_i^T \mathbf{y} - \beta^2 \mathbf{x}_i^T \mathbf{X} \mathbf{S} \mathbf{X}^T \mathbf{y}
  $$
  where $\mathbf{x}_i$ is the $i$-th feature column.

- **Update Rule for $\alpha_i$**:
  $$
  \alpha_i = \frac{s_i^2}{q_i^2 - s_i}
  $$
  If $q_i^2 - s_i \leq 0$, the feature is pruned (its $\alpha_i$ is set to infinity).

---

### **4. Precision Parameter Update**
The function `update_precisions` updates the precision parameters $(\alpha_i)$ based on the sparsity and quality parameters.

#### **Key Steps:**
1. **Identify Features**:
   - **Add**: Features not in the model but with $q_i^2 - s_i > 0$.
   - **Recompute**: Features in the model with $q_i^2 - s_i > 0$.
   - **Delete**: Features in the model with $q_i^2 - s_i \leq 0$.

2. **Update $\alpha_i$**:
   - For features to add/recompute:
     $$
     \alpha_i = \frac{s_i^2}{q_i^2 - s_i}
     $$
   - For features to delete:
     $$
     \alpha_i \to \infty
     $$

3. **Convergence Check**:
   - The algorithm stops if:
     - No features are added/deleted.
     - The change in $\alpha_i$ for existing features is below a tolerance.

---

### **5. Predictive Distribution**
The function `predict_dist` computes the predictive mean and variance for test data.

#### **Mathematical Formulation:**
- **Predictive Mean**:
  $$
  \hat{y} = \mathbf{X}_{\text{test}} \mathbf{m}
  $$
- **Predictive Variance**:

$$ 
\text{var}(\hat{y}) = \frac{1}{\beta} + \mathbf{X}_{\text{test}} \mathbf{S} \mathbf{X}_{\text{test}}^T
$$  

---

### **6. Key Equations and Intuition**
- **Marginal Likelihood Maximization**:
  The algorithm maximizes the marginal likelihood (evidence) with respect to $\alpha_i$ and $\beta$:

  $$
  p(\mathbf{y} \mid \mathbf{X}, \alpha, \beta) = \int p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}, \beta)\, p(\mathbf{w} \mid \alpha)\, d\mathbf{w}
  $$

  This is done iteratively using the **Expectation-Maximization (EM)** framework.

- **Automatic Relevance Determination (ARD)**:
  The precision parameters $\alpha_i$ act as **regularization parameters**, automatically determining the relevance of each feature. Irrelevant features have $\alpha_i \to \infty$, and their weights are pruned.

---

### **7. Summary of the Algorithm**
1. Initialize $\alpha_i$ and $\beta$.
2. Compute the posterior distribution of weights (mean and covariance).
3. Update $\alpha_i$ and $\beta$ based on sparsity/quality parameters.
4. Prune irrelevant features $(\alpha_i \to \infty$).
5. Repeat until convergence (no feature changes and small $\alpha_i$ updates).

This approach provides a **sparse, probabilistic, and robust** solution to regression/classification problems. The code efficiently implements these steps using numerical optimizations (Cholesky decomposition, Woodbury identity for matrix inversions).

---

### **8. Cross-Validation Model Selection**

When the ARD model is configured with `cv=True`, an outer layer of
K-fold cross-validation is run at **every ARD iteration** to guide model
selection.  Rather than relying on the ARD algorithm's internal convergence
criteria alone, the CV score provides an independent estimate of
generalisation performance for the current active feature set and
hyperparameter state.

#### 8.1 Predictive Log-Likelihood Score

The recommended scoring method (`cv_method='bayesian'` or `'predictive'`)
uses the **log predictive likelihood**:

$$
\mathcal{L}_{\text{CV}} = \frac{1}{K} \sum_{k=1}^{K}
\log p(\mathbf{y}_{\text{val}}^{(k)} \mid
\mathbf{X}_{\text{val}}^{(k)}, \mathcal{D}_{\text{train}}^{(k)})
$$

where $\mathcal{D}_{\text{train}}^{(k)} = (\mathbf{X}_{\text{train}}^{(k)},
\mathbf{y}_{\text{train}}^{(k)})$ is the $k$-th training fold.  For each
fold:

1. **Per-fold centering** — the training data is centered using
   **only** the training-fold statistics $\bar{\mathbf{x}}_{\text{train}}$
   and $\bar{y}_{\text{train}}$:

   $$
   \begin{aligned}
   \mathbf{X}_{\text{train}}^{(c)} &=
   \mathbf{X}_{\text{train}} - \bar{\mathbf{x}}_{\text{train}} \\[2pt]
   \mathbf{y}_{\text{train}}^{(c)} &=
   \mathbf{y}_{\text{train}} - \bar{y}_{\text{train}} \\[2pt]
   \mathbf{X}_{\text{val}}^{(c)} &=
   \mathbf{X}_{\text{val}} - \bar{\mathbf{x}}_{\text{train}}
   \end{aligned}
   $$

   This is critical: centering on the full dataset before splitting would
   leak information from the validation fold into the transformation,
   producing systematically optimistic scores.

2. **Posterior computation** — the ARD posterior is computed on the
   centered training data using the current global precision estimates
   $\alpha_i$ and $\beta$:

   $$
   \mathbf{S}^{-1} = \beta \mathbf{X}_{\text{train}}^{(c)T}
   \mathbf{X}_{\text{train}}^{(c)} + \mathbf{A},
   \qquad
   \mathbf{m} = \beta \mathbf{S} \, \mathbf{X}_{\text{train}}^{(c)T}
   \mathbf{y}_{\text{train}}^{(c)}
   $$

3. **Predictive distribution** — predictions are converted back to the
   original scale:

   $$
   \hat{\mathbf{y}}_{\text{val}} = \mathbf{X}_{\text{val}}^{(c)}
   \mathbf{m} + \bar{y}_{\text{train}}
   $$

   with predictive variance:

   $$
   \sigma^2_i = \frac{1}{\beta} + \mathbf{x}_{\text{val},i}^{(c)T}
   \mathbf{S} \, \mathbf{x}_{\text{val},i}^{(c)}
   $$

4. **Log-likelihood** — the score for the fold is:

   $$
   \ell_k = -\frac{1}{2} \sum_{i=1}^{n_{\text{val}}}
   \left[ \log(2\pi\sigma_i^2) +
   \frac{(y_{\text{val},i} - \hat{y}_{\text{val},i})^2}{\sigma_i^2} \right]
   $$

#### 8.2 Retrospective Selection

CV scores are recorded for every ARD iteration.  After the main ARD loop
completes (either by convergence or reaching `n_iter`), the iteration with
the **highest** CV score is retrospectively selected as the final model.
This is controlled by the `retrospective_selection=True` parameter (the
default).

This approach has two advantages over early-stopping based on CV:

- **Decouples ARD convergence from model selection** — the ARD algorithm
  is allowed to explore the full sparsity path without premature
  termination.
- **Avoids threshold tuning** — no `cv_tol` parameter is needed to decide
  when CV improvement has "plateaued".

#### 8.3 Ridge CV (Legacy)

The `cv_method='ridge'` option uses scikit-learn's `Ridge` regression
with default L2 penalty inside `cross_val_score`.  This evaluates the
ARD-discovered active feature set under a **non-sparse** model, which is
statistically inconsistent with the ARD framework.  It is retained for
backward compatibility but is not recommended for new work.

#### 8.4 Practical Considerations

- **Computational cost** — each CV evaluation fits the ARD posterior on
  $K$ training folds.  For high-dimensional problems ($d > 100$), this
  dominates the runtime.  Consider using `method='ard'` (no CV) for
  exploratory work.
- **Frozen hyperparameters** — the current global $\alpha_i$ and $\beta$
  are used in all folds; they are not re-optimised per fold.  This
  approximation is valid because the active set and precision estimates
  evolve smoothly across ARD iterations.
- **Reproducibility** — the KFold split uses `random_state=42`.  For
  production work, consider exposing this as a configurable parameter.
