# Understanding $v(S) = \text{Var}[\mathbb{E}[f \mid X_S]]$

The value function that underpins all Shapley and Owen effects in ShapleyX.
Read it **inside-out**.

---

## Step 1 — The inner expectation

$$\mathbb{E}[f \mid X_S = x]$$

"Given that the variables in set $S$ take the specific value $x$, what is the
average output?"

This is a **function of $x$** — for each possible combination of values for the
conditioning variables, you get a different conditional average.

**Example:** $f = \text{NSE}$ (Nash-Sutcliffe efficiency), $S = \{\text{LZFPM}\}$
(lower zone free water capacity):

| LZFPM (mm) | $\mathbb{E}[\text{NSE} \mid \text{LZFPM}]$ |
|------------|---------------------------------------------|
| 40 | 0.55 — too little storage, model underperforms |
| 200 | 0.72 — moderate storage |
| 500 | 0.78 — ample storage, model performs well |

The conditional expectation varies across the range of LZFPM. It is a single
number for each fixed LZFPM value, but it changes as LZFPM changes.

---

## Step 2 — The outer variance

$$\text{Var}[\;\cdot\;]$$

"As $X_S$ varies across its natural distribution (the prior, the posterior,
or the empirical range), how much does that conditional average bounce around?"

Take all the conditional expectations from Step 1 — one for each possible value
of $X_S$ — and compute their variance:

$$v(\{\text{LZFPM}\}) = \text{Var}[0.55, 0.72, 0.78, \ldots] \approx 0.007$$

---

## What $v(S)$ tells you

| Value | Interpretation |
|-------|---------------|
| $v(S) = 0$ | Knowing $X_S$ tells you **nothing** about $f$ — the conditional average is flat across all values of $X_S$ |
| $v(S) = \text{Var}[f]$ | Knowing $X_S$ tells you **everything** — once you fix $X_S$, there is no remaining uncertainty in $f$ |
| $0 < v(S) < \text{Var}[f]$ | Knowing $X_S$ explains **part** of the output variance |

$v(S)$ is **monotonic**: adding more variables to $S$ can only increase
$v(S)$, never decrease it. If you already know LZFPM, also knowing LZFSM
might explain additional variance — $v(\{\text{LZFPM}, \text{LZFSM}\}) \geq v(\{\text{LZFPM}\})$.

---

## The variance decomposition

For any subset $S$:

$$\text{Var}[f] = \underbrace{\text{Var}\big[\mathbb{E}[f \mid X_S]\big]}_{v(S)\;—\;\text{explained by knowing }X_S} \;+\; \underbrace{\mathbb{E}\big[\text{Var}[f \mid X_S]\big]}_{\text{residual — what remains after knowing }X_S}$$

- The **first term** is $v(S)$ — the variance of the conditional expectation.
  How much does the prediction improve (on average) when you learn $X_S$?

- The **second term** is the average remaining variance after conditioning.
  Even after fixing $X_S$, $f$ might still vary due to the other variables not in $S$.

---

## How ShapleyX estimates $v(S)$

ShapleyX uses the **pick-freeze** estimator (Owen & Prieur 2017;
Goda 2021). For a subset $S$:

1. Draw $N$ joint samples $X^{(1)}, \ldots, X^{(N)}$ from the full distribution
2. Evaluate $f$ on each: $Y_i = f(X^{(i)})$
3. For each sample, create a **conditional** draw $\tilde{X}^{(i)}$ that shares
   $X^{(i)}_S$ but redraws everything else independently
4. Evaluate $f$ again: $\tilde{Y}_i = f(\tilde{X}^{(i)})$
5. Compute the **covariance** between the two sets of outputs:

$$v(S) = \text{Cov}[Y, \tilde{Y}]$$

This covariance equals $\text{Var}[\mathbb{E}[f \mid X_S]]$ because the only
thing linking $Y$ and $\tilde{Y}$ is the shared $X_S$ — their covariance
measures exactly how much variance $X_S$ explains.
