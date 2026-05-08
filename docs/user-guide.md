# ShapleyX User Guide — From Data to Decisions

This guide assumes you have a model or a dataset, you want to know which
inputs matter most, and you're comfortable running a few lines of Python.
No sensitivity analysis background is needed.

---

## 1. What is ShapleyX and why would I use it?

You have a model (or a dataset) with several input variables and one
output.  Maybe it's a flood simulator with rainfall, soil type, and
topography as inputs.  Maybe it's a manufacturing process with
temperature, pressure, and feed rate.  The question is always the same:

> **Which inputs drive the output the most?**

ShapleyX answers this question.  It takes your data, builds a fast
mathematical approximation of your model (a *surrogate*), then decomposes
the output variability into contributions from each input.  The result is
a **ranked importance table** with error bars.

What you get out:

| Input | Importance | 95% CI |
|-------|-----------|--------|
| Rainfall | 0.45 | [0.41, 0.49] |
| Soil type | 0.32 | [0.29, 0.35] |
| Topography | 0.18 | [0.15, 0.21] |
| Temperature | 0.05 | [0.03, 0.07] |

How it works under the hood (in three moves):

1. **Build a surrogate** — approximate your model with a polynomial
   expansion (Legendre polynomials on $[0,1]^d$).  This turns your
   expensive simulator into a cheap formula.

2. **Fit it sparsely** — use Automatic Relevance Determination (ARD) or
   Orthogonal Matching Pursuit (OMP) to select only the polynomial terms
   that actually matter.  Tens or hundreds of terms survive from
   thousands of candidates.

3. **Extract sensitivity** — read off Sobol indices (variance
   decomposition) and Shapley effects (game-theory fair division) from
   the surrogate coefficients.

---

## 2. Before you start — what your data needs to look like

### Format

A CSV file (or pandas DataFrame) with one column per input, plus a
column called `Y` for the output:

```
X1,X2,X3,Y
0.51,0.23,0.89,1.87
0.12,0.67,0.34,2.34
...
```

No missing values.  Inputs don't need to be in $[0,1]$ — ShapleyX
normalises them automatically.

### How many samples?

A rough rule of thumb (for `polys=[10,5]`, ARD):

| Number of inputs | Recommended samples | Typical runtime |
|-----------------|-------------------|----------------|
| 3–5 | 200–500 | < 30 seconds |
| 6–10 | 500–2000 | 1–5 minutes |
| 11–20 | 2000–5000 | 5–30 minutes |
| 20+ | 5000+ | 30 minutes+ |

These are guidelines — simpler functions need fewer; noisier data needs
more.  The R² diagnostic (Section 7) tells you if you had enough.

---

## 3. Installation

```bash
pip install shapleyx
```

For large problems (20+ inputs), install with the optional Numba
acceleration:

```bash
pip install shapleyx[streaming]
```

Verify it works:

```python
import shapleyx
print("Ready.")
```

---

## 4. Your first analysis — the five-minute workflow

Let's walk through a complete analysis on a synthetic test function
(the cantilever beam displacement).  Copy this into a script or
notebook.

### 4.1 Prepare some data

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

# Cantilever beam: displacement = 4*L³/(E*w*t³)
L = np.random.uniform(1.0, 2.0, n)      # beam length (m)
E = np.random.uniform(60e9, 80e9, n)    # Young's modulus (Pa)
w = np.random.uniform(0.05, 0.15, n)    # beam width (m)
t = np.random.uniform(0.005, 0.015, n)  # beam thickness (m)

Y = 4 * L**3 / (E * w * t**3)

df = pd.DataFrame({'L': L, 'E': E, 'w': w, 't': t, 'Y': Y})
print(f"{n} samples, Y range: [{Y.min():.3f}, {Y.max():.3f}]")
```

### 4.2 Run the analysis

```python
from shapleyx import rshdmr

model = rshdmr(
    df,                     # your data
    polys=[10, 5],          # polynomial orders
    method='ard_cv',        # ARD with cross-validation
    cv_method='bayesian',   # Bayesian CV scoring
    n_iter=100,             # max ARD iterations
    resampling=True,        # bootstrap confidence intervals
    number_of_resamples=500,
    verbose=True,           # show progress
)

sobol, shapley, total = model.run_all()
```

### 4.3 What you see

The pipeline runs through several stages and prints progress at each:

```
====================  Transforming data to unit hypercube  ====================
  Feature: L, Min Value: 1.0012, Max Value: 1.9987
  ...

====================  Building basis functions  ====================
  Basis functions of 1 order : 40
  Basis functions of 2 order : 150
  Total basis functions in basis set : 190

====================  Running regression analysis  ====================
  running ARD
  Fit Execution Time : 2.341
  Basis functions selected: 28 / 190 (14.74%)

====================  RS-HDMR model performance statistics  ====================
  explained variance score: 0.999
  r²: 0.9998
```

Then a predicted-vs-actual scatter plot appears.  Finally:

```
====================  Completed all analysis  ====================
```

### 4.4 Your results

```python
# Sobol indices (first-order + interactions)
print(sobol[['derived_labels', 'index', 'lower', 'upper']])

# Shapley effects (scaled to sum to 1)
print(shapley[['label', 'scaled effect', 'lower', 'upper']])
```

For the cantilever beam you'd see something like:

```
Shapley effects:
  Label  scaled effect  lower  upper
0     L          0.063  0.058  0.068
1     E          0.105  0.099  0.111
2     w          0.104  0.098  0.110
3     t          0.728  0.712  0.742
```

**Thickness ($t$) dominates, at 73% of the output variance.**  This
makes physical sense — beam displacement is proportional to $1/t^3$, so
small changes in thickness produce large changes in displacement.

---

### Sidebar: What does `polys=[10, 5]` mean?

The surrogate model is built from polynomial terms:
- **First-order (main effects):** $L, L^2, \dots, L^{10}$ for each input.
  These capture *nonlinearity* in how each input alone affects the output.
- **Second-order (interactions):** $L \cdot E, L^2 \cdot E^5$, etc.,
  up to degree 5 per variable.  These capture how inputs *work together*.
- Higher-order interactions ($>2$) are usually negligible, so we don't
  model them unless you expand `polys` to three or more entries.

In general:

| `polys` | Meaning | Typical use |
|---------|---------|-------------|
| `[10, 5]` | 10th-degree main effects, 5th-degree pairwise | Default for $d$ ≤ 10 |
| `[8, 6, 2]` | 8th-degree main, 6th-degree pairwise, 2nd-degree triple | $d$ = 10–25, more interactions |
| `[10]` | 10th-degree total expansion (not HDMR) | Full polynomial chaos, $d$ ≤ 6 |
| `[6, 4, 2, 2]` | Up to 4th-order interactions | $d$ ≤ 8, rich interaction structure |

Higher degrees = more flexible surrogate, but more computation.

### Sidebar: What's a "good" R²?

The `explained variance score` (often called R²) tells you how well the
surrogate fits your data.  1.0 is perfect; 0.0 means it's no better than
guessing the mean.

| R² | Verdict |
|----|---------|
| > 0.99 | Excellent — your sensitivity results are reliable |
| 0.95–0.99 | Good — results are trustworthy |
| 0.85–0.95 | Adequate — directional conclusions are valid |
| 0.70–0.85 | Marginal — try more samples or different `polys` |
| < 0.70 | Poor — the surrogate isn't capturing your function; results may mislead |

---

## 5. Choosing your regression method

ShapleyX offers several regression methods.  Here's how to choose:

| Method string | Algorithm | Best for | Use CV? |
|--------------|-----------|----------|---------|
| `'ard'` | ARD (Bayesian sparse learning) | Feature selection quality | No |
| `'ard_cv'` | ARD + retrospective CV | Production analysis | Yes |
| `'omp'` | Orthogonal Matching Pursuit | Speed, quick exploration | No |
| `'omp_cv'` | OMP + sklearn CV | Speed with CV | Yes |
| `'omp_stream'` | Streaming OMP (memory-efficient) | Large problems ($d \ge 12$) | No |
| `'omp_cv_stream'` | Streaming OMP-CV | Large problems with CV | Yes |

### Quick decision flowchart

```
Is d ≥ 12 or do you have > 50K candidate basis functions?
  ├── Yes → Do you need cross-validation?
  │         ├── Yes → method='omp_cv_stream'
  │         └── No  → method='omp_stream'
  └── No  → Do you need the best possible feature selection?
            ├── Yes → method='ard_cv'
            └── No  → method='omp_cv'
```

### ARD vs OMP in plain language

**ARD** (Automatic Relevance Determination) is a Bayesian algorithm that
carefully weighs each candidate basis function and decides whether to
keep it, update it, or discard it.  It's thorough — ARD often finds a
slightly better sparse model than OMP — but slower.

**OMP** (Orthogonal Matching Pursuit) is greedier.  At each step it picks
the single basis function most correlated with what's left to explain,
then solves a least-squares problem.  It's fast and usually nearly as
good as ARD.

**Streaming OMP** is OMP that never builds the full design matrix in
memory — it computes basis columns on-the-fly from compact primitives.
For large problems (20 inputs, 80K candidate basis functions) this uses
~3 MB instead of ~6 GB.

### What does cross-validation do?

Cross-validation (CV) tests each candidate model against data it hasn't
seen.  Instead of picking the model ARD converged to (which may be
overfitted), CV selects the model that generalises best.

The `n_iter` parameter controls the maximum number of basis functions to
consider.  ARD typically converges within 100–300 iterations.  OMP needs
more — 50–200 is usually sufficient.

---

## 6. Understanding polynomial orders — the `polys` parameter

The `polys` parameter controls what your surrogate can express.  Here's
how the basis set grows:

| d | `polys` | Basis functions | Memory (dense) | Typical time (ARD) |
|---|---------|----------------|---------------|-------------------|
| 4 | [10, 5] | 190 | <1 MB | < 5s |
| 6 | [10, 5] | 345 | <1 MB | < 10s |
| 8 | [10, 5] | 1,000 | ~8 MB | ~30s |
| 10 | [8, 6, 4] | 9,380 | ~80 MB | ~5 min |
| 15 | [8, 6, 2] | 15,600 | ~120 MB | ~10 min |
| 20 | [8, 6, 2] | 16,120 | ~260 MB | ~25 min |
| 20 | [8, 6, 4] | 79,960 | **6.4 GB** | **use streaming** |

**Rules of thumb:**

- **Start with `[10, 5]`** for problems with $d \le 10$.  Upgrade to
  `[8, 6, 2]` if you suspect 3-way interactions.
- **If memory blows up**, switch to `method='omp_stream'` or
  `method='omp_cv_stream'`.  Streaming uses orders of magnitude less
  memory.
- **If the R² is low**, try increasing the first number (higher-degree
  main effects) or adding a third entry for higher-order interactions.
- **If the analysis is too slow**, reduce `n_iter` or switch from ARD
  to OMP.

---

## 7. Interpreting your results

### Sobol indices

The Sobol table tells you how much of the output variance each input
(or input combination) explains:

```
  derived_labels     index    lower    upper
0             X1    0.312   0.308   0.316
1          X1_X3    0.240   0.234   0.246
2             X2    0.447   0.441   0.452
```

- **`derived_labels`**: which input(s).  `X1_X3` means the interaction
  between X1 and X3.
- **`index`**: the fraction of output variance explained.  0.312 = 31.2%.
  These sum to the R² (not to 1.0).
- **`lower` / `upper`**: 95% bootstrap confidence interval.  If the
  interval includes zero, that input's effect is not statistically
  distinguishable from noise.

### Shapley effects

Shapley effects are a fair division of the *explained* variance.  They
always sum to 1 and are always positive (unlike Sobol indices, which can
sometimes be negative for small effects when estimated from noisy data).

**When Sobol and Shapley disagree:** Shapley effects distribute
interaction contributions equally among the participating inputs.  Sobol
first-order indices only count the direct effect.  If X1 and X2 interact
strongly, Sobol S₁ might be small while Shapley φ₁ is large — because
X1's importance comes from its interaction with X2, not from acting alone.

### The predicted-vs-actual plot

`model.run_all()` automatically generates a scatter plot of predicted
vs actual output values.  A good fit shows points tightly clustered
along the diagonal red line.  Signs of trouble:

- **Fan shape** (spread increases with output magnitude): your model has
  heteroscedastic errors.  Consider log-transforming the output.
- **Systematic curvature** (points bow away from the line): the
  polynomial expansion isn't flexible enough.  Increase `polys`.
- **Clusters of outliers**: check your data for measurement errors.

### Total sensitivity

`model.total` gives the total sensitivity index $S^T$ for each input —
the sum of its direct effect *plus* all interactions it participates in.
If $S^T_i \gg S_i$, that input's importance comes primarily through
interactions.

---

## 8. Sensitivity measures beyond variance

Variance-based measures (Sobol, Shapley) assume that variance is the
right summary of output uncertainty.  Sometimes it isn't:

- **Heavy-tailed outputs:** a few extreme values dominate the variance.
- **Median shifts:** an input changes the median but not the variance.
- **Multimodal outputs:** the output has two or more distinct regimes.

ShapleyX provides three **moment-free** alternatives.  All operate on the
already-fitted surrogate:

```python
# PAWN — density-based, fast screening
pawn = model.get_pawn(S=10)           # no surrogate needed

# PAWN via the surrogate (more accurate, more expensive)
pawnx = model.get_pawnx(1000, 500, 100)

# Delta — moment-independent ranking
delta = model.get_deltax(1000, 500)

# H-index — sensitive to tail behaviour
h_idx = model.get_hx(1000, 500)
```

| Measure | What it detects | Best for |
|---------|----------------|----------|
| Sobol / Shapley | Variance explained | Default, general purpose |
| PAWN | Distribution shifts (CDF distance) | Screening, provides p-values |
| Delta | Area between PDFs | Rankings (sums to 1) |
| H-index | KL divergence | Tail sensitivity / extreme events |

If your Sobol/Shapley results seem to miss an input you suspect matters,
run PAWN or Delta — they may reveal effects that variance-based measures
overlook.

---

## 9. Handling correlated inputs

Sobol indices assume inputs are **independent**.  If your inputs are
correlated (e.g., rainfall and soil moisture move together), standard
Sobol indices can be misleading.

For correlated inputs, use the **Monte Carlo Shapley** method:

```python
import numpy as np

# Define correlation matrix
corr = np.array([
    [1.0, 0.5, 0.0],    # X1–X2 correlation = 0.5
    [0.5, 1.0, 0.0],    # others independent
    [0.0, 0.0, 1.0],
])

# MC Shapley via the surrogate
mc = model.get_mc_shapley(corr=corr, N=5000, B=500)
print(mc[['variable', 'effect', 'lower', 'upper']])
```

The MC Shapley method works by:
1. Sampling input configurations respecting the correlation structure
2. Evaluating the surrogate for each configuration
3. Computing Shapley effects by averaging over random permutations

This is more expensive than the coefficient-based Shapley effects
(a few minutes vs seconds) but correctly accounts for correlation.

---

## 10. Performance — making it faster

### Numba acceleration

Install Numba to accelerate the OMP correlation scans by 3–10×:

```bash
pip install numba
```

The first run will be slightly slower (JIT compilation), but subsequent
runs use cached compiled code.  This is automatic — no code changes
needed.

### Streaming OMP for large problems

When the basis set exceeds ~50K features, switch to streaming:

```python
model = rshdmr(df, polys=[8, 6, 2],
               method='omp_cv_stream',   # streaming, not dense
               n_iter=100,
               n_jobs=4)                 # parallel CV folds
```

Streaming uses 50–100× less memory (3 MB vs 260 MB for the example
above) and is often faster because the compact data fits in CPU cache.

### Parallel CV

For CV methods, `n_jobs` runs fold evaluation in parallel:

```python
model = rshdmr(df, polys=[8, 6, 2],
               method='omp_cv_stream',
               n_iter=100,
               n_jobs=4)   # use 4 CPU cores for fold evaluation
```

4 workers typically gives ~1.4× speedup.  More workers beyond ~6 provide
diminishing returns due to memory bandwidth limits.

### Quick exploratory runs

During exploration, skip resampling for speed:

```python
model = rshdmr(df, polys=[10, 5],
               method='omp_cv',
               n_iter=50,                # fewer iterations
               resampling=False)         # skip bootstrap
```

Add `resampling=True` only for your final production run.

---

## 11. Troubleshooting

### My R² is low

1. **Need more samples?** Try doubling the sample size and see if R²
   improves.  If it does, you were data-limited.
2. **Need more flexibility?** Increase `polys` — e.g. `[10,5]` →
   `[12,6,2]` to allow higher-degree main effects and 3-way interactions.
3. **Wrong method?** ARD handles noise better than OMP.  Try
   `method='ard_cv'`.
4. **Concept check:** Is your function actually representable as a
   polynomial?  Functions with sharp discontinuities or non-smooth
   behaviour may need more samples or a different surrogate approach.

### It's using too much memory

1. Switch to streaming: `method='omp_stream'` or `'omp_cv_stream'`.
2. Reduce `polys`: `[8,6,2]` is cheaper than `[10,6,4]`.
3. If you're using ARD, the Gram matrix ($X^TX$) is the culprit.
   The current ARD implementation requires building it.  Switch to
   streaming OMP for memory-constrained situations.

### The analysis runs forever

1. **Too many ARD iterations?** Lower `n_iter` to 100 or 50 for
   exploratory work.
2. **Too high `polys`?** Halving the degree roughly quarters the number
   of candidate basis functions.
3. **ARD is slower than OMP.** Try `method='omp_cv'` instead.
4. **CV is expensive.** If you don't need CV for model selection,
   use `method='ard'` or `method='omp'` without CV.

### I get a Numba-related error

Numba is optional.  Everything works without it — the streaming OMP
path falls back to pure NumPy.  The analysis will be slower (perhaps
3–5×) but results are identical.

If you see `OSError: Cannot create a file when that file already exists`
on Windows with `n_jobs > 1`, this is a known Numba cache race condition
(fixed in v0.6.0).  Upgrade with `pip install --upgrade shapleyx`.

### My sensitivity results don't make physical sense

1. Check the R² — if it's below ~0.85, the surrogate itself is suspect.
2. For correlated inputs, standard Sobol/Shapley (coefficient-based) can
   mislead.  Use MC Shapley (Section 9).
3. Moment-free measures (Section 8) may tell a different story —
   especially if the output distribution is non-Gaussian.

---

## 12. Where to go next

**Worked examples** (Jupyter notebooks, shipped with the package):
- `Examples/ishigami.ipynb` — the Ishigami benchmark, 3 inputs
- `Examples/wing_weight.ipynb` — wing weight function, 10 inputs
- `Examples/borehole.ipynb` — borehole function, 8 inputs
- `Examples/fire_spread.ipynb` — fire spread model, 5 inputs
- `Examples/sobolG_function.ipynb` — Sobol G-function, 50 inputs
- `Examples/sobolG_function_streaming.ipynb` — same, with streaming OMP

**API reference and theory:**
- [API Reference](reference/api.md) — complete parameter documentation
- [ARD Theory](explanation/ard_theory.md) — mathematical background
- [Streaming OMP](explanation/streaming-omp.md) — how the memory-efficient
  regression works
- [Moment-Free Measures](explanation/moment-free.md) — PAWN, Delta,
  H-index theory
- [MC Shapley Guide](how-to-guides/mc-shapley.md) — correlated inputs
  in depth

**Research paper:**
- Bennett, F. (2026). *Bayesian Cross-Validation for Automatic Relevance
  Determination in Sparse RS-HDMR Surrogate Construction.*
  [`paper/ard_cv_paper.pdf`](https://github.com/frbennett/shapleyx/blob/main/paper/ard_cv_paper.pdf)
