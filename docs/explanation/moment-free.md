# Moment-Free Sensitivity Measures

ShapleyX provides three **moment-free** (distribution-based) sensitivity
measures alongside its variance-based Sobol and Shapley indices: **PAWN**,
**Delta**, and the **H-index**.  While Sobol and Shapley indices ask "which
inputs drive output *variance*?", moment-free measures ask "which inputs
change the output *distribution*?" — a question that matters whenever the
shape, spread, or tail behaviour of the output is as important as its
magnitude.

All three operate on the RS‑HDMR surrogate after it has been trained,
making them fast to compute even for expensive models.

---

## Why Moment-Free?

Variance-based measures can miss important effects when inputs

- shift the output's location without affecting variance,
- change the spread (variance) but not the mean, or
- create or dissolve multi-modal, skewed, or heavy-tailed behaviour.

In physical and environmental models these situations are common.  A
pollutant input may trigger an algal bloom only past a threshold,
changing the output distribution's shape without moving its variance
much — Sobol ≈ 0 but Delta ≫ 0.  Moment-free measures capture
these effects because they compare the full output distribution, not
just its moments.

| Concern | Variance-based (Sobol, Shapley) | Moment-free (PAWN, Delta, H) |
|---------|:---:|:---:|
| Mean shift, variance unchanged | ✗ Missed | ✓ Detected |
| Variance change, mean unchanged | ✓ Detected | ✓ Detected |
| Shape / tail change | ✗ Compressed | ✓ Detected |
| Multi-modality | ✗ Compressed | ✓ Detected |
| Threshold / tipping-point effects | ✗ Often missed | ✓ Often detected |
| Interpretation | % of output variance | Distributional influence |

---

## How They Work (Conceptual)

All three share the same core experimental design:

1.  **Unconditioned reference** — sample the surrogate model across the
    full input space to estimate the full output distribution
    \\(p(y)\\).

2.  **Conditioned distribution** — for each input \\(X_i\\), fix it to one
    value \\(x_i\\) and sample all other inputs freely to estimate the
    conditional output distribution \\(p(y \\mid X_i = x_i)\\).

3.  **Compare** — measure the distance between the conditioned
    distribution and the unconditioned reference.  If fixing \\(X_i\\)
    produces a noticeably different output distribution, \\(X_i\\) is
    influential.

4.  **Aggregate** — repeat for many values of \\(x_i\\) and average
    the distances to produce a single sensitivity index per input.

Where they differ is in **which distance metric** they use — and
therefore *what kind* of distributional change they're most sensitive to.

---

## 1. PAWN (Pianosi & Wagener, 2015, 2018)

**Core question:** *Does conditioning on \\(X_i\\) produce a
statistically distinguishable output CDF?*

### Distance Measure

PAWN uses the **two-sample Kolmogorov–Smirnov (KS) statistic**, the
maximum vertical distance between two empirical cumulative distribution
functions:

$$
D_{ks} = \\sup_y \\left| F_{Y}(y) - F_{Y|X_i}(y) \\right|
$$

where \\(F_{Y}\\) is the unconditional CDF and \\(F_{Y|X_i}\\) is the
CDF conditioned on \\(X_i\\) falling in a particular interval.

### Algorithm

1.  Divide each input variable's range into \\(S\\) equal-probability
    intervals (default \\(S = 10\\)).
2.  For each interval, subset the output samples that fall into that
    slice and compute \\(D_{ks}\\) against the full unconditioned output.
3.  Across the \\(S\\) intervals, report the **minimum, mean, median,
    and maximum** KS statistic.

The median KS statistic is the primary PAWN index used for ranking.
A higher value means the input has a larger effect on the output
distribution across a wider range of conditioning values.

### What It Returns

| Column | Meaning |
|--------|---------|
| `minimum` | Smallest KS statistic across conditioning intervals |
| `mean` | Average KS statistic |
| `median` | **Primary PAWN index** — consensus distributional effect |
| `maximum` | Strongest local effect (may be driven by an extreme interval) |
| `CV` | Coefficient of variation (std / mean) — consistency across intervals |
| `stdev` | Standard deviation of KS across intervals |
| `null hyp` | Accept/reject: is the conditioned distribution distinguishable at \\(\\alpha=0.05\\)? |

### Key Properties

- **Distribution-free** — no assumptions about the shape of the output.
- **Inexpensive** — KS test is fast; PAWN is the cheapest moment-free method.
- **Statistical significance** — provides p-values so non-influential
  inputs can be formally ruled out.
- **Insensitive to conditioning strategy** — using quantile-based intervals
  avoids sensitivity to binning choices.
- **Screening tool** — best used first to identify which variables have
  *any* distributional effect before applying more expensive measures.

### API

```python
# Direct from data (no surrogate required)
pawn = analyzer.get_pawn(S=10)

# Via the RS-HDMR surrogate (recommended — much faster)
pawn = analyzer.get_pawnx(num_unconditioned=1000, num_conditioned=500,
                           num_ks_samples=100, alpha=0.05)
```

### Reference

> Pianosi, F. & Wagener, T. (2015). A simple and efficient method for
> global sensitivity analysis based on cumulative distribution functions.
> *Environmental Modelling & Software*, 67, 1–11.

---

## 2. Delta Index (Borgonovo, 2007)

**Core question:** *What is the expected shift in the output's probability
density when you pin \\(X_i\\) to a fixed value?*

### Distance Measure

The Delta index uses the **L₁ distance between probability density
functions** — the total area between the unconditioned and conditioned
PDFs:

$$
\\delta_i = \\mathbb{E}_{X_i} \\left[
   \\frac{1}{2} \\int \\left| p(y) - p(y \\mid X_i) \\right| dy
\\right]
$$

The factor \\(\\frac{1}{2}\\) ensures \\(\\delta_i \\in [0, 1]\\).
The expectation is taken over the distribution of \\(X_i\\), so this is
the *average* PDF shift when \\(X_i\\) is varied across its range.

### Algorithm

1.  Estimate the **unconditioned PDF** of the surrogate output via
    Gaussian KDE with Silverman bandwidth.
2.  For each input \\(X_i\\), repeat \\(N\\) times:
    - Fix \\(X_i = x_i\\) (randomly drawn from its range).
    - Set all other inputs to the unconditioned reference sample but
      overwrite column \\(i\\) with \\(x_i\\).
    - Predict via the surrogate to obtain the conditioned output sample.
    - Estimate the **conditioned PDF** via Gaussian KDE.
    - Compute the area between the two KDEs via `np.trapz`.
3.  Average the areas across all \\(N\\) conditioning values — that's
    \\(\\delta_i\\).

The Delta index is then normalised so that \\(\\sum_i \\delta_i = 1\\),
giving the **normalised Delta** (`delta_norm`).

### What It Returns

| Column | Meaning |
|--------|---------|
| `delta` | Expected shift in output PDF when \\(X_i\\) is fixed |
| `delta_norm` | Normalised delta (sums to 1 across all variables) |
| `Var` | Input variable label |

### Key Properties

- **Sensitive to all distributional changes** — mean shifts, variance
  changes, shape changes, multi-modality.
- **Single interpretable number** — raw \\(\\delta\\) is the expected
  proportion of the output PDF that shifts when the input is known.
- **Always non-negative, normalisable** — natural ranking.
- **Symmetric** — the L₁ distance is symmetric, so the order of
  \\(p(y)\\) and \\(p(y|X_i)\\) doesn't matter.
- **More expensive than PAWN** — KDE construction per conditioning value
  adds computational cost.
- **General-purpose** — the best single-number moment-free importance
  measure when you don't have a specific distributional concern in mind.

### API

```python
delta = analyzer.get_deltax(num_unconditioned=1000, delta_samples=500)
print(delta[['Var', 'delta', 'delta_norm']])
```

### Reference

> Borgonovo, E. (2007). A new uncertainty importance measure.
> *Reliability Engineering & System Safety*, 92(6), 771–784.

---

## 3. H-Index (KL Divergence–Based)

**Core question:** *How much information is lost if you replace the
conditioned output distribution with the unconditioned one?*

### Distance Measure

The H-index uses **Kullback–Leibler (KL) divergence**, an
information-theoretic measure of how one probability distribution
diverges from a reference:

$$
h_i = \\mathbb{E}_{X_i} \\left[
   D_{KL}\\left( p(y \\mid X_i) \\;\\|\\; p(y) \\right)
\\right]
$$

where

$$
D_{KL}(p \\| q) = \\int p(y) \\log\\frac{p(y)}{q(y)} \\, dy
$$

### Algorithm

The H-index follows the same structure as Delta:

1.  Estimate the unconditioned PDF \\(p(y)\\) via Gaussian KDE.
2.  For each input \\(X_i\\), repeat \\(N\\) times:
    - Fix \\(X_i = x_i\\) and obtain the conditioned output sample.
    - Estimate \\(p(y \\mid X_i)\\) via Gaussian KDE.
    - Compute \\(D_{KL}(p(y \\mid X_i) \\| p(y))\\) — KL divergence
      from the *conditioned* distribution to the *unconditioned* one.
3.  Average across conditioning values — that's \\(h_i\\).

A small \\(\\epsilon\\) (default \\(10^{-10}\\)) is added to KDE
evaluations to prevent numerical issues from near-zero density regions.

### What It Returns

| Column | Meaning |
|--------|---------|
| `delta` | Mean KL divergence when \\(X_i\\) is fixed |
| `delta_norm` | Normalised (sums to 1 across all variables) |
| `Var` | Input variable label |

!!! note "Column naming"
    The H-index uses `delta` as the column name for the raw KL value,
    reflecting its implementation heritage.  The scores are **not**
    comparable to Delta index values — they use fundamentally different
    distance metrics with different scales.

### Key Properties

- **Information-theoretic** — KL divergence quantifies the information
  lost when using the unconditioned distribution as an approximation
  for the conditioned one.
- **Penalises narrow conditioned distributions** — if conditioning
  produces a tight, peaked distribution, KL divergence is large because
  the unconditioned distribution assigns low probability to those
  regions.
- **Asymmetric** — \\(D_{KL}(p \\| q) \\neq D_{KL}(q \\| p)\\).
  The H-index measures divergence *from the conditioned* to the
  *unconditioned*, so it's sensitive to cases where the input
  concentrates the output into a narrow range.
- **Sensitive to tail behaviour** — heavy or light tails in the
  conditioned distribution produce large KL values.
- **No upper bound** — unlike Delta (\\(\\in [0,1]\\)), KL divergence
  is \\(\\in [0, \\infty)\\), so absolute values are not directly
  interpretable as proportions.  Use `delta_norm` for ranking.
- **Computationally similar to Delta** — KDE construction per
  conditioning value dominates cost.

### When to Prefer the H-Index

The H-index is the best moment-free choice when:

- The output has **heavy tails** (flood risk, extreme events, financial
  loss distributions) and you care about capturing tail behaviour.
- Conditioning produces **narrower, more informative** output
  distributions — KL divergence naturally rewards this.
- You want the **most theoretically principled** information measure
  (KL divergence has deep connections to Bayesian inference and
  information geometry).

### API

```python
h_idx = analyzer.get_hx(num_unconditioned=1000, delta_samples=500)
print(h_idx[['Var', 'delta', 'delta_norm']])
```

### Reference

> The H-index is an implementation of KL-divergence-based sensitivity
> analysis.  See also: Liu, H., Chen, W., & Sudjianto, A. (2006).
> Relative entropy based method for probabilistic sensitivity analysis
> in engineering design. *Journal of Mechanical Design*, 128(2), 326–336.

---

## Comparison Summary

| | PAWN | Delta | H-index |
|------|------|-------|---------|
| **Distance measure** | KS statistic (CDF) | L₁ area between PDFs | KL divergence (information) |
| **Range of raw index** | \\([0, 1]\\) | \\([0, 1]\\) | \\([0, \\infty)\\) |
| **Statistical test?** | ✓ Yes (p-values) | ✗ No | ✗ No |
| **Interpretation** | "How distinguishable?" | "How much density shifts?" | "How much information lost?" |
| **Sensitive to** | Any CDF change | Any PDF change | Tails, narrow peaks, concentration |
| **Computational cost** | ★☆☆ Lowest | ★★☆ Medium | ★★☆ Medium |
| **Best use case** | Screening | General ranking | Tail / peak sensitivity |

---

## Practical Guidance

### Choosing a Method

- **Start with PAWN** — it's the fastest and provides statistical
  significance tests.  If PAWN says an input has no distributional
  effect, you can confidently ignore it.
- **Use Delta for ranking** — its normalisation to sum-to-1 makes it the
  most intuitive for presenting relative importance to stakeholders.
- **Use the H-index when tails matter** — flood risk, extreme events,
  failure probability, and any application where the extremes of the
  distribution are the quantity of interest.

### Sample Size Recommendations

| Method | `num_unconditioned` | `num_conditioned` / `delta_samples` | Notes |
|--------|:---:|:---:|-------|
| PAWN | 1000 | 500 | KS test has good power with moderate samples |
| Delta | 1000–5000 | 200–1000 | More samples improve KDE accuracy |
| H-index | 1000–5000 | 200–1000 | KL divergence is more sensitive to small-sample KDE noise |

### Integration with the Workflow

```python
from shapleyx import rshdmr

# Train the RS-HDMR model
analyzer = rshdmr("data.csv", polys=[10], method='ard_cv')
analyzer.run_all()

# Variance-based indices
print(analyzer.sobol_indices)

# Moment-free measures
pawn  = analyzer.get_pawnx(1000, 500, 100)
delta = analyzer.get_deltax(1000, 500)
h_idx = analyzer.get_hx(1000, 500)

# Combine all measures into one table
import pandas as pd
comparison = pd.DataFrame({
    'Variable': pawn.index,
    'Sobol_S1':   analyzer.sobol_indices['S1'],
    'Shapley':    analyzer.sobol_indices['Shapley'],
    'PAWN':       pawn['median'],
    'Delta':      delta['delta_norm'].values,
    'H-index':    h_idx['delta_norm'].values,
})
print(comparison.round(3))
```

### Limitations

- **All three methods** operate on the RS‑HDMR surrogate, so their
  accuracy depends on the quality of the surrogate fit.  Check the
  \\(R^2\\) or explained variance (`evs`) before interpreting
  moment-free results.
- **KDE-based methods** (Delta, H-index) can be unreliable with very
  small samples or strongly multi-modal outputs.  Increase
  `num_unconditioned` if the output distribution is complex.
- **Moment-free indices do not decompose output variance** — they cannot
  be used to close a variance budget or identify interaction effects.
  They are complementary to, not replacements for, Sobol and Shapley
  indices.
- **No interaction detection** — the current implementation compares
  only single-variable conditioning against the unconditioned reference.
  For interaction effects, use Owen-Shapley interactions from the
  variance-based framework.
