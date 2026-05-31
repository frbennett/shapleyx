# ShapleyX Workshop: NSE Transform Sensitivity

**SAC-SMA Model — St Helens Creek, North Queensland**

Compare parameter sensitivity between sqrt-transformed and raw Nash-Sutcliffe Efficiency using Monte Carlo Shapley effects with correlated inputs.

## Quick Start

| Venue | Click |
|-------|-------|
| **Google Colab** (recommended) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frbennett/shapleyx/blob/main/workshops/transform_sensitivity.ipynb) |
| **Binder** (no account needed) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frbennett/shapleyx/main?labpath=workshops%2Ftransform_sensitivity.ipynb) |

## What You'll Do

1. **Generate training data** — 4,096 scrambled Sobol' samples across the 14-D SAC-SMA parameter space
2. **Compute two NSE variants** — sqrt-transformed (variance-stabilised, standard practice) vs raw (high-flow dominated)
3. **Build RS-HDMR surrogates** — streaming OMP-CV for each variant
4. **Compute Shapley effects** — Monte Carlo pick-freeze with Vrugt et al. (2006) posterior correlations
5. **Compare rankings** — the transform changes which parameters dominate (Shapley correlation only 0.48)

## Key Findings

- **LZFSM overtakes PFREE as #1** under raw NSE — free-water storage dominates when peak flows are unweighted
- **REXP jumps rank 10→5** — percolation nonlinearity matters far more for raw NSE
- **LZTWM and ADIMP collapse** (rank 2→7 and 5→10) — drowned out by peak-flow errors
- **Upper-zone parameters** (UZFWM, UZK) are bottom-3 regardless of transform

## Files

| File | Purpose |
|------|---------|
| `transform_sensitivity.ipynb` | Workshop notebook (31 cells) |
| `transform_data/sacramento.py` | SAC-SMA model (Numba JIT) |
| `transform_data/st_helens_forcing_original.csv` | 46 years daily rainfall, PET, discharge |
| `transform_data/es_parameters.csv` | Parameter bounds |

## Local Setup

```bash
pip install shapleyx
jupyter notebook workshops/transform_sensitivity.ipynb
```

## Expected Runtime

| Environment | Time |
|-------------|------|
| Colab (CPU) | ~4 minutes |
| Local (8-core) | ~3 minutes |

## References

- Bennett & Roberts (2025) — RS-HDMR with ARD for GSA
- Vrugt et al. (2006) — SAC-SMA parameter correlations
- Owen & Prieur (2017) — Shapley effects for dependent inputs
