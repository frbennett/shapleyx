# SAC-SMA Flow Regime Sensitivity Workshop

**ShapleyX + Sacramento Rainfall-Runoff Model — Hands-on Workshop**

Compare parameter sensitivity between total-flow and high-flow NSE using Monte Carlo Shapley effects with correlated inputs.

## Quick Start

| Venue | Click |
|-------|-------|
| **Google Colab** (recommended) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frbennett/shapleyx/blob/main/workshops/flow_regime_sensitivity.ipynb) |
| **Binder** (no account needed) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frbennett/shapleyx/main?labpath=workshops%2Fflow_regime_sensitivity.ipynb) |

## What You'll Do

1. **Build a surrogate** — RS-HDMR with streaming OMP-CV on 4,096 Sobol' samples of the 14-parameter SAC-SMA model
2. **Define flow regimes** — total-flow NSE (all 16,398 days) vs high-flow NSE (251 days contributing 50% of total discharge)
3. **Compute Shapley effects** — Monte Carlo pick-freeze with Vrugt et al. (2006) posterior correlations
4. **Compare regimes** — which parameters matter for baseflow vs storm response?

## Files

| File | Purpose |
|------|---------|
| `flow_regime_sensitivity.ipynb` | Workshop notebook (45 cells) |
| `data/sacramento.py` | SAC-SMA model (Numba JIT) |
| `data/st_helens_forcing_original.csv` | 46 years daily rainfall, PET, discharge |
| `data/es_parameters.csv` | Parameter bounds |
| `data/flow_regime_shapley_comparison.csv` | Pre-computed results (if you skip the long computation) |

## Local Setup

```bash
pip install shapleyx
jupyter notebook workshops/flow_regime_sensitivity.ipynb
```

The notebook auto-detects Colab vs local and adjusts paths accordingly.

## Expected Runtime

| Environment | Time |
|-------------|------|
| Colab (CPU) | ~5 minutes |
| Colab (GPU, T4) | ~3 minutes |
| Local (8-core) | ~3 minutes |

## References

- Bennett & Roberts (2025) — RS-HDMR with ARD for GSA
- Vrugt et al. (2006) — SAC-SMA parameter correlations
- Owen & Prieur (2017) — Shapley effects for dependent inputs
- Burnash et al. (1973) — SAC-SMA model description
