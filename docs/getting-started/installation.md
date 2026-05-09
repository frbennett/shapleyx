# Installation

## Prerequisites

- Python 3.7 or higher
- pip package manager

## Installing ShapleyX

### Standard installation (from PyPI)

```bash
pip install shapleyx
```

This installs the core package with all required dependencies
(numpy, scipy, pandas, matplotlib, scikit-learn).

### With Numba acceleration (recommended)

```bash
pip install shapleyx[streaming]
```

Adds **Numba** for JIT-compiled computation throughout the package.
The first run compiles Numba kernels (~30--60s); subsequent runs use
cached compiled code on disk.  Without Numba everything still works via
pure-NumPy fallbacks.  Numba accelerates three independent subsystems:

| Subsystem | Modules | Typical speedup |
|-----------|---------|----------------|
| **Streaming OMP** | `utilities/streaming.py` | 4--10× on correlation scans, parallelised via `prange` |
| **Surrogate prediction** | `utilities/predictor.py` | 5--6× on single-sample MC Shapley evaluations |
| **MC Shapley bootstrap** | `utilities/mc_shapley.py` | 3--10× on exhaustive $B$-iteration bootstrap loops |

### With development tools

```bash
pip install shapleyx[dev]
```

Adds pytest, mypy, flake8, and Jupyter for local development.

### From GitHub (development version)

```bash
pip install https://github.com/frbennett/shapleyx/archive/main.zip
```

To upgrade an existing installation:

```bash
pip install --upgrade shapleyx
```

Or clone and install in development mode:

```bash
git clone https://github.com/frbennett/shapleyx.git
cd shapleyx
pip install -e .
```

## Dependencies

ShapleyX requires the following Python packages (installed automatically):

| Required | Optional |
|---|---|
| `numpy` | `numba` — accelerates streaming OMP, surrogate prediction,<br/>and MC Shapley bootstrap (`pip install shapleyx[streaming]`) |
| `scipy` | `tqdm` (progress bars during MC sampling) |
| `pandas` | |
| `matplotlib` | |
| `scikit-learn` | |

*Example with the streaming extra:*
```bash
pip install shapleyx[streaming]
```

## Verifying Installation

```python
from importlib.metadata import version
print(f"ShapleyX v{version('shapleyx')}")
```
