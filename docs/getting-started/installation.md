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

### With streaming acceleration (recommended)

```bash
pip install shapleyx[streaming]
```

Adds **Numba** for JIT-compiled regression.  Streaming Orthogonal
Matching Pursuit (OMP) runs **4–10× faster** via parallel
correlation scans.  The first run compiles Numba kernels (~30–60s);
subsequent runs use cached compiled code on disk.  Without Numba
the streaming path still works via a pure-NumPy fallback.

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
| `numpy` | `numba` (included in `pip install shapleyx[streaming]`) |
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
