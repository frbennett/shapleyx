# Installation

## Prerequisites

- Python 3.7 or higher
- pip package manager

## Installing ShapleyX

### From PyPI (recommended)

```bash
pip install shapleyx
```

To upgrade an existing installation:

```bash
pip install --upgrade shapleyx
```

### From GitHub (development version)

```bash
pip install https://github.com/frbennett/shapleyx/archive/main.zip
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
| `numpy` | `numba` (compiled bootstrap & Legendre evaluation) |
| `scipy` | `tqdm` (progress bars during MC sampling) |
| `pandas` | |
| `matplotlib` | |
| `scikit-learn` | |

## Verifying Installation

```python
from importlib.metadata import version
print(f"ShapleyX v{version('shapleyx')}")
```
