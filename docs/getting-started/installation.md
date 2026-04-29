# Installation

## Prerequisites

- Python 3.7 or higher
- pip package manager

## Installing ShapleyX

Not currently available from PyPI:



Install from source:

```bash
git clone https://github.com/frbennett/shapleyx.git
cd shapleyx
pip install .
```

## Dependencies

ShapleyX requires the following Python packages:

- numpy
- scipy
- pandas
- scikit-learn
- matplotlib

These will be installed automatically when installing ShapleyX.

## Verifying Installation

After installation, you can verify it works by running:

```python
import shapleyx
print(shapleyx.__version__)