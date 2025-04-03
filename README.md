# ShapleyX - Global Sensitivity Analysis with RS-HDMR

ShapleyX is a Python package for global sensitivity analysis using Random Sampling High-Dimensional Model Representation (RS-HDMR) with the Group Method of Data Handling (GMDH) for parameter selection and linear regression for parameter refinement.

## Features

- **Multiple Sensitivity Analysis Methods**:
  - Sobol sensitivity indices to arbitrary order
  - Shapley effects
  - Owen-Shapley interactions
  - PAWN distribution-based sensitivity analysis

- **Advanced Techniques**:
  - Legendre polynomial expansion
  - Automatic Relevance Determination (ARD) regression
  - Bootstrap resampling for confidence intervals
  - Model performance statistics and visualization

## Installation

### From GitHub (latest version)
```bash
pip install https://github.com/frbennett/shapleyx/archive/main.zip
```

To upgrade:
```bash
pip uninstall -y shapleyx
pip install https://github.com/frbennett/shapleyx/archive/main.zip
```

### Development Installation
```bash
git clone https://github.com/frbennett/shapleyx.git
cd shapleyx
python setup.py develop
```

## Dependencies
- numpy
- pandas
- matplotlib
- scipy

## Documentation

Full documentation is available at:
- [Document Home](https://frbennett.github.io/shapleyx/)
- [Quick Start Guide](https://frbennett.github.io/shapleyx/getting-started/quickstart/)
- [Theory Background](https://frbennett.github.io/shapleyx/explanation/theory/)
- [API Reference](https://frbennett.github.io/shapleyx/reference/api/)

## Examples

See the `Examples/` directory for Jupyter notebooks demonstrating usage:
- [Ishigami Function Example](Examples/ishigami.ipynb)
- [Advanced Legendre Expansion](Examples/ishigami_new_legendre.ipynb)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
