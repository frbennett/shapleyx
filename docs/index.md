---
title: "ShapleyX Documentation"
layout: single
classes: wide
mathjax: true
---
# Welcome to ShapleyX

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

## Getting Started

For a step-by-step guide on how to use ShapleyX, including code examples and visualization, please see the **[Quick Start Guide](quick_start.md)**.

## Documentation

- **[Quick Start Guide](quick_start.md)**: Basic usage and examples.
- **[Theory Background](theory.md)**: Explanation of the underlying methods and mathematics.
- **[API Reference](api.md)**: Detailed description of the classes, methods, and parameters.

## Examples

See the `Examples/` directory in the [repository](https://github.com/frbennett/shapleyx/tree/main/Examples) for Jupyter notebooks demonstrating usage:
- Ishigami Function Example
- Advanced Legendre Expansion

## Dependencies
- numpy
- pandas
- matplotlib
- scipy

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/frbennett/shapleyx/blob/main/LICENSE) file in the repository for details.
