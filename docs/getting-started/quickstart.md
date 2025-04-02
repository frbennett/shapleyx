# Quickstart Guide

This guide will walk you through a basic sensitivity analysis using ShapleyX.

## Loading Data

First, prepare your data in CSV format with columns for input parameters and one column for output (named 'Y'):

```python
import pandas as pd

# Load your data
data = pd.read_csv('input_data.csv')
```

## Running Analysis

```python
from shapleyx import rshdmr

# Initialize analyzer
analyzer = rshdmr(
    data_file='input_data.csv',  # or pass DataFrame directly
    polys=[10, 5],              # polynomial orders
    method='ard',               # regression method
    verbose=True                # show progress
)

# Run complete analysis pipeline
sobol_indices, shapley_effects, total_index = analyzer.run_all()
```

## Viewing Results

```python
# Sobol indices
print("Sobol Indices:")
print(sobol_indices)

# Shapley effects  
print("\nShapley Effects:")
print(shapley_effects)

# Total indices
print("\nTotal Indices:")
print(total_index)
```

## Plotting Results

```python
# Plot predicted vs actual
analyzer.plot_hdmr()
```

## Next Steps

- See [Tutorials](../tutorials/basic-usage.md) for more detailed examples
- Explore [How-to Guides](../how-to-guides/common-tasks.md) for customization options