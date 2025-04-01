# Quick Start Guide

This guide provides basic examples of using ShapleyX for sensitivity analysis.

## Basic Usage

```python
from shapleyx import ShapleyX

# Initialize with your model and parameters
analyzer = ShapleyX(model_function, parameter_ranges)

# Calculate sensitivity indices
results = analyzer.analyze()

# Visualize results
results.plot()
```

## Example Analysis

1. First, define your model function:
```python
def model_function(x):
    return x[:,0]**2 + x[:,1]*x[:,2]
```

2. Set up the parameter ranges:
```python
ranges = {
    'x1': [0, 1],
    'x2': [-1, 1], 
    'x3': [0.5, 1.5]
}
```

3. Run the analysis:
```python
results = ShapleyX(model_function, ranges).analyze()
```

For more detailed examples, see the [Examples](../Examples) directory.