---
title: "Quick Start Guide"
layout: single
classes: wide
mathjax: true
---

# Quick Start Guide

This guide will walk you through a basic sensitivity analysis using ShapleyX.

## Basic Usage

1. **Import the package and prepare your data**:
```python
import pandas as pd
from shapleyx import rshdmr

# Your data should be a DataFrame with features (X) and target (Y)
# Y should be in a column named 'Y'
data = pd.read_csv('your_data.csv')
```

2. **Initialize the analyzer**:
```python
analyzer = rshdmr(
    data_file=data,
    polys=[10, 5],  # Polynomial orders for expansion
    n_iter=250,     # Number of iterations
    method='ard_cv' # Regression method (ARD with cross-validation)
)
```

3. **Run the analysis**:
```python
sobol_indices, shapley_effects, total_index = analyzer.run_all()
```

4. **View results**:
```python
print("Sobol Indices:")
print(sobol_indices)

print("\nShapley Effects:")
print(shapley_effects)

print("\nTotal Index:")
print(total_index)
```

## Example with Ishigami Function

```python
import numpy as np
from scipy.stats import qmc

def ishigami_function_sample(m):
    n = 3
    a = 7
    b = 0.1
    sampler = qmc.Sobol(d=n, scramble=True, seed=123)
    S = sampler.random_base2(m=m)
    S = S * 2 * np.pi - np.pi
    Y = np.sin(S[:,0]) + a*(np.sin(S[:,1])**2) + b*S[:,2]**4 * np.sin(S[:,0])
    data = pd.DataFrame(S, columns=['X1', 'X2', 'X3'])
    data['Y'] = Y
    return data

# Generate data
data = ishigami_function_sample(8)  # 256 samples

# Run analysis
model = rshdmr(data, polys=[10, 5], n_iter=250, method='ard_cv')
sobol, shap, total = model.run_all()

# Compare different sensitivity methods
results = pd.DataFrame()
results['Sobol'] = sobol['S1']
results['Shapley'] = shap['scaled effect']
results['Total'] = total['total']
print(results)
```

## Visualizing Results

```python
import matplotlib.pyplot as plt

# Plot Sobol indices
sobol.plot(kind='bar')
plt.title('Sobol Indices')
plt.ylabel('Sensitivity')
plt.show()

# Plot Shapley effects
shap.plot(kind='bar')
plt.title('Shapley Effects')
plt.ylabel('Sensitivity')
plt.show()
```

## Next Steps
- Explore advanced options in the [API Reference](api.md)
- Learn about the theoretical foundations in [Theory Background](theory.md)
- Try the [example notebooks](https://github.com/frbennett/shapleyx/tree/main/Examples)
