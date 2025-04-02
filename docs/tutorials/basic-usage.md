# Basic Usage Tutorial

This tutorial walks through a complete sensitivity analysis workflow with ShapleyX.

## Step 1: Prepare Your Data

```python
import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.rand(1000, 5)  # 1000 samples, 5 parameters
Y = X[:,0]**2 + 2*X[:,1]*X[:,2] + np.sin(X[:,3]) + X[:,4]

# Create DataFrame
data = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])
data['Y'] = Y
data.to_csv('sample_data.csv', index=False)
```

## Step 2: Initialize the Analyzer

```python
from shapleyx import rshdmr

analyzer = rshdmr(
    data_file='sample_data.csv',
    polys=[10, 5],  # Polynomial orders
    method='ard',   # Automatic Relevance Determination
    verbose=True
)
```

## Step 3: Run the Analysis

```python
# Run complete analysis pipeline
sobol_indices, shapley_effects, total_index = analyzer.run_all()

# View results
print("Sobol Indices:")
print(sobol_indices)

print("\nShapley Effects:")
print(shapley_effects)

print("\nTotal Indices:")
print(total_index)
```

## Step 4: Visualize Results

```python
# Plot predicted vs actual
analyzer.plot_hdmr()

# Plot sensitivity indices
analyzer.plot_indices()
```

## Next Steps

- Try with your own dataset
- Experiment with different polynomial orders
- Explore advanced configuration options