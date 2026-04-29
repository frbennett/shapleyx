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

## Step 4: Visualize and Explore Results

```python
# Plot predicted vs actual
analyzer.run_plot_hdmr()

# Examine Shapley effects
print(analyzer.shap)

# Compute PAWN sensitivity (distribution-based, no surrogate needed)
pawn = analyzer.get_pawn(S=10)
print(pawn)
```

## Step 5: MC Shapley for Correlated Inputs

When inputs may be correlated, use the Monte Carlo Shapley method:

```python
# Independent inputs via the surrogate model
mc_indep = analyzer.get_mc_shapley(N=2000, B=200)

# With correlation
import numpy as np
corr = np.array([[1.0, 0.5, 0.0, 0.0, 0.0],
                 [0.5, 1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 1.0]])
mc_corr = analyzer.get_mc_shapley(corr=corr, N=2000, B=200)
```

## Next Steps

- Try with your own dataset
- Experiment with different polynomial orders and regression methods
- See the [MC Shapley guide](../how-to-guides/mc-shapley.md) for correlated inputs
- Explore the [Common Tasks](../how-to-guides/common-tasks.md) for advanced usage