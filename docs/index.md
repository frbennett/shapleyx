# ShapleyX Documentation

Welcome to the ShapleyX documentation!

ShapleyX is a Python package for global sensitivity analysis using Sparse Random Sampling — High Dimensional Model Representation (RS-HDMR) with Automatic Relevance Determination (ARD) or Orthogonal Matching Pursuit (OMP) for parameter selection and linear regression for parameter refinement.

It computes Sobol indices, Shapley effects, total sensitivity indices, and moment-free measures (PAWN, Delta, H-index). A Monte Carlo Shapley method handles correlated inputs via conditional sampling.

## Documentation Sections

- **Getting Started**: Installation and basic setup
- **Tutorials**: Step-by-step guides for common tasks
- **How-to Guides**: Solutions to specific problems, including MC Shapley for correlated inputs
- **Reference**: Complete API documentation
- **Explanation**: Background and theory

```python
import shapleyx

# Initialize RS-HDMR analyzer
analyzer = shapleyx.rshdmr(data_file='input_data.csv', polys=[10, 5], method='ard')

# Run the entire analysis pipeline
sobol_indices, shapley_effects, total_index = analyzer.run_all()

# Compute MC Shapley effects for correlated inputs
mc_results = analyzer.get_mc_shapley(N=5000, B=500)
```


