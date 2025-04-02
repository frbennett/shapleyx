# ShapleyX Documentation

Welcome to the ShapleyX documentation! 

ShapleyX is a Python package for global sensitivity analysis using Sparse Random Sampling - High Dimensional Model Representation (HDMR) with Group Method of Data Handling (GMDH) for parameter selection and linear regression for parameter refinement.

## Documentation Sections

- **Getting Started**: Installation and basic setup
- **Tutorials**: Step-by-step guides for common tasks
- **How-to Guides**: Solutions to specific problems
- **Reference**: Complete API documentation
- **Explanation**: Background and theory

```python
import shapleyx

# Initialize RS-HDMR analyzer
analyzer = shapleyx.rshdmr(data_file='input_data.csv', polys=[10, 5], method='ard')

# Run the entire analysis pipeline
sobol_indices, shapley_effects, total_index = analyzer.run_all()