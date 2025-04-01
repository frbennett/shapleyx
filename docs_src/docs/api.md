# API Reference

## ShapleyX Class

```python
class ShapleyX(model, param_ranges, n_samples=1000, order=2)
```
Main class for sensitivity analysis.

**Parameters**:
- `model`: Callable that takes parameter array and returns output
- `param_ranges`: Dict of parameter names to [min,max] ranges
- `n_samples`: Number of samples for analysis
- `order`: Maximum interaction order to compute

**Methods**:

### analyze()
```python
analyze(confidence=0.95, n_bootstrap=100)
```
Run the sensitivity analysis.

**Returns**: Results object containing:
- First-order indices
- Total-order indices
- Interaction effects
- Confidence intervals

### plot()
```python
plot(kind='bar', params=None)
```
Visualize sensitivity results.

**Parameters**:
- `kind`: Plot type ('bar', 'pie', 'waterfall')
- `params`: Specific parameters to plot (default: all)

## Results Class

Stores and processes sensitivity analysis results.

**Attributes**:
- `first_order`: DataFrame of first-order indices
- `total_order`: DataFrame of total-order indices
- `interactions`: Dict of interaction effects
- `stats`: Model performance statistics