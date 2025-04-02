# Common Tasks

## Handling Large Datasets

```python
# Process data in chunks
analyzer = rshdmr(
    data_file='large_data.csv',
    chunksize=10000,  # Process 10,000 rows at a time
    polys=[5, 3],     # Lower polynomial orders for large datasets
    method='ard'
)
```

## Customizing Polynomial Orders

```python
# Set different polynomial orders for each input
analyzer = rshdmr(
    data_file='data.csv',
    polys=[10, 5, 8, 3, 6],  # Specific orders for each parameter
    method='ard'
)
```

## Saving and Loading Results

```python
# Save results to file
import pickle
with open('sensitivity_results.pkl', 'wb') as f:
    pickle.dump({
        'sobol': sobol_indices,
        'shapley': shapley_effects,
        'total': total_index
    }, f)

# Load results
with open('sensitivity_results.pkl', 'rb') as f:
    results = pickle.load(f)
```

## Comparing Different Methods

```python
# Compare ARD and OLS methods
analyzer_ard = rshdmr(data_file='data.csv', method='ard')
analyzer_ols = rshdmr(data_file='data.csv', method='ols')

ard_results = analyzer_ard.run_all()
ols_results = analyzer_ols.run_all()
```

## Troubleshooting Common Issues

### Memory Errors
- Reduce polynomial orders
- Use smaller chunksize
- Filter less important parameters

### Convergence Issues
- Check data quality
- Try different polynomial orders
- Normalize input data