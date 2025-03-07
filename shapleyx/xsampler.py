from scipy.stats import qmc
import numpy as np 

def xsampler(num_samples: int, ranges: dict) -> np.ndarray:
    """
    Generate a Latin Hypercube sample scaled to the specified ranges.

    Args:
        num_samples (int): Number of samples to generate.
        ranges (dict): A dictionary where keys are feature names and values are tuples of (lower, upper) bounds.

    Returns:
        np.ndarray: A scaled Latin Hypercube sample of shape (num_samples, num_features).
    """
    num_features = len(ranges)
    
    # Extract lower and upper bounds from the ranges dictionary
    lower_bounds = [bounds[0] for bounds in ranges.values()]
    upper_bounds = [bounds[1] for bounds in ranges.values()]
    
    # Generate Latin Hypercube sample
    sampler = qmc.LatinHypercube(d=num_features)
    sample = sampler.random(n=num_samples)
    
    # Scale the sample to the specified ranges
    sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)
    
    return sample_scaled 


