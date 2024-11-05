from scipy.stats import qmc

def xsampler(n, ranges):
    """_summary_

    Args:
        n (_type_): _description_
        ranges (_type_): _description_

    Returns:
        _type_: _description_
    """
    num_features = len(ranges)
    lower = []
    upper = []
    sampler = qmc.LatinHypercube(d=num_features)
    sample = sampler.random(n=n)
    
    for the_range in ranges:
        lower.append(ranges[the_range][0])
        upper.append(ranges[the_range][1])

    sample_scaled = qmc.scale(sample, lower, upper)

    return sample_scaled 