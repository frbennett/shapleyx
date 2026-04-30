# Import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import math

from shapleyx.utilities.mc_shapley import (
    MultivariateNormal,
    shapley_effects,
)

from importlib.metadata import version
print(f"Running on ShapleyX v{version('shapleyx')}")


def linear_model(x):
    """Linear model f(x) = x_1 + x_2 + x_3."""
    return x[0] + x[1] + x[2]


def v_analytical(u_indices, cov):
    """Closed-form v(u) for the linear-Gaussian model.

    v(u) = Cov[f(X), f(X_u)] where X ~ N(0, cov).
    """
    d = cov.shape[0]
    u = np.asarray(list(u_indices))
    if len(u) == 0:
        return 0.0
    if len(u) == d:
        return cov.sum()

    v_idx = np.setdiff1d(np.arange(d), u)
    ones_u = np.ones(len(u))
    ones_v = np.ones(len(v_idx))

    Sigma_uu = cov[np.ix_(u, u)]
    Sigma_uv = cov[np.ix_(u, v_idx)]
    Sigma_vu = cov[np.ix_(v_idx, u)]
    Sigma_vv = cov[np.ix_(v_idx, v_idx)]

    inv_Sigma_uu = np.linalg.inv(Sigma_uu)

    term1 = ones_u @ Sigma_uu @ ones_u
    term2 = 2.0 * ones_u @ Sigma_uv @ ones_v
    term3 = ones_v @ Sigma_vu @ inv_Sigma_uu @ Sigma_uv @ ones_v

    return term1 + term2 + term3


def analytical_shapley(cov):
    """Compute analytical Shapley effects for the linear-Gaussian model.

    Returns normalised effects (summing to 1).
    """
    d = cov.shape[0]
    v_cache = {}
    for k in range(d + 1):
        for subset in combinations(range(d), k):
            v_cache[frozenset(subset)] = v_analytical(subset, cov)

    sh = np.zeros(d)
    for i in range(d):
        for k in range(d):
            for subset in combinations(set(range(d)) - {i}, k):
                u = frozenset(subset)
                u_with_i = u.union({i})
                diff = v_cache[u_with_i] - v_cache[u]
                weight = (math.factorial(k) * math.factorial(d - k - 1)
                         / math.factorial(d))
                sh[i] += weight * diff

    total_var = v_cache[frozenset(range(d))]
    return sh / total_var


def mc_shapley_run(cov, N=20000, method='exhaustive', B=500,
                   random_state=42):
    """Run MC Shapley for a given covariance.

    Returns a DataFrame with effects and CI bounds.
    """
    d = cov.shape[0]
    joint = MultivariateNormal(mean=np.zeros(d), cov=cov)
    eff, sh, var, lower, upper = shapley_effects(
        linear_model, joint, N=N, method=method,
        B=B, alpha=0.05, random_state=random_state, progress=True
    )
    return pd.DataFrame({
        'variable': [f'X{i+1}' for i in range(d)],
        'effect': eff,
        'lower': lower,
        'upper': upper,
    })


cov_indep = np.eye(3)
analytical_indep = analytical_shapley(cov_indep)
print("Analytical Shapley effects (independent):", analytical_indep)

mc_indep = mc_shapley_run(cov_indep, N=10000, B=300)
print(mc_indep)


