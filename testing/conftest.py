"""Shared test fixtures for the ShapleyX test suite."""
import pytest
import numpy as np
from shapleyx.utilities.mc_shapley import (
    MultivariateNormal,
    GaussianCopulaUniform,
    TruncatedMultivariateNormal,
)


# ---------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------

def linear_model(x):
    """f(x) = x_0 + x_1 + x_2"""
    return float(x[0] + x[1] + x[2])


def linear_model_batch(X):
    """Vectorised linear model."""
    return X[:, 0] + X[:, 1] + X[:, 2]


def quadratic_model(x):
    """f(x) = x_0 + 2*x_1**2 + 0.5*x_0*x_2"""
    return float(x[0] + 2.0 * x[1]**2 + 0.5 * x[0] * x[2])


def quadratic_model_batch(X):
    """Vectorised quadratic model."""
    return X[:, 0] + 2.0 * X[:, 1]**2 + 0.5 * X[:, 0] * X[:, 2]


# ---------------------------------------------------------------
# Distribution fixtures
# ---------------------------------------------------------------

@pytest.fixture
def mvn_independent():
    """3D independent standard normal."""
    return MultivariateNormal(mean=np.zeros(3), cov=np.eye(3))


@pytest.fixture
def mvn_correlated():
    """3D correlated normal: rho_01 = 0.7."""
    cov = np.array([
        [1.0, 0.7, 0.0],
        [0.7, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return MultivariateNormal(mean=np.zeros(3), cov=cov)


@pytest.fixture
def gaussian_copula():
    """3D Gaussian copula with uniform marginals on [-pi, pi]."""
    return GaussianCopulaUniform(
        lows=[-np.pi, -np.pi, -np.pi],
        highs=[np.pi, np.pi, np.pi],
        corr=np.eye(3),
    )


@pytest.fixture
def truncated_normal():
    """2D truncated normal on [-1, 1]^2."""
    return TruncatedMultivariateNormal(
        mean=[0.0, 0.0],
        cov=[[1.0, 0.5], [0.5, 1.0]],
        lower=[-1.0, -1.0],
        upper=[1.0, 1.0],
        joint_burn_in=20,
        cond_burn_in=5,
    )


# ---------------------------------------------------------------
# Analytical validation helpers for linear-Gaussian model
# ---------------------------------------------------------------

def v_analytical(u_indices, cov):
    """Closed-form v(u) = Cov[f(X), f(X_u)] for linear f and X ~ N(0, cov)."""
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
    """Analytical Shapley effects for the linear-Gaussian model."""
    import math
    from itertools import combinations

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


def analytical_sobol(cov):
    """Analytical first-order and total-order Sobol indices for the
    linear-Gaussian model with independent inputs."""
    import math
    from itertools import combinations

    d = cov.shape[0]
    v_cache = {}
    for k in range(d + 1):
        for subset in combinations(range(d), k):
            v_cache[frozenset(subset)] = v_analytical(subset, cov)

    v_full = v_cache[frozenset(range(d))]
    S = np.array([v_cache[frozenset([i])] / v_full for i in range(d)])
    T = np.array([
        1.0 - v_cache[frozenset([j for j in range(d) if j != i])] / v_full
        for i in range(d)
    ])
    return S, T
