# -*- coding: utf-8 -*-
"""Tests for the mixed-marginal copula classes.

Covers: the marginal spec registry (all families, lognormal parameter
forms, passthrough contract), the Gaussian kernel (correlations, exact
stream equivalence with the notebook reference implementation, Rosenblatt
reconstruction), the t kernel (composite construction, exact conditional
sampler, no RQMC yet), and end-to-end integration with ``shapley_effects``.
"""
import numpy as np
import pytest
from scipy.stats import (
    norm,
    t as student_t,
    multivariate_t,
    multivariate_normal,
    lognorm as scipy_lognorm,
    truncnorm as scipy_truncnorm,
)

from shapleyx.utilities.mc_shapley import (
    GaussianCopulaMixed,
    TCopulaMixed,
    MultivariateNormal,
)


# --------------------------------------------------------------------- #
# Fixtures — cantilever beam specification (Demange-Chryst et al. 2022)  #
# --------------------------------------------------------------------- #

def lognorm_params(mean, cv):
    s = np.sqrt(np.log(1 + cv**2))
    return np.log(mean) - 0.5 * s**2, s


def normal_params(mean, cv):
    return mean, mean * cv


CANTILEVER_MARGINALS = {
    'FX': ('lognormal', {'mean': 556.8, 'cv': 0.08}),
    'FY': ('lognormal', {'mean': 453.6, 'cv': 0.08}),
    'E':  ('lognormal', {'mean': 200e9, 'cv': 0.06}),
    'lX': ('normal', *normal_params(0.062, 0.1)),
    'lY': ('normal', *normal_params(0.0987, 0.1)),
    'L':  ('normal', *normal_params(4.29, 0.1)),
}

CANTILEVER_CORR = np.array([
    [1.0, 0.0, 0.0, 0.0,  0.0,  0.0],
    [0.0, 1.0, 0.0, 0.0,  0.0,  0.0],
    [0.0, 0.0, 1.0, 0.0,  0.0,  0.0],
    [0.0, 0.0, 0.0, 1.0, -0.55, 0.45],
    [0.0, 0.0, 0.0, -0.55, 1.0, 0.45],
    [0.0, 0.0, 0.0, 0.45, 0.45, 1.0],
])


class _RefGaussianCopulaMixed:
    """Reference implementation: the notebook's GaussianCopulaMixed (exact
    RNG-stream behaviour).  Used to prove stream equivalence, which is what
    guarantees the published cantilever numbers reproduce exactly."""

    def __init__(self, marginals, latent_corr):
        self.d = len(marginals)
        self.labels = list(marginals.keys())
        self._marginals = marginals
        self._mvn = MultivariateNormal(mean=np.zeros(self.d), cov=latent_corr)

    @staticmethod
    def _to_u(x, dist, mu, sigma):
        if dist == 'lognormal':
            x = np.clip(np.asarray(x), 1e-15, None)
            return norm.cdf((np.log(x) - mu) / sigma)
        return norm.cdf((np.asarray(x) - mu) / sigma)

    @staticmethod
    def _from_u(u, dist, mu, sigma):
        if dist == 'lognormal':
            return np.exp(mu + sigma * norm.ppf(np.clip(np.asarray(u), 1e-12, 1 - 1e-12)))
        return mu + sigma * norm.ppf(np.clip(np.asarray(u), 1e-12, 1 - 1e-12))

    def sample_joint(self, n):
        Z = self._mvn.sample_joint(n)
        X = np.zeros_like(Z)
        for j in range(self.d):
            dist, mu, sigma = self._marginals[self.labels[j]]
            X[:, j] = self._from_u(norm.cdf(Z[:, j]), dist, mu, sigma)
        return X

    def sample_conditional_batch(self, u_indices, fixed_X):
        u = np.asarray(u_indices)
        N = fixed_X.shape[0]
        Z_fixed = np.zeros((N, len(u)))
        for k, idx in enumerate(u):
            dist, mu, sigma = self._marginals[self.labels[idx]]
            Z_fixed[:, k] = norm.ppf(self._to_u(fixed_X[:, k], dist, mu, sigma))
        Z_cond = self._mvn.sample_conditional_batch(u, Z_fixed)
        X = np.zeros_like(Z_cond)
        for j in range(self.d):
            dist, mu, sigma = self._marginals[self.labels[j]]
            X[:, j] = self._from_u(norm.cdf(Z_cond[:, j]), dist, mu, sigma)
        return X


def _ref_marginals():
    return {
        'FX': ('lognormal', *lognorm_params(556.8, 0.08)),
        'FY': ('lognormal', *lognorm_params(453.6, 0.08)),
        'E':  ('lognormal', *lognorm_params(200e9, 0.06)),
        'lX': ('normal', *normal_params(0.062, 0.1)),
        'lY': ('normal', *normal_params(0.0987, 0.1)),
        'L':  ('normal', *normal_params(4.29, 0.1)),
    }


def _cantilever_gaussian():
    return GaussianCopulaMixed(CANTILEVER_MARGINALS, CANTILEVER_CORR)


def _cantilever_t(nu=5.0):
    return TCopulaMixed(CANTILEVER_MARGINALS, CANTILEVER_CORR, nu=nu)


# --------------------------------------------------------------------- #
# Marginal spec registry                                                 #
# --------------------------------------------------------------------- #

def test_registry_all_families_sample():
    """Every registered family resolves and produces finite in-support samples."""
    specs = {
        'n':  ('normal', 1.0, 2.0),
        'ln': ('lognormal', {'mean': 2.0, 'cv': 0.3}),
        'u':  ('uniform', 0.0, 1.0),
        'tn': ('truncnorm', -1.0, 3.0, 1.0, 2.0),
        'b':  ('beta', 2.0, 5.0, 0.0, 1.0),
        'g':  ('gamma', 2.0, 3.0),
        'e':  ('exponential', 2.0),
        'w':  ('weibull', 1.5, 2.0),
        'gu': ('gumbel', 0.0, 1.0),
        'f':  ('frechet', 2.0, 1.0),
        'p':  ('pareto', 2.0, 1.0),
    }
    joint = GaussianCopulaMixed(specs, np.eye(len(specs)))
    X = joint.sample_joint(50_000)
    assert np.isfinite(X).all()
    # support checks
    assert X[:, 2].min() >= 0.0 and X[:, 2].max() <= 1.0          # uniform
    assert X[:, 3].min() >= -1.0 and X[:, 3].max() <= 3.0         # truncnorm
    assert X[:, 4].min() >= 0.0 and X[:, 4].max() <= 1.0          # beta
    assert X[:, 5].min() > 0.0                                    # gamma
    assert X[:, 6].min() > 0.0                                    # exponential
    assert X[:, 7].min() > 0.0                                    # weibull
    assert X[:, 9].min() > 0.0                                    # frechet
    assert X[:, 10].min() >= 1.0                                  # pareto (scale)


def test_lognormal_mean_cv_form_moments():
    joint = GaussianCopulaMixed({'X': ('lognormal', {'mean': 556.8, 'cv': 0.08})}, np.eye(1))
    X = joint.sample_joint(200_000)
    assert abs(X.mean() - 556.8) / 556.8 < 0.01
    assert abs(X.std() - 44.54) / 44.54 < 0.03


def test_lognormal_two_forms_equivalent():
    a = GaussianCopulaMixed({'X': ('lognormal', {'mean': 556.8, 'cv': 0.08})}, np.eye(1))
    mu, s = lognorm_params(556.8, 0.08)
    b = GaussianCopulaMixed({'X': ('lognormal', {'mu': mu, 'sigma': s})}, np.eye(1))
    for u in (0.01, 0.5, 0.99):
        assert abs(a._ppf[0](u) - b._ppf[0](u)) < 1e-10 * a._ppf[0](u)


def test_lognormal_flat_tuple_raises():
    with pytest.raises(TypeError):
        GaussianCopulaMixed({'X': ('lognormal', 556.8, 0.08)}, np.eye(1))


def test_lognormal_both_or_neither_forms_raise():
    with pytest.raises(TypeError):
        GaussianCopulaMixed({'X': ('lognormal', {'mean': 1.0, 'cv': 0.1, 'mu': 0.0, 'sigma': 0.1})}, np.eye(1))
    with pytest.raises(TypeError):
        GaussianCopulaMixed({'X': ('lognormal', {})}, np.eye(1))


def test_dict_spec_form():
    joint = GaussianCopulaMixed({'X': {'family': 'lognormal', 'mean': 556.8, 'cv': 0.08}}, np.eye(1))
    X = joint.sample_joint(100_000)
    assert abs(X.mean() - 556.8) / 556.8 < 0.01
    with pytest.raises(TypeError):
        GaussianCopulaMixed({'X': {'family': 'normal', 'mean': 0.0, 'std': 1.0}}, np.eye(1))


def test_unknown_family_raises():
    with pytest.raises(TypeError):
        GaussianCopulaMixed({'X': ('cauchyish', 1.0)}, np.eye(1))


def test_passthrough_contract():
    # scipy distribution
    j1 = GaussianCopulaMixed({'X': scipy_truncnorm(-2, 2, loc=1.0, scale=0.5)}, np.eye(1))
    X = j1.sample_joint(10_000)
    assert X.min() >= 0.0 and X.max() <= 2.0
    # (cdf, ppf) tuple
    j2 = GaussianCopulaMixed({'X': (norm.cdf, norm.ppf)}, np.eye(1))
    assert np.isfinite(j2.sample_joint(1000)).all()
    # callable PPF only: joint OK, conditional raises RuntimeError
    j3 = GaussianCopulaMixed({'X': lambda u: u * 10.0}, np.eye(1))
    assert np.isfinite(j3.sample_joint(1000)).all()
    with pytest.raises(RuntimeError):
        j3.sample_conditional_batch([0], np.array([[5.0]]))
    # None / 'uniform'
    j4 = GaussianCopulaMixed({'X': None, 'Y': 'uniform'}, np.eye(2))
    X4 = j4.sample_joint(1000)
    assert X4.min() >= 0.0 and X4.max() <= 1.0


def test_corr_shape_validation():
    with pytest.raises(ValueError):
        GaussianCopulaMixed({'X': ('normal', 0.0, 1.0)}, np.eye(2))


# --------------------------------------------------------------------- #
# Gaussian kernel                                                        #
# --------------------------------------------------------------------- #

def test_gaussian_empirical_correlations():
    joint = _cantilever_gaussian()
    np.random.seed(0)
    X = joint.sample_joint(200_000)
    emp = np.corrcoef(X[:, 3:].T)
    np.testing.assert_allclose(emp, CANTILEVER_CORR[3:, 3:], atol=0.01)


def test_gaussian_conditional_preserves_fixed():
    joint = _cantilever_gaussian()
    np.random.seed(0)
    X0 = joint.sample_joint(1000)
    for u in ([3], [3, 4], [0, 3, 4, 5], [0, 1, 2]):
        C = joint.sample_conditional_batch(u, X0[:, u])
        assert np.array_equal(C[:, u], X0[:, u]), f"u={u}"


def test_gaussian_rosenblatt():
    """U1|U2 via the conditional sampler must reconstruct the Gaussian copula."""
    rho = 0.45
    joint = GaussianCopulaMixed({'A': ('normal', 0.0, 1.0), 'B': ('normal', 0.0, 1.0)},
                                np.array([[1.0, rho], [rho, 1.0]]))
    n = 200_000
    U2 = np.random.default_rng(7).uniform(size=n)
    np.random.seed(7)
    X = joint.sample_conditional_batch([1], norm.ppf(U2)[:, None])
    U1 = norm.cdf(X[:, 0])
    theo = multivariate_normal([0, 0], [[1.0, rho], [rho, 1.0]])
    errs = []
    for a in (0.25, 0.5, 0.75, 0.95):
        for b in (0.25, 0.5, 0.75, 0.95):
            emp = np.mean((U1 <= a) & (U2 <= b))
            errs.append(abs(emp - theo.cdf([norm.ppf(a), norm.ppf(b)])))
    assert max(errs) < 2.5e-3


def test_gaussian_stream_equivalence_with_notebook_reference():
    """THE regression: identical RNG streams -> identical draws as the
    notebook's GaussianCopulaMixed, hence the published cantilever numbers
    (FX=0.1504, lX=0.2818, lY=0.2616, L=0.2066 at N=1e5, B=200, seed 42)
    reproduce exactly."""
    new = _cantilever_gaussian()
    ref = _RefGaussianCopulaMixed(_ref_marginals(), CANTILEVER_CORR)

    np.random.seed(42)
    X_new = new.sample_joint(5000)
    np.random.seed(42)
    X_ref = ref.sample_joint(5000)
    np.testing.assert_allclose(X_new, X_ref, rtol=1e-12, atol=0)

    u = np.array([3, 4])
    np.random.seed(42)
    _ = new.sample_joint(5000)                      # consume identical prefix
    fixed = new.sample_joint(1000)[:, u]
    np.random.seed(42)
    _ = new.sample_joint(5000)
    C_new = new.sample_conditional_batch(u, fixed)
    np.random.seed(42)
    _ = ref.sample_joint(5000)
    C_ref = ref.sample_conditional_batch(u, fixed)
    np.testing.assert_allclose(C_new, C_ref, rtol=1e-12, atol=0)


def test_gaussian_deterministic_paths():
    joint = _cantilever_gaussian()
    U = np.random.default_rng(3).uniform(size=(2000, 6))
    X = joint.sample_joint_deterministic(U)
    assert np.isfinite(X).all()
    X0 = joint.sample_joint(500)
    u = np.array([3, 4])
    Uc = np.random.default_rng(4).uniform(size=(500, 4))
    C = joint.sample_conditional_batch_deterministic(u, X0[:, u], Uc)
    assert np.array_equal(C[:, u], X0[:, u])


# --------------------------------------------------------------------- #
# t kernel                                                               #
# --------------------------------------------------------------------- #

def test_t_nu_validation():
    with pytest.raises(ValueError):
        TCopulaMixed(CANTILEVER_MARGINALS, CANTILEVER_CORR, nu=2.0)


def test_t_empirical_correlations_within_drift():
    """Physical Pearson tracks latent R within the documented t-copula drift."""
    joint = _cantilever_t(nu=5.0)
    np.random.seed(1)
    X = joint.sample_joint(200_000)
    emp = np.corrcoef(X[:, 3:].T)
    np.testing.assert_allclose(emp, CANTILEVER_CORR[3:, 3:], atol=0.03)


def test_t_independent_block_uncorrelated():
    joint = _cantilever_t(nu=3.0)
    np.random.seed(1)
    X = joint.sample_joint(200_000)
    r = np.corrcoef(X.T)
    for i in (0, 1, 2):                            # lognormal block
        for j in (3, 4, 5):                        # coupled block
            assert abs(r[i, j]) < 0.03, f"spurious cross-block corr ({i},{j})={r[i,j]:.3f}"


def test_t_conditional_preserves_fixed():
    joint = _cantilever_t(nu=5.0)
    np.random.seed(0)
    X0 = joint.sample_joint(1000)
    for u in ([3], [3, 4], [0, 3, 4, 5], [5], [3, 5]):
        C = joint.sample_conditional_batch(u, X0[:, u])
        assert np.array_equal(C[:, u], X0[:, u]), f"u={u}"


def test_t_rosenblatt():
    """The t-copula conditional sampler reconstructs the t-copula exactly."""
    rho = 0.45
    nu = 5.0
    joint = TCopulaMixed({'A': ('normal', 0.0, 1.0), 'B': ('normal', 0.0, 1.0)},
                         np.array([[1.0, rho], [rho, 1.0]]), nu=nu)
    n = 200_000
    U2 = np.random.default_rng(7).uniform(size=n)
    np.random.seed(7)
    X = joint.sample_conditional_batch([1], norm.ppf(U2)[:, None])
    U1 = norm.cdf(X[:, 0])
    theo = multivariate_t([0, 0], [[1.0, rho], [rho, 1.0]], df=nu)
    errs = []
    for a in (0.25, 0.5, 0.75, 0.95):
        for b in (0.25, 0.5, 0.75, 0.95):
            emp = np.mean((U1 <= a) & (U2 <= b))
            errs.append(abs(emp - theo.cdf([student_t.ppf(a, nu), student_t.ppf(b, nu)])))
    assert max(errs) < 2.5e-3


def test_t_deterministic_not_implemented():
    joint = _cantilever_t(nu=5.0)
    with pytest.raises(NotImplementedError):
        joint.sample_joint_deterministic(np.random.rand(10, 6))
    with pytest.raises(NotImplementedError):
        joint.sample_conditional_batch_deterministic([0], np.random.rand(10, 1), np.random.rand(10, 5))


def test_t_tail_dependence():
    """Empirical upper-tail frequency tracks lambda_t(nu, rho)."""
    from shapleyx.utilities.mc_shapley import TCopulaMixed as _T
    nu, rho = 5.0, 0.45
    joint = TCopulaMixed({'A': ('normal', 0.0, 1.0), 'B': ('normal', 0.0, 1.0)},
                         np.array([[1.0, rho], [rho, 1.0]]), nu=nu)
    np.random.seed(3)
    X = joint.sample_joint(2_000_000)
    U = norm.cdf(X)
    m = U[:, 0] > 0.995
    lam_emp = np.mean(U[m, 1] > 0.995)
    lam_th = 2 * student_t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1)
    assert abs(lam_emp - lam_th) < 0.06


# --------------------------------------------------------------------- #
# Integration with shapley_effects                                       #
# --------------------------------------------------------------------- #

def _failure_indicator(x):
    FX, FY, E, lX, lY, L = x
    D = (4 * L**3) / (E * lX * lY) * np.sqrt((FX / lX**2)**2 + (FY / lY**2)**2)
    return 1.0 if D > 0.066 else 0.0


def _failure_indicator_batch(X):
    FX, FY, E, lX, lY, L = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    D = (4 * L**3) / (E * lX * lY) * np.sqrt((FX / lX**2)**2 + (FY / lY**2)**2)
    return (D > 0.066).astype(float)


@pytest.mark.parametrize("joint", [
    _cantilever_gaussian(),
    _cantilever_t(nu=5.0),
], ids=["gaussian", "t-nu5"])
def test_shapley_effects_integration(joint):
    from shapleyx.utilities.mc_shapley import shapley_effects
    eff, sh, var, lo, hi = shapley_effects(
        _failure_indicator, joint, N=2000, method='exhaustive', B=20,
        alpha=0.05, predict_batch=_failure_indicator_batch, random_state=42,
        progress=False,
    )
    assert np.isfinite(eff).all()
    assert abs(eff.sum() - 1.0) < 1e-6          # effects sum to 1 by construction
    assert np.all(hi >= lo)
    assert len(eff) == 6
