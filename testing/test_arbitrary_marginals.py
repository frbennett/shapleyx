"""Tests for GaussianCopulaArbitrary."""

import numpy as np
import pytest
from scipy.stats import beta, lognorm, norm as norm_dist, uniform

from shapleyx.utilities.mc_shapley import (
    GaussianCopulaArbitrary,
    GaussianCopulaUniform,
    MultivariateNormal,
    collect_shapley_data,
    shapley_from_data,
    shapley_effects,
)


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def uniform_copula():
    """3D all-uniform — should match GaussianCopulaUniform."""
    return GaussianCopulaArbitrary(
        marginals={'x0': 'uniform', 'x1': 'uniform', 'x2': 'uniform'},
        corr=np.eye(3),
    )


@pytest.fixture
def mixed_copula():
    """3D with Beta, uniform, and LogNormal marginals."""
    return GaussianCopulaArbitrary(
        marginals={
            'x0': beta(a=2, b=5),
            'x1': 'uniform',
            'x2': lognorm(s=0.5),
        },
        corr=np.eye(3),
    )


@pytest.fixture
def correlated_copula():
    """3D Beta marginals with correlation."""
    corr = np.array([
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return GaussianCopulaArbitrary(
        marginals={
            'x0': beta(a=2, b=5),
            'x1': beta(a=5, b=2),
            'x2': 'uniform',
        },
        corr=corr,
    )


@pytest.fixture
def custom_ppf_copula():
    """3D with a custom callable PPF."""
    def my_ppf(u):
        """Linearly map [0,1] → [10, 20]."""
        return 10.0 + 10.0 * u

    return GaussianCopulaArbitrary(
        marginals={
            'x0': my_ppf,
            'x1': 'uniform',
            'x2': (lambda x: x, lambda u: u),  # (cdf, ppf) tuple
        },
        corr=np.eye(3),
    )


# ── Model functions ─────────────────────────────────────────────

def _linear_model(x):
    return float(x[0] + 2 * x[1] + 3 * x[2])


def _linear_model_batch(X):
    return X[:, 0] + 2 * X[:, 1] + 3 * X[:, 2]


# ── Tests: Construction ─────────────────────────────────────────

class TestConstruction:

    def test_uniform_marginals(self):
        """All-uniform should construct without error."""
        joint = GaussianCopulaArbitrary(
            marginals={'a': 'uniform', 'b': None, 'c': 'uniform'},
            corr=np.eye(3),
        )
        assert joint.d == 3

    def test_scipy_marginals(self):
        """Scipy distributions should construct."""
        joint = GaussianCopulaArbitrary(
            marginals={'a': beta(2, 5), 'b': lognorm(s=1.0)},
            corr=np.eye(2),
        )
        assert joint.d == 2

    def test_callable_ppf(self):
        """Callable PPF should construct."""
        joint = GaussianCopulaArbitrary(
            marginals={'a': lambda u: u * 2},
            corr=np.eye(1),
        )
        assert joint.d == 1

    def test_tuple_spec(self):
        """(cdf, ppf) tuple should construct."""
        joint = GaussianCopulaArbitrary(
            marginals={'a': (lambda x: x, lambda u: u)},
            corr=np.eye(1),
        )
        assert joint.d == 1

    def test_corr_shape_mismatch(self):
        """Wrong correlation shape should raise."""
        with pytest.raises(ValueError):
            GaussianCopulaArbitrary(
                marginals={'x0': 'uniform', 'x1': 'uniform'},
                corr=np.eye(3),  # 3×3 for d=2
            )

    def test_invalid_spec_raises(self):
        """Non-callable, non-scipy spec should raise TypeError."""
        with pytest.raises(TypeError):
            GaussianCopulaArbitrary(
                marginals={'x0': 42},  # int not allowed
                corr=np.eye(1),
            )

    def test_tuple_non_callable_raises(self):
        """Tuple with non-callable ppf should raise TypeError."""
        with pytest.raises(TypeError):
            GaussianCopulaArbitrary(
                marginals={'x0': (lambda x: x, 'not callable')},
                corr=np.eye(1),
            )

    def test_empty_marginals(self):
        """d=0 should work."""
        joint = GaussianCopulaArbitrary(
            marginals={},
            corr=np.empty((0, 0)),
        )
        assert joint.d == 0


# ── Tests: Sampling shapes ──────────────────────────────────────

class TestSamplingShapes:

    def test_sample_joint_shape(self, uniform_copula):
        X = uniform_copula.sample_joint(100)
        assert X.shape == (100, 3)

    def test_sample_joint_mixed_shape(self, mixed_copula):
        X = mixed_copula.sample_joint(50)
        assert X.shape == (50, 3)

    def test_sample_conditional_shape(self, uniform_copula):
        X = uniform_copula.sample_conditional([0], np.array([0.5]))
        assert X.shape == (3,)

    def test_sample_conditional_batch_shape(self, uniform_copula):
        fixed = np.array([[0.2], [0.8], [0.5]])
        X = uniform_copula.sample_conditional_batch([0], fixed)
        assert X.shape == (3, 3)

    def test_conditional_empty_u(self, uniform_copula):
        """Conditioning on no variables = joint sample."""
        X = uniform_copula.sample_conditional([], np.array([]))
        assert X.shape == (3,)

    def test_conditional_full_u(self, uniform_copula):
        """Conditioning on all variables should return the fixed values."""
        fixed = np.array([0.1, 0.5, 0.9])
        X = uniform_copula.sample_conditional([0, 1, 2], fixed)
        np.testing.assert_allclose(X, fixed, rtol=0, atol=1e-10)


# ── Tests: Uniform behaviour ────────────────────────────────────

class TestUniformBehaviour:

    def test_joint_in_unit_interval(self, uniform_copula):
        """All samples should be in [0, 1]."""
        X = uniform_copula.sample_joint(5000)
        assert np.all(X >= 0) and np.all(X <= 1)

    def test_uniform_marginal_distribution(self, uniform_copula):
        """Empirical CDF should be approximately uniform."""
        X = uniform_copula.sample_joint(5000)
        for j in range(3):
            # Kolmogorov-Smirnov against Uniform(0,1)
            ks_stat = np.max(np.abs(
                np.sort(X[:, j]) - np.linspace(0, 1, 5000)
            ))
            assert ks_stat < 0.05  # generous bound for 5K samples

    def test_independent_means_zero_correlation(self, uniform_copula):
        """With independent copula, pairwise correlations should be ≈ 0."""
        X = uniform_copula.sample_joint(5000)
        for i in range(3):
            for j in range(i + 1, 3):
                corr_coef = np.corrcoef(X[:, i], X[:, j])[0, 1]
                assert abs(corr_coef) < 0.05


# ── Tests: Scipy marginal behaviour ─────────────────────────────

class TestScipyMarginals:

    def test_beta_marginal_mean(self, mixed_copula):
        """Beta(2,5) has mean a/(a+b) = 2/7 ≈ 0.286."""
        X = mixed_copula.sample_joint(10000)
        mean_x0 = X[:, 0].mean()
        expected = 2 / (2 + 5)
        assert abs(mean_x0 - expected) < 0.02

    def test_lognormal_marginal_median(self, mixed_copula):
        """LogNormal(s=0.5) has median = 1.0."""
        X = mixed_copula.sample_joint(10000)
        median_x2 = np.median(X[:, 2])
        assert abs(median_x2 - 1.0) < 0.05

    def test_uniform_marginal_preserved(self, mixed_copula):
        """'uniform' entries should still be in [0, 1]."""
        X = mixed_copula.sample_joint(1000)
        assert np.all(X[:, 1] >= 0) and np.all(X[:, 1] <= 1)


# ── Tests: Correlation ──────────────────────────────────────────

class TestCorrelation:

    def test_rank_correlation_matches(self, correlated_copula):
        """Spearman rank correlation should approximately match input."""
        X = correlated_copula.sample_joint(10000)
        # Rank correlation (Spearman) for Gaussian copula ≈ 6/π arcsin(ρ/2)
        # For ρ=0.5: expected ≈ 0.48
        from scipy.stats import spearmanr
        rho_01, _ = spearmanr(X[:, 0], X[:, 1])
        assert abs(rho_01 - 0.48) < 0.05

    def test_independent_pair_uncorrelated(self, correlated_copula):
        """Variables with ρ=0 should be approximately uncorrelated."""
        X = correlated_copula.sample_joint(5000)
        corr_02 = np.corrcoef(X[:, 0], X[:, 2])[0, 1]
        assert abs(corr_02) < 0.05


# ── Tests: Custom PPF ───────────────────────────────────────────

class TestCustomPPF:

    def test_custom_ppf_range(self, custom_ppf_copula):
        """Custom PPF should map [0,1] → [10, 20]."""
        X = custom_ppf_copula.sample_joint(1000)
        assert np.all(X[:, 0] >= 9.5) and np.all(X[:, 0] <= 20.5)
        assert abs(X[:, 0].mean() - 15.0) < 0.3

    def test_tuple_spec_identity(self, custom_ppf_copula):
        """(cdf, ppf) tuple with identity should give [0, 1] output."""
        X = custom_ppf_copula.sample_joint(500)
        assert np.all(X[:, 2] >= 0) and np.all(X[:, 2] <= 1)


# ── Tests: Integration with MC Shapley ──────────────────────────

class TestMCShapleyIntegration:

    def test_collect_data_uniform(self, uniform_copula):
        """Should collect data for all 2^3 - 1 = 7 subsets."""
        data = collect_shapley_data(
            _linear_model, uniform_copula,
            N=200, predict_batch=_linear_model_batch,
        )
        assert len(data) == 7

    def test_shapley_from_uniform(self, uniform_copula):
        """Shapley effects should sum to 1."""
        data = collect_shapley_data(
            _linear_model, uniform_copula,
            N=500, predict_batch=_linear_model_batch,
        )
        effects, sh, tv = shapley_from_data(data, uniform_copula.d)
        assert effects.sum() == pytest.approx(1.0, abs=0.02)
        assert tv > 0

    def test_shapley_effects_end_to_end(self, uniform_copula):
        """End-to-end shapley_effects call."""
        effects, sh, tv = shapley_effects(
            _linear_model, uniform_copula,
            N=1000, method='exhaustive', B=0, random_state=42,
        )
        assert effects.shape == (3,)
        assert effects.sum() == pytest.approx(1.0, abs=0.02)

    def test_shapley_with_beta_marginals(self, mixed_copula):
        """Should run without error on scipy marginals."""
        effects, sh, tv = shapley_effects(
            _linear_model, mixed_copula,
            N=500, method='exhaustive', B=0, random_state=42,
        )
        assert effects.shape == (3,)
        assert tv > 0

    def test_shapley_equivalent_to_uniform_when_all_uniform(self):
        """GaussianCopulaArbitrary(all='uniform') ≈ GaussianCopulaUniform."""
        corr = np.eye(3)
        lows = [0.0, 0.0, 0.0]
        highs = [1.0, 1.0, 1.0]

        joint_new = GaussianCopulaArbitrary(
            marginals={'x0': 'uniform', 'x1': 'uniform', 'x2': 'uniform'},
            corr=corr,
        )
        joint_old = GaussianCopulaUniform(lows=lows, highs=highs, corr=corr)

        np.random.seed(42)
        eff_new, _, tv_new = shapley_effects(
            _linear_model, joint_new, N=2000, method='exhaustive',
            B=0, random_state=42,
        )
        np.random.seed(42)
        eff_old, _, tv_old = shapley_effects(
            _linear_model, joint_old, N=2000, method='exhaustive',
            B=0, random_state=42,
        )

        assert eff_new == pytest.approx(eff_old, abs=0.02)
        assert tv_new == pytest.approx(tv_old, rel=0.02)


# ── Tests: Conditional sampling properties ──────────────────────

class TestConditionalSampling:

    def test_conditional_preserves_fixed(self, correlated_copula):
        """Conditional sample should match fixed values on u_indices."""
        fixed = np.array([[0.3, 0.7]])
        X = correlated_copula.sample_conditional_batch([0, 1], fixed)
        np.testing.assert_allclose(X[:, 0], 0.3, rtol=0, atol=1e-10)
        np.testing.assert_allclose(X[:, 1], 0.7, rtol=0, atol=1e-10)

    def test_conditional_batch_many(self, correlated_copula):
        """Batch conditional with N >> 1."""
        N = 1000
        fixed = np.random.RandomState(99).uniform(0.3, 0.7, (N, 1))
        X = correlated_copula.sample_conditional_batch([0], fixed)
        assert X.shape == (N, 3)
        np.testing.assert_allclose(X[:, 0], fixed[:, 0], rtol=0, atol=1e-10)

    def test_conditional_mean_converges(self, correlated_copula):
        """For Beta marginals with ρ=0.5, E[X1 | X0=0.8] > E[X1]."""
        fixed = np.full((5000, 1), 0.8)
        X = correlated_copula.sample_conditional_batch([0], fixed)
        # Beta(5,2) has mean 5/7 ≈ 0.714
        # With ρ=0.5 and X0=0.8 (> E[X0]), X1 should be above its mean
        assert X[:, 1].mean() > 0.5  # qualitative check
