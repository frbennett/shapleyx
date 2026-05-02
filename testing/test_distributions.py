"""Tests for built-in distribution classes."""
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal


class TestMultivariateNormal:
    """Tests for MultivariateNormal."""

    def test_joint_shape(self, mvn_independent):
        X = mvn_independent.sample_joint(1000)
        assert X.shape == (1000, 3)

    def test_joint_mean_cov(self, mvn_independent):
        X = mvn_independent.sample_joint(50000)
        assert_array_almost_equal(X.mean(axis=0), [0, 0, 0], decimal=1)
        assert_array_almost_equal(
            np.cov(X.T), np.eye(3), decimal=1
        )

    def test_joint_correlated(self, mvn_correlated):
        X = mvn_correlated.sample_joint(50000)
        emp_corr = np.corrcoef(X.T)
        assert abs(emp_corr[0, 1] - 0.7) < 0.03

    def test_conditional_single(self, mvn_independent):
        x_cond = mvn_independent.sample_conditional([0], [2.0])
        assert x_cond.shape == (3,)
        assert x_cond[0] == pytest.approx(2.0)

    def test_conditional_batch(self, mvn_correlated):
        N = 1000
        fixed = np.random.randn(N, 1)
        X_cond = mvn_correlated.sample_conditional_batch([0], fixed)
        assert X_cond.shape == (N, 3)
        assert_array_almost_equal(X_cond[:, 0], fixed[:, 0])

    def test_dimension(self, mvn_independent):
        assert mvn_independent.d == 3


class TestGaussianCopulaUniform:
    """Tests for GaussianCopulaUniform."""

    def test_joint_bounds(self, gaussian_copula):
        X = gaussian_copula.sample_joint(5000)
        assert np.all(X >= -np.pi)
        assert np.all(X <= np.pi)

    def test_joint_shape(self, gaussian_copula):
        X = gaussian_copula.sample_joint(100)
        assert X.shape == (100, 3)

    def test_conditional_batch_shape(self, gaussian_copula):
        X_joint = gaussian_copula.sample_joint(100)
        X_cond = gaussian_copula.sample_conditional_batch([0], X_joint[:, [0]])
        assert X_cond.shape == (100, 3)

    def test_dimension(self, gaussian_copula):
        assert gaussian_copula.d == 3


class TestTruncatedMultivariateNormal:
    """Tests for TruncatedMultivariateNormal."""

    def test_joint_bounds(self, truncated_normal):
        X = truncated_normal.sample_joint(2000)
        assert np.all(X >= -1.0)
        assert np.all(X <= 1.0)

    def test_joint_shape(self, truncated_normal):
        X = truncated_normal.sample_joint(100)
        assert X.shape == (100, 2)

    def test_conditional_batch(self, truncated_normal):
        N = 500
        fixed = np.random.uniform(-0.5, 0.5, (N, 1))
        X_cond = truncated_normal.sample_conditional_batch([0], fixed)
        assert X_cond.shape == (N, 2)
        # Conditional draws should remain within bounds
        assert np.all(X_cond >= -1.0)
        assert np.all(X_cond <= 1.0)

    def test_unbounded_edges(self):
        """Unbounded edges should work with -inf/inf."""
        from shapleyx.utilities.mc_shapley import TruncatedMultivariateNormal as TMN
        joint = TMN(
            mean=[0.0, 0.0],
            cov=np.eye(2),
            lower=[-np.inf, -np.inf],
            upper=[np.inf, np.inf],
            joint_burn_in=20,
        )
        X = joint.sample_joint(1000)
        assert X.shape == (1000, 2)
        # With unbounded edges, should approximate N(0, I)
        assert abs(X.mean()) < 0.2

    def test_dimension(self, truncated_normal):
        assert truncated_normal.d == 2
