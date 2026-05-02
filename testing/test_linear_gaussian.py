"""Analytical validation: linear-Gaussian model vs closed-form values."""
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from shapleyx.utilities.mc_shapley import (
    MultivariateNormal,
    collect_shapley_data,
    shapley_from_data,
    shapley_effects,
    sobol_from_data,
)
from conftest import linear_model, linear_model_batch, analytical_shapley, analytical_sobol


class TestLinearGaussianShapley:
    """Validate MC Shapley against analytical values for linear-Gaussian."""

    @pytest.mark.parametrize("cov,expected", [
        (np.eye(3), [1/3, 1/3, 1/3]),
        (np.diag([4.0, 1.0, 1.0]), [2/3, 1/6, 1/6]),
        (np.array([[1.0, 0.5, 0.0],
                    [0.5, 1.0, 0.0],
                    [0.0, 0.0, 1.0]]), None),  # computed analytically
    ])
    def test_shapley_against_analytical(self, cov, expected):
        """MC Shapley effects should match analytical values within MC error."""
        d = cov.shape[0]
        if expected is None:
            expected = analytical_shapley(cov)

        joint = MultivariateNormal(mean=np.zeros(d), cov=cov)
        eff, sh, var = shapley_effects(
            linear_model, joint,
            N=8000, method='exhaustive',
            predict_batch=linear_model_batch,
            random_state=42,
        )
        assert_array_almost_equal(eff, expected, decimal=1)


class TestLinearGaussianSobol:
    """Validate MC Sobol indices against analytical values."""

    def test_sobol_independent(self):
        """For independent standard normals, S_i = T_i = 1/d for linear f."""
        cov = np.eye(3)
        joint = MultivariateNormal(mean=np.zeros(3), cov=cov)
        data = collect_shapley_data(
            linear_model, joint,
            N=5000, predict_batch=linear_model_batch,
        )
        S, T = sobol_from_data(data, 3)

        # For linear model with independent standard normals,
        # each variable contributes equally
        expected = np.array([1/3, 1/3, 1/3])
        assert_array_almost_equal(S, expected, decimal=1)
        assert_array_almost_equal(T, expected, decimal=1)

    def test_sobol_weighted(self):
        """For diag(4,1,1), S_0 = 2/3, S_1 = S_2 = 1/6."""
        cov = np.diag([4.0, 1.0, 1.0])
        joint = MultivariateNormal(mean=np.zeros(3), cov=cov)
        data = collect_shapley_data(
            linear_model, joint,
            N=5000, predict_batch=linear_model_batch,
        )
        S, T = sobol_from_data(data, 3)
        S_expected, T_expected = analytical_sobol(cov)

        assert_array_almost_equal(S, S_expected, decimal=1)
        # For independent inputs with additive model, S_i == T_i
        assert_array_almost_equal(S, T, decimal=1)


class TestShapleyProperties:
    """Shapley effects should satisfy fundamental properties."""

    def test_sum_to_one(self, mvn_independent):
        """Shapley effects should sum to 1."""
        eff, _, _ = shapley_effects(
            linear_model, mvn_independent,
            N=3000, method='exhaustive',
            predict_batch=linear_model_batch,
            random_state=42,
        )
        assert eff.sum() == pytest.approx(1.0, abs=0.01)

    def test_non_negative(self, mvn_independent):
        """Shapley effects should be non-negative."""
        eff, _, _ = shapley_effects(
            linear_model, mvn_independent,
            N=3000, method='exhaustive',
            predict_batch=linear_model_batch,
            random_state=42,
        )
        assert np.all(eff >= 0)
