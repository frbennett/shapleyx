"""Tests for Sobol index extraction from MC data."""
import numpy as np
import pytest

from shapleyx.utilities.mc_shapley import (
    collect_shapley_data,
    sobol_from_data,
    bootstrap_sobol,
    MCShapley,
)
from conftest import linear_model, linear_model_batch


class TestSobolFromData:
    """Tests for sobol_from_data."""

    def test_shapes(self, mvn_independent):
        data = collect_shapley_data(
            linear_model, mvn_independent,
            N=500, predict_batch=linear_model_batch,
        )
        S, T = sobol_from_data(data, 3)
        assert S.shape == (3,)
        assert T.shape == (3,)

    def test_non_negative(self, mvn_independent):
        """Both S_i and T_i should be non-negative."""
        data = collect_shapley_data(
            linear_model, mvn_independent,
            N=500, predict_batch=linear_model_batch,
        )
        S, T = sobol_from_data(data, 3)
        assert np.all(S >= 0)
        assert np.all(T >= 0)

    def test_si_leq_ti_independent(self, mvn_independent):
        """For independent inputs, S_i <= T_i should hold."""
        data = collect_shapley_data(
            linear_model, mvn_independent,
            N=2000, predict_batch=linear_model_batch,
        )
        S, T = sobol_from_data(data, 3)
        # For a linear model with independent inputs, S_i ≈ T_i
        for i in range(3):
            assert S[i] <= T[i] + 0.15

    def test_sobol_against_shapley_from_data(self, mvn_independent):
        """Sobol indices should use the same v(u) as Shapley."""
        from shapleyx.utilities.mc_shapley import shapley_from_data
        data = collect_shapley_data(
            linear_model, mvn_independent,
            N=500, predict_batch=linear_model_batch,
        )
        S, T = sobol_from_data(data, 3)
        effects, _, total_var = shapley_from_data(data, 3)
        # S_i should contribute to Shapley (Shapley redistributes interactions)
        # so S_i <= effect when interactions are present, but for a linear model
        # with independent inputs, they should be approximately equal
        for i in range(3):
            assert S[i] == pytest.approx(effects[i], abs=0.1)


class TestBootstrapSobol:
    """Tests for bootstrap_sobol."""

    def test_returns_correct_tuple(self, mvn_independent):
        data = collect_shapley_data(
            linear_model, mvn_independent,
            N=500, predict_batch=linear_model_batch,
        )
        result = bootstrap_sobol(data, 3, B=30, alpha=0.05, random_state=42)
        assert len(result) == 6
        S_point, S_lower, S_upper, T_point, T_lower, T_upper = result
        assert S_point.shape == (3,)
        assert S_lower.shape == (3,)
        assert T_point.shape == (3,)

    def test_cis_contain_point_estimate(self, mvn_independent):
        data = collect_shapley_data(
            linear_model, mvn_independent,
            N=800, predict_batch=linear_model_batch,
        )
        S_point, S_lower, S_upper, T_point, T_lower, T_upper = bootstrap_sobol(
            data, 3, B=100, alpha=0.05, random_state=42,
        )
        assert np.all(S_lower <= S_point)
        assert np.all(S_upper >= S_point)
        assert np.all(T_lower <= T_point)
        assert np.all(T_upper >= T_point)

    def test_mcshapley_sobol_ci_columns(self, mvn_independent):
        """MCShapley.compute() should include Sobol CI columns."""
        mc = MCShapley(f=linear_model, joint=mvn_independent,
                       predict_batch=linear_model_batch)
        df = mc.compute(N=500, method='exhaustive', B=30, random_state=42)
        for col in ['sobol_first_lower', 'sobol_first_upper',
                    'sobol_total_lower', 'sobol_total_upper']:
            assert col in df.columns
            assert not df[col].isna().any()
