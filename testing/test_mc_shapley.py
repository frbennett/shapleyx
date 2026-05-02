"""Tests for core MC Shapley functions."""
import numpy as np
import pytest

from shapleyx.utilities.mc_shapley import (
    collect_shapley_data,
    shapley_from_data,
    shapley_effects,
    MCShapley,
    _wrap_predict_fn,
    _get_shapley_weights,
)
from conftest import linear_model, linear_model_batch


class TestCollectShapleyData:
    """Tests for the data-collection function."""

    def test_exhaustive_structure(self, mvn_independent):
        """Data dict should contain all 2^d - 1 non-empty subsets."""
        data = collect_shapley_data(
            linear_model, mvn_independent,
            N=500, predict_batch=linear_model_batch,
        )
        d = mvn_independent.d
        expected_subsets = 2**d - 1
        assert len(data) == expected_subsets

    def test_full_set_entry(self, mvn_independent):
        """Full set should have type 'full'."""
        data = collect_shapley_data(
            linear_model, mvn_independent,
            N=500, predict_batch=linear_model_batch,
        )
        full_key = frozenset(range(mvn_independent.d))
        assert data[full_key][0] == 'full'

    def test_partial_set_entry(self, mvn_independent):
        """Partial sets should have type 'pair' with Y1 and Y2."""
        data = collect_shapley_data(
            linear_model, mvn_independent,
            N=500, predict_batch=linear_model_batch,
        )
        key = frozenset([0])
        assert data[key][0] == 'pair'
        assert len(data[key]) == 3  # ('pair', Y1, Y2)

    def test_1d_predict_fn_wrapping(self, mvn_independent):
        """Should work without predict_batch (per-sample evaluation)."""
        data = collect_shapley_data(
            linear_model, mvn_independent,
            N=200,
        )
        assert len(data) == 2**3 - 1


class TestShapleyFromData:
    """Tests for Shapley effect extraction from collected data."""

    def test_effects_shape(self, mvn_independent):
        data = collect_shapley_data(
            linear_model, mvn_independent,
            N=500, predict_batch=linear_model_batch,
        )
        effects, sh, total_var = shapley_from_data(data, mvn_independent.d)
        assert effects.shape == (3,)
        assert sh.shape == (3,)
        assert total_var > 0

    def test_effects_sum_to_one(self, mvn_independent):
        data = collect_shapley_data(
            linear_model, mvn_independent,
            N=500, predict_batch=linear_model_batch,
        )
        effects, _, _ = shapley_from_data(data, mvn_independent.d)
        assert effects.sum() == pytest.approx(1.0, abs=0.01)


class TestShapleyEffects:
    """End-to-end tests for the main shapley_effects entry point."""

    def test_exhaustive_returns_correct_types(self, mvn_independent):
        result = shapley_effects(
            linear_model, mvn_independent,
            N=500, method='exhaustive',
            random_state=42,
        )
        assert len(result) == 3  # effects, sh, total_var
        effects, sh, total_var = result
        assert effects.shape == (3,)
        assert isinstance(total_var, float)

    def test_exhaustive_with_bootstrap(self, mvn_independent):
        result = shapley_effects(
            linear_model, mvn_independent,
            N=500, method='exhaustive',
            B=50, random_state=42,
        )
        assert len(result) == 5  # + lower, upper
        effects, sh, total_var, lower, upper = result
        assert lower.shape == (3,)
        assert upper.shape == (3,)
        assert np.all(lower <= effects)
        assert np.all(upper >= effects)

    def test_permutation_returns_correct_types(self, mvn_independent):
        result = shapley_effects(
            linear_model, mvn_independent,
            N=500, method='permutation',
            n_perm=100, random_state=42,
        )
        assert len(result) == 3

    def test_invalid_method_raises(self, mvn_independent):
        with pytest.raises(ValueError, match="method must be"):
            shapley_effects(
                linear_model, mvn_independent,
                N=500, method='invalid',
            )


class TestMCShapley:
    """Tests for the MCShapley convenience wrapper."""

    def test_dataframe_columns(self, mvn_independent):
        mc = MCShapley(f=linear_model, joint=mvn_independent,
                       predict_batch=linear_model_batch)
        df = mc.compute(N=500, method='exhaustive', B=20, random_state=42)
        expected_cols = ['variable', 'effect', 'shapley_value',
                         'sobol_first', 'sobol_total', 'total_variance',
                         'lower', 'upper',
                         'sobol_first_lower', 'sobol_first_upper',
                         'sobol_total_lower', 'sobol_total_upper']
        for col in expected_cols:
            assert col in df.columns

    def test_sobol_columns_exhaustive(self, mvn_independent):
        mc = MCShapley(f=linear_model, joint=mvn_independent,
                       predict_batch=linear_model_batch)
        df = mc.compute(N=500, method='exhaustive', B=20, random_state=42)
        assert not df['sobol_first'].isna().any()
        assert not df['sobol_total'].isna().any()

    def test_sobol_columns_permutation(self, mvn_independent):
        mc = MCShapley(f=linear_model, joint=mvn_independent,
                       predict_batch=linear_model_batch)
        df = mc.compute(N=500, method='permutation', n_perm=50, random_state=42)
        # Sobol indices should be NaN for permutation method
        assert df['sobol_first'].isna().all()
        assert df['sobol_total'].isna().all()


class TestHelpers:
    """Tests for utility functions."""

    def test_wrap_predict_fn_1d(self):
        """Wrap should convert 1D function to accept 1D and 2D input."""
        def f(x):
            return float(x[0] + x[1])

        g = _wrap_predict_fn(f)
        assert g(np.array([1.0, 2.0])) == pytest.approx(3.0)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = g(X)
        assert result.shape == (2,)
        assert result[0] == pytest.approx(3.0)

    def test_shapley_weights(self):
        """Shapley weights should be valid: sum of w_k * binom(d-1,k) = 1."""
        for d in [2, 3, 5, 8]:
            w = _get_shapley_weights(d)
            from math import comb
            total = sum(w[k] * comb(d - 1, k) for k in range(d))
            assert total == pytest.approx(1.0)
