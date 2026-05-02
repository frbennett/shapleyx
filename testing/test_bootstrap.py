"""Tests for bootstrap confidence interval functions."""
import numpy as np
import pytest

from shapleyx.utilities.mc_shapley import (
    collect_shapley_data,
    bootstrap_shapley,
    bootstrap_sobol,
    shapley_effects,
)
from conftest import linear_model, linear_model_batch


class TestBootstrapShapley:
    """Tests for bootstrap_shapley."""

    def test_returns_correct_types(self, mvn_independent):
        data = collect_shapley_data(
            linear_model, mvn_independent,
            N=500, predict_batch=linear_model_batch,
        )
        point, lower, upper = bootstrap_shapley(
            data, 3, B=30, alpha=0.05, random_state=42,
        )
        assert point.shape == (3,)
        assert lower.shape == (3,)
        assert upper.shape == (3,)

    def test_cis_contain_point(self, mvn_independent):
        """Bootstrap CIs should contain the point estimate."""
        data = collect_shapley_data(
            linear_model, mvn_independent,
            N=800, predict_batch=linear_model_batch,
        )
        point, lower, upper = bootstrap_shapley(
            data, 3, B=100, alpha=0.05, random_state=42,
        )
        assert np.all(lower <= point)
        assert np.all(upper >= point)

    def test_larger_alpha_gives_wider_cis(self, mvn_independent):
        """Smaller alpha (wider CI) should produce wider intervals."""
        data = collect_shapley_data(
            linear_model, mvn_independent,
            N=500, predict_batch=linear_model_batch,
        )
        _, l_10, u_10 = bootstrap_shapley(
            data, 3, B=50, alpha=0.10, random_state=42,
        )
        _, l_01, u_01 = bootstrap_shapley(
            data, 3, B=50, alpha=0.01, random_state=42,  # 99% CI
        )
        # 99% CI should be wider than 90% CI
        width_90 = (u_10 - l_10).mean()
        width_99 = (u_01 - l_01).mean()
        assert width_99 > width_90

    def test_reproducibility(self, mvn_independent):
        """Same seed should give same result."""
        data = collect_shapley_data(
            linear_model, mvn_independent,
            N=500, predict_batch=linear_model_batch,
        )
        p1, l1, u1 = bootstrap_shapley(data, 3, B=30, alpha=0.05, random_state=42)
        p2, l2, u2 = bootstrap_shapley(data, 3, B=30, alpha=0.05, random_state=42)
        assert np.allclose(p1, p2)
        assert np.allclose(l1, l2)
        assert np.allclose(u1, u2)


class TestBootstrapSobol:
    """Tests for bootstrap_sobol."""

    def test_cis_contain_point(self, mvn_independent):
        data = collect_shapley_data(
            linear_model, mvn_independent,
            N=800, predict_batch=linear_model_batch,
        )
        Sp, Sl, Su, Tp, Tl, Tu = bootstrap_sobol(
            data, 3, B=100, alpha=0.05, random_state=42,
        )
        assert np.all(Sl <= Sp)
        assert np.all(Su >= Sp)

    def test_reproducibility(self, mvn_independent):
        data = collect_shapley_data(
            linear_model, mvn_independent,
            N=500, predict_batch=linear_model_batch,
        )
        r1 = bootstrap_sobol(data, 3, B=30, alpha=0.05, random_state=42)
        r2 = bootstrap_sobol(data, 3, B=30, alpha=0.05, random_state=42)
        for a, b in zip(r1, r2):
            assert np.allclose(a, b)


class TestShapleyEffectsBootstrap:
    """End-to-end bootstrap via shapley_effects."""

    def test_bootstrap_increases_returns(self, mvn_independent):
        """With B > 0, shapley_effects should return 5 values."""
        result = shapley_effects(
            linear_model, mvn_independent,
            N=500, method='exhaustive',
            B=30, random_state=42,
        )
        assert len(result) == 5  # effects, sh, total_var, lower, upper
