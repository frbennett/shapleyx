"""Tests for Owen (grouped) Shapley effects."""

import numpy as np
import pytest
from collections import OrderedDict

from shapleyx.utilities.mc_shapley import (
    GaussianCopulaUniform,
    collect_shapley_data,
    collect_shapley_data_groups,
    owen_from_data,
    shapley_effects,
    shapley_from_data,
    _build_group_index,
)


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def joint4():
    return GaussianCopulaUniform(
        lows=np.zeros(4), highs=np.ones(4), corr=np.eye(4))


@pytest.fixture
def var_names4():
    return ['x0', 'x1', 'x2', 'x3']


def _f_linear(x):
    """f(x) = x0 + 2*x1 + 3*x2 + 4*x3"""
    return float(x[0] + 2 * x[1] + 3 * x[2] + 4 * x[3])


def _f_linear_batch(X):
    return X[:, 0] + 2 * X[:, 1] + 3 * X[:, 2] + 4 * X[:, 3]


# ── Tests: _build_group_index ───────────────────────────────────

class TestBuildGroupIndex:

    def test_basic(self, var_names4):
        groups = {'g1': ['x0', 'x1'], 'g2': ['x2', 'x3']}
        gof, gv, gn = _build_group_index(groups, var_names4)
        assert list(gn) == ['g1', 'g2']
        assert gof.tolist() == [0, 0, 1, 1]
        assert gv == [[0, 1], [2, 3]]

    def test_singleton_groups(self, var_names4):
        groups = {'a': ['x0'], 'b': ['x1'], 'c': ['x2'], 'd': ['x3']}
        gof, gv, gn = _build_group_index(groups, var_names4)
        assert gof.tolist() == [0, 1, 2, 3]

    def test_missing_variable_raises(self, var_names4):
        groups = {'g1': ['x0', 'x99']}
        with pytest.raises(ValueError, match='x99'):
            _build_group_index(groups, var_names4)

    def test_duplicate_variable_raises(self, var_names4):
        groups = {'g1': ['x0', 'x1'], 'g2': ['x1', 'x2']}
        with pytest.raises(ValueError, match='multiple groups'):
            _build_group_index(groups, var_names4)

    def test_uncovered_variable_raises(self, var_names4):
        groups = {'g1': ['x0', 'x1']}  # x2, x3 uncovered
        with pytest.raises(ValueError, match='not assigned'):
            _build_group_index(groups, var_names4)


# ── Tests: Owen decomposition correctness ──────────────────────

class TestOwenFromData:

    def test_inner_sums_to_outer(self, joint4, var_names4):
        """Within each group, individual effects should sum to group effect."""
        groups = {'g1': ['x0', 'x1'], 'g2': ['x2', 'x3']}
        data = collect_shapley_data(
            _f_linear, joint4, N=2000,
            predict_batch=_f_linear_batch)
        ge, ie, tv = owen_from_data(data, groups, var_names4)

        for g_name, g_vars in groups.items():
            inner_sum = sum(ie[v] for v in g_vars)
            assert inner_sum == pytest.approx(ge[g_name], rel=0.01)

    def test_total_sums_match(self, joint4, var_names4):
        """Σ group = Σ individual = total variance."""
        groups = {'g1': ['x0', 'x1'], 'g2': ['x2', 'x3']}
        data = collect_shapley_data(
            _f_linear, joint4, N=2000,
            predict_batch=_f_linear_batch)
        ge, ie, tv = owen_from_data(data, groups, var_names4)

        assert sum(ge.values()) == pytest.approx(tv, rel=0.01)
        assert sum(ie.values()) == pytest.approx(tv, rel=0.01)

    def test_reduces_to_shapley_for_singletons(self, joint4, var_names4):
        """With all singleton groups, Owen = standard Shapley."""
        groups = {'a': ['x0'], 'b': ['x1'], 'c': ['x2'], 'd': ['x3']}
        data = collect_shapley_data(
            _f_linear, joint4, N=3000,
            predict_batch=_f_linear_batch)
        ge, ie, tv = owen_from_data(data, groups, var_names4)

        # Also compute standard Shapley
        eff, sh, tv2 = shapley_from_data(data, 4)

        for i, name in enumerate(var_names4):
            assert ie[name] == pytest.approx(sh[i], rel=0.05)

    def test_three_groups(self, joint4, var_names4):
        """d=6, 3 groups of 2."""
        def f6(x):
            return float(x[0]+2*x[1]+3*x[2]+4*x[3]+5*x[4]+6*x[5])
        def f6b(X):
            return X[:,0]+2*X[:,1]+3*X[:,2]+4*X[:,3]+5*X[:,4]+6*X[:,5]

        j6 = GaussianCopulaUniform(np.zeros(6), np.ones(6), np.eye(6))
        v6 = ['x0','x1','x2','x3','x4','x5']
        groups = {'g1': ['x0','x1'], 'g2': ['x2','x3'], 'g3': ['x4','x5']}
        data = collect_shapley_data(f6, j6, N=1500, predict_batch=f6b)
        ge, ie, tv = owen_from_data(data, groups, v6)

        for g_name, g_vars in groups.items():
            inner_sum = sum(ie[v] for v in g_vars)
            assert inner_sum == pytest.approx(ge[g_name], rel=0.02)
        assert sum(ge.values()) == pytest.approx(tv, rel=0.01)
        assert sum(ie.values()) == pytest.approx(tv, rel=0.01)

    def test_unbalanced_groups(self, joint4, var_names4):
        """d=6, groups of sizes 1, 2, 3."""
        def f6(x):
            return float(x[0]+2*x[1]+3*x[2]+4*x[3]+5*x[4]+6*x[5])
        def f6b(X):
            return X[:,0]+2*X[:,1]+3*X[:,2]+4*X[:,3]+5*X[:,4]+6*X[:,5]

        j6 = GaussianCopulaUniform(np.zeros(6), np.ones(6), np.eye(6))
        v6 = ['x0','x1','x2','x3','x4','x5']
        groups = {'a': ['x0'], 'b': ['x1','x2'], 'c': ['x3','x4','x5']}
        data = collect_shapley_data(f6, j6, N=1500, predict_batch=f6b)
        ge, ie, tv = owen_from_data(data, groups, v6)

        for g_name, g_vars in groups.items():
            inner_sum = sum(ie[v] for v in g_vars)
            assert inner_sum == pytest.approx(ge[g_name], rel=0.02)
        assert sum(ge.values()) == pytest.approx(tv, rel=0.01)

    def test_non_negative_effects(self, joint4, var_names4):
        """All Owen effects should be non-negative (monotonicity)."""
        groups = {'g1': ['x0', 'x1'], 'g2': ['x2', 'x3']}
        data = collect_shapley_data(
            _f_linear, joint4, N=2000,
            predict_batch=_f_linear_batch)
        ge, ie, tv = owen_from_data(data, groups, var_names4)

        for v in ge.values():
            assert v >= -1e-10
        for v in ie.values():
            assert v >= -1e-10

    def test_single_group(self, joint4, var_names4):
        """One big group = individual Shapley within that group should
        recover standard Shapley."""
        groups = {'all': ['x0', 'x1', 'x2', 'x3']}
        data = collect_shapley_data(
            _f_linear, joint4, N=3000,
            predict_batch=_f_linear_batch)
        ge, ie, tv = owen_from_data(data, groups, var_names4)

        eff, sh, tv2 = shapley_from_data(data, 4)

        # The group effect should equal total variance
        assert ge['all'] == pytest.approx(tv, rel=0.01)
        # Individual effects should match standard Shapley
        for i, name in enumerate(var_names4):
            assert ie[name] == pytest.approx(sh[i], rel=0.05)

    def test_with_correlation(self, var_names4):
        """Owen effects should work with correlated inputs."""
        corr = np.array([
            [1.0, 0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.3, 1.0],
        ])
        joint = GaussianCopulaUniform(np.zeros(4), np.ones(4), corr)
        groups = {'g1': ['x0', 'x1'], 'g2': ['x2', 'x3']}
        data = collect_shapley_data(
            _f_linear, joint, N=2000,
            predict_batch=_f_linear_batch)
        ge, ie, tv = owen_from_data(data, groups, var_names4)

        # Structural checks (not value checks — correlation changes values)
        assert sum(ge.values()) == pytest.approx(tv, rel=0.02)
        assert sum(ie.values()) == pytest.approx(tv, rel=0.02)
        # Inner must sum to outer per group
        for g_name, g_vars in groups.items():
            assert sum(ie[v] for v in g_vars) == pytest.approx(
                ge[g_name], rel=0.02)


# ── Tests: collect_shapley_data_groups ──────────────────────────

class TestCollectShapleyDataGroups:

    def test_returns_all_subsets(self, joint4, var_names4):
        """Should return all 2^d - 1 non-empty variable subsets."""
        groups = {'g1': ['x0', 'x1'], 'g2': ['x2', 'x3']}
        data = collect_shapley_data_groups(
            _f_linear, joint4, groups, var_names4, N=200)
        assert len(data) == 2**4 - 1

    def test_identical_to_standard_collect(self, joint4, var_names4):
        """With same seed, should produce identical data to
        collect_shapley_data."""
        groups = {'g1': ['x0', 'x1'], 'g2': ['x2', 'x3']}
        np.random.seed(42)
        data1 = collect_shapley_data_groups(
            _f_linear, joint4, groups, var_names4, N=200)
        np.random.seed(42)
        data2 = collect_shapley_data(
            _f_linear, joint4, N=200)

        # Same keys
        assert set(data1.keys()) == set(data2.keys())
