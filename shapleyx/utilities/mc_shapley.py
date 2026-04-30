"""
Monte Carlo Shapley effects estimation for correlated inputs.

Implements a Monte Carlo approach for computing Shapley effects when input
variables may be correlated. Supports two methods:
- 'exhaustive': enumerates all 2^d - 1 non-empty subsets
- 'permutation': uses random permutations for computational efficiency

Both methods support bootstrap confidence intervals.

Distribution classes:
    MultivariateNormal: Correlated normal inputs with conditional sampling
        via the multivariate normal conditional distribution formula.
    GaussianCopulaUniform: Correlated uniform inputs via a Gaussian copula,
        with conditional sampling in the latent normal space.

Reference:
    Owen, A. B., & Prieur, C. (2017). On Shapley value for measuring
    importance of dependent inputs. SIAM/ASA Journal on Uncertainty
    Quantification, 5(1), 986-1004.
"""

import numpy as np
from scipy.stats import norm
import itertools
import math
import pandas as pd


# ------------------------------------------------------------
# Optional progress bar (tqdm) — graceful fallback if not installed
# ------------------------------------------------------------
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


class _Progress:
    """Single progress bar tracking total function evaluations.

    Created with a pre-computed total; updated after each batch or
    per-sample evaluation step.  Degrades to a no-op when ``tqdm``
    is not installed or ``enabled`` is ``False``.

    When *total* is zero the bar is created without a known total;
    :meth:`add_total` can grow it as work is discovered (used by the
    permutation method where subsets are evaluated lazily).
    """
    def __init__(self, total, desc='MC Shapley', enabled=True):
        self._pbar = None
        if enabled and _tqdm is not None:
            self._pbar = _tqdm(total=total if total > 0 else None,
                               desc=desc, unit='evals',
                               smoothing=0.01)

    def update(self, n=1):
        if self._pbar is not None:
            self._pbar.update(n)

    def add_total(self, n):
        """Increase the total by *n* evaluations (for lazy discovery)."""
        if self._pbar is not None:
            self._pbar.total = (self._pbar.total or 0) + n
            self._pbar.refresh()

    def close(self):
        if self._pbar is not None:
            self._pbar.close()


# ------------------------------------------------------------
# Helper: wrap a 2D predict function for 1D scalar use
# ------------------------------------------------------------
def _wrap_predict_fn(predict_fn):
    """Wrap a 1D-native predict function to also accept 2D (batch) input.

    For 1D input the function is called directly and the result cast to float.
    For 2D input each row is evaluated sequentially (per-sample loop).

    Use this wrapper when the underlying function only handles 1D arrays.
    For functions that natively support 2D (batch) input, pass them directly
    via the ``predict_batch`` parameter of the data-collection functions.

    Args:
        predict_fn: Callable ``f(x)`` where ``x`` is a 1D array, returning a
            scalar (float or 0-d array).

    Returns:
        Callable ``g(x)`` accepting 1D or 2D input.
    """
    def f(x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return float(predict_fn(x))
        return np.array([float(predict_fn(xi)) for xi in x])
    return f


# ------------------------------------------------------------
# Distribution classes
# ------------------------------------------------------------
class MultivariateNormal:
    """Correlated normal inputs with conditional sampling.

    Args:
        mean: 1D array of means, shape (d,).
        cov: 2D covariance matrix, shape (d, d).
    """
    def __init__(self, mean, cov):
        self.mean = np.asarray(mean)
        self.cov = np.asarray(cov)
        self.d = len(mean)
        assert self.cov.shape == (self.d, self.d), \
            f"cov must be ({self.d}, {self.d}), got {self.cov.shape}"

    def sample_joint(self, n):
        """Draw n joint samples from the full distribution.

        Args:
            n: Number of samples.

        Returns:
            Array of shape (n, d).
        """
        return np.random.multivariate_normal(self.mean, self.cov, n)

    def sample_conditional(self, u_indices, fixed_x):
        """Draw one sample conditioned on fixed values for variables in u.

        Samples x_v | x_u = fixed_x using the conditional multivariate
        normal formula, then returns the full vector [x_u, x_v].

        Args:
            u_indices: Indices of variables to condition on.
            fixed_x: Fixed values for the conditioned variables.

        Returns:
            1D array of shape (d,).
        """
        u = np.asarray(u_indices)
        if len(u) == 0:
            return self.sample_joint(1)[0]

        fixed_x = np.asarray(fixed_x)
        all_idx = np.arange(self.d)
        v = np.setdiff1d(all_idx, u)

        mu_u = self.mean[u]
        mu_v = self.mean[v]
        Sigma_uu = self.cov[np.ix_(u, u)]
        Sigma_uv = self.cov[np.ix_(u, v)]
        Sigma_vu = Sigma_uv.T
        Sigma_vv = self.cov[np.ix_(v, v)]

        inv_Sigma_uu = np.linalg.inv(Sigma_uu)
        cond_mean = mu_v + Sigma_vu @ inv_Sigma_uu @ (fixed_x - mu_u)
        cond_cov = Sigma_vv - Sigma_vu @ inv_Sigma_uu @ Sigma_uv

        z_v = np.random.multivariate_normal(cond_mean, cond_cov)
        x_full = np.zeros(self.d)
        x_full[u] = fixed_x
        x_full[v] = z_v
        return x_full


class GaussianCopulaUniform:
    """Correlated uniform inputs via a Gaussian copula.

    Each marginal is Uniform(lows[i], highs[i]). Dependence is induced
    through a latent multivariate normal with the given correlation matrix.

    Args:
        lows: Lower bounds for each dimension, shape (d,).
        highs: Upper bounds for each dimension, shape (d,).
        corr: Correlation matrix for the latent normal, shape (d, d).
    """
    def __init__(self, lows, highs, corr):
        self.lows = np.array(lows)
        self.highs = np.array(highs)
        self.d = len(lows)
        self.corr = np.array(corr)
        assert self.corr.shape == (self.d, self.d), \
            f"corr must be ({self.d}, {self.d}), got {self.corr.shape}"

    def sample_joint(self, n):
        """Draw n joint samples.

        Args:
            n: Number of samples.

        Returns:
            Array of shape (n, d) with uniform marginals.
        """
        Z = np.random.multivariate_normal(
            mean=np.zeros(self.d), cov=self.corr, size=n
        )
        U = norm.cdf(Z)
        return self.lows + (self.highs - self.lows) * U

    def sample_conditional(self, u_indices, fixed_x):
        """Draw one sample conditioned on fixed values for variables in u.

        Transforms to the latent normal space, conditions there, then
        transforms back to the uniform scale.

        Args:
            u_indices: Indices of variables to condition on.
            fixed_x: Fixed values for the conditioned variables.

        Returns:
            1D array of shape (d,).
        """
        u = np.asarray(u_indices)
        if len(u) == 0:
            return self.sample_joint(1)[0]

        fixed_x = np.asarray(fixed_x)
        # Transform to latent normal space
        fixed_u = (fixed_x - self.lows[u]) / (self.highs[u] - self.lows[u])
        fixed_u = np.clip(fixed_u, 1e-12, 1 - 1e-12)
        z_u = norm.ppf(fixed_u)

        all_idx = np.arange(self.d)
        v = np.setdiff1d(all_idx, u)

        Sigma_uu = self.corr[np.ix_(u, u)]
        Sigma_uv = self.corr[np.ix_(u, v)]
        Sigma_vu = Sigma_uv.T
        Sigma_vv = self.corr[np.ix_(v, v)]

        inv_Sigma_uu = np.linalg.inv(Sigma_uu)
        cond_mean = Sigma_vu @ inv_Sigma_uu @ z_u
        cond_cov = Sigma_vv - Sigma_vu @ inv_Sigma_uu @ Sigma_uv

        z_v = np.random.multivariate_normal(cond_mean, cond_cov)
        Z_full = np.zeros(self.d)
        Z_full[u] = z_u
        Z_full[v] = z_v
        U_full = norm.cdf(Z_full)
        X_full = self.lows + (self.highs - self.lows) * U_full
        return X_full


# ------------------------------------------------------------
# Core functions for exhaustive method
# ------------------------------------------------------------
def collect_shapley_data(f, joint, N=10000, predict_batch=None,
                         progress=False):
    """Compute and store outputs for all non-empty subsets (exhaustive).

    For each subset u of variable indices:
    - If |u| == d (the full set): draw N joint samples, evaluate f,
      store the outputs for variance estimation.
    - Otherwise: for each of N iterations, draw one joint sample x,
      evaluate f(x), then draw a conditional sample x_cond where
      variables in u are fixed to x[u], evaluate f(x_cond). Store
      the paired outputs for covariance estimation.

    Args:
        f: Model function f(x) taking a 1D array and returning a scalar.
        joint: Distribution object with sample_joint and sample_conditional.
        N: Number of Monte Carlo samples per subset.
        predict_batch: Optional callable that accepts a 2D array (N, d)
            and returns a 1D array of predictions. When provided, batch
            evaluation is used for unconditional draws, greatly reducing
            Python-level call overhead. When None (default), ``f`` is
            called once per sample.
        progress: If ``True``, display a single tqdm progress bar
            tracking the total number of function evaluations across
            all subsets (requires ``tqdm`` to be installed).

    Returns:
        dict mapping frozenset(u) -> tuple describing stored data.
    """
    d = joint.d
    subsets = [frozenset(s) for k in range(1, d + 1)
               for s in itertools.combinations(range(d), k)]
    n_subsets = len(subsets)          # 2^d - 1
    n_partial = n_subsets - 1         # subsets with |u| < d

    # Each partial subset costs 2N evaluations (N unconditional + N conditional);
    # the full subset costs N evaluations.  Total = N + 2N * n_partial.
    total_evals = N + 2 * N * n_partial
    pbar = _Progress(total_evals, enabled=progress)

    data = {}
    for u in subsets:
        u_list = list(u)
        if len(u) == d:
            # Full set: sample and evaluate (N evals)
            X = joint.sample_joint(N)
            if predict_batch is not None:
                Y = np.asarray(predict_batch(X), dtype=float)
            else:
                Y = np.array([f(X[i]) for i in range(N)])
            pbar.update(N)
            data[u] = ('full', Y)
        else:
            # Partial set: unconditional draws (N evals) ...
            X = joint.sample_joint(N)
            if predict_batch is not None:
                Y1 = np.asarray(predict_batch(X), dtype=float)
            else:
                Y1 = np.array([f(X[i]) for i in range(N)])
            pbar.update(N)
            # ... then conditional draws (N evals)
            Y2 = np.zeros(N)
            for i in range(N):
                x_cond = joint.sample_conditional(u_list, X[i, u_list])
                Y2[i] = f(x_cond)
                pbar.update(1)
            data[u] = ('pair', Y1, Y2)

    pbar.close()
    return data


def shapley_from_data(data, d):
    """Compute point estimates of Shapley effects from collected data.

    Uses the covariance-based formulation: v(u) = Cov[f(X), f(X_u)]
    where X_u is a conditional sample sharing the same background
    variables as X.

    Args:
        data: Dict from collect_shapley_data.
        d: Number of input dimensions.

    Returns:
        effects: Normalised Shapley effects (sums to 1), shape (d,).
        sh: Unscaled Shapley values, shape (d,).
        total_var: Estimated total variance.
    """
    v = {frozenset(): 0.0}
    for u, typ in data.items():
        if typ[0] == 'full':
            Y = typ[1]
            v[u] = np.var(Y)
        else:   # 'pair'
            Y1, Y2 = typ[1], typ[2]
            cov = np.mean(Y1 * Y2) - np.mean(Y1) * np.mean(Y2)
            v[u] = cov

    sh = np.zeros(d)
    subsets_all = [frozenset(s) for k in range(d + 1)
                   for s in itertools.combinations(range(d), k)]
    for i in range(d):
        for u in subsets_all:
            if i not in u:
                u_with_i = u.union({i})
                diff = v[u_with_i] - v[u]
                k = len(u)
                weight = (math.factorial(k) * math.factorial(d - k - 1)
                          / math.factorial(d))
                sh[i] += weight * diff
    total_var = v[frozenset(range(d))]
    effects = sh / total_var
    return effects, sh, total_var


def bootstrap_shapley(data, d, B=1000, alpha=0.05, random_state=None):
    """Bootstrap confidence intervals for the exhaustive method.

    Args:
        data: Dict from collect_shapley_data.
        d: Number of input dimensions.
        B: Number of bootstrap replications.
        alpha: Significance level (e.g., 0.05 for 95% CI).
        random_state: Seed for reproducibility.

    Returns:
        point_eff: Point estimates, shape (d,).
        lower: Lower CI bounds, shape (d,).
        upper: Upper CI bounds, shape (d,).
    """
    if random_state is not None:
        np.random.seed(random_state)

    point_eff, _, _ = shapley_from_data(data, d)

    # Determine sample size N from the data
    for typ in data.values():
        if typ[0] == 'full':
            N = len(typ[1])
            break
        elif typ[0] == 'pair':
            N = len(typ[1])
            break

    boot_effects = np.zeros((B, d))
    for b in range(B):
        idx = np.random.choice(N, size=N, replace=True)
        boot_data = {}
        for u, typ in data.items():
            if typ[0] == 'full':
                boot_data[u] = ('full', typ[1][idx])
            else:
                boot_data[u] = ('pair', typ[1][idx], typ[2][idx])
        eff_b, _, _ = shapley_from_data(boot_data, d)
        boot_effects[b] = eff_b

    lower = np.percentile(boot_effects, 100 * alpha / 2, axis=0)
    upper = np.percentile(boot_effects, 100 * (1 - alpha / 2), axis=0)
    return point_eff, lower, upper


# ------------------------------------------------------------
# Random permutation method (with caching)
# ------------------------------------------------------------
def compute_subset_data(f, joint, u, N, data_cache, predict_batch=None,
                        pbar=None):
    """Compute and store data for a subset u if not already cached.

    Args:
        f: Model function.
        joint: Distribution object.
        u: Iterable of variable indices for the subset.
        N: Sample size.
        data_cache: Dict to store computed data.
        predict_batch: Optional batch prediction callable (see
            ``collect_shapley_data``).
        pbar: Optional :class:`_Progress` instance to update after each
            evaluation step.  When ``None`` no progress is reported.
    """
    key = frozenset(u)
    if key in data_cache:
        return
    d = joint.d
    if len(u) == 0:
        return

    # Announce cost before computing — allows dynamic progress bars to
    # grow their total as subsets are discovered lazily.
    cost = N if len(u) == d else 2 * N
    if pbar is not None:
        pbar.add_total(cost)

    if len(u) == d:
        # Full set: sample and evaluate (N evals)
        X = joint.sample_joint(N)
        if predict_batch is not None:
            Y = np.asarray(predict_batch(X), dtype=float)
        else:
            Y = np.array([f(X[i]) for i in range(N)])
        if pbar is not None:
            pbar.update(N)
        data_cache[key] = ('full', Y)
    else:
        # Partial set: unconditional draws (N evals) ...
        u_list = list(u)
        X = joint.sample_joint(N)
        if predict_batch is not None:
            Y1 = np.asarray(predict_batch(X), dtype=float)
        else:
            Y1 = np.array([f(X[i]) for i in range(N)])
        if pbar is not None:
            pbar.update(N)
        # ... then conditional draws (N evals)
        Y2 = np.zeros(N)
        for i in range(N):
            x_cond = joint.sample_conditional(u_list, X[i, u_list])
            Y2[i] = f(x_cond)
            if pbar is not None:
                pbar.update(1)
        data_cache[key] = ('pair', Y1, Y2)


def get_v_from_data(data_entry):
    """Extract v(u) from a cached data entry.

    Args:
        data_entry: Tuple from data_cache, or None for the empty set.

    Returns:
        float: Estimated value v(u).
    """
    if data_entry is None:
        return 0.0
    typ = data_entry[0]
    if typ == 'full':
        return np.var(data_entry[1])
    else:
        Y1, Y2 = data_entry[1], data_entry[2]
        return np.mean(Y1 * Y2) - np.mean(Y1) * np.mean(Y2)


def shapley_effects_permutation(f, joint, N=10000, n_perm=1000,
                                B=0, alpha=0.05, random_state=None,
                                predict_batch=None, progress=False):
    """Shapley effects via random permutations.

    Instead of enumerating all 2^d subsets, this method uses random
    permutations. Subset data is computed lazily and cached, so only
    subsets that appear in the permutations are evaluated.

    Args:
        f: Model function.
        joint: Distribution object.
        N: Monte Carlo sample size per subset.
        n_perm: Number of random permutations.
        B: Bootstrap replications (0 to skip).
        alpha: Significance level for CIs.
        random_state: Seed for reproducibility.
        predict_batch: Optional batch prediction callable (see
            ``shapley_effects``).
        progress: If ``True``, display tqdm progress bars over the
            permutation loop and the bootstrap loop (if B > 0).

    Returns:
        effects: Normalised Shapley effects, shape (d,).
        sh: Unscaled Shapley values, shape (d,).
        total_var: Estimated total variance.
        (lower, upper): CI bounds, only if B > 0.
    """
    if random_state is not None:
        np.random.seed(random_state)

    d = joint.d
    perms = [np.random.permutation(d).tolist() for _ in range(n_perm)]

    # Subsets are evaluated lazily — start without a known total.
    # compute_subset_data will grow the bar as new subsets are discovered.
    pbar = _Progress(0, enabled=progress)

    data = {}

    def ensure_data(u_set):
        if u_set not in data and len(u_set) > 0:
            compute_subset_data(f, joint, u_set, N, data,
                                predict_batch=predict_batch,
                                pbar=pbar)

    # Point estimate
    contrib = np.zeros(d)
    for perm in perms:
        S = frozenset()
        v_prev = 0.0
        for idx in perm:
            S_new = S.union([idx])
            ensure_data(S_new)
            v_curr = get_v_from_data(data[S_new])
            contrib[idx] += v_curr - v_prev
            S, v_prev = S_new, v_curr

    Sh = contrib / n_perm
    total_var = get_v_from_data(data[frozenset(range(d))])
    effects = Sh / total_var

    if B == 0:
        pbar.close()
        return effects, Sh, total_var

    # Bootstrap
    for entry in data.values():
        if entry[0] == 'full':
            Nsamp = len(entry[1])
            break
        elif entry[0] == 'pair':
            Nsamp = len(entry[1])
            break

    boot_effects = np.zeros((B, d))
    for b in range(B):
        idx = np.random.choice(Nsamp, size=Nsamp, replace=True)
        boot_data = {}
        for key, val in data.items():
            typ = val[0]
            if typ == 'full':
                boot_data[key] = ('full', val[1][idx])
            else:
                boot_data[key] = ('pair', val[1][idx], val[2][idx])
        contrib_b = np.zeros(d)
        for perm in perms:
            S = frozenset()
            v_prev = 0.0
            for idx in perm:
                S_new = S.union([idx])
                v_curr = get_v_from_data(boot_data[S_new])
                contrib_b[idx] += v_curr - v_prev
                S, v_prev = S_new, v_curr
        Sh_b = contrib_b / n_perm
        total_var_b = get_v_from_data(boot_data[frozenset(range(d))])
        boot_effects[b] = Sh_b / total_var_b

    lower = np.percentile(boot_effects, 100 * alpha / 2, axis=0)
    upper = np.percentile(boot_effects, 100 * (1 - alpha / 2), axis=0)
    pbar.close()
    return effects, Sh, total_var, lower, upper


# ------------------------------------------------------------
# Unified interface
# ------------------------------------------------------------
def shapley_effects(f, joint, N=10000, method='exhaustive', n_perm=1000,
                    B=0, alpha=0.05, random_state=None,
                    predict_batch=None, progress=False):
    """Compute Shapley effects for correlated inputs via Monte Carlo.

    This is the main entry point for the MC Shapley algorithm.
    It supports two computation methods and optional bootstrap
    confidence intervals.

    Args:
        f: Model function f(x) taking a 1D array and returning a scalar.
        joint: Distribution object with ``sample_joint(n)`` and
            ``sample_conditional(u_indices, fixed_x)`` methods.
        N: Monte Carlo sample size per subset.
        method: Computation method, 'exhaustive' or 'permutation'.
        n_perm: Number of random permutations (permutation method only).
        B: Number of bootstrap replications (0 to skip CIs).
        alpha: Significance level for confidence intervals.
        random_state: Random seed for reproducibility.
        predict_batch: Optional callable that accepts a 2D array (N, d)
            and returns a 1D array of predictions. When provided, batch
            evaluation is used for unconditional draws, greatly reducing
            Python-level call overhead.
        progress: If ``True``, display tqdm progress bars (requires
            ``tqdm`` to be installed).

    Returns:
        If B == 0:
            effects: Normalised Shapley effects, shape (d,).
            sh: Unscaled Shapley values, shape (d,).
            total_var: Estimated total variance, float.
        If B > 0:
            effects, sh, total_var, lower, upper
            where lower/upper are CI bounds, shape (d,).

    Examples:
        >>> import numpy as np
        >>> from shapleyx.utilities.mc_shapley import (
        ...     GaussianCopulaUniform, shapley_effects
        ... )
        >>> def ishigami(x):
        ...     return np.sin(x[0]) + 7*np.sin(x[1])**2 + 0.1*x[2]**4*np.sin(x[0])
        >>> joint = GaussianCopulaUniform(
        ...     [-np.pi]*3, [np.pi]*3, np.eye(3)
        ... )
        >>> eff, sh, var = shapley_effects(ishigami, joint, N=5000)
    """
    if random_state is not None:
        np.random.seed(random_state)

    if method == 'exhaustive':
        data = collect_shapley_data(f, joint, N, predict_batch=predict_batch,
                                    progress=progress)
        effects, sh, total_var = shapley_from_data(data, joint.d)
        if B > 0:
            _, lower, upper = bootstrap_shapley(
                data, joint.d, B, alpha, random_state
            )
            return effects, sh, total_var, lower, upper
        return effects, sh, total_var
    elif method == 'permutation':
        return shapley_effects_permutation(
            f, joint, N=N, n_perm=n_perm,
            B=B, alpha=alpha, random_state=random_state,
            predict_batch=predict_batch,
            progress=progress
        )
    else:
        raise ValueError("method must be 'exhaustive' or 'permutation'")


# ------------------------------------------------------------
# Convenience class for integration with rshdmr
# ------------------------------------------------------------
class MCShapley:
    """Monte Carlo Shapley effects estimation wrapper.

    Provides a simple interface for computing Shapley effects.
    Can be used standalone or integrated into the ``rshdmr`` class.

    Args:
        f: Model function f(x) taking a 1D array and returning a scalar.
        joint: Distribution object with ``sample_joint`` and
            ``sample_conditional`` methods.

    Examples:
        >>> mc = MCShapley(f=my_model, joint=my_distribution)
        >>> results = mc.compute(N=10000, method='exhaustive', B=500)
    """
    def __init__(self, f, joint, predict_batch=None):
        self.f = f
        self.joint = joint
        self.d = joint.d
        self.predict_batch = predict_batch

    def compute(self, N=10000, method='exhaustive', n_perm=1000,
                B=0, alpha=0.05, random_state=None, progress=False):
        """Compute Shapley effects.

        Args:
            N: Monte Carlo sample size per subset.
            method: 'exhaustive' or 'permutation'.
            n_perm: Number of permutations (permutation method only).
            B: Bootstrap replications (0 to skip).
            alpha: Significance level for CIs.
            random_state: Random seed.
            progress: If ``True``, display tqdm progress bars.

        Returns:
            pd.DataFrame with columns:
            - 'variable': Input variable names (if provided).
            - 'effect': Normalised Shapley effects.
            - 'lower', 'upper': CI bounds (if B > 0).
        """
        result = shapley_effects(
            self.f, self.joint, N=N, method=method,
            n_perm=n_perm, B=B, alpha=alpha,
            random_state=random_state,
            predict_batch=self.predict_batch,
            progress=progress
        )

        if B > 0:
            effects, sh, total_var, lower, upper = result
        else:
            effects, sh, total_var = result

        df = pd.DataFrame({
            'variable': [f'X{i+1}' for i in range(self.d)],
            'effect': effects,
            'shapley_value': sh,
        })
        df['total_variance'] = total_var

        if B > 0:
            df['lower'] = lower
            df['upper'] = upper

        return df
