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
from scipy.stats import norm, truncnorm
import itertools
import math
import pandas as pd


# -----------------------------------------------------------------------
# Optional Numba JIT for the bootstrap hot path.
# -----------------------------------------------------------------------
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


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
# Helper: wrap a 1D predict function to also accept 2D (batch) input
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
        self._L = np.linalg.cholesky(self.cov)

    def sample_joint(self, n):
        """Draw n joint samples from the full distribution.

        Args:
            n: Number of samples.

        Returns:
            Array of shape (n, d).
        """
        return np.random.multivariate_normal(self.mean, self.cov, n)

    def sample_joint_deterministic(self, U):
        """Map [0,1]^d points to joint samples (RQMC-compatible).

        Args:
            U: Array of shape (N, d) in [0, 1].

        Returns:
            Array of shape (N, d) in physical space.
        """
        Z = norm.ppf(U)                       # (N, d) standard normal
        return self.mean + Z @ self._L.T      # (N, d) correlated normal

    def _cond_params(self, u_indices):
        """Pre-compute conditional distribution parameters for a subset.

        Returns (v_indices, mu_v, A, cond_cov, L) where:
            cond_mean(x_u) = mu_v + (x_u - mu_u) @ A
            cond_cov = Sigma_vv - Sigma_vu @ inv(Sigma_uu) @ Sigma_uv
            L = cholesky(cond_cov) for fast sampling.
        """
        u = np.asarray(u_indices)
        all_idx = np.arange(self.d)
        v = np.setdiff1d(all_idx, u)

        if len(u) == 0:
            return v, None, None, None, None

        mu_u = self.mean[u]
        mu_v = self.mean[v]
        Sigma_uu = self.cov[np.ix_(u, u)]
        Sigma_uv = self.cov[np.ix_(u, v)]
        Sigma_vu = self.cov[np.ix_(v, u)]
        Sigma_vv = self.cov[np.ix_(v, v)]

        inv_Sigma_uu = np.linalg.inv(Sigma_uu)
        # A = inv(Sigma_uu) @ Sigma_uv  so that (x_u - mu_u) @ A gives the
        # adjustment to the conditional mean
        A = inv_Sigma_uu @ Sigma_uv          # (|u|, |v|)
        cond_cov = Sigma_vv - Sigma_vu @ A    # (|v|, |v|)
        # Cholesky for efficient draw: X_v = cond_mean + Z @ L.T
        L = np.linalg.cholesky(cond_cov + 1e-10 * np.eye(len(v)))

        return v, mu_v, A, cond_cov, L

    def sample_conditional(self, u_indices, fixed_x):
        """Draw one sample conditioned on fixed values for variables in u.

        Args:
            u_indices: Indices of variables to condition on.
            fixed_x: 1D array of fixed values for the conditioned variables.

        Returns:
            1D array of shape (d,).
        """
        # Route to batch implementation for consistency
        X = self.sample_conditional_batch(
            u_indices, np.atleast_2d(np.asarray(fixed_x)))
        return X[0]

    def sample_conditional_batch(self, u_indices, fixed_X):
        """Draw N conditional samples in one operation.

        Args:
            u_indices: Indices of variables to condition on.
            fixed_X: 2D array of shape (N, |u|) — fixed values for each draw.

        Returns:
            2D array of shape (N, d).
        """
        u = np.asarray(u_indices)
        N = fixed_X.shape[0]
        fixed_X = np.asarray(fixed_X)

        if len(u) == 0:
            return self.sample_joint(N)

        v, mu_v, A, _cond_cov, L = self._cond_params(u)

        # cond_means = mu_v + (fixed_X - mu_u) @ A
        # Shape: (|v|,) + (N, |u|) @ (|u|, |v|) = (N, |v|)
        mu_u = self.mean[u]
        cond_means = mu_v + (fixed_X - mu_u) @ A

        # Draw from N(0, I) and transform: Z @ L.T ~ N(0, cond_cov)
        Z = np.random.randn(N, len(v))
        X_v = cond_means + Z @ L.T

        X_full = np.zeros((N, self.d))
        X_full[:, u] = fixed_X
        X_full[:, v] = X_v
        return X_full

    def sample_conditional_batch_deterministic(self, u_indices, fixed_X, U_cond):
        """Conditional samples with deterministic innovations (RQMC-compatible).

        Args:
            u_indices: Indices of variables to condition on.
            fixed_X: 2D array of shape (N, |u|) — fixed values for each draw.
            U_cond: Array of shape (N, |v|) in [0,1] — deterministic
                numbers for the conditional innovations.

        Returns:
            2D array of shape (N, d).
        """
        u = np.asarray(u_indices)
        N = fixed_X.shape[0]
        fixed_X = np.asarray(fixed_X)

        if len(u) == 0:
            return self.sample_joint_deterministic(U_cond)

        v, mu_v, A, _cond_cov, L = self._cond_params(u)

        mu_u = self.mean[u]
        cond_means = mu_v + (fixed_X - mu_u) @ A

        Z_std = norm.ppf(U_cond)                # (N, |v|) — deterministic
        X_v = cond_means + Z_std @ L.T

        X_full = np.zeros((N, self.d))
        X_full[:, u] = fixed_X
        X_full[:, v] = X_v
        return X_full


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
        self._L = np.linalg.cholesky(self.corr)

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

    def sample_joint_deterministic(self, U):
        """Map [0,1]^d points to joint samples (RQMC-compatible).

        Args:
            U: Array of shape (N, d) in [0, 1].

        Returns:
            Array of shape (N, d) with uniform marginals.
        """
        Z = norm.ppf(U)                       # (N, d) standard normal
        Z_corr = Z @ self._L.T                # (N, d) correlated normal
        U_corr = norm.cdf(Z_corr)             # (N, d) in [0, 1]
        return self.lows + (self.highs - self.lows) * U_corr

    def _cond_params(self, u_indices):
        """Pre-compute conditional copula parameters for a subset.

        Returns (v_indices, A, cond_cov, L) where:
            cond_mean(z_u) = z_u @ A  (no absolute mean — zero-mean latent)
            cond_cov = Sigma_vv - Sigma_vu @ inv(Sigma_uu) @ Sigma_uv
            L = cholesky(cond_cov).
        """
        u = np.asarray(u_indices)
        all_idx = np.arange(self.d)
        v = np.setdiff1d(all_idx, u)

        if len(u) == 0:
            return v, None, None, None

        Sigma_uu = self.corr[np.ix_(u, u)]
        Sigma_uv = self.corr[np.ix_(u, v)]
        Sigma_vu = self.corr[np.ix_(v, u)]
        Sigma_vv = self.corr[np.ix_(v, v)]

        inv_Sigma_uu = np.linalg.inv(Sigma_uu)
        A = inv_Sigma_uu @ Sigma_uv           # (|u|, |v|)
        cond_cov = Sigma_vv - Sigma_vu @ A     # (|v|, |v|)
        L = np.linalg.cholesky(cond_cov + 1e-10 * np.eye(len(v)))

        return v, A, cond_cov, L

    def sample_conditional(self, u_indices, fixed_x):
        """Draw one sample conditioned on fixed values for variables in u.

        Args:
            u_indices: Indices of variables to condition on.
            fixed_x: 1D array of fixed values for the conditioned variables.

        Returns:
            1D array of shape (d,).
        """
        X = self.sample_conditional_batch(
            u_indices, np.atleast_2d(np.asarray(fixed_x)))
        return X[0]

    def sample_conditional_batch(self, u_indices, fixed_X):
        """Draw N conditional samples in one operation.

        Args:
            u_indices: Indices of variables to condition on.
            fixed_X: 2D array of shape (N, |u|) — fixed values for each draw.

        Returns:
            2D array of shape (N, d).
        """
        u = np.asarray(u_indices)
        N = fixed_X.shape[0]
        fixed_X = np.asarray(fixed_X)

        if len(u) == 0:
            return self.sample_joint(N)

        # Transform to latent normal space
        span_u = self.highs[u] - self.lows[u]
        fixed_u = (fixed_X - self.lows[u]) / span_u
        fixed_u = np.clip(fixed_u, 1e-12, 1 - 1e-12)
        Z_u = norm.ppf(fixed_u)                      # (N, |u|)

        v, A, _cond_cov, L = self._cond_params(u)

        # cond_means = Z_u @ A   (zero-mean latent)
        cond_means = Z_u @ A                          # (N, |v|)

        Z_std = np.random.randn(N, len(v))
        Z_v = cond_means + Z_std @ L.T                # (N, |v|)

        Z_full = np.zeros((N, self.d))
        Z_full[:, u] = Z_u
        Z_full[:, v] = Z_v
        U_full = norm.cdf(Z_full)
        return self.lows + (self.highs - self.lows) * U_full

    def sample_conditional_batch_deterministic(self, u_indices, fixed_X, U_cond):
        """Conditional samples with deterministic innovations (RQMC-compatible).

        Args:
            u_indices: Indices of variables to condition on.
            fixed_X: 2D array of shape (N, |u|) — fixed values for each draw.
            U_cond: Array of shape (N, |v|) in [0,1] — deterministic
                numbers for the conditional innovations.

        Returns:
            2D array of shape (N, d).
        """
        u = np.asarray(u_indices)
        N = fixed_X.shape[0]
        fixed_X = np.asarray(fixed_X)

        if len(u) == 0:
            return self.sample_joint_deterministic(U_cond)

        span_u = self.highs[u] - self.lows[u]
        fixed_u = (fixed_X - self.lows[u]) / span_u
        fixed_u = np.clip(fixed_u, 1e-12, 1 - 1e-12)
        Z_u = norm.ppf(fixed_u)                      # (N, |u|)

        v, A, _cond_cov, L = self._cond_params(u)

        cond_means = Z_u @ A                          # (N, |v|)
        Z_std = norm.ppf(U_cond)                     # (N, |v|) — deterministic
        Z_v = cond_means + Z_std @ L.T                # (N, |v|)

        Z_full = np.zeros((N, self.d))
        Z_full[:, u] = Z_u
        Z_full[:, v] = Z_v
        U_full = norm.cdf(Z_full)
        return self.lows + (self.highs - self.lows) * U_full


class GaussianCopulaArbitrary:
    """Correlated inputs with arbitrary marginals via a Gaussian copula.

    Each marginal can be any of:
      - ``'uniform'`` — Uniform(0, 1) on the unit interval.
        Fast path: identity transform, no PPF calls.
      - A ``scipy.stats`` continuous distribution — uses ``.ppf()``.
      - A callable ``ppf(u) -> float`` — maps [0, 1] to physical space.
      - A ``(cdf, ppf)`` tuple of callables — full specification.
      - ``None`` — treated as ``'uniform'`` (backward-compatible default).

    Samples are returned in **physical space** — matching the convention
    of :class:`GaussianCopulaUniform` and :class:`MultivariateNormal`.
    The surrogate model normalises internally, so this is transparent
    to the MC Shapley pipeline.

    Dependence is induced through a latent multivariate normal with the
    given correlation matrix.  Conditional sampling maps physical → [0, 1]
    via the marginal CDFs, conditions in the latent normal space, and
    maps back to physical space via the marginal PPFs.

    Args:
        marginals:
            Dict mapping variable names to marginal specifications.
            Order determines the column order of output arrays.
        corr:
            Correlation matrix for the latent normal, shape (d, d).

    Examples:

        Scipy distribution:

        >>> from scipy.stats import beta
        >>> joint = GaussianCopulaArbitrary(
        ...     marginals={'x0': beta(2, 5), 'x1': 'uniform'},
        ...     corr=np.eye(2),
        ... )
        >>> X = joint.sample_joint(100)  # physical space

        Custom PPF from empirical data:

        >>> joint = GaussianCopulaArbitrary(
        ...     marginals={'x0': empirical_ppf(posterior_samples)},
        ...     corr=corr_matrix,
        ... )
    """

    def __init__(self, marginals, corr):
        # ── Validate and store correlation ──
        self.corr = np.asarray(corr, dtype=float)
        self.d = len(marginals)
        if self.corr.shape != (self.d, self.d):
            raise ValueError(
                f"corr must be ({self.d}, {self.d}), got {self.corr.shape}"
            )
        self._L = np.linalg.cholesky(self.corr)

        # ── Resolve marginals ──
        self._var_names = []
        self._ppf = {}         # {j: callable(u) -> float}
        self._cdf = {}         # {j: callable(x) -> float} (for physical→[0,1])
        self._is_identity = {} # {j: bool} — fast path for uniform

        for j, (name, spec) in enumerate(marginals.items()):
            self._var_names.append(name)
            ppf_fn, cdf_fn = self._resolve_marginal(spec, name)
            self._ppf[j] = ppf_fn
            self._cdf[j] = cdf_fn
            self._is_identity[j] = (
                spec is None or spec == 'uniform'
            )

    # ── Marginal resolution ────────────────────────────────────

    @staticmethod
    def _resolve_marginal(spec, name):
        """Resolve a marginal specification to (ppf, cdf) callables."""
        if spec is None or spec == 'uniform':
            return (lambda u: u, lambda x: x)  # identity pair

        if isinstance(spec, tuple) and len(spec) == 2:
            cdf, ppf = spec
            if callable(ppf) and callable(cdf):
                return (ppf, cdf)
            raise TypeError(
                f"Marginal '{name}': tuple must be (cdf, ppf), "
                f"both callable."
            )

        if callable(spec):
            # Callable-only: assume PPF.  CDF not available.
            return (spec, None)

        # scipy.stats distribution — duck-type: has .ppf() and .cdf()
        if hasattr(spec, 'ppf') and hasattr(spec, 'cdf'):
            return (spec.ppf, spec.cdf)

        raise TypeError(
            f"Marginal '{name}' must be 'uniform', a scipy distribution, "
            f"a callable ppf(u), a (cdf, ppf) tuple, or None. "
            f"Got {type(spec)}."
        )

    # ── Physical → [0,1] and [0,1] → physical ──────────────────

    def _to_uniform(self, X, indices=None):
        """Map physical-space samples → [0, 1] via marginal CDFs.

        Args:
            X: (N, k) array in physical space.
            indices: Optional list of variable indices (length k).
                If None, X is assumed to have d columns.
        """
        k = X.shape[1]
        cols = indices if indices is not None else range(k)
        U = np.zeros_like(X)
        for ci, j in enumerate(cols):
            if self._is_identity[j]:
                U[:, ci] = X[:, ci]
            elif self._cdf[j] is not None:
                U[:, ci] = np.clip(self._cdf[j](X[:, ci]), 1e-15, 1 - 1e-15)
            else:
                raise RuntimeError(
                    f"Variable '{self._var_names[j]}' was specified with "
                    f"a PPF only — no CDF available for physical→[0,1] "
                    f"mapping. Use a (cdf, ppf) tuple instead."
                )
        return U

    def _to_physical(self, U):
        """Map [0, 1]-space samples → physical via marginal PPFs."""
        X = np.zeros_like(U)
        for j in range(self.d):
            if self._is_identity[j]:
                X[:, j] = U[:, j]
            else:
                X[:, j] = self._ppf[j](U[:, j])  # vectorised via scipy
        return X

    # ── Sampling ───────────────────────────────────────────────

    def sample_joint(self, n):
        """Draw *n* joint samples in **physical space**.

        Returns an array of shape ``(n, d)``.
        """
        Z = np.random.multivariate_normal(
            mean=np.zeros(self.d), cov=self.corr, size=n
        )
        U = norm.cdf(Z)                       # (n, d) in [0, 1]
        return self._to_physical(U)           # (n, d) in physical space

    def sample_joint_deterministic(self, U):
        """Map [0,1]^d points to joint samples (RQMC-compatible).

        Args:
            U: Array of shape (N, d) in [0, 1].

        Returns:
            Array of shape (N, d) in physical space.
        """
        Z = norm.ppf(U)                       # (N, d) standard normal
        Z_corr = Z @ self._L.T                # (N, d) correlated normal
        U_corr = norm.cdf(Z_corr)             # (N, d) in [0, 1]
        return self._to_physical(U_corr)       # (N, d) in physical space

    # ── Conditional parameters ─────────────────────────────────

    def _cond_params(self, u_indices):
        """Pre-compute conditional copula parameters for a subset.

        Returns ``(v_indices, A, cond_cov, L)`` where::

            cond_mean(z_u) = z_u @ A   (zero-mean latent)
            cond_cov = Sigma_vv - Sigma_vu @ inv(Sigma_uu) @ Sigma_uv
            L = cholesky(cond_cov).
        """
        u = np.asarray(u_indices)
        all_idx = np.arange(self.d)
        v = np.setdiff1d(all_idx, u)

        if len(u) == 0:
            return v, None, None, None

        Sigma_uu = self.corr[np.ix_(u, u)]
        Sigma_uv = self.corr[np.ix_(u, v)]
        Sigma_vu = self.corr[np.ix_(v, u)]
        Sigma_vv = self.corr[np.ix_(v, v)]

        inv_Sigma_uu = np.linalg.inv(Sigma_uu)
        A = inv_Sigma_uu @ Sigma_uv           # (|u|, |v|)
        cond_cov = Sigma_vv - Sigma_vu @ A     # (|v|, |v|)
        L = np.linalg.cholesky(cond_cov + 1e-10 * np.eye(len(v)))

        return v, A, cond_cov, L

    # ── Conditional sampling ───────────────────────────────────

    def sample_conditional(self, u_indices, fixed_x):
        """Draw one sample conditioned on fixed values in *physical space*.

        Returns a 1D array of shape ``(d,)``.
        """
        X = self.sample_conditional_batch(
            u_indices, np.atleast_2d(np.asarray(fixed_x, dtype=float))
        )
        return X[0]

    def sample_conditional_batch(self, u_indices, fixed_X):
        """Draw *N* conditional samples.

        Args:
            u_indices: Indices of variables to condition on.
            fixed_X: 2D array of shape ``(N, |u|)`` — fixed values
                in **physical space** for each draw.

        Returns:
            2D array of shape ``(N, d)`` in **physical space**.
        """
        u = np.asarray(u_indices)
        N = fixed_X.shape[0]
        fixed_X = np.asarray(fixed_X, dtype=float)

        if len(u) == 0:
            return self.sample_joint(N)

        # Physical → [0, 1] → normal scores
        fixed_U = self._to_uniform(fixed_X, indices=u)  # (N, |u|) in [0, 1]
        fixed_U = np.clip(fixed_U, 1e-15, 1 - 1e-15)
        Z_u = norm.ppf(fixed_U)                 # (N, |u|)

        v, A, _cond_cov, L = self._cond_params(u)

        # Conditional mean: Z_u @ A  (zero-mean latent)
        cond_means = Z_u @ A                    # (N, |v|)

        Z_std = np.random.randn(N, len(v))
        Z_v = cond_means + Z_std @ L.T          # (N, |v|)

        Z_full = np.zeros((N, self.d))
        Z_full[:, u] = Z_u
        Z_full[:, v] = Z_v

        U_full = norm.cdf(Z_full)               # (N, d) in [0, 1]
        return self._to_physical(U_full)        # (N, d) in physical space

    def sample_conditional_batch_deterministic(self, u_indices, fixed_X, U_cond):
        """Conditional samples with deterministic innovations (RQMC-compatible).

        Args:
            u_indices: Indices of variables to condition on.
            fixed_X: 2D array of shape (N, |u|) — fixed values
                in physical space for each draw.
            U_cond: Array of shape (N, |v|) in [0,1] — deterministic
                numbers for the conditional innovations.

        Returns:
            2D array of shape (N, d) in physical space.
        """
        u = np.asarray(u_indices)
        N = fixed_X.shape[0]
        fixed_X = np.asarray(fixed_X, dtype=float)

        if len(u) == 0:
            return self.sample_joint_deterministic(U_cond)

        fixed_U = self._to_uniform(fixed_X, indices=u)
        fixed_U = np.clip(fixed_U, 1e-15, 1 - 1e-15)
        Z_u = norm.ppf(fixed_U)                 # (N, |u|)

        v, A, _cond_cov, L = self._cond_params(u)
        cond_means = Z_u @ A                    # (N, |v|)

        Z_std = norm.ppf(U_cond)               # (N, |v|) — deterministic
        Z_v = cond_means + Z_std @ L.T          # (N, |v|)

        Z_full = np.zeros((N, self.d))
        Z_full[:, u] = Z_u
        Z_full[:, v] = Z_v

        U_full = norm.cdf(Z_full)               # (N, d) in [0, 1]
        return self._to_physical(U_full)        # (N, d) in physical space


class TruncatedMultivariateNormal:
    """Truncated multivariate normal inputs with conditional sampling.

    Each marginal is a normal distribution truncated to
    ``[lower[i], upper[i]]``.  Dependence is induced through the
    specified covariance matrix.  Both joint and conditional sampling
    use Gibbs sampling, making the class usable for any truncation
    pattern where the truncation region is a hyper-rectangle.

    Args:
        mean: 1D array of means, shape (d,).
        cov: 2D covariance matrix, shape (d, d).
        lower: Lower truncation bounds, shape (d,).  Use ``-np.inf``
            for no lower bound.
        upper: Upper truncation bounds, shape (d,).  Use ``np.inf``
            for no upper bound.
        joint_burn_in: Number of Gibbs iterations per independent
            joint sample (default 30).
        cond_burn_in: Number of Gibbs iterations per conditional
            sample (default 5).  The chain is started at the
            untruncated conditional mean, which is well-centred, so
            a small value is sufficient.
    """
    def __init__(self, mean, cov, lower, upper,
                 joint_burn_in=30, cond_burn_in=5):
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.d = len(mean)
        self._joint_burn_in = joint_burn_in
        self._cond_burn_in = cond_burn_in

        assert self.cov.shape == (self.d, self.d), \
            f"cov must be ({self.d}, {self.d}), got {self.cov.shape}"
        assert self.lower.shape == (self.d,), \
            f"lower must be ({self.d},), got {self.lower.shape}"
        assert self.upper.shape == (self.d,), \
            f"upper must be ({self.d},), got {self.upper.shape}"
        assert np.all(self.lower < self.upper), \
            "lower must be strictly less than upper element-wise"

        # Precompute Gibbs regression coefficients for joint sampling.
        self._gibbs_betas = []
        self._gibbs_stds = []
        for j in range(self.d):
            not_j = [i for i in range(self.d) if i != j]
            Sigma_jj = self.cov[j, j]
            Sigma_j_notj = self.cov[j, not_j]
            Sigma_notj_notj = self.cov[np.ix_(not_j, not_j)]
            inv = np.linalg.inv(Sigma_notj_notj)
            beta = inv @ Sigma_j_notj            # (d-1,)
            cond_var = Sigma_jj - Sigma_j_notj @ beta
            self._gibbs_betas.append((not_j, beta))
            self._gibbs_stds.append(np.sqrt(max(cond_var, 0.0)))

    # --------------------------------------------------------
    # Conditional-distribution parameters (same formulas as
    # the untruncated multivariate normal)
    # --------------------------------------------------------
    def _cond_params(self, u_indices):
        """Pre-compute conditional distribution parameters for a subset.

        Returns (v_indices, mu_v, A, cond_cov, L) where:
            cond_mean(x_u) = mu_v + (x_u - mu_u) @ A
            cond_cov = Sigma_vv - Sigma_vu @ inv(Sigma_uu) @ Sigma_uv
            L = cholesky(cond_cov).
        """
        u = np.asarray(u_indices)
        all_idx = np.arange(self.d)
        v = np.setdiff1d(all_idx, u)

        if len(u) == 0:
            return v, None, None, None, None

        mu_u = self.mean[u]
        mu_v = self.mean[v]
        Sigma_uu = self.cov[np.ix_(u, u)]
        Sigma_uv = self.cov[np.ix_(u, v)]
        Sigma_vu = self.cov[np.ix_(v, u)]
        Sigma_vv = self.cov[np.ix_(v, v)]

        inv_Sigma_uu = np.linalg.inv(Sigma_uu)
        A = inv_Sigma_uu @ Sigma_uv          # (|u|, |v|)
        cond_cov = Sigma_vv - Sigma_vu @ A    # (|v|, |v|)
        L = np.linalg.cholesky(cond_cov + 1e-10 * np.eye(len(v)))

        return v, mu_v, A, cond_cov, L

    # --------------------------------------------------------
    # Joint sampling — vectorised Gibbs across all chains
    # --------------------------------------------------------
    def sample_joint(self, n):
        """Draw n joint samples via vectorised Gibbs sampling.

        Args:
            n: Number of samples.

        Returns:
            Array of shape (n, d).
        """
        d = self.d
        lb = self.lower
        ub = self.upper

        # Initialise all chains at the mean, clipped to bounds.
        X = np.tile(np.clip(self.mean, lb, ub), (n, 1))

        for _ in range(self._joint_burn_in):
            for j in range(d):
                not_j, beta = self._gibbs_betas[j]
                # Conditional mean for variable j (n,)
                cond_mean = (self.mean[j]
                             + (X[:, not_j] - self.mean[not_j]) @ beta)
                cond_std = self._gibbs_stds[j]

                # Standardised truncation bounds
                a = (lb[j] - cond_mean) / cond_std
                b = (ub[j] - cond_mean) / cond_std

                # Sample — fast path for unbounded variables
                inf_mask = np.isneginf(a) & np.isposinf(b)
                if np.all(inf_mask):
                    X[:, j] = np.random.normal(cond_mean, cond_std)
                else:
                    new = np.empty(n)
                    if np.any(inf_mask):
                        new[inf_mask] = np.random.normal(
                            cond_mean[inf_mask], cond_std)
                    not_inf = ~inf_mask
                    if np.any(not_inf):
                        new[not_inf] = truncnorm.rvs(
                            a[not_inf], b[not_inf],
                            loc=cond_mean[not_inf],
                            scale=cond_std)
                    X[:, j] = new

        return X

    # --------------------------------------------------------
    # Conditional sampling
    # --------------------------------------------------------
    def sample_conditional(self, u_indices, fixed_x):
        """Draw one sample conditioned on fixed values for variables in u.

        Args:
            u_indices: Indices of variables to condition on.
            fixed_x: 1D array of fixed values for the conditioned variables.

        Returns:
            1D array of shape (d,).
        """
        X = self.sample_conditional_batch(
            u_indices, np.atleast_2d(np.asarray(fixed_x, dtype=float)))
        return X[0]

    def sample_conditional_batch(self, u_indices, fixed_X):
        """Draw N conditional samples via vectorised Gibbs sampling.

        Args:
            u_indices: Indices of variables to condition on.
            fixed_X: 2D array of shape (N, |u|) — fixed values.

        Returns:
            2D array of shape (N, d).
        """
        u = np.asarray(u_indices)
        N = fixed_X.shape[0]
        fixed_X = np.asarray(fixed_X, dtype=float)

        if len(u) == 0:
            return self.sample_joint(N)

        v, mu_v, A, cond_cov, _L = self._cond_params(u)
        n_v = len(v)
        mu_u = self.mean[u]

        # Conditional means: (N, |v|)
        cond_means = mu_v + (fixed_X - mu_u) @ A

        # Truncation bounds for the v-variables
        lb_v = self.lower[v]
        ub_v = self.upper[v]

        # Precompute Gibbs parameters for cond_cov (shared across rows).
        gibbs_betas_v = []
        gibbs_stds_v = []
        for j in range(n_v):
            not_j = [i for i in range(n_v) if i != j]
            Sigma_jj = cond_cov[j, j]
            Sigma_j_notj = cond_cov[j, not_j]
            Sigma_notj_notj = cond_cov[np.ix_(not_j, not_j)]
            inv = np.linalg.inv(Sigma_notj_notj)
            beta = inv @ Sigma_j_notj
            cond_var = Sigma_jj - Sigma_j_notj @ beta
            gibbs_betas_v.append((not_j, beta))
            gibbs_stds_v.append(np.sqrt(max(cond_var, 0.0)))

        # Initialise at conditional means, clipped to bounds.
        X_v = np.clip(cond_means, lb_v, ub_v)

        for _ in range(self._cond_burn_in):
            for j in range(n_v):
                not_j, beta = gibbs_betas_v[j]
                # Conditional mean for variable j within the v-block.
                # The "unconditional" mean for the truncated v-block is
                # cond_means[:, j]; we express the conditional mean as
                #   mean_j + beta @ (x[-j] - mean[-j])
                mean_v_j = cond_means[:, j]
                cond_mean_j = (mean_v_j
                               + (X_v[:, not_j] - cond_means[:, not_j]) @ beta)
                cond_std_j = gibbs_stds_v[j]

                a = (lb_v[j] - cond_mean_j) / cond_std_j
                b = (ub_v[j] - cond_mean_j) / cond_std_j

                inf_mask = np.isneginf(a) & np.isposinf(b)
                if np.all(inf_mask):
                    X_v[:, j] = np.random.normal(cond_mean_j, cond_std_j)
                else:
                    new = np.empty(N)
                    if np.any(inf_mask):
                        new[inf_mask] = np.random.normal(
                            cond_mean_j[inf_mask], cond_std_j)
                    not_inf = ~inf_mask
                    if np.any(not_inf):
                        new[not_inf] = truncnorm.rvs(
                            a[not_inf], b[not_inf],
                            loc=cond_mean_j[not_inf],
                            scale=cond_std_j)
                    X_v[:, j] = new

        X_full = np.zeros((N, self.d))
        X_full[:, u] = fixed_X
        X_full[:, v] = X_v
        return X_full


def coalitions_up_to_k(d, k_max):
    """Generate non-empty subsets up to size k_max, plus the full set.

    The full set is always included because it is needed for total
    variance estimation.  When k_max is None or >= d, all 2^d - 1
    non-empty subsets are returned (standard exhaustive enumeration).

    Args:
        d: Number of input dimensions.
        k_max: Maximum coalition size (1 to d-1).  If None, returns
            all subsets.

    Returns:
        List of frozenset objects.
    """
    if k_max is None or k_max >= d:
        return [frozenset(s) for k in range(1, d + 1)
                for s in itertools.combinations(range(d), k)]
    subsets = []
    for k in range(1, min(k_max + 1, d)):
        for s in itertools.combinations(range(d), k):
            subsets.append(frozenset(s))
    # Always include the full set for total variance
    full = frozenset(range(d))
    if full not in subsets:
        subsets.append(full)
    return subsets


# ------------------------------------------------------------
# Core functions for exhaustive method
# ------------------------------------------------------------
def collect_shapley_data(f, joint, N=10000, predict_batch=None,
                         progress=False, k_max=None):
    """Compute and store outputs for all non-empty subsets (exhaustive).

    For each subset u of variable indices:
    - If |u| == d (the full set): draw N joint samples, evaluate f,
      store the outputs for variance estimation.
    - Otherwise: draw N joint samples X, evaluate f(X) in batch; then
      draw N conditional samples X_cond (with variables in u fixed to
      X[:, u]) and evaluate f(X_cond) in batch.  Store the paired
      outputs for covariance estimation.

    Args:
        f: Model function f(x) taking a 1D array and returning a scalar.
        joint: Distribution object with sample_joint, sample_conditional,
            and sample_conditional_batch.
        N: Number of Monte Carlo samples per subset.
        predict_batch: Optional callable that accepts a 2D array (N, d)
            and returns a 1D array of predictions. When provided, batch
            evaluation is used for both unconditional and conditional
            draws. When None, ``f`` is called once per sample.
        k_max: Optional maximum coalition size.  When set, only subsets
            up to size ``k_max`` are evaluated (plus the full set for
            total variance).  This is *exact* when the model has no
            interactions above order ``k_max``; otherwise it provides
            an approximation.  Default ``None`` evaluates all 2^d - 1
            subsets.
        progress: If ``True``, display a single tqdm progress bar.

    Returns:
        dict mapping frozenset(u) -> tuple describing stored data.
    """
    d = joint.d
    subsets = coalitions_up_to_k(d, k_max)
    n_subsets = len(subsets)
    n_partial = sum(1 for u in subsets if len(u) < d)

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
            # --- Side A: unconditional draws (N evals, batched) ---
            X = joint.sample_joint(N)
            if predict_batch is not None:
                Y1 = np.asarray(predict_batch(X), dtype=float)
            else:
                Y1 = np.array([f(X[i]) for i in range(N)])
            pbar.update(N)

            # --- Side B: conditional draws (N evals, batched) ---
            X_cond = joint.sample_conditional_batch(u_list, X[:, u_list])
            if predict_batch is not None:
                Y2 = np.asarray(predict_batch(X_cond), dtype=float)
            else:
                Y2 = np.array([f(X_cond[i]) for i in range(N)])
            pbar.update(N)
            data[u] = ('pair', Y1, Y2)

    pbar.close()
    return data


def collect_shapley_data_qmc(f, joint, N=4096, predict_batch=None,
                              progress=False, k_max=None,
                              scramble=True, seed=None):
    """Compute outputs for all non-empty subsets via RQMC (single Sobol sequence).

    Uses a single 2d-dimensional scrambled Sobol sequence.  The first *d*
    columns provide the joint samples (shared across all subsets); the
    remaining *d* columns supply conditional innovations.  This avoids
    the cross-sequence correlation that arises from independent Sobol
    samplers and guarantees that joint-sample coverage is identical for
    every subset.

    .. note::

       Sobol sequences require N to be a power of 2 for optimal
       uniformity; non-power-of-2 N triggers a warning from scipy
       but the estimator remains valid with scrambled points.

    Args:
        f: Model function f(x) → scalar.
        joint: Distribution with ``sample_joint_deterministic`` and
            ``sample_conditional_batch_deterministic``.
        N: Samples per subset (should be a power of 2).
        predict_batch: Optional batch predictor f(X) → 1D array.
        k_max: Optional maximum coalition size.
        progress: Show tqdm progress bar.
        scramble: Use Owen-scrambled Sobol (default True).
        seed: Integer seed for scrambling.

    Returns:
        dict mapping frozenset(u) → tuple, same format as
        ``collect_shapley_data``.
    """
    from scipy.stats.qmc import Sobol

    d = joint.d
    subsets = coalitions_up_to_k(d, k_max)
    n_subsets = len(subsets)
    n_partial = sum(1 for u in subsets if len(u) < d)

    total_evals = N + 2 * N * n_partial
    pbar = _Progress(total_evals, enabled=progress)

    # --- Single 2d-dimensional Sobol sequence ---
    sampler = Sobol(2 * d, scramble=scramble, seed=seed)
    U_all = sampler.random(N)           # (N, 2d)

    # Joint samples from first d columns (shared across all subsets)
    U_joint = U_all[:, :d]              # (N, d)
    X_joint = joint.sample_joint_deterministic(U_joint)
    if predict_batch is not None:
        Y_joint = np.asarray(predict_batch(X_joint), dtype=float)
    else:
        Y_joint = np.array([f(X_joint[i]) for i in range(N)])

    # Extra columns for conditional innovations: d .. 2d-1
    U_extra = U_all[:, d:]              # (N, d)

    data = {}
    for u in subsets:
        u_list = list(u)
        if len(u) == d:
            data[u] = ('full', Y_joint)
            pbar.update(N)
        else:
            v_size = d - len(u_list)

            # --- Side A: joint (already computed, shared) ---
            # --- Side B: conditional ---
            U_cond = U_extra[:, :v_size]        # (N, |v|)
            X_cond = joint.sample_conditional_batch_deterministic(
                u_list, X_joint[:, u_list], U_cond)
            if predict_batch is not None:
                Y2 = np.asarray(predict_batch(X_cond), dtype=float)
            else:
                Y2 = np.array([f(X_cond[i]) for i in range(N)])
            pbar.update(2 * N)   # N for joint (shared) + N for cond
            data[u] = ('pair', Y_joint, Y2)

    pbar.close()
    return data


def shapley_from_data(data, d, _weights=None, k_max=None):
    """Compute point estimates of Shapley effects from collected data.

    Uses the covariance-based formulation: v(u) = Cov[f(X), f(X_u)]
    where X_u is a conditional sample sharing the same background
    variables as X.

    When ``k_max`` is set and some coalitions are missing from the
    data, Shapley contributions are only computed for coalitions whose
    v(u) and v(u ∪ {i}) are both available.  The effects are then
    renormalised to sum to 1, which distributes the missing variance
    proportionally.  For models whose HDMR expansion is bounded at
    order ``k_max`` (e.g., RS-HDMR surrogates), this is exact.

    Args:
        data: Dict from collect_shapley_data.
        d: Number of input dimensions.
        _weights: Optional precomputed Shapley weights array of length d,
            where ``_weights[k] = k! (d-k-1)! / d!``.  When ``None`` the
            weights are computed on first call and cached (memoised) for
            reuse across bootstrap iterations.
        k_max: Optional maximum coalition size used during data
            collection.  Used to restrict the accumulation to
            available coalitions.

    Returns:
        effects: Normalised Shapley effects (sums to 1), shape (d,).
        sh: Unscaled Shapley values, shape (d,).
        total_var: Estimated total variance.
    """
    # ---- compute v(u) from stored data ----
    v = {frozenset(): 0.0}
    for u, typ in data.items():
        if typ[0] == 'full':
            Y = typ[1]
            v[u] = np.var(Y)
        else:   # 'pair'
            Y1, Y2 = typ[1], typ[2]
            cov = np.mean(Y1 * Y2) - np.mean(Y1) * np.mean(Y2)
            v[u] = cov

    total_var = v[frozenset(range(d))]

    # ---- precompute Shapley weights once per d (memoised) ----
    if _weights is None:
        _weights = _get_shapley_weights(d)

    # ---- accumulate Shapley values ----
    sh = np.zeros(d)
    subsets_all = [frozenset(s) for k in range(d + 1)
                   for s in itertools.combinations(range(d), k)]

    truncating = k_max is not None and k_max < d

    for i in range(d):
        for u in subsets_all:
            if i not in u:
                u_with_i = u.union({i})
                if truncating and (u not in v or u_with_i not in v):
                    # Skip coalitions where we don't have both v(u)
                    # and v(u ∪ {i}) — these will be accounted for
                    # by renormalisation.
                    continue
                diff = v[u_with_i] - v[u]
                sh[i] += _weights[len(u)] * diff

    if truncating:
        # Renormalise: the skipped coalitions would have contributed
        # additional variance.  Distribute proportionally.
        sh_sum = sh.sum()
        if sh_sum > 0:
            sh = sh * (total_var / sh_sum)

    effects = sh / total_var
    return effects, sh, total_var


# ────────────────────────────────────────────────────────────────
# Owen (two-level / grouped) Shapley effects
# ────────────────────────────────────────────────────────────────

def _build_group_index(groups, var_names):
    """Build the variable→group and group→variable index mappings.

    Args:
        groups: dict[str, list[str]] — group name → variable names.
        var_names: list[str] — ordered list of all variable names.

    Returns:
        group_of_var: np.array of shape (d,) — group index for each var.
        group_vars: list[list[int]] — variable indices in each group.
        group_names: list[str] — group names in order.
    """
    d = len(var_names)
    var_to_idx = {name: i for i, name in enumerate(var_names)}
    group_of_var = np.full(d, -1, dtype=int)
    group_vars = []
    group_names = []

    for g_idx, (g_name, g_var_names) in enumerate(groups.items()):
        group_names.append(g_name)
        indices = []
        for v_name in g_var_names:
            if v_name not in var_to_idx:
                raise ValueError(
                    f"Variable '{v_name}' in group '{g_name}' not found "
                    f"in variable list {var_names}."
                )
            vi = var_to_idx[v_name]
            if group_of_var[vi] != -1:
                raise ValueError(
                    f"Variable '{v_name}' appears in multiple groups."
                )
            group_of_var[vi] = g_idx
            indices.append(vi)
        group_vars.append(indices)

    missing = np.where(group_of_var == -1)[0]
    if len(missing) > 0:
        missing_names = [var_names[i] for i in missing]
        raise ValueError(
            f"Variables not assigned to any group: {missing_names}. "
            f"All {d} variables must be covered by the groups dict."
        )

    return group_of_var, group_vars, group_names


def _group_coalitions_to_var_subsets(K, group_vars, k_max=None):
    """Generate variable frozensets from group coalitions.

    For each group coalition C ⊆ {0,…,K-1}, build the variable subset
    S_C = ∪_{k∈C} G_k.  Always includes the full set (all variables).

    When *k_max* is set, only group coalitions up to size *k_max* are
    enumerated (plus the full-set complement when k_max ≥ K/2).
    """
    subsets = []
    if k_max is None or k_max >= K:
        for size in range(1, K + 1):
            for C in itertools.combinations(range(K), size):
                var_set = set()
                for k in C:
                    var_set.update(group_vars[k])
                subsets.append(frozenset(var_set))
    else:
        for size in range(1, min(k_max + 1, K)):
            for C in itertools.combinations(range(K), size):
                var_set = set()
                for k in C:
                    var_set.update(group_vars[k])
                subsets.append(frozenset(var_set))
        # Always include full set
        full = frozenset(range(sum(len(g) for g in group_vars)))
        if full not in subsets:
            subsets.append(full)
    return subsets


def collect_shapley_data_groups(f, joint, groups, var_names,
                                 N=10000, predict_batch=None,
                                 progress=False, k_max=None):
    """Collect pick-freeze data for Owen (grouped) Shapley effects.

    Thin wrapper around :func:`collect_shapley_data` — collects ALL
    variable-level subsets (same as standard Shapley).  The *groups*
    parameter is stored for documentation but does not reduce the
    number of subsets evaluated, because the inner Shapley (within-group)
    computation requires v(S) for all combinations of individual
    variables with background groups.

    Args:
        f: Model function.
        joint: Distribution object.
        groups: dict[str, list[str]] — group definitions (for doc).
        var_names: list[str] — ordered variable names.
        N: Samples per subset.
        predict_batch: Optional batch predictor.
        progress: Show tqdm progress bar.
        k_max: Max coalition size (variable-level).

    Returns:
        dict mapping frozenset(variable_indices) → ('full', Y) or
        ('pair', Y1, Y2).
    """
    # Owen effects need all variable-level subsets for the inner
    # Shapley computation.  The groups only affect aggregation.
    return collect_shapley_data(
        f, joint, N=N, predict_batch=predict_batch,
        progress=progress, k_max=k_max,
    )


def owen_from_data(data, groups, var_names, k_max=None):
    """Compute Owen (two-level Shapley) effects from collected data.

    No additional model evaluations — reuses the *data* dict produced
    by :func:`collect_shapley_data`.

    Uses the standard Owen value formula: for each variable *i* in group
    *G_k*, the effect is the expected marginal contribution over all
    permutations that respect the group partition (groups permuted first,
    then members within each group).

    Args:
        data: Dict from :func:`collect_shapley_data`.
        groups: dict[str, list[str]] — group definitions.
        var_names: list[str] — ordered variable names.
        k_max: Coalition truncation used during data collection
            (for renormalisation).

    Returns:
        group_effects: OrderedDict[str, float] — outer Shapley effects
            per group (sum to total variance).
        individual_effects: OrderedDict[str, float] — inner Owen effects
            per variable (sum to total variance).
        total_var: float — estimated total output variance.
    """
    d = len(var_names)
    group_of_var, group_vars, group_names = _build_group_index(
        groups, var_names)
    K = len(groups)

    # ── Compute v(u) from stored data ──
    v = {frozenset(): 0.0}
    for u, typ in data.items():
        if typ[0] == 'full':
            v[u] = np.var(typ[1])
        else:
            Y1, Y2 = typ[1], typ[2]
            v[u] = np.mean(Y1 * Y2) - np.mean(Y1) * np.mean(Y2)

    total_var = v[frozenset(range(d))]

    # ── All subsets of each group ──
    group_subsets_cache = {}
    for g_idx in range(K):
        g_size = len(group_vars[g_idx])
        g_subsets = [frozenset(s) for k in range(g_size + 1)
                     for s in itertools.combinations(range(g_size), k)]
        group_subsets_cache[g_idx] = g_subsets

    # ── Outer Shapley (K-player game over groups) ──
    group_sh = np.zeros(K)
    weights_K = _get_shapley_weights(K)
    group_subsets_all = [frozenset(s) for k in range(K + 1)
                         for s in itertools.combinations(range(K), k)]

    # Track how much weight we actually used (for renormalisation
    # when k_max truncation means some variable subsets are missing)
    group_weight_used = np.zeros(K)

    for g_idx in range(K):
        for C in group_subsets_all:
            if g_idx in C:
                continue
            C_with = C.union({g_idx})
            var_C = frozenset().union(
                *(set(group_vars[k]) for k in C))
            var_C_with = frozenset().union(
                *(set(group_vars[k]) for k in C_with))

            w = weights_K[len(C)]
            if var_C not in v or var_C_with not in v:
                continue  # subset not available (truncation)

            group_sh[g_idx] += w * (v[var_C_with] - v[var_C])
            group_weight_used[g_idx] += w

    # Renormalise outer effects when subset enumeration was truncated.
    # This is exact for surrogates whose HDMR order ≤ k_max.
    needs_renorm = (k_max is not None and k_max < d)
    if needs_renorm:
        for g_idx in range(K):
            if group_weight_used[g_idx] > 0:
                group_sh[g_idx] *= 1.0 / group_weight_used[g_idx]
        # Scale to total variance
        gs = group_sh.sum()
        if gs > 0:
            group_sh = group_sh * (total_var / gs)

    # ── Inner Owen (full two-level formula) ──
    inner_sh = np.zeros(d)

    for g_idx in range(K):
        g_indices = group_vars[g_idx]
        g_size = len(g_indices)

        if g_size == 1:
            inner_sh[g_indices[0]] = group_sh[g_idx]
            continue

        g_subsets = group_subsets_cache[g_idx]
        # Other groups
        other_groups = [k for k in range(K) if k != g_idx]

        # Precompute all subsets of other groups
        other_subsets = [frozenset(s) for k in range(len(other_groups) + 1)
                         for s in itertools.combinations(other_groups, k)]

        # For each variable in the group, iterate over:
        #   T ⊆ G_k \\ {i}  (within-group predecessors)
        #   R ⊆ other_groups (between-group predecessors)
        for gi_local in range(g_size):
            vi = g_indices[gi_local]

            for T_local in g_subsets:
                if gi_local in T_local:
                    continue
                T_with_local = T_local.union({gi_local})

                # Map local → variable indices
                T_vars = frozenset(g_indices[i] for i in T_local)
                T_with_vars = frozenset(g_indices[i]
                                        for i in T_with_local)

                for R in other_subsets:
                    # Union of groups in R
                    Q = frozenset().union(
                        *(set(group_vars[k]) for k in R))

                    key_without = frozenset(Q | T_vars)
                    key_with = frozenset(Q | T_with_vars)

                    if key_without not in v or key_with not in v:
                        continue

                    # Owen weight: combine group-level and within-group
                    t = len(T_local)
                    r = len(R)
                    weight = (math.factorial(t)
                              * math.factorial(g_size - t - 1)
                              * math.factorial(r)
                              * math.factorial(K - r - 1)
                              / (math.factorial(g_size)
                                 * math.factorial(K)))

                    inner_sh[vi] += weight * (
                        v[key_with] - v[key_without])

    # Renormalise inner effects if k_max truncation caused skipped subsets
    if k_max is not None and k_max < d:
        sh_sum = inner_sh.sum()
        if sh_sum > 0:
            inner_sh = inner_sh * (total_var / sh_sum)

    # ── Build output ──
    from collections import OrderedDict

    group_effects = OrderedDict()
    for g_idx, g_name in enumerate(group_names):
        group_effects[g_name] = float(group_sh[g_idx])

    individual_effects = OrderedDict()
    for vi, v_name in enumerate(var_names):
        individual_effects[v_name] = float(inner_sh[vi])

    return group_effects, individual_effects, float(total_var)


# Module-level cache for the Shapley weights (one per dimensionality).
_shapley_weight_cache = {}


# ------------------------------------------------------------
# Helper: extract Sobol indices from collected data
# ------------------------------------------------------------
def sobol_from_data(data, d):
    """Extract first-order and total-order Sobol indices from MC data.

    Uses the covariance formulation v(u) = Cov[f(X), f(X_u)].
    Since v(u) = V[E(f(X) | X_u)] under the Owen & Prieur (2017)
    framework, first-order and total-order Sobol indices can be
    computed without additional model evaluations.

    Args:
        data: Dict from ``collect_shapley_data``.
        d: Number of input dimensions.

    Returns:
        S: First-order Sobol indices, shape (d,).
        T: Total-order Sobol indices, shape (d,).
    """
    # Compute v(u) from stored data
    v = {frozenset(): 0.0}
    for u, typ in data.items():
        if typ[0] == 'full':
            v[u] = np.var(typ[1])
        else:
            Y1, Y2 = typ[1], typ[2]
            v[u] = np.mean(Y1 * Y2) - np.mean(Y1) * np.mean(Y2)

    v_full = v[frozenset(range(d))]

    # First-order: S_i = v({i}) / v_full
    S = np.array([v.get(frozenset([i]), 0.0) / v_full for i in range(d)])

    # Total-order: T_i = 1 - v(all_except_i) / v_full
    # When {-i} is not in data (e.g. k_max < d-1), set T_i to NaN
    T = np.empty(d)
    for i in range(d):
        key_i = frozenset([j for j in range(d) if j != i])
        if key_i in v:
            T[i] = 1.0 - v[key_i] / v_full
        else:
            T[i] = np.nan

    return S, T


# Module-level cache for the Shapley weights (one per dimensionality). _shapley_weight_cache is defined above



def _get_shapley_weights(d):
    """Return array w[k] = k! (d-k-1)! / d! for k = 0, ..., d-1, plus w[d]=0."""
    if d in _shapley_weight_cache:
        return _shapley_weight_cache[d]
    w = np.empty(d + 1)
    d_fact = math.factorial(d)
    for k in range(d):
        w[k] = (math.factorial(k) * math.factorial(d - k - 1)) / d_fact
    w[d] = 0.0   # guard — the full set is never used as u (i ∉ full set)
    _shapley_weight_cache[d] = w
    return w


# -----------------------------------------------------------------------
# Compiled array-based bootstrap helpers (Numba-accelerated when available)
# -----------------------------------------------------------------------
def _data_to_arrays(data, d):
    """Convert the data dict into flat arrays for compiled bootstrap.

    Returns (Y_full, pair_Y1, pair_Y2, subset_sizes, union_lookup) where:
    - Y_full: (N,) — outputs for the full set
    - pair_Y1, pair_Y2: (n_pairs, N) — paired outputs for partial subsets
    - subset_sizes: int array (n_all,) — |u| for each canonical subset
    - union_lookup: int array (n_all, d) — index of u ∪ {i}, or -1 if i ∈ u
      (index 0 is the empty set, index n_all-1 is the full set)
    """
    # Build canonical ordering of all subsets
    all_subsets = [frozenset(s) for k in range(d + 1)
                   for s in itertools.combinations(range(d), k)]
    mask_to_idx = {s: i for i, s in enumerate(all_subsets)}
    n_all = len(all_subsets)

    # Subset sizes and union lookup
    subset_sizes = np.array([len(s) for s in all_subsets], dtype=np.int32)
    union_lookup = np.full((n_all, d), -1, dtype=np.int32)
    for idx, s in enumerate(all_subsets):
        for i in range(d):
            if i not in s:
                u2 = s.union({i})
                union_lookup[idx, i] = mask_to_idx[u2]

    # Extract Y arrays
    N = len(data[all_subsets[-1]][1])
    Y_full = data[all_subsets[-1]][1].astype(np.float64)

    pair_subsets = all_subsets[1:-1]   # exclude empty and full
    n_pairs = len(pair_subsets)
    pair_Y1 = np.empty((n_pairs, N), dtype=np.float64)
    pair_Y2 = np.empty((n_pairs, N), dtype=np.float64)
    for j, s in enumerate(pair_subsets):
        pair_Y1[j] = data[s][1]
        pair_Y2[j] = data[s][2]

    return Y_full, pair_Y1, pair_Y2, subset_sizes, union_lookup


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _bootstrap_iter_numba(Y_full, pair_Y1, pair_Y2, subset_sizes,
                              union_lookup, weights, d, B, alpha,
                              rng_states):
        """Compiled bootstrap: B iterations of resample → Shapley.

        Args:
            Y_full, pair_Y1, pair_Y2: data arrays (see _data_to_arrays).
            subset_sizes: int32 (n_all,).
            union_lookup: int32 (n_all, d).
            weights: float64 (d,).
            d: input dimensionality.
            B: number of bootstrap replications.
            alpha: significance level.
            rng_states: int32 (B,) — per-iteration random seeds.

        Returns:
            point_eff: (d,), lower: (d,), upper: (d,).
        """
        N = Y_full.shape[0]
        n_pairs = pair_Y1.shape[0]
        n_all = n_pairs + 2  # empty (0) + pairs + full (n_pairs+1)
        boot_effects = np.zeros((B, d))

        for b in range(B):
            np.random.seed(rng_states[b])
            idx = np.random.choice(N, size=N, replace=True)

            # Compute v(u) from resampled data
            v = np.zeros(n_all)
            v[n_pairs + 1] = np.var(Y_full[idx])  # full set
            for s in range(n_pairs):
                y1 = pair_Y1[s, idx]
                y2 = pair_Y2[s, idx]
                v[s + 1] = np.mean(y1 * y2) - np.mean(y1) * np.mean(y2)

            # Shapley accumulation (start from empty set at index 0)
            sh = np.zeros(d)
            for s in range(n_all):
                k = subset_sizes[s]
                w = weights[k]
                v_s = v[s]
                for i in range(d):
                    u_idx = union_lookup[s, i]
                    if u_idx >= 0:
                        sh[i] += w * (v[u_idx] - v_s)

            total_var = v[n_pairs + 1]
            boot_effects[b] = sh / total_var

        # Point estimate (first iteration)
        point_eff = boot_effects[0]
        lower = np.zeros(d)
        upper = np.zeros(d)
        for i in range(d):
            col = np.sort(boot_effects[:, i])
            lower[i] = col[int(np.floor(B * alpha / 2))]
            upper[i] = col[int(np.ceil(B * (1 - alpha / 2))) - 1]
        return point_eff, lower, upper

else:
    # Stub for when Numba is not available
    def _bootstrap_iter_numba(*args, **kwargs):
        raise RuntimeError("Numba is not available — use the pure-Python bootstrap path")


def bootstrap_shapley(data, d, B=1000, alpha=0.05, random_state=None,
                      k_max=None):
    """Bootstrap confidence intervals for the exhaustive method.

    Args:
        data: Dict from collect_shapley_data.
        d: Number of input dimensions.
        B: Number of bootstrap replications.
        alpha: Significance level (e.g., 0.05 for 95% CI).
        random_state: Seed for reproducibility.
        k_max: Optional maximum coalition size (passed to
            ``shapley_from_data``).

    Returns:
        point_eff: Point estimates, shape (d,).
        lower: Lower CI bounds, shape (d,).
        upper: Upper CI bounds, shape (d,).
    """
    if random_state is not None:
        np.random.seed(random_state)

    point_eff, _, _ = shapley_from_data(data, d, k_max=k_max)

    # Determine sample size N
    for typ in data.values():
        if typ[0] == 'full':
            N = len(typ[1])
            break
        elif typ[0] == 'pair':
            N = len(typ[1])
            break

    # ---- compiled bootstrap path (only when k_max is None) ----
    if _NUMBA_AVAILABLE and k_max is None:
        (Y_full, pair_Y1, pair_Y2,
         subset_sizes, union_lookup) = _data_to_arrays(data, d)
        weights = _get_shapley_weights(d)
        rng_states = np.random.randint(0, 2**31, size=B, dtype=np.int32)
        _, lower, upper = _bootstrap_iter_numba(
            Y_full, pair_Y1, pair_Y2, subset_sizes, union_lookup,
            weights, d, B, alpha, rng_states,
        )
    else:
        # ---- pure-Python fallback (always used when k_max is set) ----
        boot_effects = np.zeros((B, d))
        for b in range(B):
            idx = np.random.choice(N, size=N, replace=True)
            boot_data = {}
            for u, typ in data.items():
                if typ[0] == 'full':
                    boot_data[u] = ('full', typ[1][idx])
                else:
                    boot_data[u] = ('pair', typ[1][idx], typ[2][idx])
            eff_b, _, _ = shapley_from_data(boot_data, d, k_max=k_max)
            boot_effects[b] = eff_b
        lower = np.percentile(boot_effects, 100 * alpha / 2, axis=0)
        upper = np.percentile(boot_effects, 100 * (1 - alpha / 2), axis=0)

    return point_eff, lower, upper


def bootstrap_sobol(data, d, B=1000, alpha=0.05, random_state=None):
    """Bootstrap confidence intervals for first-order and total-order
    Sobol indices from the exhaustive method.

    Args:
        data: Dict from ``collect_shapley_data``.
        d: Number of input dimensions.
        B: Number of bootstrap replications.
        alpha: Significance level (e.g., 0.05 for 95% CI).
        random_state: Seed for reproducibility.

    Returns:
        S_point: Point estimates of first-order Sobol (d,).
        S_lower, S_upper: CI bounds for first-order Sobol (d,).
        T_point: Point estimates of total-order Sobol (d,).
        T_lower, T_upper: CI bounds for total-order Sobol (d,).
    """
    if random_state is not None:
        np.random.seed(random_state)

    S_point, T_point = sobol_from_data(data, d)

    # Determine sample size N
    for typ in data.values():
        if typ[0] == 'full':
            N = len(typ[1])
            break
        elif typ[0] == 'pair':
            N = len(typ[1])
            break

    boot_S = np.zeros((B, d))
    boot_T = np.zeros((B, d))
    for b in range(B):
        idx = np.random.choice(N, size=N, replace=True)
        boot_data = {}
        for u, typ in data.items():
            if typ[0] == 'full':
                boot_data[u] = ('full', typ[1][idx])
            else:
                boot_data[u] = ('pair', typ[1][idx], typ[2][idx])
        Sb, Tb = sobol_from_data(boot_data, d)
        boot_S[b] = Sb
        boot_T[b] = Tb

    S_lower = np.percentile(boot_S, 100 * alpha / 2, axis=0)
    S_upper = np.percentile(boot_S, 100 * (1 - alpha / 2), axis=0)
    T_lower = np.percentile(boot_T, 100 * alpha / 2, axis=0)
    T_upper = np.percentile(boot_T, 100 * (1 - alpha / 2), axis=0)

    return S_point, S_lower, S_upper, T_point, T_lower, T_upper


# ------------------------------------------------------------
# Random permutation method (with caching)
# ------------------------------------------------------------
def compute_subset_data(f, joint, u, N, data_cache, predict_batch=None,
                        pbar=None):
    """Compute and store data for a subset u if not already cached.

    Both unconditional (Side A) and conditional (Side B) draws are
    batched — the full N samples are drawn and evaluated in one
    operation, eliminating the per-sample Python loop.

    Args:
        f: Model function.
        joint: Distribution object with sample_joint,
            sample_conditional_batch.
        u: Iterable of variable indices for the subset.
        N: Sample size.
        data_cache: Dict to store computed data.
        predict_batch: Optional batch prediction callable.
        pbar: Optional :class:`_Progress` instance.
    """
    key = frozenset(u)
    if key in data_cache:
        return
    d = joint.d
    if len(u) == 0:
        return

    if len(u) == d:
        # Full set
        X = joint.sample_joint(N)
        if predict_batch is not None:
            Y = np.asarray(predict_batch(X), dtype=float)
        else:
            Y = np.array([f(X[i]) for i in range(N)])
        if pbar is not None:
            pbar.update(N)
        data_cache[key] = ('full', Y)
    else:
        u_list = list(u)

        # --- Side A: unconditional draws (batched) ---
        X = joint.sample_joint(N)
        if predict_batch is not None:
            Y1 = np.asarray(predict_batch(X), dtype=float)
        else:
            Y1 = np.array([f(X[i]) for i in range(N)])
        if pbar is not None:
            pbar.update(N)

        # --- Side B: conditional draws (batched) ---
        X_cond = joint.sample_conditional_batch(u_list, X[:, u_list])
        if predict_batch is not None:
            Y2 = np.asarray(predict_batch(X_cond), dtype=float)
        else:
            Y2 = np.array([f(X_cond[i]) for i in range(N)])
        if pbar is not None:
            pbar.update(N)
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
        predict_batch: Optional batch prediction callable.
        progress: If ``True``, display tqdm progress bars.

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

    # Progress bar: use exhaustive total as the upper bound, since
    # at most 2^d - 1 unique subsets can ever be evaluated.  Caching
    # will cause the bar to complete early — showing how much work
    # was saved by subset reuse.
    max_subsets = 2**d - 1
    total_evals = N + 2 * N * (max_subsets - 1)  # full set costs N
    pbar = _Progress(total_evals, enabled=progress)

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
                    predict_batch=None, progress=False, k_max=None):
    """Compute Shapley effects for correlated inputs via Monte Carlo.

    This is the main entry point for the MC Shapley algorithm.
    It supports two computation methods and optional bootstrap
    confidence intervals.

    Args:
        f: Model function f(x) taking a 1D array and returning a scalar.
        joint: Distribution object with ``sample_joint(n)``,
            ``sample_conditional(u_indices, fixed_x)``, and
            ``sample_conditional_batch(u_indices, fixed_X)`` methods.
        N: Monte Carlo sample size per subset.
        method: Computation method, 'exhaustive' or 'permutation'.
        n_perm: Number of random permutations (permutation method only).
        B: Number of bootstrap replications (0 to skip CIs).
        alpha: Significance level for confidence intervals.
        random_state: Random seed for reproducibility.
        predict_batch: Optional callable that accepts a 2D array (N, d)
            and returns a 1D array of predictions. When provided, batch
            evaluation is used for both unconditional and conditional
            draws, greatly reducing Python-level call overhead.
        progress: If ``True``, display tqdm progress bars (requires
            ``tqdm`` to be installed).
        k_max: Optional maximum coalition size (exhaustive method only).
            When set, only subsets up to size ``k_max`` are evaluated.

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
                                    progress=progress, k_max=k_max)
        effects, sh, total_var = shapley_from_data(data, joint.d, k_max=k_max)
        if B > 0:
            _, lower, upper = bootstrap_shapley(
                data, joint.d, B, alpha, random_state, k_max=k_max,
            )
            return effects, sh, total_var, lower, upper
        return effects, sh, total_var
    elif method == 'qmc_exhaustive':
        data = collect_shapley_data_qmc(f, joint, N, predict_batch=predict_batch,
                                         progress=progress, k_max=k_max,
                                         seed=random_state)
        effects, sh, total_var = shapley_from_data(data, joint.d, k_max=k_max)
        if B > 0:
            _, lower, upper = bootstrap_shapley(
                data, joint.d, B, alpha, random_state, k_max=k_max,
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
        raise ValueError(
            "method must be 'exhaustive', 'qmc_exhaustive', "
            "or 'permutation'")


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
        predict_batch: Optional callable that accepts a 2D array (N, d)
            and returns a 1D array of predictions.

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
                B=0, alpha=0.05, random_state=None, progress=False,
                k_max=None):
        """Compute Shapley effects and Sobol indices.

        Args:
            N: Monte Carlo sample size per subset.
            method: 'exhaustive' or 'permutation'.
            n_perm: Number of permutations (permutation method only).
            B: Bootstrap replications (0 to skip).
            alpha: Significance level for CIs.
            random_state: Random seed.
            progress: If ``True``, show tqdm progress bars.
            k_max: Optional maximum coalition size.  When set, only
                subsets up to size ``k_max`` are evaluated (plus the
                full set).  Exact when the model has no interactions
                above order ``k_max``; approximate otherwise.

        Returns:
            pd.DataFrame with columns:
            - 'variable': Input variable names.
            - 'effect': Normalised Shapley effects.
            - 'shapley_value': Unscaled Shapley values.
            - 'sobol_first': First-order Sobol indices S_i.
            - 'sobol_total': Total-order Sobol indices T_i
              (NaN when k_max < d-1 since {-i} subsets are not
              evaluated).
            - 'total_variance': Estimated total variance.
            - 'lower', 'upper': CI bounds (if B > 0).

            Sobol indices are computed only when ``method='exhaustive'``
            (all subsets are evaluated).  For the permutation method
            they are returned as NaN.
        """
        if random_state is not None:
            np.random.seed(random_state)

        # --- Run data collection and Shapley computation ---
        if method == 'exhaustive':
            data = collect_shapley_data(
                self.f, self.joint, N,
                predict_batch=self.predict_batch,
                progress=progress,
                k_max=k_max,
            )
            effects, sh, total_var = shapley_from_data(
                data, self.d, k_max=k_max,
            )

            # Sobol indices from the same data
            S, T = sobol_from_data(data, self.d)

            if B > 0:
                point, lower, upper = bootstrap_shapley(
                    data, self.d, B, alpha, random_state,
                    k_max=k_max,
                )
                _, S_lower, S_upper, _, T_lower, T_upper = bootstrap_sobol(
                    data, self.d, B, alpha, random_state,
                )
            else:
                S_lower = np.full(self.d, np.nan)
                S_upper = np.full(self.d, np.nan)
                T_lower = np.full(self.d, np.nan)
                T_upper = np.full(self.d, np.nan)
        elif method == 'qmc_exhaustive':
            data = collect_shapley_data_qmc(
                self.f, self.joint, N,
                predict_batch=self.predict_batch,
                progress=progress,
                k_max=k_max,
                seed=random_state,
            )
            effects, sh, total_var = shapley_from_data(
                data, self.d, k_max=k_max,
            )

            S, T = sobol_from_data(data, self.d)

            if B > 0:
                point, lower, upper = bootstrap_shapley(
                    data, self.d, B, alpha, random_state,
                    k_max=k_max,
                )
                _, S_lower, S_upper, _, T_lower, T_upper = bootstrap_sobol(
                    data, self.d, B, alpha, random_state,
                )
            else:
                S_lower = np.full(self.d, np.nan)
                S_upper = np.full(self.d, np.nan)
                T_lower = np.full(self.d, np.nan)
                T_upper = np.full(self.d, np.nan)
        elif method == 'permutation':
            result = shapley_effects_permutation(
                self.f, self.joint, N=N, n_perm=n_perm,
                B=B, alpha=alpha, random_state=random_state,
                predict_batch=self.predict_batch,
                progress=progress,
            )
            if B > 0:
                effects, sh, total_var, lower, upper = result
            else:
                effects, sh, total_var = result
            # Sobol indices not available for permutation (lazy subsets)
            S = np.full(self.d, np.nan)
            T = np.full(self.d, np.nan)
            S_lower = np.full(self.d, np.nan)
            S_upper = np.full(self.d, np.nan)
            T_lower = np.full(self.d, np.nan)
            T_upper = np.full(self.d, np.nan)
            if B > 0:
                lower = lower
                upper = upper
        else:
            raise ValueError(
                "method must be 'exhaustive', 'qmc_exhaustive', "
                "or 'permutation'")

        # --- Build DataFrame ---
        df = pd.DataFrame({
            'variable': [f'X{i+1}' for i in range(self.d)],
            'effect': effects,
            'shapley_value': sh,
            'sobol_first': S,
            'sobol_total': T,
        })
        df['total_variance'] = total_var

        if B > 0:
            df['lower'] = lower
            df['upper'] = upper
        df['sobol_first_lower'] = S_lower
        df['sobol_first_upper'] = S_upper
        df['sobol_total_lower'] = T_lower
        df['sobol_total_upper'] = T_upper

        return df

    def owen(self, groups, var_names=None, N=10000, method='exhaustive',
             B=0, alpha=0.05, random_state=None, progress=False,
             k_max=None):
        """Compute Owen (two-level Shapley) effects with group partition.

        Args:
            groups: dict[str, list[str]] — group name → list of variable
                names. Must cover all variables exactly once.
            var_names: Optional list of variable names in column order.
                When None, defaults to ``['X1', 'X2', ..., 'Xd']``.
            N: Monte Carlo sample size per subset.
            method: 'exhaustive' (all subsets) or 'permutation'.
            B: Bootstrap replications (0 to skip CIs).
            alpha: Significance level for confidence intervals.
            random_state: Random seed.
            progress: Show tqdm progress bars.
            k_max: Maximum coalition size (variable-level truncation).

        Returns:
            dict with keys:
            - ``'group_effects'``: OrderedDict[str, float]
            - ``'individual_effects'``: OrderedDict[str, float]
            - ``'total_variance'``: float
        """
        if var_names is None:
            var_names = [f'X{i+1}' for i in range(self.d)]

        if random_state is not None:
            np.random.seed(random_state)

        if method == 'exhaustive':
            data = collect_shapley_data(
                self.f, self.joint, N,
                predict_batch=self.predict_batch,
                progress=progress, k_max=k_max,
            )
            ge, ie, tv = owen_from_data(
                data, groups, var_names, k_max=k_max)
        elif method == 'permutation':
            raise NotImplementedError(
                "Owen effects with permutation method not yet implemented. "
                "Use method='exhaustive'.")
        else:
            raise ValueError(
                "method must be 'exhaustive' or 'permutation'")

        return {
            'group_effects': ge,
            'individual_effects': ie,
            'total_variance': tv,
        }


# ────────────────────────────────────────────────────────────────
# Standalone Owen effects entry point
# ────────────────────────────────────────────────────────────────

def owen_effects(f, joint, groups, var_names=None,
                 N=10000, method='exhaustive',
                 B=0, alpha=0.05, random_state=None,
                 progress=False, k_max=None):
    """Compute Owen (two-level / grouped) Shapley effects.

    Convenience wrapper around :class:`MCShapley`.  Does not require
    a trained surrogate model.

    Args:
        f: Model function f(x) → scalar.
        joint: Distribution object.
        groups: dict[str, list[str]] — group definitions.
        var_names: Optional list of variable names.
        N: Samples per subset.
        method: 'exhaustive' (all) or 'permutation' (not yet supported).
        B: Bootstrap replications.
        alpha: CI significance level.
        random_state: Random seed.
        progress: Show tqdm bar.
        k_max: Coalition truncation.

    Returns:
        dict with 'group_effects', 'individual_effects', 'total_variance'.
    """
    mc = MCShapley(f, joint)
    return mc.owen(
        groups, var_names=var_names, N=N, method=method,
        B=B, alpha=alpha, random_state=random_state,
        progress=progress, k_max=k_max,
    )