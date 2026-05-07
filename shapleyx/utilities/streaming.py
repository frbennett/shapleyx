"""
Streaming basis expansion for RS-HDMR regression.

Provides a lazy (on-the-fly) Legendre basis matrix that never materialises
the full (n_samples × n_features) design matrix.  Paired with streaming
OMP and OMP-CV regressors that operate on the lazy representation.

This avoids the memory bottleneck in :class:`legendre.legendre_expand`
for high-dimensional problems where the basis set can contain tens of
thousands to millions of terms.

Design
------
* ``FeatureRecipes`` — compact integer descriptors of every basis term.
  One term is a product of Legendre polynomials (e.g. :math:`x_1^3 x_2^5`).
* ``LazyBasisMatrix`` — wraps pre-computed *primitive* Legendre terms and
  the feature recipes.  Columns are computed on demand via Numba (if
  available) or vectorised NumPy.
* ``StreamingOMP`` — Orthogonal Matching Pursuit that scans columns
  lazily via ``dot_with_residual`` instead of holding the full matrix.
* ``StreamingOMPCV`` — K-fold cross-validated OMP with retrospective
  selection of the optimal sparsity level.

Memory comparison (d=20, polys=[8,6,4], n=10 000)
-------------------------------------------------
==================== ========== ==========
Component            Dense     Streaming
==================== ========== ==========
Full design matrix   6.4 GB     —
Primitive terms      12.8 MB    12.8 MB
Feature recipes      —           ~1 MB
XX matrix (ARD)      51 GB      —
Active submatrix     —           ~8 MB
==================== ========== ==========

The streaming path uses ~20 MB instead of ~57 GB.

"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

try:
    from joblib import Parallel, delayed

    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional Numba acceleration
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


# ============================================================================
# FeatureRecipes
# ============================================================================


class FeatureRecipes:
    """Compact descriptor of every candidate basis function.

    Each feature (basis term) is a product of Legendre polynomials:

        φᵢ(x) = ∏_{f=1}^{k} L_{d_f}(x_{v_f})

    where *k* = ``n_factors[i]``, ``v_f`` is the variable index and
    ``d_f`` is the Legendre degree.  Instead of storing variable/degree
    pairs we store **flat primitive indices** into the pre-computed
    primitive-terms array so that the hot inner loops are simple array
    lookups.

    Parameters
    ----------
    feature_names : list of str
        Human-readable names such as ``"x0_3*x1_2"``.
    prim_indices : ndarray of int32, shape (n_features, max_factors)
        Row *i*, column *f* is the index into the primitives array for
        factor *f* of feature *i*.  Unused slots are ``-1``.
    n_factors : ndarray of int32, shape (n_features,)
        Number of factors (Legendre terms) in each feature.
    """

    def __init__(
        self,
        feature_names: list[str],
        prim_indices: np.ndarray,
        n_factors: np.ndarray,
    ) -> None:
        self.feature_names = list(feature_names)
        self.n_features = len(feature_names)
        self.prim_indices = np.asarray(prim_indices, dtype=np.int32)
        self.n_factors = np.asarray(n_factors, dtype=np.int32)
        self.max_factors = self.prim_indices.shape[1]

    def __repr__(self) -> str:
        return (
            f"FeatureRecipes(n_features={self.n_features}, "
            f"max_factors={self.max_factors})"
        )


# ============================================================================
# Numba-compiled helpers  (also NumPy fallbacks when Numba is absent)
# ============================================================================

if _NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True, nogil=True)
    def _correlations_fused(
        primitives: np.ndarray,
        prim_indices: np.ndarray,
        n_factors: np.ndarray,
        residual: np.ndarray,
    ) -> np.ndarray:
        """Compute :math:`X^T r` without materialising *X*.

        This is the streaming-OMP workhorse.  For each feature *i* the
        dot-product of its (virtual) design column with the residual
        vector is accumulated in one pass over the samples.

        The outer feature loop is parallelised via :func:`numba.prange`
        — each feature's dot product is independent and writes to a
        non-overlapping slot in the result array.

        Parameters
        ----------
        primitives : float64 (n_samples, n_prim)
            Pre-computed Legendre polynomial values for every
            ``(variable, degree)`` pair.
        prim_indices : int32 (n_features, max_factors)
            Lookup table mapping feature *i* → primitive indices.
        n_factors : int32 (n_features,)
            How many primitive terms multiply to form feature *i*.
        residual : float64 (n_samples,)
            Current residual vector.

        Returns
        -------
        correlations : float64 (n_features,)
            ``X[:, i] · residual`` for every feature.
        """
        n_features = prim_indices.shape[0]
        n_samples = primitives.shape[0]
        result = np.zeros(n_features, dtype=np.float64)

        for i in prange(n_features):
            nf = n_factors[i]
            s = 0.0
            if nf == 1:
                idx = prim_indices[i, 0]
                for j in range(n_samples):
                    s += primitives[j, idx] * residual[j]
            elif nf == 2:
                idx0 = prim_indices[i, 0]
                idx1 = prim_indices[i, 1]
                for j in range(n_samples):
                    s += (
                        primitives[j, idx0]
                        * primitives[j, idx1]
                        * residual[j]
                    )
            else:
                for j in range(n_samples):
                    prod = residual[j]
                    for f in range(nf):
                        idx = prim_indices[i, f]
                        prod *= primitives[j, idx]
                    s += prod
            result[i] = s

        return result

    @njit(cache=True)
    def _build_subset(
        primitives: np.ndarray,
        prim_indices: np.ndarray,
        n_factors: np.ndarray,
        active_indices: np.ndarray,
    ) -> np.ndarray:
        """Build the design matrix for a subset of features.

        Used during the least-squares step of OMP (active set is small)
        and for constructing the pruned data matrix in post-processing.

        Parameters
        ----------
        primitives : float64 (n_samples, n_prim)
        prim_indices : int32 (n_features, max_factors)
        n_factors : int32 (n_features,)
        active_indices : int32 (n_active,)
            Which features to materialise.

        Returns
        -------
        design : float64 (n_samples, n_active)
        """
        n_samples = primitives.shape[0]
        n_active = active_indices.shape[0]
        design = np.ones((n_samples, n_active), dtype=np.float64)

        for a in range(n_active):
            feat_idx = active_indices[a]
            nf = n_factors[feat_idx]
            for f in range(nf):
                prim_idx = prim_indices[feat_idx, f]
                design[:, a] *= primitives[:, prim_idx]

        return design

    @njit(cache=True)
    def _compute_single_column(
        primitives: np.ndarray,
        prim_indices: np.ndarray,
        n_factors: np.ndarray,
        feat_idx: int,
    ) -> np.ndarray:
        """Compute a single basis column."""
        n_samples = primitives.shape[0]
        col = np.ones(n_samples, dtype=np.float64)
        nf = n_factors[feat_idx]
        for f in range(nf):
            prim_idx = prim_indices[feat_idx, f]
            col *= primitives[:, prim_idx]
        return col

else:
    # ------------------------------------------------------------------
    # Pure NumPy fallbacks — slower but always available.
    # ------------------------------------------------------------------

    def _correlations_fused(
        primitives: np.ndarray,
        prim_indices: np.ndarray,
        n_factors: np.ndarray,
        residual: np.ndarray,
    ) -> np.ndarray:
        """NumPy fallback for the correlation scan."""
        n_features = prim_indices.shape[0]
        result = np.zeros(n_features, dtype=np.float64)

        # 1st-order features: simple dot product (vectorised)
        mask_1 = n_factors == 1
        if mask_1.any():
            idxs = prim_indices[mask_1, 0]
            result[mask_1] = residual @ primitives[:, idxs]

        # 2nd-order features: product of two primitives (vectorised)
        mask_2 = n_factors == 2
        if mask_2.any():
            idxs_0 = prim_indices[mask_2, 0]
            idxs_1 = prim_indices[mask_2, 1]
            products = primitives[:, idxs_0] * primitives[:, idxs_1]
            result[mask_2] = residual @ products

        # 3rd+ order: per-feature loop (these are a minority)
        mask_n = n_factors >= 3
        for i in np.where(mask_n)[0]:
            nf = n_factors[i]
            col = residual.copy()
            for f in range(nf):
                col *= primitives[:, prim_indices[i, f]]
            result[i] = col.sum()

        return result

    def _build_subset(
        primitives: np.ndarray,
        prim_indices: np.ndarray,
        n_factors: np.ndarray,
        active_indices: np.ndarray,
    ) -> np.ndarray:
        """NumPy fallback for building a feature subset."""
        n_samples = primitives.shape[0]
        n_active = len(active_indices)
        design = np.ones((n_samples, n_active), dtype=np.float64)

        for a, feat_idx in enumerate(active_indices):
            nf = n_factors[feat_idx]
            for f in range(nf):
                design[:, a] *= primitives[:, prim_indices[feat_idx, f]]

        return design

    def _compute_single_column(
        primitives: np.ndarray,
        prim_indices: np.ndarray,
        n_factors: np.ndarray,
        feat_idx: int,
    ) -> np.ndarray:
        """NumPy fallback for a single column."""
        return _build_subset(primitives, prim_indices, n_factors, np.array([feat_idx]))[:, 0]


# ============================================================================
# LazyBasisMatrix
# ============================================================================


class LazyBasisMatrix:
    """A Legendre basis matrix that computes columns on demand.

    Instead of storing the full ``(n_samples × n_features)`` design
    matrix, this class keeps the much smaller *primitive terms* array
    (every ``(variable, degree)`` pair evaluated across all samples)
    plus a compact recipe describing how to combine primitives into
    each basis term.

    The interface is deliberately narrow — just enough to support OMP
    regression, CV, pruned-data construction, and label access.

    Parameters
    ----------
    primitives : ndarray of float64, shape (n_samples, n_prim)
        Pre-computed shifted Legendre values.  The column ordering
        follows :meth:`legendre.legendre_expand.do_expand`:

            ``x0_1, x0_2, …, x0_max, x1_1, x1_2, …, x(d-1)_max``

        so ``prim_idx(v, deg) = v * max_poly + (deg - 1)``.

    recipes : FeatureRecipes
        Descriptors for every basis term.
    """

    def __init__(
        self, primitives: np.ndarray, recipes: FeatureRecipes
    ) -> None:
        self.primitives = np.asarray(primitives, dtype=np.float64)
        self.recipes = recipes

    # -- Properties ----------------------------------------------------------

    @property
    def columns(self) -> list[str]:
        """Column labels (matching dense ``X_T_L.columns``)."""
        return self.recipes.feature_names

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_samples, n_features)`` — the *virtual* shape."""
        return (self.primitives.shape[0], self.recipes.n_features)

    @property
    def n_features(self) -> int:
        return self.recipes.n_features

    @property
    def n_samples(self) -> int:
        return self.primitives.shape[0]

    # -- Core lazy operations ------------------------------------------------

    def dot_with_residual(self, residual: np.ndarray) -> np.ndarray:
        """Compute ``Xᵀ r`` — the correlation of every feature with *r*.

        This is the hot path in OMP.  For 80K features and 10K samples
        it performs ~2.4 GFLOP and completes in ~0.3 s (Numba).

        Parameters
        ----------
        residual : ndarray of float64, shape (n_samples,)
            Current residual vector.

        Returns
        -------
        correlations : ndarray of float64, shape (n_features,)
        """
        residual = np.asarray(residual, dtype=np.float64)
        return _correlations_fused(
            self.primitives,
            self.recipes.prim_indices,
            self.recipes.n_factors,
            residual,
        )

    def active_submatrix(
        self, active_indices: np.ndarray | list[int]
    ) -> np.ndarray:
        """Build the design submatrix for selected features.

        Used by the OMP least-squares step and for pruned-data
        construction.  The active set is typically small (1–200 columns).

        Parameters
        ----------
        active_indices : 1-D array-like of int
            Indices of the features to materialise.

        Returns
        -------
        design : ndarray of float64, shape (n_samples, n_active)
        """
        active_indices = np.asarray(active_indices, dtype=np.int32)
        return _build_subset(
            self.primitives,
            self.recipes.prim_indices,
            self.recipes.n_factors,
            active_indices,
        )

    def column(self, idx: int) -> np.ndarray:
        """Compute a single basis column (for debugging / ad-hoc use)."""
        return _compute_single_column(
            self.primitives,
            self.recipes.prim_indices,
            self.recipes.n_factors,
            idx,
        )

    def column_batch(
        self, indices: list[int]
    ) -> np.ndarray:
        """Compute several basis columns at once."""
        return self.active_submatrix(indices)

    def __repr__(self) -> str:
        return (
            f"LazyBasisMatrix(shape=({self.n_samples}, {self.n_features}), "
            f"primitives=({self.primitives.shape[1]} cols), "
            f"recipes={self.recipes})"
        )


# ============================================================================
# StreamingOMP
# ============================================================================


class StreamingOMP:
    """Orthogonal Matching Pursuit on a lazy basis matrix.

    At each iteration the algorithm scans all features to find the one
    most correlated with the current residual.  The scan uses
    :meth:`LazyBasisMatrix.dot_with_residual` so the full design matrix
    is never materialised.

    The active-set least-squares solve uses ``numpy.linalg.lstsq`` on
    the (small) active submatrix.

    Parameters
    ----------
    lazy_basis : LazyBasisMatrix
        The basis to operate on.
    n_nonzero_coefs : int, optional
        Maximum number of features to select.  Default 300.
    fit_intercept : bool, optional
        Whether to fit an intercept term.  When ``True`` (default) the
        algorithm centres *y* for the correlation scan and solves the
        full least-squares problem ``[1 | X_active] w ≈ y`` so that the
        intercept is correctly accounted for.  This matches sklearn's
        :class:`~sklearn.linear_model.OrthogonalMatchingPursuit` default.
    tol : float, optional
        Early-stopping tolerance.  If the residual norm drops below
        ``tol × ‖y_c‖`` the algorithm terminates early.  Default 1e-12.

    Attributes
    ----------
    coef_ : ndarray of float64, shape (n_features,)
        Sparse coefficient vector (zeros for inactive features).
        Does **not** include the intercept term.
    intercept_ : float
        Fitted intercept (0.0 when ``fit_intercept=False``).
    active_ : ndarray of int32
        Indices of the selected features, in selection order.
    n_nonzero_coefs_ : int
        Actual number of features selected.
    n_iter_ : int
        Number of iterations performed.
    """

    def __init__(
        self,
        lazy_basis: LazyBasisMatrix,
        n_nonzero_coefs: int = 300,
        fit_intercept: bool = True,
        tol: float = 1e-12,
        verbose: bool = False,
    ) -> None:
        self.lazy_basis = lazy_basis
        self.n_nonzero_coefs = int(n_nonzero_coefs)
        self.fit_intercept = bool(fit_intercept)
        self.tol = float(tol)
        self.verbose = bool(verbose)

        # Set after fit
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.active_: np.ndarray | None = None
        self.n_nonzero_coefs_: int = 0
        self.n_iter_: int = 0

    def fit(self, y: np.ndarray) -> "StreamingOMP":
        """Run OMP on target vector *y*.

        Parameters
        ----------
        y : array-like of float, shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=np.float64)
        n_features = self.lazy_basis.n_features
        n_samples = len(y)

        # Prepare working residual (centred when fitting intercept)
        if self.fit_intercept:
            y_mean = float(np.mean(y))
            y_c = np.asarray(y - y_mean, dtype=np.float64)
        else:
            y_mean = 0.0
            y_c = y.copy()

        residual_work = y_c.copy()
        active: list[int] = []
        coef_active: np.ndarray | None = None

        max_iter = min(self.n_nonzero_coefs, n_features)
        y_norm_sq = float(np.dot(y_c, y_c))

        for iteration in range(max_iter):
            # --- correlation scan (the streaming bottleneck) ---
            corr = self.lazy_basis.dot_with_residual(residual_work)

            # Mask already-selected features
            if active:
                mask = np.ones(n_features, dtype=bool)
                mask[active] = False
                corr[~mask] = 0.0

            best = int(np.argmax(np.abs(corr)))
            if np.abs(corr[best]) < 1e-15:
                # No remaining feature has meaningful correlation
                break

            active.append(best)

            # --- least-squares solve on active submatrix ---
            X_raw = self.lazy_basis.active_submatrix(active)

            if self.fit_intercept:
                # Augment with a column of ones (intercept)
                X_aug = np.column_stack(
                    [np.ones(n_samples, dtype=np.float64), X_raw]
                )
                coef_full = np.linalg.lstsq(X_aug, y, rcond=None)[0]
                self.intercept_ = float(coef_full[0])
                coef_active = coef_full[1:]
                # residual_work for next correlation scan:
                #   use centred y to keep the correlation scan
                #   well-conditioned (same as sklearn's approach)
                residual_work = y_c - X_raw @ coef_active
                y_pred = X_aug @ coef_full
            else:
                coef_active = np.linalg.lstsq(X_raw, y, rcond=None)[0]
                self.intercept_ = 0.0
                residual_work = y - X_raw @ coef_active
                y_pred = X_raw @ coef_active

            # --- early stopping (against original y) ---
            rss = float(np.dot(y - y_pred, y - y_pred))
            if rss < self.tol * y_norm_sq:
                break

            if self.verbose:
                n_active = len(active)
                r2 = float(1.0 - rss / float(np.dot(y - y.mean(), y - y.mean())))
                print(
                    f"  OMP iter {iteration + 1:<4d}/{max_iter}"
                    f"  active={n_active:<4d}" 
                    f"  R²={r2:.6f}"
                )

        # Pack results
        n_active = len(active)
        self.active_ = np.array(active, dtype=np.int32)
        self.coef_ = np.zeros(n_features, dtype=np.float64)
        if n_active > 0:
            self.coef_[active] = coef_active
        self.n_nonzero_coefs_ = n_active
        self.n_iter_ = iteration + 1

        return self

    def predict(self, lazy_basis: LazyBasisMatrix) -> np.ndarray:
        """Predict using the fitted coefficients.

        Parameters
        ----------
        lazy_basis : LazyBasisMatrix
            Basis for the prediction points (can be the same object
            used for training, or a different one for test data).

        Returns
        -------
        y_pred : ndarray of float64, shape (n_samples,)
        """
        if self.coef_ is None or self.active_ is None:
            raise RuntimeError("StreamingOMP has not been fitted.")
        if len(self.active_) == 0:
            return np.full(
                lazy_basis.n_samples, self.intercept_, dtype=np.float64
            )

        X_active = lazy_basis.active_submatrix(self.active_)
        return X_active @ self.coef_[self.active_] + self.intercept_

    def __repr__(self) -> str:
        fitted = self.coef_ is not None
        nz = self.n_nonzero_coefs_ if fitted else "?"
        return (
            f"StreamingOMP(max_nz={self.n_nonzero_coefs}, "
            f"n_selected={nz}, fitted={fitted})"
        )


# ============================================================================
# StreamingOMPCV
# ============================================================================


# ============================================================================
# Module-level fold evaluator (must be picklable for joblib)
# ============================================================================


def _evaluate_fold_path(
    train_prim: np.ndarray,
    val_prim: np.ndarray,
    prim_indices: np.ndarray,
    n_factors: np.ndarray,
    feature_names: list[str],
    train_y: np.ndarray,
    val_y: np.ndarray,
    max_iter: int,
    fit_intercept: bool,
    scoring: str,
) -> np.ndarray:
    """Evaluate the full OMP sparsity path for a single CV fold.

    Returns an array of scores, one per sparsity level (1 … max_iter).
    Called by :class:`StreamingOMPCV` via :mod:`joblib` for parallel
    fold evaluation.

    Parameters
    ----------
    train_prim : ndarray of float64 (n_train, n_prim)
    val_prim : ndarray of float64 (n_val, n_prim)
    prim_indices : ndarray of int32 (n_features, max_factors)
    n_factors : ndarray of int32 (n_features,)
    feature_names : list of str
    train_y : ndarray of float64 (n_train,)
    val_y : ndarray of float64 (n_val,)
    max_iter : int
    fit_intercept : bool
    scoring : str — ``'r2'`` or ``'mse'``.

    Returns
    -------
    scores : ndarray of float64, shape (max_iter,)
    """
    # Reconstruct recipes and lazy matrices inside the worker
    recipes = FeatureRecipes(feature_names, prim_indices, n_factors)
    train_lazy = LazyBasisMatrix(train_prim, recipes)
    val_lazy = LazyBasisMatrix(val_prim, recipes)

    n_train = len(train_y)
    scores = np.empty(max_iter, dtype=np.float64)

    for n_nz in range(1, max_iter + 1):
        omp = StreamingOMP(train_lazy, n_nonzero_coefs=n_nz,
                           fit_intercept=fit_intercept)
        omp.fit(train_y)

        if omp.n_nonzero_coefs_ == 0:
            y_pred = np.full(len(val_prim), omp.intercept_,
                             dtype=np.float64)
        else:
            y_pred = omp.predict(val_lazy)

        if scoring == "r2":
            ss_res = np.sum((val_y - y_pred) ** 2)
            ss_tot = np.sum((val_y - np.mean(val_y)) ** 2)
            if ss_tot < 1e-15:
                scores[n_nz - 1] = 0.0
            else:
                scores[n_nz - 1] = 1.0 - ss_res / ss_tot
        elif scoring == "mse":
            scores[n_nz - 1] = -np.mean((val_y - y_pred) ** 2)
        else:
            raise ValueError(f"Unknown scoring: {scoring}")

    return scores


class StreamingOMPCV:
    """Cross-validated Orthogonal Matching Pursuit on a lazy basis.

    Runs the full OMP path (1 … ``max_iter`` non-zero coefficients) and
    evaluates each sparsity level via K-fold cross-validation.  The
    level with the best mean CV score is retrospectively selected and
    the model is refit on the full dataset at that sparsity.

    Folds share the same primitive array via NumPy view slicing —
    zero data copies.

    Parameters
    ----------
    lazy_basis : LazyBasisMatrix
    cv : int, optional
        Number of CV folds.  Default 10.
    max_iter : int, optional
        Maximum number of non-zero coefficients to consider.  Default 300.
    fit_intercept : bool, optional
        Whether to fit an intercept term (default ``True``).  Passed
        through to the inner :class:`StreamingOMP` instances.
    scoring : str, optional
        Scoring metric.  ``'r2'`` (default) or ``'mse'`` (negative MSE).
    random_state : int, optional
        Seed for the KFold split.  Default 42.
    n_jobs : int, optional
        Number of parallel jobs for CV fold evaluation (not yet implemented).
        Default 1.

    Attributes
    ----------
    coef_ : ndarray of float64, shape (n_features,)
    intercept_ : float
    active_ : ndarray of int32
    n_nonzero_coefs_ : int
        Optimal sparsity level.
    best_cv_score_ : float
        Best mean CV score across the path.
    cv_scores_ : list of (n_nonzero, mean_score, std_score)
        The full cross-validation path.
    """

    def __init__(
        self,
        lazy_basis: LazyBasisMatrix,
        cv: int = 10,
        max_iter: int = 300,
        fit_intercept: bool = True,
        scoring: str = "r2",
        random_state: int = 42,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> None:
        self.lazy_basis = lazy_basis
        self.cv = int(cv)
        self.max_iter = int(max_iter)
        self.fit_intercept = bool(fit_intercept)
        self.scoring = str(scoring)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.verbose = bool(verbose)

        # Set after fit
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.active_: np.ndarray | None = None
        self.n_nonzero_coefs_: int = 0
        self.best_cv_score_: float | None = None
        self.cv_scores_: list[tuple[int, float, float]] = []

    def fit(self, y: np.ndarray) -> "StreamingOMPCV":
        """Run the full OMP-CV path and select the best sparsity level.

        When ``n_jobs > 1`` and :mod:`joblib` is installed, CV folds
        are evaluated in parallel.  Each fold computes the full sparsity
        path independently; scores are then aggregated across folds.

        Parameters
        ----------
        y : array-like of float, shape (n_samples,)

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=np.float64)
        n = len(y)

        kf = KFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        fold_splits = list(kf.split(np.arange(n)))

        recipes = self.lazy_basis.recipes

        # ------------------------------------------------------------------
        # Parallel path
        # ------------------------------------------------------------------
        if self.n_jobs > 1:
            if not _JOBLIB_AVAILABLE:
                raise ImportError(
                    "joblib is required for n_jobs > 1. "
                    "Install it with: pip install joblib"
                )

            # Dispatch one job per fold.  Each fold computes the full
            # sparsity path (1…max_iter) independently.  On Linux the
            # primitive arrays are shared read-only across forked
            # workers (zero-copy).
            jl_verbosity = 10 if self.verbose else 0
            fold_results: list[np.ndarray] = Parallel(
                n_jobs=self.n_jobs, verbose=jl_verbosity
            )(
                delayed(_evaluate_fold_path)(
                    train_prim=self.lazy_basis.primitives[train_idx],
                    val_prim=self.lazy_basis.primitives[val_idx],
                    prim_indices=recipes.prim_indices,
                    n_factors=recipes.n_factors,
                    feature_names=recipes.feature_names,
                    train_y=y[train_idx],
                    val_y=y[val_idx],
                    max_iter=self.max_iter,
                    fit_intercept=self.fit_intercept,
                    scoring=self.scoring,
                )
                for train_idx, val_idx in fold_splits
            )
            # fold_results: list of arrays, each shape (max_iter,)

            # Aggregate: mean + std across folds for each sparsity level
            fold_array = np.column_stack(fold_results)  # (max_iter, n_folds)
            mean_scores = fold_array.mean(axis=1)
            std_scores = fold_array.std(axis=1)

            if self.verbose:
                print("  Fold evaluation complete.  Aggregating CV scores...")

            self.cv_scores_ = []
            best_score = -np.inf
            best_n_nonzero = 0
            for n_nz in range(1, self.max_iter + 1):
                ms = float(mean_scores[n_nz - 1])
                ss = float(std_scores[n_nz - 1])
                self.cv_scores_.append((n_nz, ms, ss))
                if self.verbose:
                    marker = " *" if ms > best_score else ""
                    print(
                        f"  CV nz={n_nz:<4d}/{self.max_iter}"
                        f"  score={ms:.4f} ±{ss:.4f}{marker}"
                    )
                if ms > best_score:
                    best_score = ms
                    best_n_nonzero = n_nz

            self.best_cv_score_ = best_score

        # ------------------------------------------------------------------
        # Sequential path (n_jobs=1)
        # ------------------------------------------------------------------
        else:
            best_score = -np.inf
            best_n_nonzero = 0
            self.cv_scores_ = []

            for n_nz in range(1, self.max_iter + 1):
                fold_scores: list[float] = []

                for train_idx, val_idx in fold_splits:
                    train_prim = self.lazy_basis.primitives[train_idx]
                    val_prim = self.lazy_basis.primitives[val_idx]

                    train_lazy = LazyBasisMatrix(train_prim, recipes)
                    val_lazy = LazyBasisMatrix(val_prim, recipes)

                    omp = StreamingOMP(
                        train_lazy,
                        n_nonzero_coefs=n_nz,
                        fit_intercept=self.fit_intercept,
                    )
                    omp.fit(y[train_idx])

                    if omp.n_nonzero_coefs_ == 0:
                        y_pred = np.full(
                            len(val_idx), omp.intercept_,
                            dtype=np.float64
                        )
                    else:
                        y_pred = omp.predict(val_lazy)

                    if self.scoring == "r2":
                        score = r2_score(y[val_idx], y_pred)
                    elif self.scoring == "mse":
                        score = -np.mean((y[val_idx] - y_pred) ** 2)
                    else:
                        raise ValueError(
                            f"Unknown scoring: {self.scoring}"
                        )

                    fold_scores.append(score)

                mean_score = float(np.mean(fold_scores))
                std_score = float(np.std(fold_scores))
                self.cv_scores_.append((n_nz, mean_score, std_score))

                if self.verbose:
                    marker = " *" if mean_score > best_score else ""
                    print(
                        f"  CV nz={n_nz:<4d}/{self.max_iter}"
                        f"  score={mean_score:.4f} ±{std_score:.4f}{marker}"
                    )

                if mean_score > best_score:
                    best_score = mean_score
                    best_n_nonzero = n_nz

            self.best_cv_score_ = best_score

        # Refit on full data at the optimal sparsity level
        omp = StreamingOMP(
            self.lazy_basis,
            n_nonzero_coefs=best_n_nonzero,
            fit_intercept=self.fit_intercept,
        )
        omp.fit(y)

        self.coef_ = omp.coef_
        self.intercept_ = omp.intercept_
        self.active_ = omp.active_
        self.n_nonzero_coefs_ = best_n_nonzero

        return self

    def predict(self, lazy_basis: LazyBasisMatrix) -> np.ndarray:
        """Predict using the fitted coefficients."""
        if self.coef_ is None or self.active_ is None:
            raise RuntimeError("StreamingOMPCV has not been fitted.")
        if len(self.active_) == 0:
            return np.full(
                lazy_basis.n_samples, self.intercept_, dtype=np.float64
            )
        X_active = lazy_basis.active_submatrix(self.active_)
        return X_active @ self.coef_[self.active_] + self.intercept_

    def __repr__(self) -> str:
        fitted = self.coef_ is not None
        nz = self.n_nonzero_coefs_ if fitted else "?"
        return (
            f"StreamingOMPCV(max_iter={self.max_iter}, cv={self.cv}, "
            f"n_selected={nz}, fitted={fitted})"
        )
