import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from .legendre import shift_legendre

# -----------------------------------------------------------------------
# Try to import Numba for optional JIT compilation of the predict hot-path.
# If Numba is unavailable or incompatible, fall back to pure NumPy.
# -----------------------------------------------------------------------
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


# -----------------------------------------------------------------------
# Numba-compatible shifted Legendre polynomial (recurrence-based)
# -----------------------------------------------------------------------
if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _eval_legendre_numba(n, x):
        """Evaluate the normalised shifted Legendre polynomial of degree n.

        Uses the standard Legendre recurrence applied to t = 2x - 1.
        The normalisation factor sqrt(2n + 1) is applied at the end.

        Args:
            n: Polynomial degree (non-negative integer).
            x: 1D float64 array of evaluation points in [0, 1].

        Returns:
            1D float64 array of polynomial values.
        """
        if n == 0:
            return np.ones_like(x)
        if n == 1:
            return np.sqrt(3.0) * (2.0 * x - 1.0)

        p0 = np.ones_like(x)          # P̂_0(x) = 1
        p1 = 2.0 * x - 1.0            # P̂_1(x) = 2x - 1

        for k in range(1, n):
            # (k+1) P̂_{k+1}(x) = (2k+1)(2x-1) P̂_k(x) - k P̂_{k-1}(x)
            t = 2.0 * x - 1.0
            p2 = ((2.0 * k + 1.0) * t * p1 - k * p0) / (k + 1.0)
            p0 = p1
            p1 = p2

        return np.sqrt(2.0 * n + 1.0) * p1


    @njit(cache=True)
    def _build_design_numba(X_norm, term_cols, term_degs, term_n_factors):
        """Build the Legendre design matrix (Numba-compiled).

        Args:
            X_norm: float64 array (n_rows, d) — inputs normalised to [0, 1].
            term_cols: int32 array (n_terms, max_factors) — column index per factor.
            term_degs: int32 array (n_terms, max_factors) — degree per factor.
            term_n_factors: int32 array (n_terms,) — number of factors per term.

        Returns:
            float64 array (n_rows, n_terms) — design matrix.
        """
        n_rows = X_norm.shape[0]
        n_terms = term_cols.shape[0]
        design = np.ones((n_rows, n_terms))

        for i in range(n_terms):
            nf = term_n_factors[i]
            for f in range(nf):
                col = term_cols[i, f]
                deg = term_degs[i, f]
                design[:, i] *= _eval_legendre_numba(deg, X_norm[:, col])

        return design


    @njit(cache=True)
    def _predict_numba(X_arr, mins, maxs, term_cols, term_degs,
                       term_n_factors, coef, intercept):
        """Compiled surrogate prediction.

        Args:
            X_arr: float64 array (n_rows, d) — raw input data.
            mins: float64 array (d,) — lower bounds of each feature.
            maxs: float64 array (d,) — upper bounds of each feature.
            term_cols, term_degs, term_n_factors: see _build_design_numba.
            coef: float64 array (n_terms,) — ridge coefficients.
            intercept: float64 scalar.

        Returns:
            float64 array (n_rows,) — predictions.
        """
        span = maxs - mins
        X_norm = (X_arr - mins) / span
        design = _build_design_numba(X_norm, term_cols, term_degs,
                                     term_n_factors)
        return design @ coef + intercept

else:
    # Numba not available — the numba functions are never called,
    # but we define stubs so the module can still be imported.
    def _eval_legendre_numba(n, x):        # pragma: no cover
        raise RuntimeError("Numba is not available")

    def _build_design_numba(*args):        # pragma: no cover
        raise RuntimeError("Numba is not available")

    def _predict_numba(*args):             # pragma: no cover
        raise RuntimeError("Numba is not available")


class surrogate():

    def __init__(self, non_zero_coefficients, ranges):
        self.non_zero_coefficients = non_zero_coefficients
        self.ranges = ranges

        # Pre-parse coefficient labels once to avoid repeated string splitting
        # Each label like "x1_3*x2_5" becomes [(col, deg), (col, deg), ...]
        self._parsed_labels = []
        for label in self.non_zero_coefficients['labels']:
            terms = []
            for term in label.split('*'):
                col, deg = term.split('_')
                terms.append((col, int(deg)))
            self._parsed_labels.append(terms)

    def fit(self, X, y):
        self.X_columns = X.columns

        # Pre-compute normalization ranges as numpy arrays
        self._mins = np.array([self.ranges[col][0]
                               for col in self.X_columns], dtype=np.float64)
        self._maxs = np.array([self.ranges[col][1]
                               for col in self.X_columns], dtype=np.float64)
        self._ranges_span = self._maxs - self._mins

        # Build a column name → index lookup
        self._col_index = {col: i for i, col in enumerate(self.X_columns)}

        # Normalize and build design matrix
        X_norm = (X.values - self._mins) / self._ranges_span
        fitX = self._build_design_matrix(X_norm)

        ridgereg = Ridge()
        ridgereg.fit(fitX, y)
        # Store coefficients directly — avoids sklearn.predict() overhead
        # in the hot path (called millions of times by MC Shapley)
        self._coef = ridgereg.coef_.ravel().astype(np.float64)
        self._intercept = float(ridgereg.intercept_)

        # ---- Prepare Numba-compatible data structures for compiled path ----
        if _NUMBA_AVAILABLE:
            self._prepare_numba_data()

    def _prepare_numba_data(self):
        """Convert the parsed labels into integer arrays for Numba.

        Each basis term is a product of Legendre polynomials.
        We flatten this into two padded arrays:
        - term_cols[i, f] : column index of the f-th factor in term i
        - term_degs[i, f] : degree of that factor
        - term_n_factors[i]: number of factors in term i

        Unused slots are set to -1.
        """
        n_terms = len(self._parsed_labels)
        max_factors = max(len(t) for t in self._parsed_labels)

        cols = np.full((n_terms, max_factors), -1, dtype=np.int32)
        degs = np.full((n_terms, max_factors), -1, dtype=np.int32)
        n_factors = np.zeros(n_terms, dtype=np.int32)

        for i, terms in enumerate(self._parsed_labels):
            n_factors[i] = len(terms)
            for f, (col_name, degree) in enumerate(terms):
                cols[i, f] = self._col_index[col_name]
                degs[i, f] = degree

        self._term_cols = cols
        self._term_degs = degs
        self._term_n_factors = n_factors

    def _build_design_matrix(self, X_norm):
        """Build the Legendre basis design matrix from normalized inputs.

        Uses the original SciPy-based shift_legendre.  Kept as a fallback
        and for use during fitting.

        X_norm: 2D numpy array (n_samples, n_features) already in [0,1].
        Returns: 2D numpy array (n_samples, n_terms).
        """
        n_rows = X_norm.shape[0]
        n_cols = len(self._parsed_labels)
        design = np.ones((n_rows, n_cols))

        for i, terms in enumerate(self._parsed_labels):
            for col_name, degree in terms:
                col_idx = self._col_index[col_name]
                design[:, i] *= shift_legendre(degree, X_norm[:, col_idx])

        return design

    def predict(self, X, use_numba=True):
        """Evaluate the surrogate model on new input data.

        Args:
            X: 2D array-like (n_samples, n_features) or 1D for a single point.
            use_numba: If ``True`` (default) and Numba is available, use the
                compiled prediction path.  Set to ``False`` to force the
                original pure-Python path.

        Returns:
            1D array of predictions if 2D input, or scalar if 1D input.
        """
        # Convert to numpy, ensuring column order matches self.X_columns
        if isinstance(X, pd.DataFrame):
            X_arr = X[self.X_columns].values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        # Handle 1D input (single sample)
        single_input = X_arr.ndim == 1
        if single_input:
            X_arr = X_arr.reshape(1, -1)

        # Choose the prediction path based on batch size.
        # - Compiled path: faster per-call for small batches (no Python overhead,
        #   no SciPy dispatch).  Ideal for MC Shapley where N is typically
        #   500–3000.
        # - Original NumPy path: column-vectorised Legendre evaluation wins
        #   for large batches (N ≫ 1000) where SciPy's C-level loops dominate.
        n_rows = X_arr.shape[0]
        if use_numba and _NUMBA_AVAILABLE and n_rows <= 1000:
            result = _predict_numba(
                X_arr,
                self._mins,
                self._maxs,
                self._term_cols,
                self._term_degs,
                self._term_n_factors,
                self._coef,
                self._intercept,
            )
        else:
            # Original path (also used when Numba is unavailable or batch is large)
            X_norm = (X_arr - self._mins) / self._ranges_span
            design = self._build_design_matrix(X_norm)
            result = design @ self._coef + self._intercept

        if single_input:
            return float(result[0])
        return result
