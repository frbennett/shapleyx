import math
import numpy as np
import scipy.special as sp
import pandas as pd
from sklearn.linear_model import Ridge
from .legendre import shift_legendre


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

        # Pre-compute normalization ranges as numpy arrays for fast vectorized transforms
        self._mins = np.array([self.ranges[col][0] for col in self.X_columns])
        self._maxs = np.array([self.ranges[col][1] for col in self.X_columns])
        self._ranges_span = self._maxs - self._mins

        # Build a column name → index lookup for fast access
        self._col_index = {col: i for i, col in enumerate(self.X_columns)}

        # Normalize and build design matrix
        X_norm = (X.values - self._mins) / self._ranges_span
        fitX = self._build_design_matrix(X_norm)

        ridgereg = Ridge()
        ridgereg.fit(fitX, y)
        self.ridgereg = ridgereg

    def _build_design_matrix(self, X_norm):
        """Build the Legendre basis design matrix from normalized inputs.

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

    def predict(self, X):
        """Evaluate the surrogate model on new input data.

        Args:
            X: 2D array-like (n_samples, n_features) or scalar-like for a single point.

        Returns:
            1D array of predictions if 2D input, or scalar if 1D input.
        """
        # Convert to numpy, ensuring column order matches self.X_columns
        if isinstance(X, pd.DataFrame):
            X_arr = X[self.X_columns].values.astype(float)
        else:
            X_arr = np.asarray(X, dtype=float)

        # Handle 1D input (single sample)
        single_input = X_arr.ndim == 1
        if single_input:
            X_arr = X_arr.reshape(1, -1)

        # Normalize to unit hypercube
        X_norm = (X_arr - self._mins) / self._ranges_span

        # Build design matrix using pre-parsed labels
        design = self._build_design_matrix(X_norm)

        result = self.ridgereg.predict(design)

        if single_input:
            return float(result[0])
        return result
