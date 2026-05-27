import math
import numpy as np
import scipy.special as sp
import pandas as pd 
from itertools import combinations, product 
from math import comb, factorial

def calculate_hdmr_basis_set_size(dims, poly_degrees):
    total = 0
    for k in range(1, len(poly_degrees) + 1):
        combinations = comb(dims, k)
        degrees = poly_degrees[k - 1] ** k
        sub_total = combinations * degrees
        print(f"Basis functions of {k} order : {sub_total}")
        total += sub_total
    print(f"Total basis functions in basis set : {total}")
    

def calculate_PC_basis_set_size(dims, max_poly):
    total = math.factorial(dims+max_poly)/(math.factorial(dims)*math.factorial(max_poly)) -1
    print(f"Total basis functions in basis set : {int(total)}")
    

def shift_legendre(n, x):
        """
        Computes the shifted Legendre polynomial of degree `n` evaluated at `x` and scales
        by a normalization factor.

        Args:
            n (int): Degree of the shifted Legendre polynomial.
            x (float or array-like): Point(s) at which the polynomial is evaluated.

        Returns:
            float or array-like: Value of the shifted Legendre polynomial at `x`.
        """
        normalization_factor = math.sqrt(2 * n + 1)
        polynomial_value = sp.eval_sh_legendre(n, x)
        return normalization_factor * polynomial_value

def get_hdmr_features(labels, poly_degrees):
    """Helper method to create meaningful feature names"""
    feature_names = []
    order = len(poly_degrees) 
    dims = len(labels)
    
    for current_order in range(1, order + 1):
        poly_degree = poly_degrees[current_order - 1] 
        for vars_ in combinations(range(dims), current_order):
            for degrees in product(range(1, poly_degree + 1), repeat=current_order):
                name_parts = [f'{labels[var]}_{degree}' for var, degree in zip(vars_, degrees)]
                feature_names.append('*'.join(name_parts))
    return feature_names
    
    # Polynomial chaos expansion

def generate_multi_indices_sum(total_degree, current_dim):
    if current_dim == 1:
        return [(total_degree,)]
    indices = []
    for i in range(total_degree + 1):
        for mi in generate_multi_indices_sum(total_degree - i, current_dim - 1):
            indices.append((i,) + mi)
    return indices

def generate_multi_indices(max_degree, dim):
    indices = []
    for total_degree in range(max_degree + 1):
        indices += generate_multi_indices_sum(total_degree, dim)
    return indices


def get_polynomial_chaos_features(labels, max_degree):
    feature_names = []
    dims = len(labels)
    for vars_ in generate_multi_indices(max_degree, dims):
        name_parts = [f'{labels[i]}_{degree}' for i, degree in enumerate(vars_) if degree > 0]
        feature_names.append('*'.join(name_parts))
    feature_names.pop(0)
    return feature_names


class legendre_expand():

    def __init__(self, X_T, polys):
        self.X_T = X_T
        self.polys = polys
        self.max_1st = max(polys)
        self.data_length = len(X_T) 
    

        self.primitive_variables = []
        self.poly_orders = []
        self.X_T_L = pd.DataFrame()
    
    def do_expand(self):
        # Step 1: Compute all of the required Legendre polynomial terms 
        for column in self.X_T:
            for n in range(1, self.max_1st + 1):
                self.primitive_variables.append(column)
                self.poly_orders.append(n)
                column_heading = f"{column}_{n}"
                self.X_T_L[column_heading] = self.X_T[column].apply(lambda x: shift_legendre(n, x))

    def build_basis_set(self):
        dims = len(self.X_T.columns)
        if len(self.polys) == 1:
            calculate_PC_basis_set_size(dims, self.polys[0])
            features = get_polynomial_chaos_features(self.X_T.columns, self.polys[0]) 
        else:
            calculate_hdmr_basis_set_size(dims, self.polys)
            features = get_hdmr_features(self.X_T.columns, self.polys)
        num_features = len(features) 
        print(f"Total number of features in basis set is {num_features}")
        self.do_expand()
        basis_set = np.ones((self.data_length, num_features))
        for index, feature in enumerate(features):
            terms = feature.split('*')
            for term in terms:
                var, degree = term.split('_')
                basis_set[:, index] *= self.X_T_L[f"{var}_{degree}"]

        self.X_T_L = pd.DataFrame(basis_set, columns=features)


    def build_basis_set_streaming(self):
        """Build the Legendre basis in streaming (lazy) format.

        Instead of materialising the full ``(n_samples × n_features)``
        design matrix, this method returns a :class:`LazyBasisMatrix`
        that computes columns on demand from pre-computed primitive
        Legendre terms and recipe descriptors.

        This is the entry point for ``method='omp_stream'`` and
        ``method='omp_cv_stream'``.

        Returns
        -------
        LazyBasisMatrix
            Wraps primitive terms and feature recipes.
        """
        from .streaming import FeatureRecipes, LazyBasisMatrix

        dims = len(self.X_T.columns)
        labels = list(self.X_T.columns)

        # Build a mapping: variable name → column index
        var_to_idx = {name: i for i, name in enumerate(labels)}

        if len(self.polys) == 1:
            calculate_PC_basis_set_size(dims, self.polys[0])
            features = get_polynomial_chaos_features(labels, self.polys[0])
        else:
            calculate_hdmr_basis_set_size(dims, self.polys)
            features = get_hdmr_features(labels, self.polys)

        num_features = len(features)
        print(f"Total number of features in basis set is {num_features}")

        # Compute primitive Legendre terms (same as do_expand)
        self.do_expand()
        primitives = self.X_T_L.values.astype(np.float64)
        # primitives shape: (n_samples, dims × max_1st)
        # Column ordering: var0_1, var0_2, ..., var0_max, var1_1, ...

        # Parse feature name strings into primitive-index recipes
        # A feature like "x0_3*x1_2" → primitive indices [2, 9]
        #   x0_3: var_idx=0, deg=3 → 0*8 + (3-1) = 2
        #   x1_2: var_idx=1, deg=2 → 1*8 + (2-1) = 9

        # max_factors: maximum number of '*' separated terms any feature can have.
        # For PCE (len(polys)==1): up to min(max_order, d) factors.
        # For RS-HDMR: up to len(polys) factors (one per interaction order).
        if len(self.polys) == 1:
            max_factors = min(self.polys[0], dims)
        else:
            max_factors = len(self.polys)
        prim_indices = np.full(
            (num_features, max_factors), -1, dtype=np.int32
        )
        n_factors = np.zeros(num_features, dtype=np.int32)

        for i, feature in enumerate(features):
            terms = feature.split("*")
            n_factors[i] = len(terms)
            for f, term in enumerate(terms):
                # "x0_3" → var_name="x0", deg=3
                # Split at the LAST underscore (variable names may contain
                # underscores, but degrees are just integers at the end)
                parts = term.rsplit("_", 1)
                var_name = parts[0]
                deg = int(parts[1])
                var_idx = var_to_idx[var_name]
                prim_idx = var_idx * self.max_1st + (deg - 1)
                prim_indices[i, f] = prim_idx

        recipes = FeatureRecipes(
            feature_names=features,
            prim_indices=prim_indices,
            n_factors=n_factors,
        )

        # Store feature names for downstream label access
        self._feature_names = features

        return LazyBasisMatrix(primitives, recipes)

    def get_expanded(self):
        return self.X_T_L
    
    def get_primitive_variables(self):
        return self.primitive_variables
    
    def get_poly_orders(self):
        return self.poly_orders

