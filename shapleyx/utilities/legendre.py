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


    def get_expanded(self):
        return self.X_T_L
    
    def get_primitive_variables(self):
        return self.primitive_variables
    
    def get_poly_orders(self):
        return self.poly_orders

