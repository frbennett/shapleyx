import math
import numpy as np
import scipy.special as sp
import pandas as pd 
from itertools import combinations 

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


class legendre_expand():
    """
    Expands the input features using Legendre polynomials and generates polynomial combinations.
    
    This method performs the following steps:
    1. For each column in `self.X_T`, it computes Legendre polynomial expansions up to the order specified by `self.max_1st`.
       The results are stored in `self.X_T_L` as new columns, with column names in the format `<column>_<order>`.
    2. Generates polynomial combinations of the expanded features based on the polynomial orders specified in `self.polys`.
    3. Constructs a matrix of polynomial terms and concatenates them into a final DataFrame, which is stored in `self.X_T_L`.
    
    Attributes:
        self.primitive_variables (list): A list of primitive variable names used in the expansion.
        self.poly_orders (list): A list of polynomial orders corresponding to each primitive variable.
        self.X_T_L (pd.DataFrame): A DataFrame containing the expanded Legendre polynomial terms and their combinations.
    
    Steps:
        1. For each column in `self.X_T`:
            - Compute Legendre polynomial expansions for orders from 1 to `self.max_1st`.
            - Append the primitive variable name and polynomial order to `self.primitive_variables` and `self.poly_orders`.
            - Store the expanded terms in `self.X_T_L` with appropriate column headings.
        2. For each polynomial order in `self.polys`:
            - Generate a basis set of terms for the current polynomial order.
            - Create valid combinations of terms, ensuring that each combination contains unique primitive variables.
            - Compute the polynomial terms by multiplying the corresponding columns in `self.X_T_L`.
            - Store the computed terms in a matrix and concatenate them into the final DataFrame `generated_set`.
        3. Update `self.X_T_L` with the final generated set of polynomial terms.
    
    Notes:
        - The method assumes that `self.X_T`, `self.max_1st`, `self.polys`, and `self.Y` are properly initialized.
        - The `self.shift_legendre` method is used to compute the Legendre polynomial values.
        - The `combinations` function from the `itertools` module is used to generate term combinations.
        - The method prints the number of terms generated for each polynomial order.
    
    Example:
        If `self.X_T` contains columns 'A' and 'B', and `self.max_1st` is 2, the method will compute:
        - Legendre polynomial expansions for 'A_1', 'A_2', 'B_1', and 'B_2'.
        - If `self.polys` is [1, 2], it will generate:
            - First-order terms: 'A_1', 'B_1'.
            - Second-order terms: 'A_1*B_1', 'A_2*B_1', etc.
    """
    
        # Method implementation...
    def __init__(self, X, X_T, max_1st, polys, Y):
        """
        Initializes the legendre_expand class with the given parameters.

        Args:
            X (pd.DataFrame): Original input features.
            X_T (pd.DataFrame): Transformed input features.
            max_1st (int): Maximum order for the first set of Legendre polynomial expansions.
            polys (list): List of polynomial orders for generating combinations.
            Y (pd.Series): Target variable.

        """
        # Initialize attributes
        self.X = X 
        self.X_T = X_T
        self.max_1st = max_1st
        self.polys = polys
        self.Y = Y

        self.primitive_variables = []
        self.poly_orders = []
        self.X_T_L = pd.DataFrame()

    def do_expand(self):
        # Step 1: Compute Legendre polynomial expansions
        for column in self.X_T:
            for n in range(1, self.max_1st + 1):
                self.primitive_variables.append(column)
                self.poly_orders.append(n)
                column_heading = f"{column}_{n}"
                self.X_T_L[column_heading] = self.X_T[column].apply(lambda x: shift_legendre(n, x))

        # Step 2: Generate polynomial combinations
        generated_set = pd.DataFrame()
        for order, max_poly_order in enumerate(self.polys, start=1):
            basis_set = [f"{x}_{j+1}" for x in self.X.columns for j in range(max_poly_order)]
            
            # Generate valid combinations
            combo_list = [
                combo for combo in combinations(basis_set, order)
                if len(set(term.split('_')[0] for term in combo)) == order
            ]
            
            total_combinations = len(combo_list)
            print(f"Number of terms of order {order} is {total_combinations}")

            # Compute polynomial terms
            matrix = np.zeros((len(self.Y), total_combinations))
            term_labels = []
            
            for term_index, combination in enumerate(combo_list):
                term_label = "*".join(combination)
                term_labels.append(term_label)
                
                # Multiply terms in the combination
                product = np.ones(len(self.Y))
                for term in combination:
                    product *= self.X_T_L[term]
                matrix[:, term_index] = product

            # Store results in a DataFrame
            em = pd.DataFrame(matrix, columns=term_labels)
            generated_set = pd.concat([generated_set, em], axis=1)

        # Step 3: Update the final DataFrame
        self.X_T_L = generated_set

    def get_expanded(self):
        return self.X_T_L
    
    def get_primitive_variables(self):
        return self.primitive_variables
    
    def get_poly_orders(self):
        return self.poly_orders 