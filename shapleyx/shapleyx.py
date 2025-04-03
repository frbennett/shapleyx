"""
*******************************************************************************
Global sensitivity analysis using a Sparse Random Sampling - High Dimensional 
Model Representation (HDMR) using the Group Method of Data Handling (GMDH) for 
parameter selection and linear regression for parameter refinement
*******************************************************************************

author: 'Frederick Bennett'

"""

import pandas as pd
from scipy import stats 

from .pyquotegen import quotes
import textwrap

from .utilities import ( 
    legendre,
    predictor,
    transformation,
    regression, 
    stats,
    indicies,  
    resampling, 
    pawn,
)


import warnings
warnings.filterwarnings('ignore')

def print_heading(text):
    """
    Prints a heading with the given text, surrounded by a border of equal signs.

    Args:
        text (str): The text to be displayed as the heading.
    """
    border = '=' * 60
    print(f"\n{border}\n{text}\n{border}\n")
    

class rshdmr():
    """Global Sensitivity Analysis using RS-HDMR with GMDH and linear regression.
    **Examples:**

    This class implements a global sensitivity analysis framework combining:

    - Sparse Random Sampling (SRS)
    - High Dimensional Model Representation (HDMR)
    - Group Method of Data Handling (GMDH) for parameter selection
    - Linear regression for parameter refinement

    Args:
        data_file (str or pd.DataFrame): Input data file path or DataFrame.
        polys (list, optional): Polynomial orders for expansion. Defaults to [10, 5].
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        test_size (float, optional): Test set size ratio. Defaults to 0.25.
        limit (float, optional): Coefficient limit. Defaults to 2.0.
        k_best (int, optional): Number of best features. Defaults to 1.
        p_average (int, optional): Parameter for averaging. Defaults to 2.
        n_iter (int, optional): Number of iterations. Defaults to 300.
        verbose (bool, optional): Verbosity flag. Defaults to False.
        method (str, optional): Regression method ('ard', 'omp', etc.). Defaults to 'ard'. 
            can take values

            - 'ard' - Automatic Relevance Determination
            - 'omp' - Orthogonal Matching Pursuit from sklearn.linear_model
            - 'ard_cv' - Automatic Relevance Determination with cross-validation
            - 'omp_cv' - Orthogonal Matching Pursuit with cross-validation from sklearn.linear_model
            - 'ard_sk' - Automatic Relevance Determination from sklearn.linear_model

        starting_iter (int, optional): Starting iteration. Defaults to 5.
        resampling (bool, optional): Enable bootstrap resampling. Defaults to True.
        CI (float, optional): Confidence interval percentage. Defaults to 95.0.
        number_of_resamples (int, optional): Number of bootstrap samples. Defaults to 1000.
        cv_tol (float, optional): Cross-validation tolerance. Defaults to 0.05.

    Attributes:
        X (pd.DataFrame): Input features dataframe.
        Y (pd.Series): Target variable series.
        X_T (pd.DataFrame): Transformed features in unit hypercube.
        ranges (list): Ranges of transformed data.
        X_T_L (pd.DataFrame): Expanded features with Legendre polynomials.
        coef_ (np.array): Regression coefficients.
        y_pred (np.array): Predicted values.
        evs (dict): Model evaluation statistics.
        results (pd.DataFrame): Sobol indices results.
        non_zero_coefficients (pd.DataFrame): Non-zero coefficients.
        shap (pd.DataFrame): Shapley effects.
        total (pd.DataFrame): Total sensitivity indices.
        surrogate_model: Trained surrogate model for predictions.
        primitive_variables: Primitive variables from Legendre expansion.
        poly_orders: Polynomial orders used in Legendre expansion.
        delta_instance: Instance of pawn.DeltaX or pawn.hX for delta/h indices calculation.

    **Examples:**

    ```python
    # Initialize analyzer
    analyzer = rshdmr(data_file='input.csv', polys=[10,5], method='ard')
    
    # Run analysis
    sobol, shapley, total = analyzer.run_all()
    
    # Make predictions
    predictions = analyzer.predict(new_data)
    
    # Get sensitivity indices
    pawn_results = analyzer.get_pawn(S=10)
    ```

     **Todo:**

     - Improve memory management for large expansions


    """

    
    # Rest of the code...
    
    
    def __init__(self,data_file, polys = [10, 5],
                 n_jobs = -1,
                 test_size = 0.25,
                 limit = 2.0,
                 k_best = 1,
                 p_average = 2,
                 n_iter = 300,
                 verbose = False,
                 method = 'ard',
                 starting_iter = 5,
                 resampling = True,
                 CI=95.0,
                 number_of_resamples=1000,
                 cv_tol = 0.05):

        self.read_data(data_file)
        self.n_jobs = n_jobs
        self.test_size = test_size 
        self.limit = limit 
        self.k_best = k_best 
        self.p_average =  p_average
        self.polys =  polys
        self.max_1st = max(polys) 
        self.n_iter = n_iter
        self.verbose = verbose
        self.method = method
        self.starting_iter = starting_iter
        self.resampling = resampling
        self.CI = CI
        self.number_of_resamples = number_of_resamples
        self.cv_tol = cv_tol 
        
    def read_data(self, data_file):
        """Reads data from a file or DataFrame.

        Initializes the `self.X` (features) and `self.Y` (target) attributes.

        Args:
            data_file (str or pd.DataFrame): Path to the data file (CSV) or a pandas DataFrame.

        Raises:
            ValueError: If `data_file` is not a string path or a DataFrame.
        """
        if isinstance(data_file, pd.DataFrame):
            print('Found a DataFrame')
            df = data_file
        elif isinstance(data_file, str):
            df = pd.read_csv(data_file)
        else:
            raise ValueError("data_file must be either a pandas DataFrame or a file path (str).")
        
        self.Y = df['Y']
        self.X = df.drop('Y', axis=1)
        
        # Clean up the original DataFrame to save memory
        del df
        
        
    def transform_data(self):
        """Transforms the input data `self.X` into a unit hypercube.

        Updates the following attributes:
            self.ranges (list): The ranges (min, max) of the original data features.
            self.X_T (pd.DataFrame): The transformed data matrix within the unit hypercube.
        """
        transformed_data = transformation.transformation(self.X)
        transformed_data.do_transform()
        self.ranges = transformed_data.get_ranges()
        self.X_T = transformed_data.get_X_T()
       
            
    def legendre_expand(self):
        """Performs Legendre polynomial expansion on the transformed data `self.X_T`.

        Uses the `legendre.legendre_expand` utility.

        Updates the following attributes:
            self.primitive_variables: Primitive variables from the expansion.
            self.poly_orders: Polynomial orders used in the expansion.
            self.X_T_L (pd.DataFrame): The expanded data matrix including Legendre terms.
        """
        expansion_data = legendre.legendre_expand(self.X_T, self.polys)
        expansion_data.build_basis_set() 

        self.primitive_variables = expansion_data.get_primitive_variables() 
        self.poly_orders = expansion_data.get_poly_orders()
        self.X_T_L = expansion_data.get_expanded()  


    def run_regression(self):
        """Runs the regression analysis using the specified method.

        Uses the `regression.regression` utility based on `self.method`.

        Updates the following attributes:
            self.coef_ (np.array): The regression coefficients obtained from the fit.
            self.y_pred (np.array): The predicted values based on the fitted model.
        """
        regression_instance = regression.regression(
            X_T_L=self.X_T_L,
            Y=self.Y,
            method=self.method,
            n_iter=self.n_iter,
            verbose=self.verbose,
            cv_tol=self.cv_tol,
            starting_iter=self.starting_iter
        )
        self.coef_, self.y_pred = regression_instance.run_regression()

    def stats(self):
        """Calculates and stores evaluation statistics for the fitted model.

        Uses the `stats.stats` utility.

        Updates the following attributes:
            self.evs (dict): A dictionary containing evaluation statistics (e.g., R^2, MSE).
        """
        self.evs = stats.stats(self.Y, self.y_pred, self.coef_)

    def plot_hdmr(self):
        """
        Plots the High-Dimensional Model Representation (HDMR) of the model's predictions.

        This method uses the `plot_hdmr` function from the `stats` module to visualize the 
        HDMR of the actual values (`self.Y`) against the predicted values (`self.y_pred`).

        Returns:
            None
        """
        stats.plot_hdmr(self.Y, self.y_pred)
 


    def eval_all_indices(self):
        """Evaluates Sobol indices, Shapley effects, and total sensitivity indices.

        Uses the `indicies.eval_indices` utility.

        Updates the following attributes:
            self.results (pd.DataFrame): DataFrame containing Sobol indices results.
            self.non_zero_coefficients (pd.DataFrame): DataFrame of non-zero coefficients.
            self.shap (pd.DataFrame): DataFrame containing Shapley effects.
            self.total (pd.DataFrame): DataFrame containing total sensitivity indices.
        """
        eval_indicies = indicies.eval_indices(self.X_T_L, self.Y, self.coef_, self.evs) 
        self.results = eval_indicies.get_sobol_indicies()
        self.non_zero_coefficients = eval_indicies.get_non_zero_coefficients() 

        self.shap = eval_indicies.eval_shapley(self.X.columns)
        self.total = eval_indicies.eval_total_index(self.X.columns)


    def get_pruned_data(self):
        """
        Generates a pruned dataset containing only the features with non-zero coefficients.

        This method creates a new DataFrame that includes only the columns from the original
        dataset (`X_T_L`) that correspond to the labels with non-zero coefficients. Additionally,
        it includes the target variable (`Y`).

        Returns:
            pd.DataFrame: A DataFrame containing the pruned data with selected features and the target variable.
        """
        pruned_data = pd.DataFrame()
        for label in self.non_zero_coefficients['labels'] :
            pruned_data[label] = self.X_T_L[label]
        pruned_data['Y'] = self.Y
        return pruned_data

    def run_all(self):
        """
    Execute a complete sequence of steps for RS-HDMR (Random Sampling High-Dimensional Model Representation) analysis.

    This method performs the following steps in sequence:

    1. Transforms the input data to a unit hypercube.
    2. Builds basis functions using Legendre polynomials.
    3. Runs regression analysis to fit the model.
    4. Calculates and displays RS-HDMR model performance statistics.
    5. Evaluates Sobol indices to quantify the contribution of each input variable to the output variance.
    6. Calculates Shapley effects to measure the importance of each input variable.
    7. Computes the total index to assess the overall impact of input variables.
    8. If resampling is enabled, performs bootstrap resampling to estimate confidence intervals for Sobol indices and Shapley effects.
    9. Prints a completion message with a randomly selected quote.

    Returns:
        tuple: A tuple containing three elements:

            - sobol_indices (pd.DataFrame): A DataFrame containing Sobol indices for each input variable.
            - shapley_effects (pd.DataFrame): A DataFrame containing Shapley effects for each input variable.
            - total_index (pd.DataFrame): A DataFrame containing the total index for each input variable.

    Notes:
        - The method assumes that the necessary data and configurations are already set in the class instance.
        - If resampling is enabled (`self.resampling` is True), confidence intervals (CIs) are calculated for Sobol indices and Shapley effects.
        - The method uses helper functions like `transform_data`, `legendre_expand`, `run_regression`, `stats`, `plot_hdmr`, `eval_sobol_indices`, `get_shapley`, and `get_total_index` to perform specific tasks.
        - The completion message includes a randomly selected quote for a touch of inspiration.

    Example:
        >>> sobol_indices, shapley_effects, total_index = instance.run_all()
        >>> print(sobol_indices)
        >>> print(shapley_effects)
        >>> print(total_index)
    """
        # Define a helper function to print headings
        def print_step(step_name):
            print_heading(step_name)

        # Step 1: Transform data to unit hypercube
        print_step('Transforming data to unit hypercube')
        self.transform_data()

        # Step 2: Build basis functions
        print_step('Building basis functions')
        self.legendre_expand()

        # Step 3: Run regression analysis
        print_step('Running regression analysis')
        self.run_regression()

        # Step 4: Calculate RS-HDMR model performance statistics
        print_step('RS-HDMR model performance statistics')
        self.stats() 
        print()
        self.plot_hdmr()
        
        # Step 5: Evaluate Sobol indices
        self.eval_all_indices()
        sobol_indices = self.results.drop(columns=['labels', 'coeff'])

        # Step 6: Calculate Shapley effects
        shapley_effects = self.shap

        # Step 7: Calculate total index 
        total_index = self.total

        # Step 8: Perform resampling if enabled
        if self.resampling:
            print_step(f'Running bootstrap resampling {self.number_of_resamples} samples for {self.CI}% CI') 
            do_resampling = resampling.resampling(self.get_pruned_data(), self.number_of_resamples, self.X.columns)
            do_resampling.do_resampling() 
            sobol_indices = do_resampling.get_sobol_quantiles(sobol_indices, self.CI)
            # Calculate quantiles for Shapley effects
            shapley_effects = do_resampling.get_shap_quantiles(shapley_effects, self.CI) 
            print_step('Completed bootstrap resampling')

        # Step 9: Print completion message with a quote
        quote = quotes.get_quote() 
        message = (
            "                  Completed all analysis\n"
            "                 ------------------------\n\n"
            f"{textwrap.fill(quote, 58)}"
        )
        print_step(message)
               
        return sobol_indices, shapley_effects, total_index

    def predict(self, X):
        """Predicts output for new input data using the trained surrogate model.

        If the surrogate model (`self.surrogate_model`) doesn't exist, it first
        creates and fits one using `predictor.surrogate`.

        Args:
            X (array-like): Input data for which predictions are to be made.
                Should have the same features as the original training data.

        Returns:
            array-like: Predicted output values.
        """
        if not hasattr(self, 'surrogate_model'):
            self.surrogate_model = predictor.surrogate(self.non_zero_coefficients, self.ranges)
            self.surrogate_model.fit(self.X, self.Y)
        return self.surrogate_model.predict(X) 

    def get_deltax(self, num_unconditioned: int, delta_samples: int) -> pd.DataFrame:      
        """
        Calculate delta indices for the given number of unconditioned variables and delta samples.

        This method initializes a DeltaX instance using the provided data and parameters,
        then computes the delta indices based on the specified number of unconditioned variables
        and delta samples.

        Args:
            num_unconditioned (int): The number of unconditioned variables.
            delta_samples (int): The number of delta samples to generate.

        Returns:
            pd.DataFrame: A DataFrame containing the computed delta indices.
        """
        self.delta_instance = pawn.DeltaX(self.X, self.Y, self.ranges, self.non_zero_coefficients)
        delta_indices = self.delta_instance.get_deltax(num_unconditioned, delta_samples)
        return delta_indices
    
    def get_hx(self, num_unconditioned: int, delta_samples: int) -> pd.DataFrame:      
        """
        Calculate delta indices for the given number of unconditioned variables and delta samples.

        This method initializes a DeltaX instance using the provided data and parameters,
        then computes the delta indices based on the specified number of unconditioned variables
        and delta samples.

        Args:
            num_unconditioned (int): The number of unconditioned variables.
            delta_samples (int): The number of delta samples to generate.

        Returns:
            pd.DataFrame: A DataFrame containing the computed delta indices.
        """
        self.delta_instance = pawn.hX(self.X, self.Y, self.ranges, self.non_zero_coefficients)
        delta_indices = self.delta_instance.get_hx(num_unconditioned, delta_samples)
        return delta_indices
    
    
    def get_pawnx(self, num_unconditioned: int, num_conditioned: int, num_ks_samples: int, alpha: float = 0.05) -> pd.DataFrame:
        """Calculates PAWN sensitivity indices using the surrogate model.

        Uses the `pawn.pawnx` utility.

        Args:
            num_unconditioned (int): Number of unconditioned samples for PAWN.
            num_conditioned (int): Number of conditioned samples for PAWN.
            num_ks_samples (int): Number of samples for the Kolmogorov-Smirnov test.
            alpha (float, optional): Significance level for the KS test. Defaults to 0.05.

        Returns:
            pd.DataFrame: DataFrame containing the PAWN sensitivity indices.
        """
        pawn_instance = pawn.pawnx(self.X, self.Y, self.ranges, self.non_zero_coefficients)
        pawn_indices = pawn_instance.get_pawnx(num_unconditioned, num_conditioned, num_ks_samples, alpha) 
        return pawn_indices 
    
    def get_pawn(self, S=10) :
        """Estimates PAWN sensitivity indices directly from data.

        Uses the `pawn.estimate_pawn` utility.

        Args:
            S (int, optional): Number of slices/intervals for the PAWN estimation. Defaults to 10.

        Returns:
            dict: Dictionary containing the PAWN sensitivity indices for each feature.
        """

        num_features = self.X.shape[1] 
        pawn_results = pawn.estimate_pawn(self.X.columns, num_features, self.X.values, self.Y, S=S)
        return pawn_results


    
 