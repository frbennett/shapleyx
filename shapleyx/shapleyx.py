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

    """
    *******************************************************************************
    Global Sensitivity Analysis using Sparse Random Sampling - High Dimensional 
    Model Representation (HDMR) with Group Method of Data Handling (GMDH) for 
    parameter selection and linear regression for parameter refinement.
    *******************************************************************************
    
    This module implements a global sensitivity analysis (GSA) framework using 
    Sparse Random Sampling (SRS) combined with High Dimensional Model Representation 
    (HDMR). The method employs the Group Method of Data Handling (GMDH) for parameter 
    selection and linear regression for parameter refinement. The framework is designed 
    to analyze the sensitivity of model outputs to input parameters, providing insights 
    into the relative importance of each parameter and their interactions.
    
    The module includes functionality for:
    - Reading and preprocessing input data.
    - Transforming data to a unit hypercube.
    - Building basis functions using Legendre polynomials.
    - Running regression analysis using various methods (ARD, OMP, etc.).
    - Evaluating Sobol indices, Shapley effects, and total indices.
    - Performing bootstrap resampling for confidence intervals.
    - Calculating PAWN indices for sensitivity analysis.
    - Predicting model outputs based on input parameters.
    
    Author: Frederick Bennett
    
    Classes:
        rshdmr: Main class for performing global sensitivity analysis using RS-HDMR.
    
    Methods:
        __init__: Initializes the RS-HDMR object with input data and parameters.
        read_data: Reads and preprocesses input data.
        transform_data: Transforms input data to a unit hypercube.
        legendre_expand: Expands the input data using Legendre polynomials.
        run_regression: Runs regression analysis using specified method.
        stats: Computes and prints model performance statistics.
        plot_hdmr: Plots predicted vs. experimental values.
        eval_sobol_indices: Evaluates Sobol indices for sensitivity analysis.
        get_shapley: Computes Shapley effects for sensitivity analysis.
        get_total_index: Computes total sensitivity indices.
        get_pruned_data: Returns pruned dataset based on non-zero coefficients.
        get_pawn: Computes PAWN indices for sensitivity analysis.
        run_all: Runs the entire RS-HDMR analysis pipeline.
        predict: Predicts model outputs based on input parameters.
        get_pawnx: Computes PAWN indices with additional statistical analysis.
    
    Example:
        # Initialize RS-HDMR object
        analyzer = rshdmr(data_file='input_data.csv', polys=[10, 5], method='ard')
        
        # Run the entire analysis pipeline
        sobol_indices, shapley_effects, total_index = analyzer.run_all()
        
        # Predict model outputs for new input data
        predictions = analyzer.predict(new_input_data)
        
        # Compute PAWN indices
        pawn_results = analyzer.get_pawn(S=10)
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
        """
        Reads data from a file or DataFrame and initializes the X and Y attributes.
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
        """
        Transforms the data using into a unit hypercube.

        This method applies a transformation to the data stored in `self.X`, 
        updates the transformed data, and retrieves the ranges and transformed 
        data matrix.

        Attributes:
            self.X (DataFrame or ndarray): The original data to be transformed.
            self.ranges (list): The ranges of the transformed data.
            self.X_T (DataFrame or ndarray): The transformed data matrix.

        Returns:
            None
        """
        transformed_data = transformation.transformation(self.X)
        transformed_data.do_transform()
        self.ranges = transformed_data.get_ranges()
        self.X_T = transformed_data.get_X_T()
       
            
    def legendre_expand(self):
        """
        Perform Legendre expansion on the input data.
        This method uses the `legendre_expand` function from the `legendre` module to
        expand the input data `X` and `X_T` up to the specified maximum order `max_1st`
        using the provided polynomial basis `polys` and target values `Y`.
        The expanded data is then stored in the instance variables:
        - `primitive_variables`: The primitive variables obtained from the expansion.
        - `poly_orders`: The polynomial orders used in the expansion.
        - `X_T_L`: The expanded data.
        Returns:
            None
        """
        expansion_data = legendre.legendre_expand(self.X_T, self.polys)
        expansion_data.build_basis_set() 

        self.primitive_variables = expansion_data.get_primitive_variables() 
        self.poly_orders = expansion_data.get_poly_orders()
        self.X_T_L = expansion_data.get_expanded()  


    def run_regression(self):
        """
        Runs the regression using the specified method and parameters.

        This method initializes a regression instance with the provided
        parameters and runs the regression to obtain the coefficients and
        predicted values.

        Attributes:
            X_T_L (array-like): The transformed feature matrix.
            Y (array-like): The target variable.
            method (str): The regression method to use.
            n_iter (int): The number of iterations for the regression algorithm.
            verbose (bool): If True, enables verbose output.
            cv_tol (float): The tolerance for cross-validation.
            starting_iter (int): The starting iteration for the regression algorithm.

        Returns:
            None: The method updates the instance attributes `coef_` and `y_pred`
            with the regression coefficients and predicted values, respectively.
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
        """
        Calculate and store evaluation statistics for the model.

        This method computes evaluation statistics using the actual target values (self.Y),
        the predicted values (self.y_pred), and the model coefficients (self.coef_). The
        results are stored in the instance variable `self.evs`.

        Returns:
            None
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
        """
        Evaluate all indices and store the results.
        This method performs the following evaluations:
        1. Evaluates indices using the provided data and model coefficients.
        2. Retrieves Sobol indices and stores them in `self.results`.
        3. Retrieves non-zero coefficients and stores them in `self.non_zero_coefficients`.
        4. Evaluates Shapley values for the features and stores them in `self.shap`.
        5. Evaluates the total index for the features and stores it in `self.total`.
        Returns:
            None
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
        """
        Predict the output for the given input data using the surrogate model.

        Parameters:
        X (array-like): Input data for which predictions are to be made.

        Returns:
        array-like: Predicted output for the input data.

        Notes:
        If the predictive-surrogate model does not exist, it will be created and trained using
        the non-zero coefficients and ranges provided during initialization.
        """
        if not hasattr(self, 'surrogate_model'):
            self.surrogate_model = predictor.surrogate(self.non_zero_coefficients, self.ranges)
            self.surrogate_model.fit(self.X, self.Y)
        return self.surrogate_model.predict(X) 

    def get_deltax(self, num_unconditioned: int, delta_samples: int) -> pd.DataFrame:      
        self.delta_instance = pawn.DeltaX(self.X, self.Y, self.ranges, self.non_zero_coefficients)
        delta_indices = self.delta_instance.get_deltax(num_unconditioned, delta_samples)
        return delta_indices
    
    def get_pawnx(self, num_unconditioned: int, num_conditioned: int, num_ks_samples: int, alpha: float = 0.05) -> pd.DataFrame:
        """
        Calculate the PAWN sensitivity indices for the RS-HDMR surrogate model.

        Parameters:
        -----------
        num_unconditioned : int
            The number of unconditioned samples to be used in the PAWN analysis.
        num_conditioned : int
            The number of conditioned samples to be used in the PAWN analysis.
        num_ks_samples : int
            The number of samples to be used in the Kolmogorov-Smirnov test.
        alpha : float, optional
            The significance level for the Kolmogorov-Smirnov test (default is 0.05).

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the PAWN sensitivity indices.
        """
        pawn_instance = pawn.pawnx(self.X, self.Y, self.ranges, self.non_zero_coefficients)
        pawn_indices = pawn_instance.get_pawnx(num_unconditioned, num_conditioned, num_ks_samples, alpha) 
        return pawn_indices 
    
    def get_pawn(self, S=10) :
        """
        Estimate the PAWN sensitivity indices for the features in the dataset.
        Parameters:
        -----------
        S : int, optional
            The number of slides to use for the estimation. Default is 10.
        Returns:
        --------
        pawn_results : dict
            A dictionary containing the PAWN sensitivity indices for each feature.
        """

        num_features = self.X.shape[1] 
        pawn_results = pawn.estimate_pawn(self.X.columns, num_features, self.X.values, self.Y, S=S)
        return pawn_results


    
 