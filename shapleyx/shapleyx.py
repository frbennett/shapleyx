"""
*******************************************************************************
Global sensitivity analysis using a Sparse Random Sampling - High Dimensional 
Model Representation (HDMR) using the Group Method of Data Handling (GMDH) for 
parameter selection and linear regression for parameter refinement
*******************************************************************************

author: 'Frederick Bennett'

"""
#from typing import Self
from .ARD import RegressionARD
from .resampling import *
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, ARDRegression 
import pandas as pd
import math
import numpy as np
import scipy.special as sp
from scipy import stats 
from itertools import combinations
import matplotlib
import matplotlib.pyplot as plt 
from sklearn import metrics
from scipy.stats import linregress 
# from numba import jit
import time
import json
from scipy.stats import bootstrap
from sklearn.model_selection import cross_validate 
from sklearn.model_selection import cross_val_score

from .pawn import estimate_pawn
from .xsampler import xsampler
from .pyquotegen import quotes
import textwrap
from .utilities import ( 
    legendre,
    transformation,
    regression,  
) 

#from ARD import RegressionARD
#from sklearn.linear_model import ARDRegression 

from collections import Counter 
# import gmdh

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

        This method reads input data from either a CSV file or a pandas DataFrame. 
        The input data is expected to contain a column labeled 'Y' representing the 
        target variable. The remaining columns are treated as input features (X).

        Parameters:
        -----------
        data_file : str or pd.DataFrame
            The file path to the CSV file or a pandas DataFrame containing the data.
            If a string is provided, it is assumed to be the path to a CSV file.
            If a DataFrame is provided, it is used directly.

        Attributes:
        -----------
        self.Y : pd.Series
            A pandas Series containing the target variable 'Y' extracted from the input data.
        self.X : pd.DataFrame
            A pandas DataFrame containing the input features (all columns except 'Y').

        Notes:
        ------
        - If the input is a CSV file, it is read into a DataFrame using `pd.read_csv`.
        - The original DataFrame (`df`) is deleted after extracting `X` and `Y` to save memory.
        - The method assumes the presence of a column named 'Y' in the input data.

        Example:
        --------
        # Initialize the class
        analyzer = rshdmr()

        # Read data from a CSV file
        analyzer.read_data('input_data.csv')

        # Read data from a DataFrame
        import pandas as pd
        data = pd.DataFrame({'X1': [1, 2, 3], 'X2': [4, 5, 6], 'Y': [7, 8, 9]})
        analyzer.read_data(data)
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
        Linearly transforms the input dataset to a unit hypercube.

        This method applies a transformation to the input dataset `self.X` and stores the transformed data
        in `self.X_T`. It also calculates and stores the ranges of the transformed data in `self.ranges`.

        Side Effects:
        - Updates `self.ranges` with the ranges of the transformed data.
        - Updates `self.X_T` with the transformed dataset.
        """
        transformed_data = transformation.transformation(self.X)
        transformed_data.do_transform()
        self.ranges = transformed_data.get_ranges()
        self.X_T = transformed_data.get_X_T()
       
            
    def legendre_expand(self):
        """
        Expands the input data using Legendre polynomials.

        This method applies Legendre polynomial expansion to the input data `self.X` and the transformed data `self.X_T`.
        It stores the expanded data in `self.X_T_L`, along with the primitive variables and polynomial orders.

        Side Effects:
        - Updates `self.primitive_variables` with the primitive variables from the expansion.
        - Updates `self.poly_orders` with the polynomial orders from the expansion.
        - Updates `self.X_T_L` with the expanded dataset.
        """

        expansion_data = legendre.legendre_expand(self.X, self.X_T, self.max_1st, self.polys, self.Y)
        expansion_data.do_expand() 

        self.primitive_variables = expansion_data.get_primitive_variables() 
        self.poly_orders = expansion_data.get_poly_orders()
        self.X_T_L = expansion_data.get_expanded()  


    def run_regression(self):
        regression_data = regression.regression(self.X_T_L, self.Y, self.method, self.n_iter, self.verbose, self.cv_tol, self.starting_iter)
        self.coef_, self.y_pred = regression_data.run_regression() 


    def stats(self):
        model_coefficients = self.coef_
        sum_of_coeffs_squared = np.sum(model_coefficients**2)
        data_variance = (np.std(self.Y))**2 
        var_ratio = sum_of_coeffs_squared/data_variance
        print("variance of data        : {data_variance:0.3f}".format(data_variance=data_variance))
        print("sum of coefficients^2   : {sum_of_coeffs_squared:0.3f}".format(sum_of_coeffs_squared=sum_of_coeffs_squared))
        print("variance ratio          : {var_ratio:0.3f}".format(var_ratio=var_ratio))
        
        print("===============================")
        y_pred = self.y_pred 
        mse = metrics.mean_squared_error(y_pred,self.Y)
        mae = metrics.mean_absolute_error(y_pred,self.Y)
        evs = metrics.explained_variance_score(y_pred,self.Y)
        self.evs = evs
        slope, intercept, r_value, p_value, std_err = linregress(self.Y, y_pred)
        print("mae error on test set   : {mae:0.3f}".format(mae=mae))
        print("mse error on test set   : {mse:0.3f}".format(mse=mse))
        print("explained variance score: {evs:0.3f}".format(evs=evs))
        print("===============================")
        print("slope     : ", slope)
        print("r value   : ", r_value)
        print("r^2       : ", r_value*r_value)
        print("p value   : ", p_value)
        print("std error : ", std_err)

    def plot_hdmr(self):
        plt.scatter(self.Y,self.y_pred)
        plt.ylabel('Predicted')
        plt.xlabel('Experimental')
        plt.show()

    def get_derived_labels(self, labels):
        derived_label_list = []
        for label in labels :
            if '*' in label :
                label_list = []
                sp1 = label.split('*')
                for i in sp1:
                    label_list.append(i.split('_')[0])
                derived_label = '_'.join(label_list)
                

            if '*' not in label :
                sp1 = label.split('_')  
                derived_label = sp1[0]
            derived_label_list.append(derived_label)
        return derived_label_list  
 


    def eval_sobol_indices(self):
        # Create a DataFrame for coefficients with labels and coefficients
        coefficients = pd.DataFrame({
            'labels': self.X_T_L.columns,
            'coeff': self.coef_
        })
        
        # Filter out non-zero coefficients and reset the index
        non_zero_coefficients = coefficients[coefficients['coeff'] != 0].copy()
        non_zero_coefficients.reset_index(drop=True, inplace=True)
        
        # Add derived labels to the DataFrame
        non_zero_coefficients['derived_labels'] = self.get_derived_labels(non_zero_coefficients['labels'])
        
        # Calculate the index (squared coefficients)
        non_zero_coefficients['index'] = non_zero_coefficients['coeff'] ** 2.0
        
        # Store the non-zero coefficients in the instance
        self.non_zero_coefficients = non_zero_coefficients
        
        # Group by derived labels and sum the indices
        self.results = non_zero_coefficients.groupby('derived_labels', as_index=False).sum()
        
        # Calculate the total modelled variance
        modelled_variance = self.results['index'].sum()
        
        # Normalize the indices by the modelled variance and multiply by explained variance (evs)
        self.results['index'] = (self.results['index'] / modelled_variance) * self.evs
        
        # Store a copy of non_zero_coefficients in the instance (for debugging or further use)
        self.non_zero_coefficients.copy = non_zero_coefficients.copy()
    
        

        
    def get_shapley(self):
        """
        Calculate and store SHAP (SHapley Additive exPlanations) values for each feature in the dataset.

        This method computes the contribution of each feature to the model's predictions using a SHAP-based approach.
        The SHAP values are calculated by iterating over the results of the model and distributing the contribution
        of each feature based on its presence in derived labels. The resulting SHAP values are stored in a DataFrame
        and scaled to represent relative importance.

        Attributes:
            self.shap (pd.DataFrame): A DataFrame containing the following columns:
                - 'label': The name of the feature.
                - 'effect': The raw SHAP value for the feature.
                - 'scaled effect': The SHAP value scaled by the sum of all SHAP values, representing relative importance.

        Steps:
            1. Initialize an empty DataFrame to store SHAP values.
            2. Iterate over each feature in the dataset (self.X.columns).
            3. For each feature, calculate its SHAP value by iterating over the model results (self.results).
            - If the feature is present in the derived labels of a result, its contribution is added proportionally.
            4. Store the feature names and their corresponding SHAP values in a list of tuples.
            5. Create a DataFrame from the list of tuples and store it in self.shap.
            6. Scale the SHAP values by dividing each value by the sum of all SHAP values to represent relative importance.

        Example:
            Assuming `self.results` contains model results with derived labels and indices, and `self.X` contains the feature matrix:
            >>> self.get_shapley()
            >>> print(self.shap)
                label  effect  scaled effect
            0   feature1  0.123      0.456
            1   feature2  0.234      0.567
            2   feature3  0.345      0.678

        Notes:
            - The SHAP values are calculated based on the assumption that the contribution of a feature is evenly distributed
            among all features in its derived labels.
            - The scaled effect provides a normalized measure of feature importance, summing to 1 across all features.
        """
        # Initialize an empty DataFrame for SHAP values
        self.shap = pd.DataFrame(columns=['label', 'effect', 'scaled effect'])
        
        # Calculate SHAP values for each feature
        shap_values = []
        for feature in self.X.columns:
            shap = 0
            for _, result in self.results.iterrows():
                derived_labels = result['derived_labels'].split('_')
                if feature in derived_labels:
                    shap += result['index'] / len(derived_labels)
            shap_values.append((feature, shap))
        
        # Create DataFrame from the calculated SHAP values
        self.shap['label'], self.shap['effect'] = zip(*shap_values)
        
        # Scale the SHAP values
        self.shap['scaled effect'] = self.shap['effect'] / self.shap['effect'].sum()

            
    def get_total_index(self):
        # Initialize an empty DataFrame with columns 'label' and 'total'
        self.total = pd.DataFrame(columns=['label', 'total'])
        
        # Use a list comprehension to calculate the total index for each column in self.X
        label_list = []
        total_list = []
        
        for column in self.X.columns:
            # Calculate the total index for the current column
            total = sum(
                row['index'] for _, row in self.results.iterrows()
                if column in row['derived_labels'].split('_')
            )
            label_list.append(column)
            total_list.append(total)
        
        # Assign the lists to the DataFrame
        self.total['label'] = label_list
        self.total['total'] = total_list


    def get_pruned_data(self):
        pruned_data = pd.DataFrame()
        for label in self.non_zero_coefficients['labels'] :
            pruned_data[label] = self.X_T_L[label]
        pruned_data['Y'] = self.Y
        return pruned_data

    def get_pawn(self, S=10) :
        """
    Estimate the PAWN sensitivity indices for the model's features.

    This method calculates the PAWN sensitivity indices, which are used to assess the influence of each input feature 
    on the output of the model. The PAWN method is a variance-based sensitivity analysis technique that provides 
    insights into the relative importance of features.

    Parameters:
    -----------
    S : int, optional
        Number of intervals to divide the range of each input variable. Default is 10.

    Returns:
    --------
    pawn_results : dict or array-like
        The results of the PAWN sensitivity analysis. The structure of the output depends on the implementation 
        of the `estimate_pawn` function. Typically, it includes sensitivity indices for each feature.

    Notes:
    ------
    - The method relies on the `estimate_pawn` function to perform the actual sensitivity analysis.
    - The input features (`self.X`) and output values (`self.Y`) are assumed to be preprocessed and available as 
      attributes of the class instance.
    - The number of features is automatically determined from the shape of `self.X`.

    Example:
    --------
    >>> model = MyModel()
    >>> model.X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    >>> model.Y = np.array([10, 20, 30])
    >>> pawn_results = model.get_pawn(num_samples=20)
    >>> print(pawn_results)
    {'feature1': 0.75, 'feature2': 0.25}
        """
        num_features = self.X.shape[1] 
        pawn_results = estimate_pawn(self.X.columns, num_features, self.X.values, self.Y, S=S)
        return pawn_results


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
        self.eval_sobol_indices()
        sobol_indices = self.results.drop(columns=['labels', 'coeff'])

        # Step 6: Calculate Shapley effects
        self.get_shapley()
        shapley_effects = self.shap

        # Step 7: Calculate total index
        self.get_total_index()
        total_index = self.total

        # Step 8: Perform resampling if enabled
        if self.resampling:
            number_of_resamples = self.number_of_resamples
            upper = 100-(100-self.CI)/2
            lower = 100-upper

            print_step(f'Running bootstrap resampling {number_of_resamples} samples for {self.CI}% CI')
            pruned_data = self.get_pruned_data()
            resampling_results =  resampling(pruned_data, number_of_resamples = number_of_resamples)

            quantiles = resampling_results.quantile([lower/100,0.5,upper/100], axis=1).T
            quantiles.columns = ['lower', 'mean', 'upper']
            
            # Calculate quantiles for Sobol indices
            sobol_indices['lower'] = quantiles['lower'].values - quantiles['mean'].values + sobol_indices['index'].values
            sobol_indices['upper'] = quantiles['upper'].values - quantiles['mean'].values + sobol_indices['index'].values

            # Calculate quantiles for Shapley effects
            shaps = get_shap(resampling_results, self.X.columns)
            shaps = shaps.div(shaps.sum(axis=0), axis=1)  # Normalize Shapley effects 
#            for i in shaps.columns:
#                shaps[i] = shaps[i]/(shaps[i].sum())

            quantiles = shaps.quantile([0.025,0.5,0.975], axis=1).T
#            quantiles = quantiles.T
            quantiles.columns = ['lower', 'mean', 'upper']

            shapley_effects['lower'] = quantiles['lower'].values - quantiles['mean'].values + shapley_effects['scaled effect'].values
            shapley_effects['upper'] = quantiles['upper'].values - quantiles['mean'].values + shapley_effects['scaled effect'].values
            
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
    
    # put X into a data frame and transform
    def predict(self, X):
        X = pd.DataFrame(X, columns=self.X.columns)
        X_T = pd.DataFrame()
        for column in self.X.columns:
            max = self.ranges[column][1]
            min = self.ranges[column][0]
            X_T[column] = (X[column] - min) / (max-min)

        #build regression model
        prunedX = self.get_pruned_data()
        prunedY = prunedX['Y']
        del prunedX['Y']
        ridgereg =  Ridge()
        ridgereg.fit(prunedX, prunedY)

        # Expand X_T into pruned basis
        labels = self.get_pruned_data().columns
        labels = [i for i in labels if i != 'Y']

        # create predict dataframe
        num_rows = len(X)
        num_columns = len(labels)
        predictX = np.ones((num_rows, num_columns))

        for i in range(num_columns):
            label = labels[i]
            for function in label.split('*'):
                func_args = function.split('_')
                predictX[:,i] *= self.shift_legendre(int(func_args[1]), X_T[func_args[0]])

        return ridgereg.predict(predictX)
    
    def get_pawnx(self, num_unconditioned: int, num_conditioned: int, num_ks_samples: int, alpha: float = 0.05) -> pd.DataFrame:
        """
        Calculate PAWN indices for the RS-HDMR surrogate function.
    
        Args:
            num_unconditioned (int): Number of unconditioned samples.
            num_conditioned (int): Number of conditioned samples.
            num_ks_samples (int): Number of KS samples.
            alpha (float, optional): p-value for KS test. Defaults to 0.05.
    
        Returns:
            pd.DataFrame: DataFrame containing PAWN indices and statistics.
        """
        # Calculate critical value for KS test
        calpha = np.sqrt(-np.log(alpha / 2) / 2)
        dnm = np.sqrt((num_unconditioned + num_conditioned) / (num_unconditioned * num_conditioned))
        critical_value = dnm * calpha
        print(f'For the Kolmogorov–Smirnov test with alpha = {alpha:.3f}, the critical value is {critical_value:.3f}')
    
        # Initialize dictionaries to store results
        results = {}
        results_p = {}
        feature_labels = self.X.columns
        num_features = len(self.ranges)
    
        # Generate reference set
        x_ref = xsampler(num_unconditioned, self.ranges)
        y_ref = self.predict(x_ref)
        print(f"Number of features: {num_features}")
    
        # Iterate over each feature
        for j in range(num_features):
            accept = 'accept'
            ks_stats = []
            ks_p_values = []
            parameter_range = self.ranges[feature_labels[j]]
    
            # Perform KS test for each sample
            for _ in range(num_ks_samples):
                xi = np.random.uniform(parameter_range[0], parameter_range[1])
                xn = xsampler(num_conditioned, self.ranges)
                xn[:, j] = xi
                yn = self.predict(xn)
                ks_result = ks_2samp(y_ref, yn)
                ks_stats.append(ks_result.statistic)
                ks_p_values.append(ks_result.pvalue)
    
            # Calculate summary statistics for KS statistics
            stats_summary = {
                'minimum': np.min(ks_stats),
                'mean': np.mean(ks_stats),
                'median': np.median(ks_stats),
                'maximum': np.max(ks_stats),
                'stdev': np.std(ks_stats),
                'null hyp': 'accept' if np.min(ks_p_values) >= alpha else 'reject'
            }
    
            # Calculate summary statistics for p-values
            p_values_summary = {
                'minimum': np.min(ks_p_values),
                'mean': np.mean(ks_p_values),
                'median': np.median(ks_p_values),
                'maximum': np.max(ks_p_values),
                'stdev': np.std(ks_p_values),
                'null hyp': stats_summary['null hyp']
            }
    
            # Store results for the current feature
            results[feature_labels[j]] = stats_summary
            results_p[feature_labels[j]] = p_values_summary
    
            # Print progress
            print(f"Feature {j + 1}: Median KS Statistic = {stats_summary['median']:.3f}, Std Dev = {stats_summary['stdev']:.3f}")
    
        # Convert results to DataFrames
        results_df = pd.DataFrame(results).T
        results_p_df = pd.DataFrame(results_p).T
    
        return results_df
    
 