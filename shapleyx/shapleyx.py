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

from scipy.stats import ks_2samp 

#from ARD import RegressionARD
#from sklearn.linear_model import ARDRegression 

from collections import Counter 
# import gmdh

import warnings
warnings.filterwarnings('ignore')

def print_heading(text):
    print()
    print('==========================================================')
    print(text)
    print('==========================================================')
    print() 
    

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
        

        

            
    def shift_legendre(self, n, x):
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

    
        
    def transform_data(self):
        """
        Linearly transforms the input dataset to a unit hypercube.
    
        This method scales each feature in the dataset to the range [0, 1] using min-max scaling.
        It also stores the original min and max values for each feature in the `ranges` attribute.
    
        Attributes:
            self.X_T (pd.DataFrame): Transformed dataset with features scaled to [0, 1].
            self.ranges (dict): Dictionary storing the original min and max values for each feature.
        """
        self.X_T = pd.DataFrame()
        self.ranges = {}
    
        for column in self.X.columns:
            feature_min = self.X[column].min()
            feature_max = self.X[column].max()
    
            # Log the min and max values for debugging or informational purposes
            print(f"{column}: min = {feature_min}, max = {feature_max}")
    
            # Perform min-max scaling to transform the feature to [0, 1]
            self.X_T[column] = (self.X[column] - feature_min) / (feature_max - feature_min)
    
            # Store the original min and max values for potential inverse transformations
            self.ranges[column] = [feature_min, feature_max]
    
        
            
    def legendre_expand(self):
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
    
        self.primitive_variables = []
        self.poly_orders = []
        self.X_T_L = pd.DataFrame()
        for column in self.X_T:
            for n in range (1,self.max_1st+1):
                self.primitive_variables.append(column)
                self.poly_orders.append(n)
                column_heading = column + "_" + str(n)
                self.X_T_L[column_heading] = [self.shift_legendre(n, x) for x in self.X_T[column]]
                
        polys = self.polys
        generated_set = pd.DataFrame() 
        max_poly = max(polys)
        order = 0
        for i in polys:
            order += 1
            max_poly_order = polys[order-1] 
            basis_set = []
            for x in  self.X.columns :
                for j in range(max_poly_order):
                    basis_set.append(x + '_' + str(j+1))
                    
            combo_list = []
            for combo in combinations(basis_set, order):
                primitive_list = []
                for i in combo:
                    primitive_list.append(i.split('_')[0])
                if len(np.unique(primitive_list)) == order:
                    combo_list.append(combo)
                    
            total_combinations = len(combo_list)
            print(' number of terms of order ', str(order), 'is ', str(total_combinations)) 
    
            matrix = np.zeros([len(self.Y),total_combinations])
            derived_labels = []
            term_labels = []
            term_index = 0
            
            for combination in combo_list:
#            for combination in combinations(basis_set, order): 
                term_id = 1
                for term in combination:
                    if term_id == 1:
                        matrix[:,term_index] = self.X_T_L[term] 
                        term_label = term
                        term_id +=1
                    else :
                        matrix[:,term_index] = matrix[:,term_index] * self.X_T_L[term]
                        term_label = term_label + '*' + term
                term_labels.append(term_label)         
                term_index += 1
            em = pd.DataFrame(matrix) 
            em.columns = term_labels 
    
            generated_set = pd.concat([generated_set, em], axis = 1)
        
        self.X_T_L = generated_set

    def run_regression(self):
        start_time = time.perf_counter()
        if self.method == 'ard':
            print('running ARD')
            self.clf = RegressionARD(n_iter=self.n_iter, verbose=self.verbose, cv=False)
            
        if self.method == 'ard_cv':
            print('running ARD')
            self.clf = RegressionARD(n_iter=self.n_iter, verbose=self.verbose, cv_tol=self.cv_tol, cv=True)
            
        elif self.method == 'omp':
            print('running OMP')
            self.clf = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_iter)
        elif self.method == 'ompcv':
            print('running OMP_CV')
            self.clf = OrthogonalMatchingPursuitCV(max_iter=self.n_iter, cv=10) 
        elif self.method == 'ardsk':
            print('running ARD_SK')
            self.clf = ARDRegression(max_iter=self.n_iter, compute_score=True) 
            
        elif self.method == 'ardcv':
            print('running ARD with cross validation')
            # Use OMPCV to get a ballpark figure for n_iter
 #           clf = OrthogonalMatchingPursuitCV(max_iter=self.n_iter, cv=5) 
 #           clf.fit(self.X_T_L,self.Y)
            
            best_score = -100
            best_score_iter = 2
            # set the starting iteration a few steps earlier than the OMPCV estimate
 #           iteration = clf.n_nonzero_coefs_ - 5
            iteration = self.starting_iter 
            converged = False
            
            while not converged:
                print(iteration)
                clf = RegressionARD(n_iter=iteration, verbose=False)
                results = cross_val_score(clf, self.X_T_L,self.Y, cv=5)
                test = np.mean(results)
                if test > best_score :
                    best_score = test
                    best_score_iter = iteration

                if ((iteration -best_score_iter) >= 10) or (iteration == self.n_iter):
                    converged = True
        
                iteration += 1
            print('the best iteration ',     best_score_iter)   
            
            self.clf = RegressionARD(n_iter=best_score_iter, verbose=self.verbose)

        elif self.method == 'ardompcv':
            print('running ARD OMP cross validation')
            clf = OrthogonalMatchingPursuitCV(max_iter=self.n_iter, cv=5) 
            clf.fit(self.X_T_L,self.Y)
            num_iterations = clf.n_nonzero_coefs_
            self.clf = RegressionARD(num_iterations, verbose=self.verbose)
             
#        self.clf = ARDRegression(n_iter=self.n_iter, verbose=True, tol=1.0e-3)
        self.clf.fit(self.X_T_L,self.Y)
        end_time = time.perf_counter()

        if self.method == 'ompcv':
            print('Number of non-zero coefficeints from OMP_CV : ', self.clf.n_nonzero_coefs_)

#        print('number of iterations ', self.clf.n_iter_)
        print(f"Fit Execution Time : {end_time - start_time:0.6f}" ) 
        print('--') 
        print(" ")
        print(" Model complete ")

        print(" ")

    def stats(self):
        model_coefficients = self.clf.coef_
        sum_of_coeffs_squared = np.sum(model_coefficients**2)
        data_variance = (np.std(self.Y))**2 
        var_ratio = sum_of_coeffs_squared/data_variance
        print("variance of data        : {data_variance:0.3f}".format(data_variance=data_variance))
        print("sum of coefficients^2   : {sum_of_coeffs_squared:0.3f}".format(sum_of_coeffs_squared=sum_of_coeffs_squared))
        print("variance ratio          : {var_ratio:0.3f}".format(var_ratio=var_ratio))
        
        print("===============================")
        y_pred = self.clf.predict(self.X_T_L)
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
        y_pred = self.clf.predict(self.X_T_L)
        plt.scatter(self.Y,y_pred)
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
 

    def eval_sobol_indicesxxx(self):
        coefficients = pd.DataFrame()
        coefficients['labels'] =self.X_T_L.columns
        coefficients['coeff'] = self.clf.coef_
    
        non_zero_coefficients = (coefficients[coefficients['coeff'] != 0]).copy() 
        non_zero_coefficients['std_devs'] = np.diag(np.sqrt(self.clf.sigma_) ) 
        non_zero_coefficients.reset_index(drop=True, inplace=True)
        non_zero_coefficients['derived_labels'] =  self.get_derived_labels(non_zero_coefficients['labels']) 

        
        posterior_indicies = pd.DataFrame()
        for i in range(1000) : 
            posterior_indicies['sample_' + str(i+1)] = np.random.normal(non_zero_coefficients['coeff'], non_zero_coefficients['std_devs'])
            
        self.posterior_indicies = posterior_indicies
            
#        non_zero_coefficients['lower'] = posterior_indices.quantile(0.025, axis=1)
#        non_zero_coefficients['upper'] = posterior_indices.quantile(0.975, axis=1)
#        non_zero_coefficients['median'] = posterior_indices.quantile(0.5, axis=1)

        non_zero_coefficients['index'] = (non_zero_coefficients['coeff'])**2.0
#        non_zero_coefficients['index'] = (non_zero_coefficients['coeff']/np.std(self.Y))**2.0
        non_zero_coefficients['lower'] = posterior_indicies.quantile(0.025, axis=1)
        non_zero_coefficients['upper'] = posterior_indicies.quantile(0.975, axis=1)
        non_zero_coefficients['median'] = posterior_indicies.quantile(0.975, axis=1)
        
        self.non_zero_coefficients = non_zero_coefficients
        
        self.results = non_zero_coefficients.groupby(['derived_labels']).sum()
        modelled_variance = self.results['index'].sum() 
        self.results['index'] = self.results['index']/modelled_variance * self.evs
        self.results['median'] = self.results['median']/modelled_variance * self.evs
        self.results['lower'] = self.results['lower']/modelled_variance * self.evs
        self.results['upper'] = self.results['upper']/modelled_variance * self.evs


        self.ttt = non_zero_coefficients.copy() 
        print(self.results) 

    def eval_sobol_indices(self):
        coefficients = pd.DataFrame()
        coefficients['labels'] =self.X_T_L.columns
        coefficients['coeff'] = self.clf.coef_
    
        non_zero_coefficients = (coefficients[coefficients['coeff'] != 0]).copy()  
        non_zero_coefficients.reset_index(drop=True, inplace=True)
        non_zero_coefficients['derived_labels'] =  self.get_derived_labels(non_zero_coefficients['labels']) 

        
            
#        non_zero_coefficients['lower'] = posterior_indices.quantile(0.025, axis=1)
#        non_zero_coefficients['upper'] = posterior_indices.quantile(0.975, axis=1)
#        non_zero_coefficients['median'] = posterior_indices.quantile(0.5, axis=1)

        non_zero_coefficients['index'] = (non_zero_coefficients['coeff'])**2.0
#        non_zero_coefficients['index'] = (non_zero_coefficients['coeff']/np.std(self.Y))**2.0
        
        self.non_zero_coefficients = non_zero_coefficients
        
        self.results = non_zero_coefficients.groupby(['derived_labels']).sum()
        modelled_variance = self.results['index'].sum() 
        self.results['index'] = self.results['index']/modelled_variance * self.evs


        self.ttt = non_zero_coefficients.copy()  
        

        
    def get_shapley(self):
        self.shap = pd.DataFrame()
        label_list = []
        shap_list = []
        for i in self.X.columns:
            shap=0
            for j, k in self.results.iterrows():
                if i in j.split('_'):
                    shap += k['index']/len(j.split('_'))
            label_list.append(i)
            shap_list.append(shap)
        self.shap['label'] = label_list
        self.shap['effect'] = shap_list
        self.shap['scaled effect'] = self.shap['effect']/self.shap['effect'].sum()
            
    def get_total_index(self):
        self.total = pd.DataFrame()
        label_list = []
        total_list = []
        for i in self.X.columns:
            total=0
            for j, k in self.results.iterrows():
                if i in j.split('_'):
                    total += k['index']
            label_list.append(i)
            total_list.append(total)  
        self.total['label'] = label_list
        self.total['total'] = total_list

    def get_pruned_data(self):
        pruned_data = pd.DataFrame()
        for label in self.non_zero_coefficients['labels'] :
            pruned_data[label] = self.X_T_L[label]
        pruned_data['Y'] = self.Y
        return pruned_data

    def get_pawn(self, S=10) :
        num_features = len(self.X.columns)
        pawn_results = estimate_pawn(self.X.columns, num_features, self.X.values, self.Y, S=S)
        return pawn_results


    def run_all(self):
        print_heading('Transforming data to unit hypercube')
        self.transform_data()
    
        print_heading('Building basis functions')
        self.legendre_expand()
    
        print_heading('Running regression analysis')
        self.run_regression() 
    
        print_heading('RS-HDMR model performance statistics')
        self.stats() 
        print()
        self.plot_hdmr() 

        self.eval_sobol_indices()
        sobol_indices = self.results

        del sobol_indices['labels']
        del sobol_indices['coeff']

        self.get_shapley()
        shapley_effects = self.shap
    
        self.get_total_index()
        total_index = self.total

        if self.resampling:
            number_of_resamples = self.number_of_resamples

            upper = 100-(100-self.CI)/2
            lower = 100-upper



            print_heading('Running bootstrap resampling ' + str(number_of_resamples) + ' samples for ' + str(self.CI) + '% CI')
            pruned_data = self.get_pruned_data()
            resampling_results =  resampling(pruned_data, number_of_resamples = number_of_resamples)


            quantiles = resampling_results.quantile([lower/100,0.5,upper/100], axis=1)
            quantiles = quantiles.T
            quantiles.columns = ['lower', 'mean', 'upper']

            sobol_indices['lower'] = quantiles['lower'] - quantiles['mean'] + sobol_indices['index']
            sobol_indices['upper'] = quantiles['upper'] - quantiles['mean'] + sobol_indices['index']

            shaps = get_shap(resampling_results, self.X.columns)
            for i in shaps.columns:
                shaps[i] = shaps[i]/(shaps[i].sum())

            quantiles = shaps.quantile([0.025,0.5,0.975], axis=1)
            quantiles = quantiles.T
            quantiles.columns = ['lower', 'mean', 'upper']

            shapley_effects['lower'] = quantiles['lower'].values - quantiles['mean'].values + shapley_effects['scaled effect'].values
            shapley_effects['upper'] = quantiles['upper'].values - quantiles['mean'].values + shapley_effects['scaled effect'].values
            
            print_heading('Completed bootstrap resampling')

        quote = quotes.get_quote() 
        print_heading('                  Completed all analysis \n' + 
                      '                 ------------------------ \n\n ' + 
                      textwrap.fill(quote, 58))

 #       quote = quotes.get_quote()
 #       print_heading(textwrap.fill(quote, 58))
        
#        print(textwrap.fill(quote, 50))
#        print('')
        
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
    
    def get_pawnx(self, Nu : int, Nc : int, M : int, alpha=0.05):
        """
        Calculate PAWN indices for the RS-HDMR surrogate function

        Args:
            Nu (int): Number of unconditioned samples
            Nc (int): Number of conditioned samples
            M (int): Number of KS samples
            alpha (float, optional): p value for KS test. Defaults to 0.05.

        Returns:
            _type_: Dataframe
        """
        
        calpha = np.sqrt(-np.log(alpha/2)/2)
        Dnm = np.sqrt((Nu+Nc)/(Nu*Nc))
        critical_value = Dnm*calpha
        print(f'For the Kolmogorov–Smirnov test with alpha = {alpha:.3f} the critical value is {critical_value:.3f}')
        
        results = {} 
        resultsp = {}
        labels = self.X.columns
        num_features = len(self.ranges)
        #generate reference set
        x_ref = xsampler(Nu, self.ranges)
        y_ref = self.predict(x_ref)
        print(num_features)
        for j in range(num_features):
            accept = 'accept'
            all_stats = []
            all_p = []
            for i in range(M):
                Xi = np.random.rand()
                Xn = xsampler(Nc, self.ranges)
                Xn[:,j] = Xi
                Yn = self.predict(Xn)
                ks = ks_2samp(y_ref, Yn)
                all_stats.append(ks.statistic)
                all_p.append(ks.pvalue)

            min = np.min(all_stats)
            mean = np.mean(all_stats)
            median = np.median(all_stats)
            max = np.max(all_stats)
            std = np.std(all_stats)

            minp = np.min(all_p)
            meanp = np.mean(all_p)
            medianp = np.median(all_p)
            maxp = np.max(all_p)
            stdp = np.std(all_p)
            if minp < alpha :
                accept = 'reject'

            results[labels[j]] = [min, mean, median, max, std, accept] 
            resultsp[labels[j]] = [minp, meanp, medianp, maxp, stdp, accept] 
            print(j+1, np.median(all_stats),np.std(all_stats))

        headings = ['minimum', 'mean', 'median', 'maximum', 'stdev', 'null hyp']
        results = pd.DataFrame(results).T
        resultsp = pd.DataFrame(resultsp).T

        results.columns = headings
        resultsp.columns = headings
        return results  
 