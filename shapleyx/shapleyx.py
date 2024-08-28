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
#from ARD import RegressionARD
#from sklearn.linear_model import ARDRegression 

from collections import Counter 
# import gmdh

import warnings
warnings.filterwarnings('ignore')

def print_heading(text):
    print()
    print('=====================================================')
    print(text)
    print('=====================================================')
    print() 
    

class rshdmr():
    
    def __init__(self,data_file, polys = [10, 5],
                n_jobs = -1,
                test_size = 0.25,
                 limit = 2.0,
                 k_best = 1,
                 p_average = 2,
                 n_iter = 300,
                 verbose = False,
                 method = 'ard',
                 starting_iter = 5):

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
        
    def read_data(self,data_file):
        """
        dsdsd
        """
        if isinstance(data_file, pd.DataFrame):
            print(' found a dataframe')
            df = data_file
        if isinstance(data_file, str):
            df = pd.read_csv(data_file)
        self.Y = df['Y']
        self.X = df.drop('Y', axis=1) 
        # we can clean up the original dataframe
        del df
        
    def shift_legendre(self,n,x):
        funct = math.sqrt(2*n+1) * sp.eval_sh_legendre(n,x)
        return funct
    
        
    def transform_data(self):
        """
        Linear transform input dataset to unit hypercube
        """
        self.X_T = pd.DataFrame()
        self.ranges = {}
        feature_names = list(self.X.columns.values)
        print(feature_names)
        for column in feature_names:
            max = self.X[column].max()
            min = self.X[column].min()
            print(column + " : min " + str(min) + " max " + str(max)) 
            self.X_T[column] = (self.X[column] - min) / (max-min)
            self.ranges[column] = [min,max]
        
            
    def legendre_expand(self):
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
            self.clf = RegressionARD(n_iter=self.n_iter, verbose=self.verbose)
        elif self.method == 'omp':
            print('running OMP')
            self.clf = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_iter)
        elif self.method == 'ompcv':
            print('running OMP_CV')
            self.clf = OrthogonalMatchingPursuitCV(max_iter=self.n_iter, cv=5) 
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
                results = cross_val_score(clf, self.X_T_L,self.Y, cv=3)
                test = np.mean(results)
                if test > best_score :
                    best_score = test
                    best_score_iter = iteration

                if ((iteration -best_score_iter) >= 10) or (iteration == self.n_iter):
                    converged = True
        
                iteration += 1
            print('the best iteration ',     best_score_iter)   
            
            self.clf = RegressionARD(n_iter=best_score_iter, verbose=self.verbose)
             
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
    
        return sobol_indices, shapley_effects, total_index
 