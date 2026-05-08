from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, ARDRegression 
import pandas as pd
from .ARD import RegressionARD 
import math
import numpy as np
import scipy.special as sp
from scipy import stats 
from itertools import combinations
from sklearn import metrics
from scipy.stats import linregress 
# from numba import jit
import time
import json
from scipy.stats import bootstrap
from sklearn.model_selection import cross_validate 
from sklearn.model_selection import cross_val_score

class regression():

    def __init__(self, X_T_L, Y, method, n_iter, verbose, cv_tol, starting_iter, cv_method, n_jobs=1,
                 ard_algorithm='sequential', threshold_lambda=1e4,
                 cv_folds=10, return_std=False, ard_tol=1e-3):
        self.X_T_L = X_T_L
        self.Y = Y
        self.method = method
        self.n_iter = n_iter
        self.verbose = verbose
        self.cv_tol = cv_tol
        self.starting_iter = starting_iter 
        self.cv_method = cv_method 
        self.n_jobs = n_jobs
        self.ard_algorithm = ard_algorithm
        self.threshold_lambda = threshold_lambda
        self.cv_folds = cv_folds
        self.return_std = return_std
        self.ard_tol = ard_tol

    def run_regression(self, lazy_basis=None):
        start_time = time.perf_counter()
        if self.method == 'ard':
            print('running ARD')
            self.clf = RegressionARD(
                n_iter=self.n_iter, verbose=self.verbose, cv=False,
                algorithm=self.ard_algorithm,
                threshold_lambda=self.threshold_lambda,
                tol=self.ard_tol,
            )
            
        if self.method == 'ard_cv':
            print('running ARD')
            self.clf = RegressionARD(
                n_iter=self.n_iter, verbose=self.verbose, cv=True,
                cv_folds=self.cv_folds, cv_method=self.cv_method,
                algorithm=self.ard_algorithm,
                threshold_lambda=self.threshold_lambda,
                return_std=self.return_std,
                tol=self.ard_tol,
            )
            
        elif self.method == 'omp':
            print('running OMP')
            self.clf = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_iter)
        elif self.method == 'omp_cv':
            print('running OMP_CV')
            self.clf = OrthogonalMatchingPursuitCV(max_iter=self.n_iter, cv=10) 

        elif self.method == 'omp_stream':
            print('running Streaming OMP')
            from .streaming import StreamingOMP
            if lazy_basis is None:
                raise ValueError(
                    "lazy_basis is required for method='omp_stream'"
                )
            self.clf = StreamingOMP(
                lazy_basis, n_nonzero_coefs=self.n_iter,
                verbose=self.verbose,
            )

        elif self.method == 'omp_cv_stream':
            print('running Streaming OMP-CV')
            from .streaming import StreamingOMPCV
            if lazy_basis is None:
                raise ValueError(
                    "lazy_basis is required for method='omp_cv_stream'"
                )
            self.clf = StreamingOMPCV(
                lazy_basis, cv=10, max_iter=self.n_iter,
                n_jobs=self.n_jobs, verbose=self.verbose,
            )
            
        elif self.method == 'ard_sk':
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

        elif self.method == 'ard_omp_cv':
            print('running ARD OMP cross validation')
            clf = OrthogonalMatchingPursuitCV(max_iter=self.n_iter, cv=5) 
            clf.fit(self.X_T_L,self.Y)
            num_iterations = clf.n_nonzero_coefs_
            self.clf = RegressionARD(num_iterations, verbose=self.verbose)
             
#        self.clf = ARDRegression(n_iter=self.n_iter, verbose=True, tol=1.0e-3)
        if self.method in ('omp_stream', 'omp_cv_stream'):
            self.clf.fit(self.Y)
        else:
            self.clf.fit(self.X_T_L,self.Y)
        end_time = time.perf_counter()

        if self.method == 'ompcv':
            print('Number of non-zero coefficeints from OMP_CV : ', self.clf.n_nonzero_coefs_)

#        print('number of iterations ', self.clf.n_iter_)
        print(f"Fit Execution Time : {end_time - start_time:0.6f}" )
        n_selected = int(np.sum(self.clf.coef_ != 0))
        n_total = self.clf.coef_.shape[0]
        pct = 100.0 * n_selected / n_total
        print(f"Basis functions selected: {n_selected} / {n_total} ({pct:.2f}%)")
        print('--') 
        print(" ")
        print(" Model complete ")

        print(" ")

        if self.method in ('omp_stream', 'omp_cv_stream'):
            y_pred = self.clf.predict(lazy_basis)
        else:
            y_pred = self.clf.predict(self.X_T_L)

        return self.clf.coef_ , y_pred 