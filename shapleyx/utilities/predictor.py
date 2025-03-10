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

    def fit(self, X, y):
        self.X_columns = X.columns
        X_T = pd.DataFrame()
        for column in X.columns:
            max = self.ranges[column][1]
            min = self.ranges[column][0]
            X_T[column] = (X[column] - min) / (max-min)

        num_columns = len(self.non_zero_coefficients['labels'])
        num_rows = len(X)
        fitX = np.ones((num_rows, num_columns))

        for i in range(num_columns):
            label = self.non_zero_coefficients['labels'][i]
            for function in label.split('*'):
                func_args = function.split('_')
                fitX[:,i] *= shift_legendre(int(func_args[1]), X_T[func_args[0]])
        
        ridgereg =  Ridge()
        ridgereg.fit(fitX, y)
        self.ridgereg = ridgereg 

    def predict(self, X):
        X = pd.DataFrame(X, columns=self.X_columns) 
        X_T = pd.DataFrame()
        for column in X.columns:
            max = self.ranges[column][1]
            min = self.ranges[column][0]
            X_T[column] = (X[column] - min) / (max-min)

        num_columns = len(self.non_zero_coefficients['labels'])
        num_rows = len(X)
        predictX = np.ones((num_rows, num_columns))

        for i in range(num_columns):
            label = self.non_zero_coefficients['labels'][i]
            for function in label.split('*'):
                func_args = function.split('_')
                predictX[:,i] *= shift_legendre(int(func_args[1]), X_T[func_args[0]])
        result = self.ridgereg.predict(predictX) 
        return result 
    
    