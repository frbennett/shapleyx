
from sklearn import metrics
from scipy.stats import linregress 

import matplotlib.pyplot as plt  
import numpy as np 

def stats(Y, y_pred, model_coefficients):
    sum_of_coeffs_squared = np.sum(model_coefficients**2)
    data_variance = (np.std(Y))**2 
    var_ratio = sum_of_coeffs_squared/data_variance
    print("variance of data        : {data_variance:0.3f}".format(data_variance=data_variance))
    print("sum of coefficients^2   : {sum_of_coeffs_squared:0.3f}".format(sum_of_coeffs_squared=sum_of_coeffs_squared))
    print("variance ratio          : {var_ratio:0.3f}".format(var_ratio=var_ratio))
    
    print("===============================")
    y_pred = y_pred 
    mse = metrics.mean_squared_error(y_pred,Y)
    mae = metrics.mean_absolute_error(y_pred,Y)
    evs = metrics.explained_variance_score(y_pred,Y)
    slope, intercept, r_value, p_value, std_err = linregress(Y, y_pred)
    print("mae error on test set   : {mae:0.3f}".format(mae=mae))
    print("mse error on test set   : {mse:0.3f}".format(mse=mse))
    print("explained variance score: {evs:0.3f}".format(evs=evs))
    print("===============================")
    print("slope     : ", slope)
    print("r value   : ", r_value)
    print("r^2       : ", r_value*r_value)
    print("p value   : ", p_value)
    print("std error : ", std_err)

    return evs 

def plot_hdmr(Y, y_pred):
    plt.scatter(Y,y_pred)
    plt.ylabel('Predicted')
    plt.xlabel('Experimental')
    plt.show()
