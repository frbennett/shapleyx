from sklearn.linear_model import Ridge
from sklearn import metrics
import pandas as pd
import numpy as np

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()    

def get_derived_labels(labels):
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


def resampling(dataset, number_of_resamples = 1000):
    number_of_coefficients = len(dataset.columns)-1
    sample_array = np.zeros((number_of_resamples, number_of_coefficients))
    derived_lables = get_derived_labels(dataset.columns)
    derived_lables = [i for i in derived_lables if i != 'Y']
    
    for i in range(number_of_resamples):
        printProgressBar(i+1, number_of_resamples, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r")
        sample = dataset.sample(frac=1, replace=True).copy()
        sampleY = sample['Y']
        del sample['Y']
    
        ridgereg =  Ridge() 
        ridgereg.fit(sample, sampleY)
        y_pred = ridgereg.predict(sample)
        evs = metrics.explained_variance_score(y_pred,sampleY)
        indicies = ridgereg.coef_**2
        modelled_variance = np.sum(indicies)
    
        sample_array[i,:] = indicies / modelled_variance * evs
        
    sample_array_df = pd.DataFrame(sample_array.T, index=derived_lables)
    sample_array_df_group = sample_array_df.groupby([sample_array_df.index]).sum()
    
    return sample_array_df_group


def get_shap(resamples, labels):

    number_of_resamples = np.shape(resamples)[1]
    number_of_labels = len(labels)
    sample_array = np.zeros((number_of_labels,number_of_resamples))
    
    column = 0
    for i in labels:
        for j, k in resamples.iterrows():
            if i in j.split('_'):
                sample_array[column,:] += k/len(j.split('_'))
        column+=1
    shaps = pd.DataFrame(sample_array, index=labels)
    return shaps