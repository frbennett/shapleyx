import pandas as pd

class transformation():
    def __init__(self, X):
        self.X = X
        self.X_T = pd.DataFrame()
        self.ranges = {}

    def do_transform(self):
        for column in self.X.columns:
            feature_min = self.X[column].min()
            feature_max = self.X[column].max()
    
            # Log the min and max values for debugging or informational purposes
            print(f"Feature: {column}, Min Value: {feature_min:.4f}, Max Value: {feature_max:.4f}") 
    
            # Perform min-max scaling to transform the feature to [0, 1]
            self.X_T[column] = (self.X[column] - feature_min) / (feature_max - feature_min)
    
            # Store the original min and max values for potential inverse transformations
            self.ranges[column] = [feature_min, feature_max]

    def get_ranges(self):
        return self.ranges
    
    def get_X_T(self):
        return self.X_T 