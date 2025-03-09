import pandas as pd 

class eval_indices():

    def __init__(self, X_T_L, Y, coef_, evs):
        self.X_T_L = X_T_L
        self.Y = Y
        self.coef_ = coef_
        self.evs = evs

        self.eval_sobol_indices() 

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

    def eval_shapley(self,features):
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
        for feature in features:
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

        return self.shap
    
    def eval_total_index(self, features):
        # Initialize an empty DataFrame with columns 'label' and 'total'
        self.total = pd.DataFrame(columns=['label', 'total'])
        
        # Use a list comprehension to calculate the total index for each column in self.X
        label_list = []
        total_list = []
        
        for column in features:
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

        return self.total 



    def get_sobol_indicies(self):
        return self.results  
    
    def get_non_zero_coefficients(self):
        return self.non_zero_coefficients 




