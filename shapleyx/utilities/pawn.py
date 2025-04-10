
from scipy.stats import ks_2samp
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from .predictor import surrogate 
from scipy.stats import qmc


def xsampler(num_samples: int, ranges: dict) -> np.ndarray:
    """
    Generate a Latin Hypercube sample scaled to the specified ranges.

    Args:
        num_samples (int): Number of samples to generate.
        ranges (dict): A dictionary where keys are feature names and values are tuples of (lower, upper) bounds.

    Returns:
        np.ndarray: A scaled Latin Hypercube sample of shape (num_samples, num_features).
    """
    num_features = len(ranges)
    
    # Extract lower and upper bounds from the ranges dictionary
    lower_bounds = [bounds[0] for bounds in ranges.values()]
    upper_bounds = [bounds[1] for bounds in ranges.values()]
    
    # Generate Latin Hypercube sample
    sampler = qmc.LatinHypercube(d=num_features)
    sample = sampler.random(n=num_samples)
    
    # Scale the sample to the specified ranges
    sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)
    
    return sample_scaled 


def estimate_pawn(
    var_names,
    D: int,
    X: np.ndarray,
    Y: np.ndarray,
    S: int = 10,
    print_to_console: bool = False,
    seed: int = None,
) -> pd.DataFrame:
    """
    Estimate the PAWN sensitivity indices for a given set of input variables and output responses.

    The PAWN method is a global sensitivity analysis technique that quantifies the influence of input variables 
    on the output of a model. It uses the Kolmogorov-Smirnov (KS) statistic to compare the empirical 
    cumulative distribution functions (CDFs) of the output conditioned on different intervals of the input variables.

    Parameters:
        var_names (list or array-like): Names or labels of the input variables (dimensions).
        D (int): Number of input variables (dimensions).
        X (np.ndarray): Input data matrix of shape (N, D), where N is the number of samples and D is the number of variables.
        Y (np.ndarray): Output response vector of shape (N,), corresponding to the input data matrix X.
        S (int, optional): Number of intervals to divide the range of each input variable. Default is 10.
        print_to_console (bool, optional): If True, prints intermediate results to the console. Default is False.
        seed (int, optional): Seed for the random number generator to ensure reproducibility. Default is None.

    Returns:
        pd.DataFrame: A DataFrame containing the PAWN sensitivity indices for each input variable. 
                      The columns include:
                      - 'minimum': Minimum KS statistic across intervals.
                      - 'mean': Mean KS statistic across intervals.
                      - 'median': Median KS statistic across intervals.
                      - 'maximum': Maximum KS statistic across intervals.
                      - 'CV': Coefficient of variation (standard deviation divided by mean).
                      - 'stdev': Standard deviation of the KS statistic across intervals.
                      The index of the DataFrame corresponds to the input variable names provided in `var_names`.

    Notes:
        - The KS statistic measures the maximum distance between the empirical CDFs of the output conditioned 
          on different intervals of the input variable.
        - A higher KS statistic indicates a stronger influence of the input variable on the output.
        - The PAWN method is particularly useful for non-linear and non-monotonic models.
        - The function uses `scipy.stats.ks_2samp` to compute the KS statistic.

    References:
        1. Pianosi, F., & Wagener, T. (2015). A simple and efficient method for global sensitivity analysis 
           based on cumulative distribution functions. Environmental Modelling & Software, 67, 1–11.
        2. Pianosi, F., & Wagener, T. (2018). Distribution-based sensitivity analysis from a generic input-output 
           sample. Environmental Modelling & Software, 108, 197–207.
        3. Saltelli, A., et al. (2008). Global Sensitivity Analysis: The Primer. Wiley.
        4. Pianosi, F., et al. (2016). Sensitivity analysis of environmental models: A systematic review with 
           practical workflow. Environmental Modelling & Software, 79, 214–232.

    Example:
        >>> var_names = ['x1', 'x2', 'x3']
        >>> X = np.random.rand(100, 3)  # 100 samples, 3 variables
        >>> Y = X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 0.1, 100)  # Output response
        >>> results = estimate_pawn(var_names, 3, X, Y, S=10, seed=42)
        >>> print(results)
    """
    if seed is not None:
        np.random.seed(seed)

    results = np.full((D, 6), np.nan)
    temp_pawn = np.full((S, D), np.nan)
    step = 1 / S

    for d_i in range(D):
        X_di = X[:, d_i]
        X_q = np.nanquantile(X_di, np.arange(0, 1 + step, step))

        for s in range(S):
            mask = (X_di >= X_q[s]) & (X_di < X_q[s + 1])
            Y_sel = Y[mask]

            if len(Y_sel) == 0:
                continue  # Skip if no samples are available

            ks_statistic = ks_2samp(Y_sel, Y).statistic
            temp_pawn[s, d_i] = ks_statistic

        p_ind = temp_pawn[:, d_i]
        results[d_i, :] = [
            np.nanmin(p_ind),
            np.nanmean(p_ind),
            np.nanmedian(p_ind),
            np.nanmax(p_ind),
            np.nanstd(p_ind) / np.nanmean(p_ind),
            np.nanstd(p_ind),
        ]

    results_df = pd.DataFrame(
        results,
        columns=["minimum", "mean", "median", "maximum", "CV", "stdev"],
        index=var_names,
    )

    if print_to_console:
        print(results_df)

    return results_df


class pawnx():
    def __init__(self, X, y, ranges, non_zero_coefficients):
        """
        Initialize the PAWN sensitivity analysis class with the given parameters.

        Args:
            X (pd.DataFrame): Input features for the surrogate model.
            predict (callable): Function to predict the output of the surrogate model.
            ranges (dict): Dictionary containing the parameter ranges for each feature.
        """
        self.X = X
        self.y = y 
        self.ranges = ranges
        self.non_zero_coefficients = non_zero_coefficients 
        self.predict = surrogate(self.non_zero_coefficients, self.ranges)
        self.predict.fit(self.X, self.y)

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
        y_ref = self.predict.predict(x_ref)
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
                yn = self.predict.predict(xn)
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
    
#****************************************************************************
#                            Delta moment-free method                       *
#****************************************************************************

def kde_area_between(kde1, kde2,D1, D2, grid_resolution=1000):
    """
    Computes the area between two KDEs using SciPy's gaussian_kde.
    
    Parameters:
    - D1, D2: Input datasets (1D numpy arrays)
    - bw_method1, bw_method2: Bandwidth methods for each KDE (optional)
    - grid_resolution: Number of points for evaluation grid
    
    Returns:
    - Area between the two KDEs
    """
    # Create KDE objects

    # Determine integration range with 3x max bandwidth padding
    h = max(kde1.covariance_factor() * np.std(D1, ddof=1),
            kde2.covariance_factor() * np.std(D2, ddof=1))
    
    x_min = min(D1.min(), D2.min()) - 3*h
    x_max = max(D1.max(), D2.max()) + 3*h
    x = np.linspace(x_min, x_max, grid_resolution)

    # Evaluate KDEs
    kde1_vals = kde1(x)
    kde2_vals = kde2(x)

    # Compute absolute differences and integrate
    return np.trapz(np.abs(kde1_vals - kde2_vals), x)


class DeltaX():
    """
    A class to calculate delta statistics for feature importance analysis.

    The DeltaX class is designed to compute delta statistics, which measure the influence of each feature
    on the output of a surrogate model. The delta statistic is calculated by comparing the distribution
    of the model's predictions when a feature is fixed to a specific value versus when it is unconditioned.

    Attributes:
    -----------
    X : pd.DataFrame
        The input feature matrix used for training the surrogate model.
    y : pd.Series or np.ndarray
        The target values corresponding to the input feature matrix.
    ranges : dict
        A dictionary specifying the range of values for each feature. Keys are feature labels, and values
        are tuples of (min, max) values.
    non_zero_coefficients : np.ndarray
        An array of non-zero coefficients used by the surrogate model.
    predict : callable
        A surrogate model trained on the input data (X, y) using the provided non-zero coefficients.

    Methods:
    --------
    get_deltax(num_unconditioned: int, num_samples: int) -> pd.DataFrame:
        Calculates the delta statistics for each feature in the dataset.
    """
    def __init__(self, X, y, ranges, non_zero_coefficients):

        self.X = X
        self.y = y 
        self.ranges = ranges
        self.non_zero_coefficients = non_zero_coefficients 
        self.predict = surrogate(self.non_zero_coefficients, self.ranges)
        self.predict.fit(self.X, self.y)

    def get_deltax(self,num_unconditioned: int,  num_samples: int) -> pd.DataFrame:
        """
        Calculate the delta statistics for each feature in the dataset.

        The delta statistic quantifies the influence of each feature on the model's predictions by comparing
        the distribution of predictions when the feature is fixed to a specific value versus when it is
        unconditioned. The statistic is computed using kernel density estimation (KDE) and the area between
        the KDEs of the two distributions.

        Parameters:
        -----------
        num_unconditioned : int
            The number of unconditioned samples to generate for the reference set.
        num_samples : int
            The number of samples to generate for each feature to calculate the delta statistic.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the following columns:
            - 'Var': The feature labels.
            - 'delta': The median delta statistic for each feature.
            - 'delta_norm': The normalized delta statistic for each feature.
        """
        # Initialize dictionaries to store results
        results = {}
        results_p = {}
        feature_labels = self.X.columns
        num_features = len(self.ranges)
    
        # Generate reference set
        x_ref = xsampler(num_unconditioned, self.ranges)
        D1 = self.predict.predict(x_ref)
        kde1 = gaussian_kde(D1, bw_method='silverman')

        print(f"Number of features: {num_features}")
        
        delta_stats = [] 
        # Iterate over each feature
        for j in range(num_features):
            parameter_range = self.ranges[feature_labels[j]]
    
            # Perform calc test for each sample
            areas = []
            for _ in range(num_samples):
                xi = np.random.uniform(parameter_range[0], parameter_range[1])
                xn = x_ref.copy() 
                xn[:, j] = xi
                D2 = self.predict.predict(xn)
                kde2 = gaussian_kde(D2, bw_method='silverman')
                area = kde_area_between(kde1, kde2,D1, D2, grid_resolution=1000)
                areas.append(area)
            delta_stat = np.mean(areas)
            delta_stats.append(delta_stat)
            print(f"Feature {feature_labels[j]}: Expectation value of delta Statistic = {delta_stat:.3f}")

        results = pd.DataFrame()
        results['Var'] = feature_labels
        results['delta'] = delta_stats
        results['delta_norm'] = delta_stats/np.sum(delta_stats)
    
        return results
    

#****************************************************************************
#                            h moment-free method                       *
#****************************************************************************

def kl_divergence(kde_p, kde_q ,samples_p, samples_q, num_points=1000, epsilon=1e-10):
    """
    Compute KL(P || Q) where P and Q are KDEs estimated from samples_p and samples_q.
    
    Args:
        samples_p: 1D array of samples from distribution P
        samples_q: 1D array of samples from distribution Q
        epsilon: Small value to avoid numerical issues
        num_points: Number of evaluation points for numerical integration
        
    Returns:
        KL divergence value
    """
    # Fit KDEs to the samples
    kde_p = gaussian_kde(samples_p)
    kde_q = gaussian_kde(samples_q)
    
    # Define evaluation points spanning both datasets
    min_val = min(np.min(samples_p), np.min(samples_q)) - 1.0
    max_val = max(np.max(samples_p), np.max(samples_q)) + 1.0
    grid_points = np.linspace(min_val, max_val, num_points)
    
    # Evaluate KDEs (add epsilon to avoid zeros)
    p = kde_p(grid_points) + epsilon
    q = kde_q(grid_points) + epsilon
    
    # Normalize to ensure valid probability distributions
    p /= np.trapz(p, grid_points)
    q /= np.trapz(q, grid_points)
    
    # Compute KL divergence: ∫ p(x) * log(p(x)/q(x)) dx
    kl_integrand = p * (np.log(p) - np.log(q))
    kl = np.trapz(kl_integrand, grid_points)
    
    return kl


class hX():
    """
    A class to calculate delta statistics for feature importance analysis.

    The DeltaX class is designed to compute delta statistics, which measure the influence of each feature
    on the output of a surrogate model. The delta statistic is calculated by comparing the distribution
    of the model's predictions when a feature is fixed to a specific value versus when it is unconditioned.

    Attributes:
    -----------
    X : pd.DataFrame
        The input feature matrix used for training the surrogate model.
    y : pd.Series or np.ndarray
        The target values corresponding to the input feature matrix.
    ranges : dict
        A dictionary specifying the range of values for each feature. Keys are feature labels, and values
        are tuples of (min, max) values.
    non_zero_coefficients : np.ndarray
        An array of non-zero coefficients used by the surrogate model.
    predict : callable
        A surrogate model trained on the input data (X, y) using the provided non-zero coefficients.

    Methods:
    --------
    get_deltax(num_unconditioned: int, num_samples: int) -> pd.DataFrame:
        Calculates the delta statistics for each feature in the dataset.
    """
    def __init__(self, X, y, ranges, non_zero_coefficients, epsilon=1e-10):

        self.X = X
        self.y = y 
        self.ranges = ranges
        self.non_zero_coefficients = non_zero_coefficients 
        self.predict = surrogate(self.non_zero_coefficients, self.ranges)
        self.predict.fit(self.X, self.y)
        self.epsilon = epsilon

    def get_hx(self,num_unconditioned: int,  num_samples: int) -> pd.DataFrame:
        """
        Calculate the delta statistics for each feature in the dataset.

        The delta statistic quantifies the influence of each feature on the model's predictions by comparing
        the distribution of predictions when the feature is fixed to a specific value versus when it is
        unconditioned. The statistic is computed using kernel density estimation (KDE) and the area between
        the KDEs of the two distributions.

        Parameters:
        -----------
        num_unconditioned : int
            The number of unconditioned samples to generate for the reference set.
        num_samples : int
            The number of samples to generate for each feature to calculate the delta statistic.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the following columns:
            - 'Var': The feature labels.
            - 'delta': The median delta statistic for each feature.
            - 'delta_norm': The normalized delta statistic for each feature.
        """
        # Initialize dictionaries to store results
        results = {}
        results_p = {}
        feature_labels = self.X.columns
        num_features = len(self.ranges)
    
        # Generate reference set
        x_ref = xsampler(num_unconditioned, self.ranges)
        D1 = self.predict.predict(x_ref)
        kde1 = gaussian_kde(D1, bw_method='silverman')

        print(f"Number of features: {num_features}")
        
        delta_stats = [] 
        # Iterate over each feature
        for j in range(num_features):
            parameter_range = self.ranges[feature_labels[j]]
    
            # Perform calc test for each sample
            areas = []
            for _ in range(num_samples):
                xi = np.random.uniform(parameter_range[0], parameter_range[1])
                xn = x_ref.copy() 
                xn[:, j] = xi
                D2 = self.predict.predict(xn)
                kde2 = gaussian_kde(D2, bw_method='silverman')
                area = kl_divergence(kde1, kde2,D1, D2, num_points=1000, epsilon=self.epsilon)
                areas.append(area)
            delta_stat = np.mean(areas)
            delta_stats.append(delta_stat)
            print(f"Feature {feature_labels[j]}: Expectation value of delta Statistic = {delta_stat:.3f}")

        results = pd.DataFrame()
        results['Var'] = feature_labels
        results['delta'] = delta_stats
        results['delta_norm'] = delta_stats/np.sum(delta_stats)
    
        return results