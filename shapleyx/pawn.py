
from scipy.stats import ks_2samp
import numpy as np
import pandas as pd

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
