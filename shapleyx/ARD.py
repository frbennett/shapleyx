from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.linear_model._base import LinearModel, LinearClassifierMixin
from sklearn.utils import check_X_y,check_array,as_float_array

from sklearn import linear_model
from sklearn.model_selection import cross_val_score

from numpy.linalg import LinAlgError

from scipy.linalg import solve_triangular
from scipy.linalg import pinvh
import numpy as np 
import warnings
import logging
class RegressionARD(RegressorMixin, LinearModel):
    """Regression with Automatic Relevance Determination (ARD) using Sparse Bayesian Learning.

    This class implements a fast version of ARD regression, which is a Bayesian approach
    to regression that automatically determines the relevance of each feature. It is based
    on the Sparse Bayesian Learning (SBL) algorithm, which promotes sparsity in the model
    by estimating the precision of the coefficients.

    Args:
        n_iter (int, optional): Maximum number of iterations for the optimization algorithm.
            Defaults to 300.
        tol (float, optional): Convergence threshold. If the absolute change in the precision
            parameter for the weights is below this threshold, the algorithm terminates.
            Defaults to 1e-3.
        fit_intercept (bool, optional): Whether to calculate the intercept for this model.
            If set to False, no intercept will be used in calculations (e.g., data is expected
            to be already centered). Defaults to True.
        copy_X (bool, optional): If True, X will be copied; else, it may be overwritten.
            Defaults to True.
        verbose (bool, optional): If True, the algorithm will print progress messages during
            fitting. Defaults to False.
        cv_tol (float, optional): Tolerance for cross-validation. If the percentage change in
            cross-validation score is below this threshold, the algorithm terminates.
            Defaults to 0.1.
        cv (bool, optional): If True, cross-validation will be used to determine the optimal
            number of features. Defaults to False.

    Attributes:
        coef_ (array): Coefficients of the regression model (mean of the posterior distribution).
            Shape (n_features,).
        alpha_ (float): Estimated precision of the noise.
        active_ (array): Boolean array indicating which features are active (non-zero coefficients).
            Shape (n_features,), dtype=bool.
        lambda_ (array): Estimated precisions of the coefficients. Shape (n_features,).
        sigma_ (array): Estimated covariance matrix of the weights, computed only for non-zero
            coefficients. Shape (n_features, n_features).
        scores_ (list): List of cross-validation scores if `cv` is True.

    References:
        [1] Tipping, M. E., & Faul, A. C. (2003). Fast marginal likelihood maximisation for
            sparse Bayesian models. In Proceedings of the Ninth International Workshop on
            Artificial Intelligence and Statistics (pp. 276-283).
            
        [2] Tipping, M. E., & Faul, A. C. (2001). Analysis of sparse Bayesian learning. In
            Advances in Neural Information Processing Systems (pp. 383-389).

    Note:
        The RegressionARD class code has been adapted from the original implementation by Amazasp Shaumyan
        https://github.com/AmazaspShumik/sklearn-bayes
    """
    
    def __init__( self, n_iter = 300, tol = 1e-3, fit_intercept = True, 
                  copy_X = True, verbose = False, cv_tol = 0.1, cv=False):
        self.n_iter          = n_iter
        self.tol             = tol
        self.scores_         = list()
        self.fit_intercept   = fit_intercept
        self.copy_X          = copy_X
        self.verbose         = verbose
        self.cv              = cv
        self.cv_tol          = cv_tol
        
        
    def _center_data(self,X,y):
        ''' Centers data'''
        X     = as_float_array(X, copy=self.copy_X)
        # normalisation should be done in preprocessing!
        X_std = np.ones(X.shape[1], dtype = X.dtype)
        if self.fit_intercept:
            X_mean = np.average(X,axis = 0)
            y_mean = np.average(y,axis = 0)
            X     -= X_mean
            y      = y - y_mean
        else:
            X_mean = np.zeros(X.shape[1],dtype = X.dtype)
            y_mean = 0. if y.ndim == 1 else np.zeros(y.shape[1], dtype=X.dtype)
        return X,y, X_mean, y_mean, X_std
        
  
    def fit(self,X,y):
        '''
        Fit the ARD regression model to the data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, matrix of explanatory variables.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        '''
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        X, y, X_mean, y_mean, X_std = self._center_data(X, y)
        n_samples, n_features = X.shape
        cv_list = []
        cv_score_history = [] 
        current_r = 0

        #  precompute X'*Y , X'*X for faster iterations & allocate memory for
        #  sparsity & quality vectors
        XY     = np.dot(X.T,y)
        XX     = np.dot(X.T,X)
        XXd    = np.diag(XX)

        #  initialise precision of noise & and coefficients
        var_y  = np.var(y)
        
        # check that variance is non zero !!!
        if var_y == 0 :
            beta = 1e-2
        else:
            beta = 1. / np.var(y)
        
        A      = np.PINF * np.ones(n_features)
        active = np.zeros(n_features , dtype = bool)
        
        # in case of almost perfect multicollinearity between some features
        # start from feature 0
        if np.sum( XXd - X_mean**2 < np.finfo(np.float32).eps ) > 0:
            A[0]       = np.finfo(np.float16).eps
            active[0]  = True
        else:
            # start from a single basis vector with largest projection on targets
            proj  = XY**2 / XXd
            start = np.argmax(proj)
            active[start] = True
            A[start]      = XXd[start]/( proj[start] - var_y)
 
        warning_flag = 0
        for i in range(self.n_iter):
            XXa     = XX[active,:][:,active]
            XYa     = XY[active]
            Aa      =  A[active]
            
            # mean & covariance of posterior distribution
            Mn,Ri,cholesky  = self._posterior_dist(Aa,beta,XXa,XYa)
            if cholesky:
                Sdiag  = np.sum(Ri**2,0)
            else:
                Sdiag  = np.copy(np.diag(Ri)) 
                warning_flag += 1
            
            # raise warning in case cholesky failes
            if warning_flag == 1:
                warnings.warn(("Cholesky decomposition failed ! Algorithm uses pinvh, "
                               "which is significantly slower, if you use RVR it "
                               "is advised to change parameters of kernel"))
                
            # compute quality & sparsity parameters            
            s,q,S,Q = self._sparsity_quality(XX,XXd,XY,XYa,Aa,Ri,active,beta,cholesky)
                
            # update precision parameter for noise distribution
            rss     = np.sum( ( y - np.dot(X[:,active] , Mn) )**2 )
            beta    = n_samples - np.sum(active) + np.sum(Aa * Sdiag )
            beta   /= ( rss + np.finfo(np.float32).eps )

            # update precision parameters of coefficients
            A,converged  = update_precisions(Q,S,q,s,A,active,self.tol,
                                             n_samples,False)
# ***************************************            
            if self.cv:
                # Select features based on the 'active' mask
                # Assumes X is a numpy array for efficient slicing
                X_active = X[:, active]

                # Define the model for cross-validation (instantiated fresh each time)
                cv_model = linear_model.Ridge()

                try:
                    # Perform 10-fold cross-validation, explicitly using R^2 scoring
                    # Ensure 'y' corresponds correctly to 'X_active'
                    cv_scores = cross_val_score(cv_model, X_active, y, cv=10, scoring='r2')
                    new_cv_score = np.mean(cv_scores) # Use numpy mean for clarity

                    # Calculate percentage change, handling division by zero
                    # Assumes current_cv_score is initialized (e.g., to None or 0.0) before the loop
                    if current_cv_score is not None and current_cv_score != 0:
                        percentage_change = (new_cv_score - current_cv_score) / current_cv_score * 100
                    elif new_cv_score == 0 and (current_cv_score is None or current_cv_score == 0):
                        percentage_change = 0.0 # No change if both old and new scores are zero
                    else:
                        # Handle cases where current_cv_score is None (first iteration) or zero
                        percentage_change = np.inf # Indicate a large change if starting from zero/None

                    # Optional: Replace print with logging for better control in applications
                    # Assumes 'i' is an iteration counter from an outer loop
                    print(f"Iteration {i}: CV Score = {new_cv_score:.4f}, % Change = {percentage_change:.2f}%")

                    # Check for convergence based on the absolute percentage change
                    # Assumes cv_tol is a positive threshold for the magnitude of change
                    # Assumes 'converged' is initialized (e.g., to False) before the loop
                    if current_cv_score is not None and abs(percentage_change) < self.cv_tol:
                        converged = True
                        # Consider adding a 'break' here if the loop should terminate immediately upon convergence

                    # Update the current score and history
                    # Assumes cv_score_history is initialized (e.g., as []) before the loop
                    current_cv_score = new_cv_score
                    cv_score_history.append(new_cv_score)

                except ValueError as ve:
                    # Catch specific errors, e.g., if X_active becomes empty or has incompatible dimensions
                    print(f"Warning: Cross-validation failed at iteration {i} due to ValueError: {ve}")
                    # Decide how to handle: stop, skip, assign default score?
                    # Example: Treat as no improvement or break
                    percentage_change = np.nan # Mark as invalid
                    # converged = True # Option: Stop if CV fails
                except Exception as e:
                    # Catch other potential errors during cross-validation
                    print(f"Warning: Cross-validation failed unexpectedly at iteration {i}: {e}")
                    percentage_change = np.nan
                    # converged = True # Option: Stop if CV fails

            
            # Calculate active features once per iteration
            num_active_features = np.sum(active)

            if self.verbose:
                # Use logging (assuming logger is configured) and f-string for iteration progress
                # import logging  # Ensure logging is imported at the top of the file
                logging.info(f"Iteration: {i}, Active Features: {num_active_features}")

            # Check for convergence or max iterations to terminate
            if converged or i == self.n_iter - 1:
                # Construct the final status message
                final_status = f"Finished at Iteration: {i}, Active Features: {num_active_features}."
                if converged:
                    log_level = logging.INFO # Normal convergence
                    final_status += " Algorithm converged."
                    # The original code printed "Algorithm converged !" only if verbose.
                    # Logging INFO level covers this sufficiently. Add DEBUG if more detail needed.
                    # if self.verbose:
                    #    logging.debug("Convergence details: ...")
                else: # i == self.n_iter - 1
                    log_level = logging.WARNING # Reached max iterations without converging
                    final_status += f" Reached maximum iterations ({self.n_iter})."

                # Log the final status
                logging.log(log_level, final_status)
                break # Exit the loop
        
        #print(('Iteration: {0}, number of features '
        #               'in the model: {1}').format(i,np.sum(active)))      
          
        # after last update of alpha & beta update parameters
        # of posterior distribution
        XXa,XYa,Aa         = XX[active,:][:,active],XY[active],A[active]
        Mn, Sn, cholesky   = self._posterior_dist(Aa,beta,XXa,XYa,True)
        self.coef_         = np.zeros(n_features)
        self.coef_[active] = Mn
        self.sigma_        = Sn
        self.active_       = active
        self.lambda_       = A
        self.alpha_        = beta
        self._set_intercept(X_mean,y_mean,X_std)
        if self.cv :
        #    print(max(enumerate(cv_list), key=lambda x: x[1]))
            print(('Iteration: {0}, number of features '
                       'in the model: {1}').format(i,np.sum(active))) 
        return self
        
        
    def _posterior_dist(self,A,beta,XX,XY,full_covar=False):
        '''
        Calculate the mean and covariance matrix of the posterior distribution of coefficients.

        Parameters
        ----------
        A : array, shape (n_features,)
            Precision parameters for the coefficients.

        beta : float
            Precision of the noise.

        XX : array, shape (n_features, n_features)
            X' * X matrix.

        XY : array, shape (n_features,)
            X' * y vector.

        full_covar : bool, optional (default=False)
            If True, return the full covariance matrix; otherwise, return the inverse of the
            lower triangular matrix from the Cholesky decomposition.

        Returns
        -------
        Mn : array, shape (n_features,)
            Mean of the posterior distribution.

        Sn : array, shape (n_features, n_features)
            Covariance matrix of the posterior distribution.

        cholesky : bool
            Whether the Cholesky decomposition was successful.
        '''
        # compute precision matrix for active features
        Sinv = beta * XX
        np.fill_diagonal(Sinv, np.diag(Sinv) + A)
        cholesky = True
        # try cholesky, if it fails go back to pinvh
        try:
            # find posterior mean : R*R.T*mean = beta*X.T*Y
            # solve(R*z = beta*X.T*Y) => find z => solve(R.T*mean = z) => find mean
            R    = np.linalg.cholesky(Sinv)
            Z    = solve_triangular(R,beta*XY, check_finite=False, lower = True)
            Mn   = solve_triangular(R.T,Z, check_finite=False, lower = False)
            
            # invert lower triangular matrix from cholesky decomposition
            Ri   = solve_triangular(R,np.eye(A.shape[0]), check_finite=False, lower=True)
            if full_covar:
                Sn   = np.dot(Ri.T,Ri)
                return Mn,Sn,cholesky
            else:
                return Mn,Ri,cholesky
        except LinAlgError:
            cholesky = False
            Sn   = pinvh(Sinv)
            Mn   = beta*np.dot(Sinv,XY)
            return Mn, Sn, cholesky
    

    def _sparsity_quality(self,XX,XXd,XY,XYa,Aa,Ri,active,beta,cholesky):
        '''
        Calculate sparsity and quality parameters for each feature.

        Parameters
        ----------
        XX : array, shape (n_features, n_features)
            X' * X matrix.

        XXd : array, shape (n_features,)
            Diagonal of X' * X matrix.

        XY : array, shape (n_features,)
            X' * y vector.

        XYa : array, shape (n_active_features,)
            X' * y vector for active features.

        Aa : array, shape (n_active_features,)
            Precision parameters for active features.

        Ri : array, shape (n_active_features, n_active_features)
            Inverse of the lower triangular matrix from the Cholesky decomposition or the
            covariance matrix.

        active : array, dtype=bool, shape (n_features,)
            Boolean array indicating which features are active.

        beta : float
            Precision of the noise.

        cholesky : bool
            Whether the Cholesky decomposition was successful.

        Returns
        -------
        si : array, shape (n_features,)
            Sparsity parameters.

        qi : array, shape (n_features,)
            Quality parameters.

        S : array, shape (n_features,)
            Intermediate sparsity parameters.

        Q : array, shape (n_features,)
            Intermediate quality parameters.

        Theoretical Note:
        -----------------
        Here we used Woodbury Identity for inverting covariance matrix
        of target distribution 
        C    = 1/beta + 1/alpha * X' * X
        C^-1 = beta - beta^2 * X * Sn * X'
        '''

        bxy        = beta*XY
        bxx        = beta*XXd
        if cholesky:
            # here Ri is inverse of lower triangular matrix obtained from cholesky decomp
            xxr    = np.dot(XX[:,active],Ri.T)
            rxy    = np.dot(Ri,XYa)
            S      = bxx - beta**2 * np.sum( xxr**2, axis=1)
            Q      = bxy - beta**2 * np.dot( xxr, rxy)
        else:
            # here Ri is covariance matrix
            XXa    = XX[:,active]
            XS     = np.dot(XXa,Ri)
            S      = bxx - beta**2 * np.sum(XS*XXa,1)
            Q      = bxy - beta**2 * np.dot(XS,XYa)
        # Use following:
        # (EQ 1) q = A*Q/(A - S) ; s = A*S/(A-S), so if A = np.PINF q = Q, s = S
        qi         = np.copy(Q)
        si         = np.copy(S) 
        #  If A is not np.PINF, then it should be 'active' feature => use (EQ 1)
        Qa,Sa      = Q[active], S[active]
        qi[active] = Aa * Qa / (Aa - Sa )
        si[active] = Aa * Sa / (Aa - Sa )
        return [si,qi,S,Q]
    
    def predict_dist(self,X):
        '''
        Computes predictive distribution for test set.
        Predictive distribution for each data point is one dimensional
        Gaussian and therefore is characterised by mean and variance.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples_test, n_features)
            Test data, matrix of explanatory variables.

        Returns
        -------
        y_hat : array, shape (n_samples_test,)
            Estimated values of targets on the test set (mean of the predictive distribution).

        var_hat : array, shape (n_samples_test,)
            Variance of the predictive distribution.
        '''
        y_hat     = self._decision_function(X)
        var_hat   = 1./self.alpha_
        var_hat  += np.sum( np.dot(X[:,self.active_],self.sigma_) * X[:,self.active_], axis = 1)
        return y_hat, var_hat




def update_precisions(Q,S,q,s,A,active,tol,n_samples,clf_bias):
    '''
    Updates the precision parameters (alpha) for features in a sparse Bayesian learning model
    by selecting a feature to add, recompute, or delete based on its impact on the log marginal
    likelihood. The function also checks for convergence.

    Parameters:
    -----------
    Q : numpy.ndarray
        Quality parameters for all features.
    S : numpy.ndarray
        Sparsity parameters for all features.
    q : numpy.ndarray
        Quality parameters for features currently in the model.
    s : numpy.ndarray
        Sparsity parameters for features currently in the model.
    A : numpy.ndarray
        Precision parameters (alpha) for all features.
    active : numpy.ndarray (bool)
        Boolean array indicating whether each feature is currently in the model.
    tol : float
        Tolerance threshold for determining convergence based on changes in precision.
    n_samples : int
        Number of samples in the dataset, used to normalize the change in log marginal likelihood.
    clf_bias : bool
        Flag indicating whether the model includes a bias term (used in classification tasks).

    Returns:
    --------
    list
        A list containing two elements:
        - Updated precision parameters (A) for all features.
        - A boolean flag indicating whether the model has converged.

    Notes:
    ------
    The function performs the following steps:
    1. Computes the change in log marginal likelihood for adding, recomputing, or deleting features.
    2. Identifies the feature that causes the largest change in likelihood.
    3. Updates the precision parameter (alpha) for the selected feature.
    4. Checks for convergence based on whether no features are added/deleted and changes in precision
       are below the specified tolerance.
    5. Returns the updated precision parameters and convergence status.

    Convergence is determined by two conditions:
    - No features are added or deleted.
    - The change in precision for features already in the model is below the tolerance threshold.

    The function ensures that the bias term is not removed in classification tasks.
    '''
    # initialise vector holding changes in log marginal likelihood
    deltaL = np.zeros(Q.shape[0])
    
    # identify features that can be added , recomputed and deleted in model
    theta        =  q**2 - s 
    add          =  (theta > 0) * (active == False)
    recompute    =  (theta > 0) * (active == True)
    delete       = ~(add + recompute)
    
    # compute sparsity & quality parameters corresponding to features in 
    # three groups identified above
    Qadd,Sadd      = Q[add], S[add]
    Qrec,Srec,Arec = Q[recompute], S[recompute], A[recompute]
    Qdel,Sdel,Adel = Q[delete], S[delete], A[delete]
    
    # compute new alpha's (precision parameters) for features that are 
    # currently in model and will be recomputed
    Anew           = s[recompute]**2/ ( theta[recompute] + np.finfo(np.float32).eps)
    delta_alpha    = (1./Anew - 1./Arec)
    
    # compute change in log marginal likelihood 
    deltaL[add]       = ( Qadd**2 - Sadd ) / Sadd + np.log(Sadd/Qadd**2 )
    deltaL[recompute] = Qrec**2 / (Srec + 1. / delta_alpha) - np.log(1 + Srec*delta_alpha)
    deltaL[delete]    = Qdel**2 / (Sdel - Adel) - np.log(1 - Sdel / Adel)
    deltaL            = deltaL  / n_samples
    
    # find feature which caused largest change in likelihood
    feature_index = np.argmax(deltaL)
             
    # no deletions or additions
    same_features  = np.sum( theta[~recompute] > 0) == 0
    
    # changes in precision for features already in model is below threshold
    no_delta       = np.sum( abs( Anew - Arec ) > tol ) == 0
    
    # check convergence: if no features to add or delete and small change in 
    #                    precision for current features then terminate
    converged = False
    if same_features and no_delta:
        converged = True
        return [A,converged]
    
    # if not converged update precision parameter of weights and return
    if theta[feature_index] > 0:
        A[feature_index] = s[feature_index]**2 / theta[feature_index]
        if active[feature_index] == False:
            active[feature_index] = True
    else:
        # at least two active features
        if active[feature_index] == True and np.sum(active) >= 2:
            # do not remove bias term in classification 
            # (in regression it is factored in through centering)
            if not (feature_index == 0 and clf_bias):
               active[feature_index] = False
               A[feature_index]      = np.PINF
                
    return [A,converged]