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
            cv_tol (float, optional): DEPRECATED - Tolerance for cross-validation early stopping.
                If the percentage change in cross-validation score is below this threshold,
                the algorithm terminates. Defaults to 0.1. Note: Early stopping based on CV
                is deprecated; use `retrospective_selection=True` instead.
            cv (bool, optional): If True, cross-validation will be used for model selection.
                Defaults to False.
            cv_method (str, optional): Method for cross-validation scoring. Options:
                'ridge' - Uses ridge regression (legacy, not recommended),
                'bayesian' - Uses Bayesian marginal likelihood (recommended),
                'predictive' - Uses predictive log likelihood,
                'hybrid' - Combines multiple metrics.
                Defaults to 'bayesian'.
            cv_folds (int, optional): Number of folds for cross-validation. Defaults to 10.
            retrospective_selection (bool, optional): If True, runs all iterations and
                retrospectively selects the best model based on CV score. If False, uses
                early stopping (deprecated). Defaults to True.
            store_history (bool, optional): If True, stores model states at each iteration
                for debugging/analysis. Increases memory usage. Defaults to False.

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
        history_ (dict): Dictionary containing iteration history if `store_history=True`.
            Includes 'iterations', 'cv_scores', 'n_features', and optionally 'states'.
        best_iteration_ (int): Index of the best iteration selected via retrospective selection.
        best_cv_score_ (float): Best cross-validation score across all iterations.

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
                  copy_X = True, verbose = False, cv_tol = 0.1, cv=False,
                  cv_method='bayesian', cv_folds=10, retrospective_selection=True,
                  store_history=False):
        self.n_iter          = n_iter
        self.tol             = tol
        self.scores_         = list()
        self.fit_intercept   = fit_intercept
        self.copy_X          = copy_X
        self.verbose         = verbose
        self.cv              = cv
        self.cv_tol          = cv_tol
        self.cv_method       = cv_method
        self.cv_folds        = cv_folds
        self.retrospective_selection = retrospective_selection
        self.store_history   = store_history
        
        # Initialize history attributes
        self.history_ = {
            'iterations': [],
            'cv_scores': [],
            'n_features': [],
            'states': [] if store_history else None
        }
        self.best_iteration_ = None
        self.best_cv_score_ = None
        
        # Warn about deprecated cv_tol if retrospective_selection is True
        if retrospective_selection and cv_tol != 0.1:
            warnings.warn(
                "cv_tol parameter is deprecated when retrospective_selection=True. "
                "Early stopping based on CV tolerance is disabled. "
                "Set retrospective_selection=False to use cv_tol for early stopping.",
                DeprecationWarning
            )
        
        
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
        
        # Initialize history storage
        self.history_ = {
            'iterations': [],
            'cv_scores': [],
            'n_features': [],
            'states': [] if self.store_history else None
        }
        self.best_iteration_ = None
        self.best_cv_score_ = None
        
        # For backward compatibility
        cv_list = []
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
        
        A      = np.inf * np.ones(n_features)
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
        
        # Store best model state for retrospective selection
        best_state = None
        best_cv_score = -np.inf
        best_iteration = -1
        
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
            
            # --- Cross-validation scoring (if enabled) ---
            cv_score = None
            if self.cv:
                cv_score = self._compute_cv_score(X, y, active, beta, A, XX, XY,
                                                 X_mean, y_mean, X_std)
                
                # Store in history
                self.history_['iterations'].append(i)
                self.history_['cv_scores'].append(cv_score)
                self.history_['n_features'].append(np.sum(active))
                
                if self.store_history:
                    # Store model state
                    state = {
                        'active': active.copy(),
                        'coef': np.zeros(n_features),
                        'coef_active': Mn.copy(),
                        'lambda': A.copy(),
                        'alpha': beta,
                        'sigma': Ri.copy() if not cholesky else None,
                        'cholesky': cholesky
                    }
                    state['coef'][active] = Mn
                    self.history_['states'].append(state)
                
                # Update best model for retrospective selection
                if cv_score is not None and cv_score > best_cv_score:
                    best_cv_score = cv_score
                    best_iteration = i
                    # Store best state
                    best_state = {
                        'active': active.copy(),
                        'A': A.copy(),
                        'beta': beta,
                        'XXa': XXa.copy(),
                        'XYa': XYa.copy(),
                        'Aa': Aa.copy(),
                        'X_mean': X_mean.copy(),
                        'y_mean': y_mean,
                        'X_std': X_std.copy()
                    }
                
                # For backward compatibility
                cv_list.append(cv_score)
                if i == 0:
                    current_r = cv_score if cv_score is not None else 0
                
                # Print CV status
                if self.verbose:
                    print(f'Iteration: {i:<4}  CV Score: {cv_score:.6f}  '
                          f'Active features: {np.sum(active)}')
            
            # --- Verbose output for main iteration progress ---
            if self.verbose and not self.cv:
                # Use f-string for consistency and clarity
                print(f'Iteration: {i:<5}, number of features remaining: {np.sum(active)}')

            # --- Check for convergence (ARD criteria only, no CV early stopping) ---
            # Note: CV-based early stopping is disabled when retrospective_selection=True
            if converged or i == self.n_iter - 1:
                if self.verbose:
                    print(f'Finished ARD iterations at iteration {i+1}.')
                    if converged:
                        print('Algorithm converged (ARD criteria).')
                    elif i == self.n_iter - 1:
                        print('Reached maximum number of iterations without full convergence.')
                
                # If using retrospective selection and we have a best state, restore it
                if self.cv and self.retrospective_selection and best_state is not None:
                    if self.verbose:
                        print(f'Restoring best model from iteration {best_iteration} '
                              f'with CV score: {best_cv_score:.6f}')
                    
                    # Restore best state
                    active = best_state['active']
                    A = best_state['A']
                    beta = best_state['beta']
                    XXa = best_state['XXa']
                    XYa = best_state['XYa']
                    Aa = best_state['Aa']
                    X_mean = best_state['X_mean']
                    y_mean = best_state['y_mean']
                    X_std = best_state['X_std']
                    
                    # Update best iteration attributes
                    self.best_iteration_ = best_iteration
                    self.best_cv_score_ = best_cv_score
                
                # Break only if not using retrospective selection, if we're at the last iteration, or if converged
                if not self.retrospective_selection or i == self.n_iter - 1 or converged:
                    break
        
            
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
        
        # Store scores for backward compatibility
        self.scores_ = cv_list if self.cv else []
        
        if self.cv and self.verbose:
            print(('Number of features in the model: {0}').format(np.sum(active)))
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
        # (EQ 1) q = A*Q/(A - S) ; s = A*S/(A-S), so if A = np.inf q = Q, s = S
        qi         = np.copy(Q)
        si         = np.copy(S) 
        #  If A is not np.inf, then it should be 'active' feature => use (EQ 1)
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

    def _compute_cv_score(self, X, y, active, beta, A, XX, XY, X_mean, y_mean, X_std):
        '''
        Compute cross-validation score using the selected method.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.
        y : array, shape (n_samples,)
            Target values.
        active : array, dtype=bool, shape (n_features,)
            Boolean array indicating active features.
        beta : float
            Noise precision.
        A : array, shape (n_features,)
            Coefficient precisions.
        XX : array, shape (n_features, n_features)
            X' * X matrix.
        XY : array, shape (n_features,)
            X' * y vector.
        X_mean : array, shape (n_features,)
            Mean of X used for centering.
        y_mean : float
            Mean of y used for centering.
        X_std : array, shape (n_features,)
            Standard deviation of X (not used in current implementation).
            
        Returns
        -------
        cv_score : float
            Cross-validation score (higher is better).
        '''
        if not self.cv:
            return None
            
        if self.cv_method == 'ridge':
            # Legacy ridge regression CV
            return self._ridge_cv_score(X, y, active)
        elif self.cv_method == 'bayesian':
            # Bayesian marginal likelihood CV
            return self._bayesian_cv_score(X, y, active, beta, A, XX, XY)
        elif self.cv_method == 'predictive':
            # Predictive log likelihood CV
            return self._predictive_cv_score(X, y, active, beta, A, XX, XY, X_mean, y_mean, X_std)
        elif self.cv_method == 'hybrid':
            # Hybrid scoring combining multiple metrics
            return self._hybrid_cv_score(X, y, active, beta, A, XX, XY, X_mean, y_mean, X_std)
        else:
            raise ValueError(f"Unknown cv_method: {self.cv_method}. "
                           f"Supported methods: 'ridge', 'bayesian', 'predictive', 'hybrid'")

    def _ridge_cv_score(self, X, y, active):
        '''
        Legacy ridge regression cross-validation score.
        Maintains backward compatibility with original implementation.
        '''
        X_active = X[:, active]
        if X_active.shape[1] == 0:
            return -np.inf  # No active features, poor score
            
        cv_model = linear_model.Ridge()
        cv_scores = cross_val_score(cv_model, X_active, y, cv=self.cv_folds)
        return cv_scores.mean()

    def _bayesian_cv_score(self, X, y, active, beta, A, XX, XY):
        '''
        Bayesian marginal likelihood cross-validation score.
        
        Computes the log marginal likelihood of the ARD model using k-fold CV.
        This is conceptually aligned with the ARD framework as it uses the same
        probabilistic model for scoring.
        '''
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        log_likelihoods = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit ARD on training fold (simplified - using current state as approximation)
            # In practice, we would re-fit but that's expensive. Instead, we compute
            # marginal likelihood using the current model parameters.
            active_train = active.copy()
            if not np.any(active_train):
                # No active features, use simple Gaussian likelihood
                var_y = np.var(y_train)
                if var_y == 0:
                    var_y = 1e-10
                ll = -0.5 * len(y_val) * np.log(2 * np.pi * var_y) - 0.5 * np.sum((y_val - np.mean(y_train))**2) / var_y
            else:
                # Compute marginal likelihood using ARD model
                XX_train = np.dot(X_train[:, active_train].T, X_train[:, active_train])
                XY_train = np.dot(X_train[:, active_train].T, y_train)
                A_active = A[active_train]
                
                # Compute posterior for training data
                Mn, Sn, _ = self._posterior_dist(A_active, beta, XX_train, XY_train, full_covar=True)
                
                # Predictive mean and variance for validation data
                X_val_active = X_val[:, active_train]
                y_pred = np.dot(X_val_active, Mn)
                sigma2 = 1.0 / beta + np.sum(np.dot(X_val_active, Sn) * X_val_active, axis=1)
                
                # Add small epsilon for numerical stability
                sigma2 = np.maximum(sigma2, np.finfo(np.float64).eps)
                
                # Log likelihood of validation data under predictive distribution
                ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + (y_val - y_pred)**2 / sigma2)
            
            log_likelihoods.append(ll)
        
        return np.mean(log_likelihoods)

    def _predictive_cv_score(self, X, y, active, beta, A, XX, XY, X_mean, y_mean, X_std):
        '''
        Predictive log likelihood cross-validation score.
        
        Uses the predictive distribution of the ARD model on validation data.
        This is a proper scoring rule that accounts for both predictive accuracy
        and uncertainty calibration.
        
        Note: This method works in centered space (same as _bayesian_cv_score)
        to maintain consistency. The input X and y are already centered from fit().
        '''
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        predictive_lls = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # X and y are already centered from fit(), use directly
            # No need to subtract X_mean and y_mean again
            
            # Fit ARD on training fold
            active_train = active.copy()
            if not np.any(active_train):
                # No active features
                predictive_lls.append(-np.inf)
                continue
                
            XX_train = np.dot(X_train[:, active_train].T, X_train[:, active_train])
            XY_train = np.dot(X_train[:, active_train].T, y_train)
            A_active = A[active_train]
            
            # Compute posterior
            Mn, Sn, _ = self._posterior_dist(A_active, beta, XX_train, XY_train, full_covar=True)
            
            # Predictive distribution for validation data
            # X_val is already centered, use directly
            X_val_active = X_val[:, active_train]
            
            # Predictions in centered space (consistent with y_val)
            y_pred = np.dot(X_val_active, Mn)
            sigma2 = 1.0 / beta + np.sum(np.dot(X_val_active, Sn) * X_val_active, axis=1)
            
            # Add small epsilon for numerical stability
            sigma2 = np.maximum(sigma2, np.finfo(np.float64).eps)
            
            # Log predictive likelihood
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + (y_val - y_pred)**2 / sigma2)
            predictive_lls.append(ll)
        
        return np.mean(predictive_lls)

    def _hybrid_cv_score(self, X, y, active, beta, A, XX, XY, X_mean, y_mean, X_std):
        '''
        Hybrid cross-validation score combining multiple metrics.
        
        Combines:
        1. Predictive log likelihood (weight: 0.6)
        2. Number of active features (sparsity penalty, weight: 0.2)
        3. Mean squared error (weight: 0.2)
        
        Returns a composite score where higher is better.
        '''
        # Get predictive score
        predictive_score = self._predictive_cv_score(X, y, active, beta, A, XX, XY, X_mean, y_mean, X_std)
        
        # Normalize predictive score (rough normalization)
        if predictive_score > -100:
            norm_predictive = predictive_score / 100.0
        else:
            norm_predictive = -1.0
            
        # Sparsity score (penalize too many features)
        n_active = np.sum(active)
        n_features = X.shape[1]
        sparsity_score = 1.0 - (n_active / n_features)  # Higher for sparser models
        
        # MSE score (simplified)
        if n_active > 0:
            X_active = X[:, active]
            XXa = XX[active, :][:, active]
            XYa = XY[active]
            Aa = A[active]
            Mn, _, _ = self._posterior_dist(Aa, beta, XXa, XYa, full_covar=False)
            y_pred = np.dot(X_active, Mn) + y_mean
            mse = np.mean((y - y_pred) ** 2)
            mse_score = 1.0 / (1.0 + mse)  # Higher for lower MSE
        else:
            mse_score = 0.0
            
        # Composite score
        composite_score = (0.6 * norm_predictive +
                          0.2 * sparsity_score +
                          0.2 * mse_score)
        
        return composite_score




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
               A[feature_index]      = np.inf
                
    return [A,converged]