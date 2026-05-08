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
                'bayesian' - Uses predictive log-likelihood CV (recommended),
                'predictive' - Alias for 'bayesian'.
                Defaults to 'bayesian'.
            cv_folds (int, optional): Number of folds for cross-validation. Defaults to 10.
            retrospective_selection (bool, optional): If True, runs all iterations and
                retrospectively selects the best model based on CV score. If False, uses
                early stopping (deprecated). Defaults to True.
            store_history (bool, optional): If True, stores model states at each iteration
                for debugging/analysis. Increases memory usage. Defaults to False.
            threshold_lambda (float, optional): Post-fit pruning threshold.  After ARD
                converges (and after retrospective CV selection, if enabled), any feature
                whose estimated precision ``lambda_[i]`` exceeds this threshold has its
                coefficient set to exactly zero.  This mirrors the pruning step in
                sklearn's ``ARDRegression`` and produces sparser models for problems
                where the SBL algorithm retains marginal features with large but finite
                precisions.  Defaults to 10\\,000.  Set to ``np.inf`` to disable.
            algorithm (str, optional): Which ARD variant to use.
                ``'sequential'`` (default) — Tipping & Faul (2003) fast sequential
                SBL: picks one feature per iteration (add/recompute/delete).
                ``'em'`` — batch evidence maximization (`MacKay 1992`_):
                updates all weights simultaneously with Gamma hyper-priors
                (``alpha_1, alpha_2, lambda_1, lambda_2``) and per-iteration
                pruning via ``threshold_lambda``.  Typically produces sparser
                models matching sklearn's ``ARDRegression`` behaviour.
            alpha_1, alpha_2 : float, optional
                Gamma hyper-prior shape/inverse-scale for the noise precision.
                Only used when ``algorithm='em'``.  Default 1e-6.
            lambda_1, lambda_2 : float, optional
                Gamma hyper-prior shape/inverse-scale for the weight precisions.
                Only used when ``algorithm='em'``.  Default 1e-6.
    
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
                  store_history=False, threshold_lambda=1e4,
                  algorithm='sequential',
                  alpha_1=1e-6, alpha_2=1e-6,
                  lambda_1=1e-6, lambda_2=1e-6):
        self.n_iter          = n_iter
        self.tol             = tol
        self.fit_intercept   = fit_intercept
        self.copy_X          = copy_X
        self.verbose         = verbose
        self.cv              = cv
        self.cv_tol          = cv_tol
        self.cv_method       = cv_method
        self.cv_folds        = cv_folds
        self.retrospective_selection = retrospective_selection
        self.store_history   = store_history
        self.threshold_lambda = float(threshold_lambda)
        self.algorithm        = algorithm
        self.alpha_1          = float(alpha_1)
        self.alpha_2          = float(alpha_2)
        self.lambda_1         = float(lambda_1)
        self.lambda_2         = float(lambda_2)

        if self.algorithm not in ('sequential', 'em'):
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Expected 'sequential' or 'em'."
            )

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
        # ── Route to the appropriate algorithm ─────────────────
        if self.algorithm == 'em':
            return self._fit_em(X, y)
        # ── Sequential SBL (Tipping & Faul 2003) ──────────────
        # Save original-scale data for cross-validation (prevents leakage
        # from full-dataset centering into per-fold evaluation).
        X_orig, y_orig = X.copy(), y.copy()
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
                cv_score = self._compute_cv_score(X_orig, y_orig, active, beta, A, XX, XY,
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
        
        # Post-fit pruning: zero out coefficients whose estimated
        # precision exceeds threshold_lambda (mirrors sklearn ARDRegression).
        if np.isfinite(self.threshold_lambda):
            prune_mask = (self.lambda_ > self.threshold_lambda) & self.active_
            self.coef_[prune_mask] = 0.0
            self.active_[prune_mask] = False
            if self.verbose and np.any(prune_mask):
                print(f"  Pruned {np.sum(prune_mask)} features with "
                      f"lambda > {self.threshold_lambda:.0f}")

        self._set_intercept(X_mean,y_mean,X_std)
        
        # Store scores for backward compatibility
        self.scores_ = cv_list if self.cv else []
        
        if self.cv and self.verbose:
            print(('Number of features in the model: {0}').format(np.sum(active)))
        return self
        
        
    def _fit_em(self, X_orig, y_orig):
        """Batch evidence-maximization ARD (MacKay 1992).

        Updates all weights, precisions, and the noise precision
        simultaneously at each iteration, with Gamma hyper-priors
        driving sparsity.  Mirrors sklearn's ``ARDRegression``.

        Parameters
        ----------
        X_orig : ndarray (n_samples, n_features) — original scale
        y_orig : ndarray (n_samples,) — original scale

        Returns
        -------
        self
        """
        import numpy as np
        from scipy.linalg import pinvh

        n_samples, n_features = X_orig.shape
        dtype = X_orig.dtype

        # Centre data if fitting intercept
        X = X_orig.copy()
        y = y_orig.copy()
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = float(np.mean(y))
            X -= X_mean
            y -= y_mean
        else:
            X_mean = np.zeros(n_features, dtype=dtype)
            y_mean = 0.0
        X_std = np.ones(n_features, dtype=dtype)

        # ── Initialisation ────────────────────────────────────
        eps = np.finfo(np.float64).eps
        alpha_ = 1.0 / (np.var(y) + eps)   # noise precision
        lambda_ = np.ones(n_features, dtype=np.float64)  # weight precisions
        coef_ = np.zeros(n_features, dtype=np.float64)
        keep_lambda = np.ones(n_features, dtype=bool)

        a1, a2 = self.alpha_1, self.alpha_2
        l1, l2 = self.lambda_1, self.lambda_2
        thresh = self.threshold_lambda
        scores = []

        coef_old = None

        # ── Main loop ─────────────────────────────────────────
        for it in range(self.n_iter):
            # --- Posterior covariance (Woodbury when p > n) ---
            active_idx = np.where(keep_lambda)[0]
            if len(active_idx) == 0:
                break

            Xa = X[:, keep_lambda]
            try:
                # Standard route: (β XᵀX + Λ)⁻¹
                XTX = np.dot(Xa.T, Xa)
                Sinv = alpha_ * XTX + np.diag(lambda_[keep_lambda])
                sigma_ = pinvh(Sinv)
            except Exception:
                # Fallback: use Woodbury when ill-conditioned
                sigma_ = self._sigma_woodbury(X, alpha_, lambda_, keep_lambda)

            # --- Weight update ---
            coef_[:] = 0.0
            coef_[keep_lambda] = alpha_ * np.linalg.multi_dot(
                [sigma_, Xa.T, y]
            )

            # --- Update precisions with Gamma hyper-priors ---
            gamma_ = 1.0 - lambda_[keep_lambda] * np.diag(sigma_)
            gamma_ = np.maximum(gamma_, 0.0)

            lambda_[keep_lambda] = (gamma_ + 2.0 * l1) / (
                coef_[keep_lambda] ** 2 + 2.0 * l2
            )
            # sse = ||y - X @ coef||²
            sse_ = np.dot(y - np.dot(Xa, coef_[keep_lambda]),
                          y - np.dot(Xa, coef_[keep_lambda]))
            alpha_ = (n_samples - gamma_.sum() + 2.0 * a1) / (
                sse_ + 2.0 * a2
            )

            # --- Prune features whose precision exceeds threshold ---
            keep_lambda = lambda_ < thresh
            coef_[~keep_lambda] = 0.0

            # --- Progress ---
            if self.verbose:
                n_active = int(np.sum(keep_lambda))
                r2 = 1.0 - sse_ / np.dot(y, y) if np.dot(y, y) > 0 else 0.0
                print(f"  EM iter {it+1:<4d}/{self.n_iter}"
                      f"  active={n_active:<4d}  R²={r2:.4f}")

            # --- Convergence check ---
            if it > 0 and coef_old is not None:
                delta = np.sum(np.abs(coef_old - coef_))
                if delta < self.tol:
                    if self.verbose:
                        print(f"  Converged at iteration {it+1}")
                    break
            coef_old = coef_.copy()

        # ── Final state ────────────────────────────────────────
        if np.any(keep_lambda):
            Xa = X[:, keep_lambda]
            XTX = np.dot(Xa.T, Xa)
            Sinv = alpha_ * XTX + np.diag(lambda_[keep_lambda])
            sigma_ = pinvh(Sinv)
            coef_[:] = 0.0
            coef_[keep_lambda] = alpha_ * np.linalg.multi_dot(
                [sigma_, Xa.T, y]
            )
        else:
            sigma_ = np.zeros((0, 0))

        self.coef_ = coef_
        self.alpha_ = float(alpha_)
        self.sigma_ = sigma_
        self.lambda_ = lambda_
        self.active_ = keep_lambda
        self.scores_ = scores
        self.n_iter_ = it + 1
        self._set_intercept(X_mean, y_mean, X_std)

        return self

    @staticmethod
    def _sigma_woodbury(X, alpha_, lambda_, keep_lambda):
        """Compute posterior covariance via Woodbury identity.

        Uses the Woodbury matrix identity to compute
        (Λ + α XᵀX)⁻¹ efficiently when n < p_split.
        """
        keep_idx = np.where(keep_lambda)[0]
        Lambda_inv = np.diag(1.0 / lambda_[keep_lambda])
        Xa = X[:, keep_lambda]
        # sigma = Λ⁻¹ - Λ⁻¹ Xᵀ (α⁻¹ I + X Λ⁻¹ Xᵀ)⁻¹ X Λ⁻¹
        X_Linv = np.dot(Xa, Lambda_inv)
        inner = np.eye(X.shape[0]) / alpha_ + np.dot(X_Linv, Xa.T)
        try:
            inner_inv = np.linalg.inv(inner)
        except np.linalg.LinAlgError:
            inner_inv = pinvh(inner)
        return Lambda_inv - np.linalg.multi_dot([Lambda_inv, Xa.T, inner_inv, X_Linv.T])

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
            Training data in ORIGINAL scale (not centered).
        y : array, shape (n_samples,)
            Target values in ORIGINAL scale (not centered).
        active : array, dtype=bool, shape (n_features,)
            Boolean array indicating active features.
        beta : float
            Noise precision.
        A : array, shape (n_features,)
            Coefficient precisions.
        XX : array, shape (n_features, n_features)
            X' * X matrix (precomputed from centered data; not used by most methods).
        XY : array, shape (n_features,)
            X' * y vector (precomputed from centered data; not used by most methods).
        X_mean : array, shape (n_features,)
            Mean of X (full-data centering; NOT used — per-fold centering avoids leakage).
        y_mean : float
            Mean of y (full-data centering; NOT used — per-fold centering avoids leakage).
        X_std : array, shape (n_features,)
            Standard deviation of X (not used by current methods).

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
        elif self.cv_method in ('bayesian', 'predictive'):
            # Predictive log-likelihood CV (per-fold centering, no data leakage)
            return self._predictive_cv_score(X, y, active, beta, A)
        else:
            raise ValueError(f"Unknown cv_method: {self.cv_method}. "
                           f"Supported methods: 'ridge', 'bayesian', 'predictive'")

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

    def _predictive_cv_score(self, X, y, active, beta, A):
        '''
        Predictive log-likelihood cross-validation score.

        Uses K-fold CV with per-fold centering to eliminate data leakage.
        Within each fold the training data is centered independently; the
        validation data is transformed with the training-fold statistics.
        The score is the log predictive likelihood of the validation data
        under the ARD posterior fitted on the training fold.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data in ORIGINAL scale (not pre-centered).
        y : array, shape (n_samples,)
            Target values in ORIGINAL scale (not pre-centered).
        active : array, dtype=bool, shape (n_features,)
            Boolean array indicating active features.
        beta : float
            Noise precision (from current ARD iteration).
        A : array, shape (n_features,)
            Coefficient precisions (from current ARD iteration).

        Returns
        -------
        cv_score : float
            Mean predictive log-likelihood across folds (higher is better).
        '''
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        active_features = active.copy()

        log_likelihoods = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # --- per-fold centering (no leakage from validation data) ---
            X_train_mean = np.mean(X_train, axis=0)
            y_train_mean = np.mean(y_train)
            X_train_c = X_train - X_train_mean
            y_train_c = y_train - y_train_mean
            X_val_c = X_val - X_train_mean

            if not np.any(active_features):
                # No active features: predict with training mean
                var_y = np.var(y_train_c)
                if var_y < np.finfo(np.float64).eps:
                    var_y = 1e-10
                ll = -0.5 * len(y_val) * np.log(2 * np.pi * var_y) \
                     - 0.5 * np.sum((y_val - y_train_mean)**2) / var_y
                log_likelihoods.append(ll)
                continue

            # Subset to active features
            X_train_a = X_train_c[:, active_features]
            X_val_a   = X_val_c[:, active_features]
            A_active  = A[active_features]

            # Compute posterior on training fold
            XX_train = np.dot(X_train_a.T, X_train_a)
            XY_train = np.dot(X_train_a.T, y_train_c)
            Mn, Sn, _ = self._posterior_dist(A_active, beta, XX_train, XY_train,
                                             full_covar=True)

            # Predict validation data (convert back to original scale)
            y_pred = np.dot(X_val_a, Mn) + y_train_mean

            # Predictive variance
            sigma2 = 1.0 / beta + np.sum(np.dot(X_val_a, Sn) * X_val_a, axis=1)
            sigma2 = np.maximum(sigma2, np.finfo(np.float64).eps)

            # Log predictive likelihood
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2)
                               + (y_val - y_pred)**2 / sigma2)
            log_likelihoods.append(ll)

        return np.mean(log_likelihoods)




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