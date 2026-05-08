"""
ARD — Automatic Relevance Determination (Sparse Bayesian Learning)

A scikit-learn-compatible regressor implementing Sparse Bayesian
Learning with retrospective Bayesian cross-validation for model
selection.  Class API:::

    from shapleyx.ard import RegressionARD

    ard = RegressionARD(n_iter=200, cv=True, cv_method='bayesian')
    ard.fit(X_train, y_train)
    y_pred = ard.predict(X_test)
    r2 = ard.score(X_test, y_test)

Reference
---------
Tipping, M. E., & Faul, A. C. (2003).  Fast marginal likelihood
maximisation for sparse Bayesian models.  *Proc. 9th Int. Workshop
on AI and Statistics*.

Bennett, F. (2026).  Bayesian Cross-Validation for Automatic
Relevance Determination in Sparse RS-HDMR Surrogate Construction.
*arXiv preprint* (forthcoming).
"""

from ._sbl import RegressionARD, update_precisions

__all__ = ["RegressionARD", "update_precisions"]
