"""
Backward-compatibility shim — delegates to `shapleyx.ard._sbl`.

The canonical implementation of `RegressionARD` and `update_precisions`
now lives at `shapleyx.ard._sbl`.  This module is kept so that existing
code with ``from shapleyx.utilities.ARD import RegressionARD`` continues
to work.
"""

from ..ard._sbl import RegressionARD, update_precisions

__all__ = ["RegressionARD", "update_precisions"]
