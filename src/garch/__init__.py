"""GARCH package wrapper.

This package exposes submodules under `src.garch.*`. To keep imports
lightweight and avoid optional dependencies at import time, no heavy
re-exports are performed here. Import from the specific subpackages, e.g.:

    from src.garch.garch_diagnostic import diagnostics
    from src.garch.garch_params import estimation

New academic modules:
    from src.garch.benchmark.realized_volatility import compute_realized_measures

"""

from __future__ import annotations

# Realized volatility estimators using HLOC prices
from src.garch.benchmark.realized_volatility import (
    compare_realized_estimators,
    compute_realized_measures,
    efficiency_ratio,
    garman_klass_estimator,
    parkinson_estimator,
    realized_variance_returns,
    rogers_satchell_estimator,
    yang_zhang_estimator,
)

__all__ = [
    "compare_realized_estimators",
    "compute_realized_measures",
    "efficiency_ratio",
    "garman_klass_estimator",
    "parkinson_estimator",
    "realized_variance_returns",
    "rogers_satchell_estimator",
    "yang_zhang_estimator",
]
