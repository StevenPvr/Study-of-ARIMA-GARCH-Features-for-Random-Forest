"""Statsmodels utilities for ARIMA/SARIMA models.

Provides utilities for managing statsmodels-specific concerns such as warning
suppression during model fitting. Used across ARIMA/SARIMA pipelines to maintain
clean, warning-free execution while preserving important user-facing diagnostics.
"""

from __future__ import annotations

import warnings

__all__ = ["suppress_statsmodels_warnings"]


def suppress_statsmodels_warnings() -> None:
    """Suppress common statsmodels warnings for ARIMA/SARIMA models.

    Suppresses frequent but uninformative warnings emitted by statsmodels during
    ARIMA/SARIMA model fitting. These warnings typically relate to:
    - Lack of supported date index (when using integer indices)
    - Date index frequency information
    - General UserWarnings from the statsmodels module

    This function should be called at the beginning of functions that fit SARIMA
    models to avoid cluttering logs with repetitive warnings that do not indicate
    actual problems. Important errors and convergence issues will still be raised
    as exceptions.

    Warning categories suppressed:
        - UserWarning from statsmodels module
        - No supported index available warnings
        - Date index has been provided warnings
        - Frequency information warnings

    Returns:
        None. This function has side effects on the warnings filter.

    Examples:
        Basic usage in SARIMA fitting function:
        >>> suppress_statsmodels_warnings()
        >>> # Now fit SARIMA model without index/frequency warnings
        >>> model = SARIMAX(data, order=(1,1,1), seasonal_order=(0,0,0,0))
        >>> results = model.fit(disp=False)

        Use at module initialization for entire pipeline:
        >>> def optimize_sarima_parameters(data):
        ...     suppress_statsmodels_warnings()
        ...     # Multiple model fits follow without repetitive warnings
        ...     for params in param_grid:
        ...         model = SARIMAX(data, order=params['order'])
        ...         model.fit(disp=False)

    Notes:
        - This function modifies the global warnings filter
        - Suppressed warnings are common and typically not actionable
        - Serious issues (convergence failures, invalid parameters) are still
          raised as exceptions and will not be suppressed
        - Call this function once per module/function that fits SARIMA models

    Usage in project:
        - src/arima/evaluation_arima/evaluation_arima.py
        - src/arima/optimisation_arima/main.py
        - src/arima/training_arima/main.py
        - Any module fitting SARIMAX models
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
    warnings.filterwarnings("ignore", message=".*No supported index is available.*")
    warnings.filterwarnings("ignore", message=".*date index has been provided.*")
    warnings.filterwarnings("ignore", message=".*frequency information.*")
