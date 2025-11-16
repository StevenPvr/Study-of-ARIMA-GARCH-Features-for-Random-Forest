"""Initial variance estimation functions for EGARCH models.

This module provides different strategies for estimating initial variance
before starting the EGARCH recursion:

- sample: Sample variance (simple, numerically stable)
- mean_squared: Mean of squared residuals
- rolling: Rolling window variance (more robust to outliers)
- unconditional: Unconditional EGARCH variance (requires parameters)
"""

from __future__ import annotations

import numpy as np

from src.constants import GARCH_MIN_INIT_VAR


def estimate_initial_variance_sample(residuals: np.ndarray) -> float:
    """Estimate initial variance using sample variance.

    This is the simplest and most numerically stable method.

    Args:
        residuals: Residual series.

    Returns:
        Sample variance (at least GARCH_MIN_INIT_VAR).
    """
    residuals_arr = np.asarray(residuals, dtype=float).ravel()
    if residuals_arr.size == 0:
        return GARCH_MIN_INIT_VAR

    variance = float(np.var(residuals_arr, ddof=1))
    return max(variance, GARCH_MIN_INIT_VAR)


def estimate_initial_variance_mean_squared(residuals: np.ndarray) -> float:
    """Estimate initial variance using mean of squared residuals.

    Args:
        residuals: Residual series.

    Returns:
        Mean squared residual (at least GARCH_MIN_INIT_VAR).
    """
    residuals_arr = np.asarray(residuals, dtype=float).ravel()
    if residuals_arr.size == 0:
        return GARCH_MIN_INIT_VAR

    variance = float(np.mean(residuals_arr**2))
    return max(variance, GARCH_MIN_INIT_VAR)


def estimate_initial_variance_rolling(
    residuals: np.ndarray,
    window: int = 20,
) -> float:
    """Estimate initial variance using rolling window.

    More robust to outliers by using median of rolling variances.

    Args:
        residuals: Residual series.
        window: Rolling window size (default: 20).

    Returns:
        Median rolling variance (at least GARCH_MIN_INIT_VAR).
    """
    residuals_arr = np.asarray(residuals, dtype=float).ravel()
    n = residuals_arr.size

    if n < window:
        # Fall back to sample variance
        return estimate_initial_variance_sample(residuals_arr)

    # Compute rolling variances
    rolling_vars = []
    for i in range(n - window + 1):
        window_data = residuals_arr[i : i + window]
        rolling_vars.append(np.var(window_data, ddof=1))

    # Use median for robustness
    variance = float(np.median(rolling_vars))
    return max(variance, GARCH_MIN_INIT_VAR)


def estimate_initial_variance_unconditional(
    omega: float,
    beta: float | tuple[float, float],
) -> float:
    """Estimate initial variance using unconditional EGARCH variance.

    For EGARCH, the unconditional variance is:
        E[σ²] = exp(ω / (1 - Σβ))

    This requires estimated parameters, so it's only useful for warm starts.

    Args:
        omega: EGARCH omega parameter.
        beta: EGARCH beta parameter(s).

    Returns:
        Unconditional variance (at least GARCH_MIN_INIT_VAR).
    """
    # Sum beta parameters
    if isinstance(beta, tuple):
        beta_sum = sum(beta)
    else:
        beta_sum = float(beta)

    # Check stationarity
    if abs(beta_sum) >= 1.0:
        # Non-stationary: unconditional variance undefined
        return GARCH_MIN_INIT_VAR

    # Compute unconditional log-variance
    log_var_uncond = omega / (1.0 - beta_sum)

    # Exponential to get variance
    variance = float(np.exp(log_var_uncond))

    # Ensure valid
    if not np.isfinite(variance) or variance <= 0:
        return GARCH_MIN_INIT_VAR

    return max(variance, GARCH_MIN_INIT_VAR)


def estimate_initial_variance(
    residuals: np.ndarray,
    method: str = "sample",
    **kwargs,
) -> float:
    """Estimate initial variance using specified method.

    Args:
        residuals: Residual series.
        method: Estimation method ('sample', 'mean_squared', 'rolling', 'unconditional').
        **kwargs: Additional arguments passed to method function.
            - window: Rolling window size (for 'rolling')
            - omega, beta: EGARCH parameters (for 'unconditional')

    Returns:
        Initial variance estimate (at least GARCH_MIN_INIT_VAR).

    Raises:
        ValueError: If method is unknown or required kwargs missing.
    """
    method_lower = method.lower()

    if method_lower == "sample":
        return estimate_initial_variance_sample(residuals)

    if method_lower == "mean_squared":
        return estimate_initial_variance_mean_squared(residuals)

    if method_lower == "rolling":
        window = kwargs.get("window", 20)
        return estimate_initial_variance_rolling(residuals, window=window)

    if method_lower == "unconditional":
        omega = kwargs.get("omega")
        beta = kwargs.get("beta")
        if omega is None or beta is None:
            msg = "unconditional method requires 'omega' and 'beta' kwargs"
            raise ValueError(msg)
        return estimate_initial_variance_unconditional(omega, beta)

    msg = f"Unknown initialization method: {method}"
    raise ValueError(msg)


__all__ = [
    "estimate_initial_variance",
    "estimate_initial_variance_sample",
    "estimate_initial_variance_mean_squared",
    "estimate_initial_variance_rolling",
    "estimate_initial_variance_unconditional",
]
