"""Diagnostic tests for EGARCH model quality during optimization.

This module provides functions to compute diagnostic penalties based on
statistical tests of standardized residuals. These penalties are used
in the composite objective function to ensure model quality.
"""

from __future__ import annotations

import numpy as np

from src.constants import (
    GARCH_OPTIMIZATION_ARCH_LM_LAGS,
    GARCH_OPTIMIZATION_DIAGNOSTIC_PVALUE_THRESHOLD,
)
from src.garch.garch_params.core import egarch_variance
from src.utils import get_logger

logger = get_logger(__name__)


def _compute_arch_lm_statistic(residuals_squared: np.ndarray, lags: int) -> tuple[float, float]:
    """Compute ARCH-LM test statistic and p-value.

    Args:
        residuals_squared: Squared standardized residuals.
        lags: Number of lags for ARCH-LM test.

    Returns:
        Tuple of (lm_statistic, p_value).
    """
    try:
        from scipy.stats import chi2  # type: ignore
    except ImportError:
        logger.warning("SciPy not available for ARCH-LM test, returning penalty")
        return float("inf"), 0.0

    n = len(residuals_squared)
    if n < lags + 1:
        return float("inf"), 0.0

    # Regress squared residuals on lagged squared residuals
    # y_t = c + b1*y_{t-1} + ... + bp*y_{t-p} + e_t
    y = residuals_squared[lags:]
    X = np.ones((n - lags, lags + 1))
    for i in range(lags):
        X[:, i + 1] = residuals_squared[lags - i - 1 : n - i - 1]

    # Compute R-squared from regression
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_fitted = X @ beta
        ss_res = np.sum((y - y_fitted) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # LM statistic = n * R^2 ~ chi2(lags)
        lm_stat = float((n - lags) * r_squared)
        p_value = float(1.0 - chi2.cdf(lm_stat, df=lags))

        return lm_stat, p_value
    except np.linalg.LinAlgError:
        return float("inf"), 0.0


def _standardize_residuals(
    residuals: np.ndarray,
    sigma2: np.ndarray,
) -> np.ndarray:
    """Standardize residuals by conditional standard deviation.

    Args:
        residuals: Raw residuals.
        sigma2: Conditional variance.

    Returns:
        Standardized residuals.
    """
    # Filter valid values
    mask = np.isfinite(residuals) & np.isfinite(sigma2) & (sigma2 > 0)
    if not np.any(mask):
        return np.array([])

    std_residuals = residuals[mask] / np.sqrt(sigma2[mask])
    return std_residuals


def compute_diagnostic_penalty(
    residuals: np.ndarray,
    params: dict[str, float],
    o: int,
    p: int,
    dist: str,
) -> float:
    """Compute diagnostic penalty for model quality.

    The penalty is based on ARCH-LM test for remaining ARCH effects
    in standardized residuals. Higher penalties indicate worse model fit.

    Args:
        residuals: Raw residuals.
        params: Estimated EGARCH parameters.
        o: ARCH order.
        p: GARCH order.
        dist: Distribution name.

    Returns:
        Diagnostic penalty value (0 = perfect, higher = worse).
    """
    # Extract parameters
    omega = params.get("omega", 0.0)
    alpha = params.get("alpha", 0.0)
    gamma = params.get("gamma", 0.0)
    beta = params.get("beta", 0.0)
    nu = params.get("nu")
    lambda_skew = params.get("lambda")

    # Compute conditional variance
    try:
        sigma2 = egarch_variance(
            residuals,
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            dist=dist,
            nu=nu,
            lambda_skew=lambda_skew,
            init=None,
            o=o,
            p=p,
        )
    except Exception as ex:
        logger.debug("Failed to compute variance for diagnostics: %s", ex)
        return 1.0  # Max penalty

    # Standardize residuals
    std_residuals = _standardize_residuals(residuals, sigma2)
    if len(std_residuals) < GARCH_OPTIMIZATION_ARCH_LM_LAGS + 1:
        return 1.0  # Max penalty

    # Compute ARCH-LM test on standardized residuals
    std_residuals_squared = std_residuals**2
    lm_stat, p_value = _compute_arch_lm_statistic(
        std_residuals_squared, GARCH_OPTIMIZATION_ARCH_LM_LAGS
    )

    # Penalty based on p-value:
    # - If p > threshold: no remaining ARCH effects, penalty = 0
    # - If p <= threshold: remaining ARCH effects, penalty proportional to (1 - p)
    if p_value > GARCH_OPTIMIZATION_DIAGNOSTIC_PVALUE_THRESHOLD:
        penalty = 0.0
    else:
        # Scale penalty from 0 to 1 based on how far below threshold
        penalty = 1.0 - (p_value / GARCH_OPTIMIZATION_DIAGNOSTIC_PVALUE_THRESHOLD)

    logger.debug(
        "Diagnostic: ARCH-LM p-value=%.4f, penalty=%.4f",
        p_value,
        penalty,
    )

    return float(penalty)


def compute_aic_penalty(n_obs: int, loglik: float, n_params: int) -> float:
    """Compute AIC (Akaike Information Criterion).

    Args:
        n_obs: Number of observations.
        loglik: Log-likelihood value.
        n_params: Number of model parameters.

    Returns:
        AIC value (lower is better).
    """
    # AIC = -2*log(L) + 2*k
    # But we have negative log-likelihood, so: AIC = 2*(-loglik) + 2*k
    aic = -2.0 * loglik + 2.0 * n_params
    return float(aic)


def normalize_aic_penalty(aic: float, n_obs: int) -> float:
    """Normalize AIC to [0, 1] range for composite objective.

    Args:
        aic: Raw AIC value.
        n_obs: Number of observations.

    Returns:
        Normalized AIC penalty in [0, 1].
    """
    # Normalize by number of observations to make comparable across datasets
    # Use sigmoid-like transformation to bound to [0, 1]
    normalized = aic / n_obs
    # Apply tanh for soft bounding
    penalty = float(np.tanh(normalized / 10.0))  # Scale factor of 10 for reasonable range
    return max(0.0, min(1.0, penalty))


__all__ = [
    "compute_diagnostic_penalty",
    "compute_aic_penalty",
    "normalize_aic_penalty",
]
