"""Statistical tests for model comparison in volatility forecasting.

This module provides statistical tests for comparing forecast accuracy
between different volatility models. The main test implemented is the
Diebold-Mariano (1995) test for equal predictive accuracy.

Academic References:
    - Diebold & Mariano (1995): "Comparing predictive accuracy"
      Journal of Business & Economic Statistics, 13(3), 253-263.
    - Newey & West (1987): "A simple, positive semi-definite,
      heteroskedasticity and autocorrelation consistent covariance matrix"
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.utils import get_logger

logger = get_logger(__name__)


def _calculate_dm_variance_with_autocorrelation(d: np.ndarray, h: int) -> float:
    """Calculate variance with autocorrelation correction for Diebold-Mariano test.

    Uses Newey-West HAC variance estimator:
    V = gamma_0 + 2 * sum_{k=1}^{h-1} gamma_k

    Args:
        d: Loss differential array d_t = L_t(model1) - L_t(model2).
        h: Forecast horizon.

    Returns:
        Variance of the mean with autocorrelation correction.
    """
    n = len(d)
    d_mean = float(np.mean(d))
    d_centered = d - d_mean

    # Variance (gamma_0)
    gamma_0 = float(np.var(d_centered, ddof=1))

    if h <= 1:
        return gamma_0 / n

    # Calculate autocovariances for k = 1 to h-1
    gamma_sum = 0.0
    for k in range(1, min(h, n)):
        if n - k > 0:
            gamma_k = float(np.mean(d_centered[k:] * d_centered[:-k]))
            gamma_sum += gamma_k

    variance_hac = gamma_0 + 2.0 * gamma_sum
    return variance_hac / n


def diebold_mariano_test(
    e: np.ndarray,
    sigma2_model1: np.ndarray,
    sigma2_model2: np.ndarray,
    *,
    loss_function: str = "qlike",
    h: int = 1,
) -> dict[str, float | str | int]:
    """Diebold-Mariano test for comparing volatility forecast accuracy.

    Tests H0: Equal predictive accuracy between two volatility models.
    H1: Models have different predictive accuracy.

    Specialized for volatility forecasting with losses computed on variance forecasts:
    - QLIKE: log(σ²) + ε²/σ²
    - MSE: (ε² - σ²)²
    - MAE: |ε² - σ²|

    Args:
        e: Realized residuals (aligned with forecasts).
        sigma2_model1: Variance forecasts from model 1.
        sigma2_model2: Variance forecasts from model 2.
        loss_function: Type of loss ("qlike", "mse", "mae").
        h: Forecast horizon (default: 1 for one-step-ahead).

    Returns:
        Dict with dm_statistic, p_value, better_model, mean_loss_diff, n.

    Note:
        - Positive DM statistic → Model 2 is better (lower loss)
        - Negative DM statistic → Model 1 is better (lower loss)
        - p < 0.05 → Significant difference in predictive accuracy

    Example:
        >>> e = np.random.randn(100)
        >>> sigma2_1 = np.abs(np.random.randn(100)) + 0.1
        >>> sigma2_2 = np.abs(np.random.randn(100)) + 0.1
        >>> result = diebold_mariano_test(e, sigma2_1, sigma2_2, loss_function="qlike")
        >>> print(f"Better model: {result['better_model']}, p-value: {result['p_value']:.4f}")

    Reference:
        Diebold, F. X., & Mariano, R. S. (1995). "Comparing predictive accuracy."
        Journal of Business & Economic Statistics, 13(3), 253-263.
    """
    e = np.asarray(e, dtype=float).ravel()
    s2_1 = np.asarray(sigma2_model1, dtype=float).ravel()
    s2_2 = np.asarray(sigma2_model2, dtype=float).ravel()

    # Filter finite and positive variances
    m = np.isfinite(e) & np.isfinite(s2_1) & np.isfinite(s2_2) & (s2_1 > 0) & (s2_2 > 0)
    if not np.any(m):
        return {
            "dm_statistic": float("nan"),
            "p_value": float("nan"),
            "better_model": "unknown",
            "mean_loss_diff": float("nan"),
            "n": 0,
        }

    e = e[m]
    s2_1 = s2_1[m]
    s2_2 = s2_2[m]
    n = len(e)

    # Compute loss differentials
    if loss_function == "qlike":
        loss1 = np.log(s2_1) + (e**2) / s2_1
        loss2 = np.log(s2_2) + (e**2) / s2_2
    elif loss_function == "mse":
        loss1 = (e**2 - s2_1) ** 2
        loss2 = (e**2 - s2_2) ** 2
    elif loss_function == "mae":
        loss1 = np.abs(e**2 - s2_1)
        loss2 = np.abs(e**2 - s2_2)
    else:
        msg = f"Unknown loss function: {loss_function}"
        raise ValueError(msg)

    # Loss differential: positive means model 2 is better
    d = loss1 - loss2
    d_mean = float(np.mean(d))

    # HAC variance estimation
    variance = _calculate_dm_variance_with_autocorrelation(d, h)

    # DM statistic
    if variance <= 0.0 or not np.isfinite(variance):
        return {
            "dm_statistic": 0.0,
            "p_value": 1.0,
            "better_model": "equal",
            "mean_loss_diff": d_mean,
            "n": n,
        }

    dm_stat = float(d_mean / np.sqrt(variance))

    # P-value (two-tailed)
    if n > 1000:
        p_value = float(2 * (1 - stats.norm.cdf(np.abs(dm_stat))))
    else:
        p_value = float(2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1)))

    # Interpretation
    if dm_stat > 0:
        better_model = "model_2"
    elif dm_stat < 0:
        better_model = "model_1"
    else:
        better_model = "equal"

    return {
        "dm_statistic": float(dm_stat),
        "p_value": float(p_value),
        "better_model": better_model,
        "mean_loss_diff": d_mean,
        "n": n,
        "loss_function": loss_function,
    }


__all__ = [
    "diebold_mariano_test",
]
