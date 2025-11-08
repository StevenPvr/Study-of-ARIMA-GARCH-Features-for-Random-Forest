"""EGARCH(1,1) parameter estimation via conditional MLE.

Implements conditional maximum likelihood estimation:
- Assumes parametric distribution for innovations zt (Normal, Student-t, Skew-t)
- Maximizes conditional log-likelihood by recursing conditional variance σt²
- Numerical optimization performed by software libraries

Intended for ARIMA residuals (mean ~ 0).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.garch.garch_params.utils import (
    compute_normal_loglikelihood,
    compute_skewt_loglikelihood,
    compute_student_loglikelihood,
    compute_variance_step,
    egarch_finalize,
    egarch_kappa,
    egarch_setup,
    initialize_variance,
    optimize_with_fallback,
    validate_beta,
    validate_series,
    validate_skewt_params,
    validate_student_params,
)
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------- EGARCH(1,1) ----------------------


def egarch11_variance(
    e: np.ndarray,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    *,
    dist: str = "normal",
    nu: float | None = None,
    lambda_skew: float | None = None,
    init: float | None = None,
) -> np.ndarray:
    """Compute EGARCH(1,1) variance path.

    log(sigma_t^2) = omega + beta * log(sigma_{t-1}^2)
    + alpha * (|z_{t-1}| - kappa) + gamma * z_{t-1}
    where z_{t-1} = e_{t-1}/sigma_{t-1} and kappa = E|Z|
    under the chosen innovation distribution.
    """
    ee = np.asarray(e, dtype=float).ravel()
    n = ee.size
    s2 = np.empty(n, dtype=float)
    s2[0] = initialize_variance(ee, init)
    kappa = egarch_kappa(dist, nu, lambda_skew)
    for t in range(1, n):
        s2[t] = compute_variance_step(
            ee[t - 1], s2[t - 1], omega, alpha, gamma, beta, kappa
        )
        if not np.isfinite(s2[t]) or s2[t] <= 0:
            return np.full(n, np.nan)
    return s2


def _negloglik_egarch_normal(params: np.ndarray, e: np.ndarray) -> float:
    """Compute negative log-likelihood for EGARCH(1,1) with Normal innovations.

    Returns large penalty (1e50) for invalid parameters or numerical issues.
    """
    omega, alpha, gamma, beta = params
    if not validate_beta(beta):
        return 1e50
    s2 = egarch11_variance(e, omega, alpha, gamma, beta, dist="normal", nu=None)
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return 1e50
    try:
        ll = compute_normal_loglikelihood(e, s2)
        return -ll
    except Exception:
        return 1e50


def _negloglik_egarch_student(params: np.ndarray, e: np.ndarray) -> float:
    """Compute negative log-likelihood for EGARCH(1,1) with Student-t innovations.

    Returns large penalty (1e50) for invalid parameters or numerical issues.
    """
    omega, alpha, gamma, beta, nu = params
    if not validate_student_params(beta, nu):
        return 1e50
    s2 = egarch11_variance(e, omega, alpha, gamma, beta, dist="student", nu=nu)
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return 1e50
    try:
        ll = compute_student_loglikelihood(e, s2, nu)
        return -float(ll)
    except Exception:
        return 1e50


def _negloglik_egarch_skewt(params: np.ndarray, e: np.ndarray) -> float:
    """Compute negative log-likelihood for EGARCH(1,1) with Skew-t innovations.

    Returns large penalty (1e50) for invalid parameters or numerical issues.
    """
    omega, alpha, gamma, beta, nu, lambda_skew = params
    if not validate_skewt_params(beta, nu, lambda_skew):
        return 1e50
    s2 = egarch11_variance(
        e, omega, alpha, gamma, beta, dist="skewt", nu=nu, lambda_skew=lambda_skew
    )
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return 1e50
    try:
        ll = compute_skewt_loglikelihood(e, s2, nu, lambda_skew)
        return -float(ll)
    except Exception:
        return 1e50


def estimate_egarch_mle(
    residuals: np.ndarray,
    *,
    dist: str = "normal",
    x0: Iterable[float] | None = None,
) -> dict[str, float]:
    """Estimate EGARCH(1,1) parameters via conditional maximum likelihood.

    Assumes parametric distribution for innovations zt (Normal, Student-t, or Skew-t)
    and maximizes the conditional log-likelihood by recursing the conditional
    variance σt². Numerical optimization is performed using SciPy.

    Args:
        residuals: Residual series εt from mean model (SARIMA).
        dist: Distribution for innovations: 'normal', 'student', or 'skewt'.
        x0: Optional initial parameter vector.

    Returns:
        Dictionary with estimated parameters (omega, alpha, gamma, beta, nu, lambda)
        and optimization results (loglik, converged).
    """
    e = validate_series(residuals)
    v = float(np.var(e))
    x0_arr, bounds, fun, fallback_x0 = egarch_setup(
        e, dist, x0, v, _negloglik_egarch_normal, _negloglik_egarch_student, _negloglik_egarch_skewt
    )
    logger.info("Starting EGARCH(1,1) MLE: dist=%s", dist)
    res = optimize_with_fallback(fun, x0_arr, fallback_x0, bounds)
    out = egarch_finalize(dist, res)
    extra_params = ""
    if "nu" in out:
        extra_params = f", nu={out['nu']:.2f}"
    if "lambda" in out:
        extra_params += f", lambda={out['lambda']:.4f}"
    logger.info(
        "Finished EGARCH MLE (success=%s): omega=%.6g, alpha=%.4f, gamma=%.4f, beta=%.4f%s",
        out["converged"],
        out["omega"],
        out["alpha"],
        out["gamma"],
        out["beta"],
        extra_params,
    )
    return out


__all__ = [
    "egarch11_variance",
    "estimate_egarch_mle",
]
