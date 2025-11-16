"""EGARCH Maximum Likelihood Estimation (MLE).

This module provides MLE estimation for EGARCH models:
- Negative log-likelihood functions
- SLSQP optimization
- Parameter estimation for EGARCH(o,p)
- Support for Normal, Student-t, and Skew-t distributions

References:
- Nelson (1991): EGARCH specification
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np

from src.constants import (
    GARCH_ESTIMATION_EPS,
    GARCH_ESTIMATION_FTOL,
    GARCH_ESTIMATION_MAXITER,
    GARCH_ESTIMATION_PENALTY_VALUE,
)
from src.garch.garch_params.core import (
    compute_loglik_normal,
    compute_loglik_skewt,
    compute_loglik_student,
    egarch_variance,
    validate_beta,
    validate_residuals,
    validate_skewt_params,
    validate_student_params,
)
from src.garch.garch_params.estimation.cache import get_global_cache
from src.garch.garch_params.estimation.convergence import (
    ConvergenceResult,
    extract_convergence_info,
)
from src.garch.garch_params.estimation.initialization import (
    build_bounds,
    build_initial_params,
    build_result_dict,
    compute_initial_omega,
    extract_params_from_array,
)
from src.utils import get_logger

logger = get_logger(__name__)


def run_slsqp_optimizer(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
) -> Any:
    """Run SciPy SLSQP optimizer with proper settings.

    Args:
        objective: Objective function to minimize.
        x0: Initial parameter vector.
        bounds: Parameter bounds.

    Returns:
        Optimization result object.

    Raises:
        RuntimeError: If SciPy is not available.
    """
    try:
        from scipy.optimize import minimize  # type: ignore
    except ImportError as exc:
        msg = "SciPy required for MLE estimation"
        raise RuntimeError(msg) from exc

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Values in x were outside bounds during a minimize step",
            category=RuntimeWarning,
        )
        options = {
            "maxiter": GARCH_ESTIMATION_MAXITER,
            "ftol": GARCH_ESTIMATION_FTOL,
            "eps": GARCH_ESTIMATION_EPS,
        }
        return minimize(objective, x0=x0, method="SLSQP", bounds=bounds, options=options)


def _validate_beta_order_1(
    beta: float | tuple[float, float] | tuple[float, float, float],
) -> float | None:
    """Validate beta for GARCH order p=1."""
    if isinstance(beta, tuple):
        return None
    beta_val = float(beta)
    return beta_val if validate_beta(beta_val) else None


def _validate_beta_order_2(
    beta: float | tuple[float, float] | tuple[float, float, float],
) -> float | None:
    """Validate beta for GARCH order p=2."""
    if not isinstance(beta, tuple) or len(beta) != 2:
        return None
    beta1, beta2 = float(beta[0]), float(beta[1])
    return beta1 if (validate_beta(beta1) and validate_beta(beta2)) else None


def _validate_beta_order_3(
    beta: float | tuple[float, float] | tuple[float, float, float],
) -> float | None:
    """Validate beta for GARCH order p=3."""
    if not isinstance(beta, tuple) or len(beta) != 3:
        return None
    beta1, beta2, beta3 = float(beta[0]), float(beta[1]), float(beta[2])
    is_valid = validate_beta(beta1) and validate_beta(beta2) and validate_beta(beta3)
    return beta1 if is_valid else None


def _validate_beta_for_negloglik(
    beta: float | tuple[float, float] | tuple[float, float, float], p: int
) -> float | None:
    """Validate beta and return representative value.

    Args:
        beta: Beta parameter(s).
        p: GARCH order.

    Returns:
        Representative beta if valid, None otherwise.
    """
    validators = {
        1: _validate_beta_order_1,
        2: _validate_beta_order_2,
        3: _validate_beta_order_3,
    }

    validator = validators.get(p)
    return validator(beta) if validator else None


def _validate_dist_params(
    dist: str, beta_val: float, nu: float | None, lambda_skew: float | None
) -> bool:
    """Validate distribution-specific parameters.

    Args:
        dist: Distribution name.
        beta_val: Representative beta value.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter.

    Returns:
        True if valid.
    """
    if dist == "student" and nu is not None:
        return validate_student_params(beta_val, nu)
    if dist == "skewt" and nu is not None and lambda_skew is not None:
        return validate_skewt_params(beta_val, nu, lambda_skew)
    return True


def _compute_loglik_safe(
    residuals: np.ndarray,
    variances: np.ndarray,
    dist: str,
    nu: float | None,
    lambda_skew: float | None,
) -> float | None:
    """Compute log-likelihood, returning None if invalid.

    Args:
        residuals: Residual series.
        variances: Variance series.
        dist: Distribution name.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter.

    Returns:
        Log-likelihood if valid, None otherwise.
    """
    if dist == "normal":
        return compute_loglik_normal(residuals, variances)
    if dist == "student":
        if nu is None:
            return None
        return compute_loglik_student(residuals, variances, nu)
    # skewt
    if nu is None or lambda_skew is None:
        return None
    return compute_loglik_skewt(residuals, variances, nu, lambda_skew)


def create_negloglik_function(
    residuals: np.ndarray, o: int, p: int, dist: str
) -> Callable[[np.ndarray], float]:
    """Create negative log-likelihood function for optimization.

    Args:
        residuals: Residual series.
        o: ARCH order.
        p: GARCH order.
        dist: Distribution name.

    Returns:
        Negative log-likelihood function.
    """

    def negloglik(params: np.ndarray) -> float:
        """Compute negative log-likelihood.

        Returns large penalty for invalid parameters.
        """
        try:
            omega, alpha, gamma, beta, nu, lambda_skew = extract_params_from_array(
                params, o, p, dist
            )

            # Validate beta
            beta_val = _validate_beta_for_negloglik(beta, p)
            if beta_val is None:
                return GARCH_ESTIMATION_PENALTY_VALUE

            # Validate distribution parameters
            if not _validate_dist_params(dist, beta_val, nu, lambda_skew):
                return GARCH_ESTIMATION_PENALTY_VALUE

            # Compute variance path
            variances = egarch_variance(
                residuals,
                omega,
                alpha,
                gamma,
                beta,
                dist=dist,
                nu=nu,
                lambda_skew=lambda_skew,
                o=o,
                p=p,
            )
            if not np.all(np.isfinite(variances)) or np.any(variances <= 0):
                return GARCH_ESTIMATION_PENALTY_VALUE

            # Compute log-likelihood
            loglik = _compute_loglik_safe(residuals, variances, dist, nu, lambda_skew)
            if loglik is None:
                return GARCH_ESTIMATION_PENALTY_VALUE

            return -float(loglik)

        except Exception:
            return GARCH_ESTIMATION_PENALTY_VALUE

    return negloglik


def _log_estimation_start(o: int, p: int, dist: str) -> None:
    """Log estimation start message.

    Args:
        o: ARCH order.
        p: GARCH order.
        dist: Distribution name.
    """
    logger.info("Starting EGARCH(%d,%d) MLE estimation: dist=%s", o, p, dist)


def _log_estimation_result(
    o: int, p: int, result_dict: dict[str, float], convergence: ConvergenceResult
) -> None:
    """Log estimation result.

    Args:
        o: ARCH order.
        p: GARCH order.
        result_dict: Result dictionary.
        convergence: Convergence result.
    """
    logger.info(
        "EGARCH(%d,%d) MLE completed: converged=%s, loglik=%.2f, iterations=%s",
        o,
        p,
        convergence.converged,
        convergence.final_loglik,
        convergence.n_iterations if convergence.n_iterations is not None else "N/A",
    )


def estimate_egarch_mle(
    residuals: np.ndarray,
    *,
    o: int = 1,
    p: int = 1,
    dist: str = "normal",
    x0: np.ndarray | None = None,
    use_cache: bool = True,
) -> tuple[dict[str, float], ConvergenceResult]:
    """Estimate EGARCH(o,p) parameters via Maximum Likelihood Estimation.

    Uses conditional MLE with SLSQP optimizer. Supports EGARCH(o,p) with
    o, p ∈ {1,2,3} and distributions: Normal, Student-t, Skew-t.

    Args:
        residuals: Residual series from mean model (SARIMA).
        o: ARCH order (1, 2, or 3).
        p: GARCH order (1, 2, or 3).
        dist: Distribution name ('normal', 'student', 'skewt').
        x0: Optional initial parameter vector.
        use_cache: Whether to use MLE cache (default True).

    Returns:
        Tuple of (parameter_dict, convergence_result).

    Raises:
        ValueError: If orders are invalid or insufficient data.
    """
    # Handle invalid distribution explicitly by returning penalty values
    allowed_dists = {"normal", "student", "skewt"}
    if dist.lower() not in allowed_dists:
        penalty = -float(GARCH_ESTIMATION_PENALTY_VALUE)
        logger.warning("Invalid distribution '%s'; returning penalty values", dist)
        penalty_result = {
            "omega": 0.0,
            "alpha": 0.0,
            "gamma": 0.0,
            "beta": 0.0,
            "loglik": penalty,
            "converged": False,
        }
        conv = ConvergenceResult(
            converged=False,
            n_iterations=None,
            final_loglik=penalty,
            message=f"invalid dist: {dist}",
        )
        return penalty_result, conv

    # Check cache first if enabled
    if use_cache:
        cache = get_global_cache()
        cached_result = cache.get(residuals, o, p, dist)
        if cached_result is not None:
            params, convergence = cached_result
            logger.debug("Using cached MLE result for o=%d, p=%d, dist=%s", o, p, dist)
            return params, convergence

    # Validate inputs
    if o not in (1, 2, 3):
        msg = f"ARCH order o={o} not supported (only o=1, 2, or 3)"
        raise ValueError(msg)
    if p not in (1, 2, 3):
        msg = f"GARCH order p={p} not supported (only p=1, 2, or 3)"
        raise ValueError(msg)

    residuals_arr = validate_residuals(residuals)
    variance_est = float(np.var(residuals_arr))

    # Build initial parameters and bounds
    omega_init = compute_initial_omega(variance_est)
    if x0 is None:
        x0 = build_initial_params(omega_init, o, p, dist)
    bounds = build_bounds(o, p, dist)

    # Create objective function
    negloglik = create_negloglik_function(residuals_arr, o, p, dist)

    # Run optimization
    _log_estimation_start(o, p, dist)
    opt_result = run_slsqp_optimizer(negloglik, x0, bounds)

    # Extract results
    convergence = extract_convergence_info(opt_result)
    result_dict = build_result_dict(
        opt_result.x, o, p, dist, convergence.final_loglik, convergence.converged
    )

    _log_estimation_result(o, p, result_dict, convergence)

    if not convergence.converged:
        logger.warning("EGARCH(%d,%d) MLE failed to converge: %s", o, p, convergence.message)

    # Store in cache if enabled
    if use_cache:
        cache = get_global_cache()
        cache.put(residuals, o, p, dist, result_dict, convergence)

    return result_dict, convergence
