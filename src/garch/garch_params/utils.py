"""Utility functions for GARCH parameter estimation.

Contains helper functions for:
- Series validation
- Variance initialization and computation
- Distribution parameter validation
- Log-likelihood computations
- Optimization utilities
- Data loading and preparation
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from src.constants import (
    GARCH_DATASET_FILE,
    GARCH_MIN_INIT_VAR,
    GARCH_SKEWT_LAMBDA_INIT,
    GARCH_SKEWT_LAMBDA_MAX,
    GARCH_SKEWT_LAMBDA_MIN,
    GARCH_STUDENT_NU_INIT,
    GARCH_STUDENT_NU_MAX,
    GARCH_STUDENT_NU_MIN,
)
from src.garch.structure_garch.utils import load_garch_dataset, prepare_residuals
from src.utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# Series Validation Utilities
# ============================================================================


def validate_series(e: np.ndarray) -> np.ndarray:
    """Return residual series as 1D float array and validate length.

    Raises ValueError if fewer than 10 observations are provided.

    Args:
        e: Residual series to validate.

    Returns:
        Validated 1D float array.
    """
    arr = np.asarray(e, dtype=float).ravel()
    if arr.size < 10:
        msg = "Need at least 10 observations to estimate EGARCH(1,1)."
        raise ValueError(msg)
    return arr


# ============================================================================
# EGARCH Kappa Computation Utilities
# ============================================================================


def _compute_student_kappa(nu: float) -> float:
    """Compute Student-t kappa value.

    Args:
        nu: Degrees of freedom.

    Returns:
        Kappa value for Student-t.
    """
    from scipy.special import gammaln  # type: ignore

    ln_num = 0.5 * np.log(max(nu - 2.0, 1e-12)) + gammaln(0.5 * (nu - 1.0))
    ln_den = 0.5 * np.log(np.pi) + gammaln(0.5 * nu)
    return float(np.exp(ln_num - ln_den))


def _compute_skewt_kappa(nu: float, lambda_skew: float) -> float:
    """Compute Skew-t kappa value.

    Args:
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter.

    Returns:
        Kappa value for Skew-t.
    """
    nu_val = float(nu)
    lambda_val = float(lambda_skew)
    kappa_t = _compute_student_kappa(nu_val)
    # Adjust for skewness (lambda affects asymmetry)
    # When lambda < 0 (left skew), E[|Z|] tends to be slightly higher
    # Simple approximation: kappa_skewt ≈ kappa_t * (1 + 0.1 * |lambda|)
    kappa_adj = kappa_t * (1.0 + 0.1 * abs(lambda_val))
    return float(kappa_adj)


def _try_compute_skewt_kappa(nu: float, lambda_skew: float) -> float | None:
    """Try to compute Skew-t kappa, return None on failure."""
    try:
        return _compute_skewt_kappa(nu, lambda_skew)
    except Exception as ex:
        logger.debug("Falling back to Student-t kappa; scipy unavailable or failed: %s", ex)
        return None


def _try_compute_student_kappa(nu: float) -> float | None:
    """Try to compute Student-t kappa, return None on failure."""
    try:
        return _compute_student_kappa(nu)
    except Exception as ex:
        logger.debug("Falling back to Normal kappa; scipy unavailable or failed: %s", ex)
        return None


def _is_valid_skewt_params(nu: float | None, lambda_skew: float | None) -> bool:
    """Check if parameters are valid for Skew-t."""
    return nu is not None and lambda_skew is not None and nu > 2.0


def _is_valid_student_params(nu: float | None) -> bool:
    """Check if parameters are valid for Student-t."""
    return nu is not None and nu > 2.0


def _compute_skewt_kappa_safe(nu: float, lambda_skew: float) -> float | None:
    """Safely compute Skew-t kappa or return None."""
    if not _is_valid_skewt_params(nu, lambda_skew):
        return None
    return _try_compute_skewt_kappa(nu, lambda_skew)


def _compute_student_kappa_safe(nu: float) -> float | None:
    """Safely compute Student-t kappa or return None."""
    if not _is_valid_student_params(nu):
        return None
    return _try_compute_student_kappa(nu)


def _get_skewt_kappa(nu: float | None, lambda_skew: float | None) -> float | None:
    """Get Skew-t kappa if parameters are valid."""
    if nu is not None and lambda_skew is not None:
        return _compute_skewt_kappa_safe(nu, lambda_skew)
    return None


def _get_student_kappa(nu: float | None) -> float | None:
    """Get Student-t kappa if parameters are valid."""
    if nu is not None:
        return _compute_student_kappa_safe(nu)
    return None


def egarch_kappa(dist: str, nu: float | None, lambda_skew: float | None = None) -> float:
    """Return E[|Z|] for standardized innovations used in EGARCH.

    For Normal: sqrt(2/pi)
    For Student-t (variance=1 standardized): sqrt(nu-2) * Gamma((nu-1)/2) / (sqrt(pi) * Gamma(nu/2))
    For Skew-t (Hansen): computed numerically or approximated

    Args:
        dist: Distribution name ('normal', 'student', or 'skewt').
        nu: Degrees of freedom (for Student-t/Skew-t).
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
        Kappa value E[|Z|].
    """
    dist_l = dist.lower()
    if dist_l == "skewt":
        kappa = _get_skewt_kappa(nu, lambda_skew)
        if kappa is not None:
            return kappa
    if dist_l == "student":
        kappa = _get_student_kappa(nu)
        if kappa is not None:
            return kappa
    # Default to Normal constant
    return float(np.sqrt(2.0 / np.pi))


# ============================================================================
# Variance Computation Utilities
# ============================================================================


def initialize_variance(ee: np.ndarray, init: float | None) -> float:
    """Initialize variance for EGARCH recursion.

    Args:
        ee: Residual series.
        init: Optional initial variance value.

    Returns:
        Initial variance value.
    """
    if init is not None and init > 0:
        return float(init)
    v = float(np.var(ee))
    return max(v, GARCH_MIN_INIT_VAR)


def compute_variance_step(
    ee: np.ndarray,
    s2_prev: float,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    kappa: float,
) -> float:
    """Compute next variance step in EGARCH recursion.

    Args:
        ee: Previous residual value.
        s2_prev: Previous variance.
        omega: EGARCH omega parameter.
        alpha: EGARCH alpha parameter.
        gamma: EGARCH gamma parameter.
        beta: EGARCH beta parameter.
        kappa: E[|Z|] constant.

    Returns:
        Next variance value.
    """
    z_prev = float(ee / np.sqrt(max(s2_prev, GARCH_MIN_INIT_VAR)))
    ln_next = (
        omega
        + beta * np.log(max(s2_prev, GARCH_MIN_INIT_VAR))
        + alpha * (abs(z_prev) - kappa)
        + gamma * z_prev
    )
    # Clip ln_next to prevent overflow in exp (ln(700) ≈ 6.55 is safe)
    ln_next_clipped = min(ln_next, 700.0)
    return float(np.exp(ln_next_clipped))


# ============================================================================
# Parameter Validation Utilities
# ============================================================================


def validate_beta(beta: float) -> bool:
    """Validate beta parameter for stationarity.

    Args:
        beta: EGARCH beta parameter.

    Returns:
        True if parameter is valid for stationarity.
    """
    return -0.999 < beta < 0.999


def validate_student_params(beta: float, nu: float) -> bool:
    """Validate Student-t distribution parameters.

    Args:
        beta: EGARCH beta parameter.
        nu: Degrees of freedom.

    Returns:
        True if parameters are valid.
    """
    return (-0.999 < beta < 0.999) and nu > 2.0


def validate_skewt_params(beta: float, nu: float, lambda_skew: float) -> bool:
    """Validate Skew-t distribution parameters.

    Args:
        beta: EGARCH beta parameter.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter.

    Returns:
        True if parameters are valid.
    """
    return (
        (-0.999 < beta < 0.999)
        and nu > 2.0
        and GARCH_SKEWT_LAMBDA_MIN < lambda_skew < GARCH_SKEWT_LAMBDA_MAX
    )


# ============================================================================
# Log-Likelihood Computation Utilities
# ============================================================================


def compute_normal_loglikelihood(e: np.ndarray, s2: np.ndarray) -> float:
    """Compute Normal log-likelihood given variance.

    Args:
        e: Residual series.
        s2: Variance series.

    Returns:
        Log-likelihood value.

    Raises:
        ValueError: If overflow occurs in computation.
    """
    with np.errstate(divide="ignore", over="ignore"):
        z2 = (e**2) / s2
    if not np.all(np.isfinite(z2)):
        raise ValueError("Overflow in z2 computation")
    ll = -0.5 * (np.log(2.0 * np.pi) + np.log(s2) + z2).sum()
    return float(ll)


def compute_student_loglikelihood(e: np.ndarray, s2: np.ndarray, nu: float) -> float:
    """Compute Student-t log-likelihood given variance and degrees of freedom.

    Args:
        e: Residual series.
        s2: Variance series.
        nu: Degrees of freedom.

    Returns:
        Log-likelihood value.

    Raises:
        ValueError: If overflow occurs in computation.
    """
    from scipy.special import gammaln  # type: ignore

    c_log = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * (
        np.log(np.pi) + np.log(nu - 2.0)
    )
    # Suppress overflow warnings but keep exact calculations
    # If real overflow occurs (inf values), we'll detect it below
    with np.errstate(divide="ignore", over="ignore"):
        z2_scaled = (e**2) / (s2 * (nu - 2.0))
    # Only reject if actual overflow occurred (infinite values)
    if not np.all(np.isfinite(z2_scaled)):
        raise ValueError("Overflow in z2_scaled computation")
    ll_terms = c_log - 0.5 * np.log(s2) - 0.5 * (nu + 1.0) * np.log1p(z2_scaled)
    return np.sum(ll_terms)


def compute_skewt_loglikelihood(
    e: np.ndarray, s2: np.ndarray, nu: float, lambda_skew: float
) -> float:
    """Compute Skew-t (Hansen) log-likelihood given variance, nu, and lambda.

    Skew-t standardized (variance=1) density:
    f(z) = bc * (1 + 1/(nu-2) * ((b*z+a)/(1-lambda))^2)^(-(nu+1)/2)  if z < -a/b
    f(z) = bc * (1 + 1/(nu-2) * ((b*z+a)/(1+lambda))^2)^(-(nu+1)/2)  if z >= -a/b

    where:
    - c = Gamma((nu+1)/2) / (sqrt(pi*(nu-2)) * Gamma(nu/2))
    - a = 4*lambda*c*(nu-2)/(nu-1)
    - b = sqrt(1 + 3*lambda^2 - a^2)

    Args:
        e: Residual series.
        s2: Variance series.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter.

    Returns:
        Log-likelihood value.

    Raises:
        ValueError: If invalid Skew-t parameters.
    """
    from scipy.special import gammaln  # type: ignore

    z = e / np.sqrt(s2)
    lambda_val = float(lambda_skew)
    nu_val = float(nu)

    # Compute constants
    c_log = gammaln(0.5 * (nu_val + 1.0)) - gammaln(0.5 * nu_val) - 0.5 * (
        np.log(np.pi) + np.log(nu_val - 2.0)
    )
    c = np.exp(c_log)
    a = 4.0 * lambda_val * c * (nu_val - 2.0) / (nu_val - 1.0)
    b_sq = 1.0 + 3.0 * lambda_val**2 - a**2
    if b_sq <= 0:
        raise ValueError("Invalid Skew-t parameters: b^2 <= 0")
    b = np.sqrt(b_sq)
    threshold = -a / b

    # Compute log-likelihood terms
    ll_terms = np.empty_like(z)
    mask_left = z < threshold
    mask_right = ~mask_left

    # Left tail: z < -a/b
    if np.any(mask_left):
        z_left = z[mask_left]
        denom = 1.0 - lambda_val
        z_scaled = (b * z_left + a) / denom
        z2_scaled = z_scaled**2 / (nu_val - 2.0)
        ll_terms[mask_left] = (
            c_log + np.log(b)
            - 0.5 * np.log(s2[mask_left])
            - 0.5 * (nu_val + 1.0) * np.log1p(z2_scaled)
        )

    # Right tail: z >= -a/b
    if np.any(mask_right):
        z_right = z[mask_right]
        denom = 1.0 + lambda_val
        z_scaled = (b * z_right + a) / denom
        z2_scaled = z_scaled**2 / (nu_val - 2.0)
        ll_terms[mask_right] = (
            c_log + np.log(b)
            - 0.5 * np.log(s2[mask_right])
            - 0.5 * (nu_val + 1.0) * np.log1p(z2_scaled)
        )

    return float(np.sum(ll_terms))


# ============================================================================
# Optimization Utilities
# ============================================================================


def minimize_slsqp(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: Sequence[tuple[float | None, float | None]],
    constraints: Sequence[dict] | None = None,
) -> Any:
    """Run SciPy SLSQP minimize with local import to keep optional dep isolated.

    Args:
        fun: Objective function to minimize.
        x0: Initial parameter vector.
        bounds: Parameter bounds.
        constraints: Optional constraints.

    Returns:
        Optimization result object.

    Raises:
        RuntimeError: If SciPy is not available.
    """
    try:
        from scipy.optimize import minimize  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        msg = "SciPy required for MLE estimation"
        raise RuntimeError(msg) from exc
    return minimize(fun, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)


def optimize_with_fallback(
    fun: Callable[[np.ndarray], float],
    primary_x0: np.ndarray,
    fallback_x0: np.ndarray,
    bounds: Sequence[tuple[float | None, float | None]],
    constraints: Sequence[dict] | None = None,
) -> Any:
    """Try optimization with primary x0, fallback to alternative on failure.

    Args:
        fun: Objective function to minimize.
        primary_x0: Primary initial parameter vector.
        fallback_x0: Fallback initial parameter vector.
        bounds: Parameter bounds.
        constraints: Optional constraints.

    Returns:
        Optimization result object.
    """
    try:
        return minimize_slsqp(fun, primary_x0, bounds, constraints)
    except Exception as exc:
        logger.info("Warm-start failed, retrying with defaults: %s", exc)
        return minimize_slsqp(fun, fallback_x0, bounds, constraints)


def _build_normal_setup(
    e: np.ndarray,
    x0: Iterable[float] | None,
    w0: float,
    negloglik_normal: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[np.ndarray, Sequence[tuple[float | None, float | None]], Callable, np.ndarray]:
    """Build setup for Normal distribution."""
    b0, a0, g0 = 0.95, 0.1, 0.0
    if x0 is not None:
        x0_arr = np.array(list(x0), dtype=float)
    else:
        x0_arr = np.array([w0, a0, g0, b0], dtype=float)
    bounds = [(-50.0, 50.0), (-5.0, 5.0), (-5.0, 5.0), (-0.999, 0.999)]

    def fun(p):
        return negloglik_normal(p, e)

    fallback_x0 = np.array([w0, a0, g0, b0], dtype=float)
    return x0_arr, bounds, fun, fallback_x0


def _build_student_setup(
    e: np.ndarray,
    x0: Iterable[float] | None,
    w0: float,
    negloglik_student: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[np.ndarray, Sequence[tuple[float | None, float | None]], Callable, np.ndarray]:
    """Build setup for Student-t distribution."""
    b0, a0, g0 = 0.95, 0.1, 0.0
    x0_arr = (
        np.array(list(x0), dtype=float)
        if x0 is not None
        else np.array([w0, a0, g0, b0, GARCH_STUDENT_NU_INIT], dtype=float)
    )
    bounds = [
        (-50.0, 50.0),
        (-5.0, 5.0),
        (-5.0, 5.0),
        (-0.999, 0.999),
        (GARCH_STUDENT_NU_MIN, GARCH_STUDENT_NU_MAX),
    ]

    def fun(p):
        return negloglik_student(p, e)

    fallback_x0 = np.array([w0, a0, g0, b0, GARCH_STUDENT_NU_INIT], dtype=float)
    return x0_arr, bounds, fun, fallback_x0


def _build_skewt_setup(
    e: np.ndarray,
    x0: Iterable[float] | None,
    w0: float,
    negloglik_skewt: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[np.ndarray, Sequence[tuple[float | None, float | None]], Callable, np.ndarray]:
    """Build setup for Skew-t distribution."""
    b0, a0, g0 = 0.95, 0.1, 0.0
    x0_arr = (
        np.array(list(x0), dtype=float)
        if x0 is not None
        else np.array(
            [w0, a0, g0, b0, GARCH_STUDENT_NU_INIT, GARCH_SKEWT_LAMBDA_INIT],
            dtype=float,
        )
    )
    bounds = [
        (-50.0, 50.0),
        (-5.0, 5.0),
        (-5.0, 5.0),
        (-0.999, 0.999),
        (GARCH_STUDENT_NU_MIN, GARCH_STUDENT_NU_MAX),
        (GARCH_SKEWT_LAMBDA_MIN, GARCH_SKEWT_LAMBDA_MAX),
    ]

    def fun(p):
        return negloglik_skewt(p, e)

    fallback_x0 = np.array(
        [w0, a0, g0, b0, GARCH_STUDENT_NU_INIT, GARCH_SKEWT_LAMBDA_INIT], dtype=float
    )
    return x0_arr, bounds, fun, fallback_x0


def egarch_setup(
    e: np.ndarray,
    dist: str,
    x0: Iterable[float] | None,
    v: float,
    negloglik_normal: Callable[[np.ndarray, np.ndarray], float],
    negloglik_student: Callable[[np.ndarray, np.ndarray], float],
    negloglik_skewt: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[
    np.ndarray,
    Sequence[tuple[float | None, float | None]],
    Callable[[np.ndarray], float],
    np.ndarray,
]:
    """Build init vector, bounds, objective and fallback x0 for EGARCH(1,1).

    Args:
        e: Residual series.
        dist: Distribution name ('normal', 'student', or 'skewt').
        x0: Optional initial parameter vector.
        v: Initial variance estimate.
        negloglik_normal: Negative log-likelihood function for Normal.
        negloglik_student: Negative log-likelihood function for Student-t.
        negloglik_skewt: Negative log-likelihood function for Skew-t.

    Returns:
        Tuple of (x0_arr, bounds, fun, fallback_x0).

    Raises:
        ValueError: If distribution is invalid.
    """
    b0 = 0.95
    w0 = (1.0 - b0) * np.log(max(v, GARCH_MIN_INIT_VAR))
    dist_l = dist.lower()

    if dist_l == "normal":
        return _build_normal_setup(e, x0, w0, negloglik_normal)
    if dist_l == "student":
        return _build_student_setup(e, x0, w0, negloglik_student)
    if dist_l == "skewt":
        return _build_skewt_setup(e, x0, w0, negloglik_skewt)

    msg = "dist must be 'normal', 'student', or 'skewt'."
    raise ValueError(msg)


def egarch_finalize(dist: str, res: Any) -> dict[str, float]:
    """Finalize EGARCH estimation results.

    Args:
        dist: Distribution name.
        res: Optimization result object.

    Returns:
        Dictionary with estimated parameters and optimization results.
    """
    out: dict[str, float] = {
        "omega": float(res.x[0]),
        "alpha": float(res.x[1]),
        "gamma": float(res.x[2]),
        "beta": float(res.x[3]),
        "loglik": float(-res.fun),
        "converged": bool(res.success),
    }
    dist_l = dist.lower()
    if dist_l == "student":
        out["nu"] = float(res.x[4])
    elif dist_l == "skewt":
        out["nu"] = float(res.x[4])
        out["lambda"] = float(res.x[5])
    return out


# ============================================================================
# Data Loading and Preparation Utilities
# ============================================================================


def load_and_prepare_data() -> tuple[np.ndarray, np.ndarray]:
    """Load dataset and prepare training/test residuals.

    Returns:
        Tuple of (training_residuals, test_residuals).

    Raises:
        ValueError: If data loading or preparation fails.
    """
    try:
        df = load_garch_dataset(str(GARCH_DATASET_FILE))
    except Exception as ex:
        logger.error("Failed to load GARCH dataset: %s", ex)
        raise

    df_train = df.loc[df["split"] == "train"].copy()
    if df_train.empty:
        msg = "No training data found in dataset"
        logger.error(msg)
        raise ValueError(msg)

    try:
        resid_train = prepare_residuals(df_train, use_test_only=False)
        resid_test = prepare_residuals(df, use_test_only=True)
    except Exception as ex:
        logger.error("Failed to prepare residuals: %s", ex)
        raise

    resid_train = resid_train[np.isfinite(resid_train)]
    resid_test = resid_test[np.isfinite(resid_test)]

    if resid_train.size < 10:
        msg = f"Insufficient training residuals: {resid_train.size} < 10"
        logger.error(msg)
        raise ValueError(msg)

    return resid_train, resid_test


def estimate_single_model(
    resid_train: np.ndarray, dist: str
) -> tuple[str, dict[str, float] | None]:
    """Estimate a single EGARCH model (helper for parallel execution).

    Args:
        resid_train: Training residuals from SARIMA model.
        dist: Distribution name ('normal', 'student', or 'skewt').

    Returns:
        Tuple of (distribution_name, parameter_dict or None).
    """
    from src.garch.garch_params.estimation import estimate_egarch_mle

    try:
        logger.info("Optimizing EGARCH(1,1) with %s innovations...", dist.capitalize())
        result = estimate_egarch_mle(resid_train, dist=dist)
        return dist, result
    except Exception as ex:
        logger.warning("EGARCH MLE (%s) failed: %s", dist, ex)
        return dist, None


def estimate_egarch_models(
    resid_train: np.ndarray,
) -> tuple[
    dict[str, float] | None,
    dict[str, float] | None,
    dict[str, float] | None,
]:
    """Estimate EGARCH models for normal, student, and skewt distributions.

    Optimizes all three models in parallel using conditional MLE.

    Args:
        resid_train: Training residuals from SARIMA model.

    Returns:
        Tuple of (egarch_normal, egarch_student, egarch_skewt) parameter dicts.
    """
    distributions = ["normal", "student", "skewt"]
    results: dict[str, dict[str, float] | None] = {}

    # Optimize all three models in parallel
    logger.info("Starting parallel optimization of 3 EGARCH models...")
    with ProcessPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_dist = {
            executor.submit(estimate_single_model, resid_train, dist): dist
            for dist in distributions
        }

        # Collect results as they complete
        for future in as_completed(future_to_dist):
            dist, result = future.result()
            results[dist] = result

    return results.get("normal"), results.get("student"), results.get("skewt")

