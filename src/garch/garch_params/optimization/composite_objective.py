"""Composite objective function for EGARCH hyperparameter optimization.

This module implements a composite objective that combines:
1. QLIKE loss (forecast accuracy)
2. AIC penalty (model complexity)
3. Diagnostic penalty (model quality via ARCH-LM test)

The weights are configurable via constants.
"""

from __future__ import annotations

import numpy as np

from src.constants import (
    GARCH_OPTIMIZATION_AIC_WEIGHT,
    GARCH_OPTIMIZATION_DIAGNOSTIC_WEIGHT,
    GARCH_OPTIMIZATION_QLIKE_WEIGHT,
)
from src.garch.garch_params.optimization.diagnostics import (
    compute_aic_penalty,
    compute_diagnostic_penalty,
    normalize_aic_penalty,
)
from src.utils import get_logger

logger = get_logger(__name__)


def _count_model_parameters(o: int, p: int, dist: str) -> int:
    """Count number of parameters in EGARCH(o,p) model.

    Args:
        o: ARCH order.
        p: GARCH order.
        dist: Distribution name.

    Returns:
        Number of parameters.
    """
    # Base parameters: omega (1) + alpha (o) + gamma (o) + beta (p)
    n_params = 1 + o + o + p

    # Distribution parameters
    if dist == "student":
        n_params += 1  # nu
    elif dist == "skewt":
        n_params += 2  # nu + lambda

    return n_params


def compute_composite_objective(
    qlike: float,
    residuals: np.ndarray,
    params: dict[str, float],
    loglik: float,
    o: int,
    p: int,
    dist: str,
) -> tuple[float, dict[str, float]]:
    """Compute composite objective function value.

    The composite objective is a weighted sum of:
    - QLIKE loss (normalized)
    - AIC penalty (normalized)
    - Diagnostic penalty (ARCH-LM test)

    Args:
        qlike: QLIKE loss value.
        residuals: Residuals for diagnostic tests.
        params: Estimated parameters.
        loglik: Log-likelihood value.
        o: ARCH order.
        p: GARCH order.
        dist: Distribution name.

    Returns:
        Tuple of (composite_objective, components_dict).
    """
    # Component 1: QLIKE (already normalized, lower is better)
    qlike_normalized = float(qlike)

    # Component 2: AIC penalty
    n_params = _count_model_parameters(o, p, dist)
    n_obs = len(residuals)
    aic_raw = compute_aic_penalty(n_obs, loglik, n_params)
    aic_normalized = normalize_aic_penalty(aic_raw, n_obs)

    # Component 3: Diagnostic penalty (ARCH-LM test)
    diagnostic_penalty = compute_diagnostic_penalty(residuals, params, o, p, dist)

    # Weighted composite objective
    composite = (
        GARCH_OPTIMIZATION_QLIKE_WEIGHT * qlike_normalized
        + GARCH_OPTIMIZATION_AIC_WEIGHT * aic_normalized
        + GARCH_OPTIMIZATION_DIAGNOSTIC_WEIGHT * diagnostic_penalty
    )

    # Verify weights sum to 1.0
    total_weight = (
        GARCH_OPTIMIZATION_QLIKE_WEIGHT
        + GARCH_OPTIMIZATION_AIC_WEIGHT
        + GARCH_OPTIMIZATION_DIAGNOSTIC_WEIGHT
    )
    if not np.isclose(total_weight, 1.0):
        logger.warning("Composite objective weights do not sum to 1.0: %.4f", total_weight)

    components = {
        "qlike": qlike_normalized,
        "aic": aic_normalized,
        "aic_raw": aic_raw,
        "diagnostic": diagnostic_penalty,
        "composite": composite,
    }

    logger.debug(
        "Composite objective: QLIKE=%.4f (w=%.2f), AIC=%.4f (w=%.2f), "
        "Diagnostic=%.4f (w=%.2f) => Total=%.4f",
        qlike_normalized,
        GARCH_OPTIMIZATION_QLIKE_WEIGHT,
        aic_normalized,
        GARCH_OPTIMIZATION_AIC_WEIGHT,
        diagnostic_penalty,
        GARCH_OPTIMIZATION_DIAGNOSTIC_WEIGHT,
        composite,
    )

    return float(composite), components


__all__ = [
    "compute_composite_objective",
]
