"""Parameter extraction utilities for GARCH diagnostics.

Unified functions for extracting parameters from estimation results,
handling both new (nested) and legacy formats.
"""

from __future__ import annotations

from src.utils import get_logger

logger = get_logger(__name__)


def extract_params_dict(best: dict) -> dict:
    """Extract parameters dictionary from estimation results.

    Handles both formats:
    - New format: best["params"] (nested)
    - Legacy format: best (direct)

    Args:
        best: Best estimation results dictionary.

    Returns:
        Parameters dictionary.
    """
    params = best.get("params")
    if isinstance(params, dict):
        return params
    return best


def extract_nu_from_params(best: dict) -> float | None:
    """Extract nu parameter from estimation results.

    Handles both formats:
    - New format: best["params"]["nu"]
    - Legacy format: best["nu"]

    Args:
        best: Best estimation results dictionary.

    Returns:
        Nu (degrees of freedom) parameter or None if not found.
    """
    # Try new format first (params nested)
    params = best.get("params", {})
    if isinstance(params, dict):
        nu_value = params.get("nu")
        if nu_value is not None:
            return float(nu_value)

    # Fallback to legacy format
    nu_value = best.get("nu")
    return float(nu_value) if nu_value is not None else None


def extract_lambda_skew_from_params(
    best: dict,
    dist: str | None = None,
    required: bool = False,
) -> float | None:
    """Extract lambda_skew parameter from estimation results.

    Handles both formats:
    - New format: best["params"]["lambda_skew"] or best["params"]["lambda"]
    - Legacy format: best["lambda_skew"] or best["lambda"]

    Args:
        best: Best estimation results dictionary.
        dist: Distribution name. If "skewt" and required=True, raises error if not found.
        required: If True and dist="skewt", raises ValueError when lambda_skew not found.

    Returns:
        Lambda_skew parameter or None if not found.

    Raises:
        ValueError: If required=True, dist="skewt", and lambda_skew not found.
    """
    # Try new format first (params nested)
    params = best.get("params", {})
    if isinstance(params, dict):
        lambda_value = params.get("lambda_skew") or params.get("lambda")
        if lambda_value is not None:
            return float(lambda_value)

    # Fallback to legacy format
    lambda_value = best.get("lambda_skew") or best.get("lambda")
    if lambda_value is not None:
        return float(lambda_value)

    # Handle required parameter for Skew-t distribution
    if required and dist and dist.lower() == "skewt":
        available_keys = list(params.keys()) if isinstance(params, dict) else list(best.keys())
        msg = (
            "Skew-t distribution requires 'lambda_skew' (or 'lambda') parameter. "
            f"Available keys: {available_keys}"
        )
        raise ValueError(msg)

    return None
