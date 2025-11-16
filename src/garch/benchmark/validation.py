"""Validation functions for volatility backtest parameters."""

from __future__ import annotations

import src.constants as C


def validate_ewma_lambda(ewma_lambda: float) -> None:
    """Validate EWMA lambda parameter.

    Args:
        ewma_lambda: EWMA decay factor.

    Raises:
        ValueError: If lambda is not in (0, 1).
    """
    if not (0.0 < ewma_lambda < 1.0):
        msg = f"ewma_lambda must be in (0, 1), got {ewma_lambda}"
        raise ValueError(msg)


def validate_rolling_window(rolling_window: int) -> None:
    """Validate rolling window parameter.

    Args:
        rolling_window: Window size.

    Raises:
        ValueError: If window is less than 1.
    """
    if rolling_window < 1:
        msg = f"rolling_window must be >= 1, got {rolling_window}"
        raise ValueError(msg)


def validate_var_alphas(var_alphas: list[float]) -> None:
    """Validate VaR alphas parameter.

    Args:
        var_alphas: List of VaR alpha values.

    Raises:
        ValueError: If alphas are invalid.
    """
    if not isinstance(var_alphas, list):
        msg = f"var_alphas must be a list, got {type(var_alphas)}"
        raise ValueError(msg)
    if not all(isinstance(a, (int, float)) and 0.0 < a < 1.0 for a in var_alphas):
        msg = f"var_alphas must be a list of floats in (0, 1), got {var_alphas}"
        raise ValueError(msg)


def validate_backtest_params(
    ewma_lambda: float | None,
    rolling_window: int | None,
    var_alphas: list[float] | None,
) -> tuple[float, int]:
    """Validate and resolve backtest parameters.

    Args:
        ewma_lambda: EWMA decay factor (lambda).
        rolling_window: Window size for rolling baselines.
        var_alphas: VaR alphas (validated but not returned).

    Returns:
        Tuple of (validated ewma_lambda, validated rolling_window).

    Raises:
        ValueError: If parameters are invalid.
    """
    if ewma_lambda is None:
        ewma_lambda = float(C.VOL_EWMA_LAMBDA_DEFAULT)
    else:
        ewma_lambda = float(ewma_lambda)

    if rolling_window is None:
        rolling_window = int(C.VOL_ROLLING_WINDOW_DEFAULT)
    else:
        rolling_window = int(rolling_window)

    validate_ewma_lambda(ewma_lambda)
    validate_rolling_window(rolling_window)
    if var_alphas is not None:
        validate_var_alphas(var_alphas)

    return ewma_lambda, rolling_window
