"""Seasonal decomposition utilities for visualization."""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from src.constants import (
    SEASONAL_DEFAULT_PERIOD_DAILY,
    SEASONAL_DEFAULT_PERIOD_MONTHLY,
    SEASONAL_DEFAULT_PERIOD_WEEKLY,
)


def _maybe_resample(series: pd.Series, resample_to: str | None) -> pd.Series:
    """Resample series to a regular grid if requested, returning a clean Series.

    Args:
        series: Series to resample.
        resample_to: Target frequency string (e.g., 'W', 'B', 'D') or None.

    Returns:
        Resampled Series or original if resample_to is None.
    """
    if resample_to is None or series.empty:
        return series
    return series.resample(resample_to).mean().dropna()


def _infer_seasonal_period(resample_to: str | None, override: int | None) -> int:
    """Infer a reasonable seasonal period from the target frequency unless overridden.

    Args:
        resample_to: Target frequency string (e.g., 'B', 'D', 'M').
        override: Override period value.

    Returns:
        Seasonal period as integer.
    """
    if override is not None:
        return int(override)
    default_map = {
        "B": SEASONAL_DEFAULT_PERIOD_DAILY,
        "D": 7,
        "M": SEASONAL_DEFAULT_PERIOD_MONTHLY,
    }
    key = (resample_to or "").upper()
    return int(default_map.get(key, SEASONAL_DEFAULT_PERIOD_WEEKLY))


def _decompose_seasonal_component(series: pd.Series, *, model: str, period: int) -> pd.Series:
    """Run seasonal decomposition and return the seasonal component as a Series.

    Args:
        series: Time series to decompose.
        model: Decomposition model ('additive' or 'multiplicative').
        period: Seasonal period.

    Returns:
        Seasonal component as Series.
    """
    result = seasonal_decompose(series, model=model, period=int(period))
    return result.seasonal
