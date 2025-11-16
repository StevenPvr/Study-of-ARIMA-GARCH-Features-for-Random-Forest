from __future__ import annotations

"""Compatibility wrapper for legacy data visualisation imports."""

from src.arima.data_visualisation import data_visualisation as _wrapped_module

# Explicitly re-export functions for type checker visibility
from src.arima.data_visualisation.data_visualisation import (
    plot_acf_pacf,
    plot_seasonality_daily,
    plot_seasonality_for_year,
    plot_seasonality_full_period,
    plot_seasonality_monthly,
    plot_stationarity,
    plot_weighted_series,
)

# Dynamically copy remaining attributes for backward compatibility
for _name, _value in vars(_wrapped_module).items():
    if _name == "__builtins__" or _name in globals():
        continue
    globals()[_name] = _value

__all__ = [
    "plot_acf_pacf",
    "plot_seasonality_daily",
    "plot_seasonality_for_year",
    "plot_seasonality_full_period",
    "plot_seasonality_monthly",
    "plot_stationarity",
    "plot_weighted_series",
]
