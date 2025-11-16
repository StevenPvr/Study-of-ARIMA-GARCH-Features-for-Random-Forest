"""GARCH data visualization module.

Provides functions for visualizing returns, volatility clustering, and autocorrelation.
"""

from __future__ import annotations

from src.garch.garch_data_visualisation.plots import (
    plot_returns_autocorrelation,
    save_returns_and_squared_plots,
)
from src.garch.garch_data_visualisation.utils import (
    extract_dates_from_dataframe,
    prepare_test_dataframe,
    prepare_x_axis,
)

__all__ = [
    "plot_returns_autocorrelation",
    "save_returns_and_squared_plots",
    "extract_dates_from_dataframe",
    "prepare_test_dataframe",
    "prepare_x_axis",
]
