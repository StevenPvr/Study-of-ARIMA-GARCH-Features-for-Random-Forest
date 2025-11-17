from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Force headless backend for CI
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src package is importable when running tests directly
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.arima.data_visualisation.plotting import (  # noqa: E402  (import after sys.path update)
    plot_seasonal_decomposition_daily,
    plot_seasonal_decomposition_monthly,
)
from src.constants import (  # noqa: E402
    ARIMA_SEASONALITY_DAILY_PERIOD,
    ARIMA_SEASONALITY_MONTHLY_PERIOD,
)


def _build_sample_series() -> pd.Series:
    """Create a deterministic seasonal series long enough for both periods."""

    min_length = max(ARIMA_SEASONALITY_DAILY_PERIOD, ARIMA_SEASONALITY_MONTHLY_PERIOD) * 6
    date_index = pd.date_range("2021-01-01", periods=min_length, freq="B")
    values = np.sin(np.linspace(0, 6 * np.pi, min_length))
    return pd.Series(values, index=date_index)


class TestSeasonalDecompositionPlots:
    """Headless tests for seasonal decomposition helpers."""

    def test_plot_seasonal_decomposition_daily_returns_matplotlib_objects(self) -> None:
        series = _build_sample_series()

        figure, axes = plot_seasonal_decomposition_daily(series)

        assert isinstance(figure, Figure)
        assert axes
        assert all(isinstance(axis, Axes) for axis in axes)

        plt.close(figure)

    def test_plot_seasonal_decomposition_monthly_returns_matplotlib_objects(self) -> None:
        series = _build_sample_series()

        figure, axes = plot_seasonal_decomposition_monthly(series)

        assert isinstance(figure, Figure)
        assert axes
        assert all(isinstance(axis, Axes) for axis in axes)

        plt.close(figure)
