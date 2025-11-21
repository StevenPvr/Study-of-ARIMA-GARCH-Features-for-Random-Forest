"""Pytest fixtures for visualization tests."""

from __future__ import annotations

import pytest

# Configure matplotlib to use non-interactive backend for tests
import matplotlib

matplotlib.use("Agg")


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    import numpy as np
    import pandas as pd

    # Create sample time series data
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    values = np.random.randn(100).cumsum()
    return pd.Series(values, index=dates)


@pytest.fixture
def sample_figure_and_axes():
    """Provide a sample matplotlib figure and axes."""
    from src.visualization.plotting_utils import create_figure_canvas

    fig, canvas, axes = create_figure_canvas((10, 6))
    return fig, axes
