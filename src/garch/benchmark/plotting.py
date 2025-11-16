"""Plotting utilities for volatility backtest."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import src.constants as C
from src.utils import get_logger

logger = get_logger(__name__)


def _plot_single_model(
    ax: Any,
    dates: np.ndarray,
    realized_var: np.ndarray,
    s2_forecast: np.ndarray,
    model_name: str,
) -> None:
    """Plot a single model forecast vs realized variance.

    Args:
        ax: Matplotlib axes object.
        dates: Test dates array.
        realized_var: Realized variance array.
        s2_forecast: Forecast variance array.
        model_name: Name of the model.
    """
    m = np.isfinite(realized_var) & np.isfinite(s2_forecast)

    ax.plot(
        dates[m],
        realized_var[m],
        label="Realized (e²)",
        alpha=0.6,
        linewidth=1.5,
        color="#1f77b4",
    )
    ax.plot(
        dates[m],
        s2_forecast[m],
        label="Forecast (σ²)",
        alpha=0.8,
        linewidth=1.5,
        color="#ff7f0e",
    )
    ax.set_title(f"{model_name}", fontweight="bold")
    ax.set_ylabel("Variance")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_all_models(
    fig: Any,
    axes: Any,
    dates: np.ndarray,
    realized_var: np.ndarray,
    forecasts: pd.DataFrame,
) -> None:
    """Plot all model forecasts on subplots.

    Args:
        fig: Matplotlib figure object.
        axes: Matplotlib axes array.
        dates: Test dates array.
        realized_var: Realized variance array.
        forecasts: DataFrame with all model forecasts.
    """
    models_config = [
        ("arima_garch", "ARIMA-GARCH", axes[0, 0]),
        ("har3", "HAR(3)", axes[0, 1]),
        ("ewma", "EWMA", axes[1, 0]),
        ("arch1", "ARCH(1)", axes[1, 1]),
        ("roll_var", "Rolling Variance", axes[2, 0]),
        ("roll_std", "Rolling Std", axes[2, 1]),
    ]

    for model_key, model_name, ax in models_config:
        s2_col = f"s2_{model_key}" if model_key != "arima_garch" else "s2_arima_garch"
        if s2_col not in forecasts.columns:
            continue
        s2_forecast = forecasts[s2_col].to_numpy()
        _plot_single_model(ax, dates, realized_var, s2_forecast, model_name)


def plot_volatility_forecasts(
    dates: np.ndarray,
    e_test: np.ndarray,
    forecasts: pd.DataFrame,
) -> None:
    """Plot volatility forecasts comparison for all models.

    Args:
        dates: Test dates array.
        e_test: Test residuals.
        forecasts: DataFrame with all model forecasts.

    Raises:
        ImportError: If matplotlib or seaborn are not available.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore
    except ImportError as e:
        msg = (
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install matplotlib seaborn"
        )
        raise ImportError(msg) from e

    realized_var = e_test**2
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    _plot_all_models(fig, axes, dates, realized_var, forecasts)

    fig.suptitle(
        "Volatility Forecasts Comparison: Realized vs Forecasted Variance",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    C.VOL_BACKTEST_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.VOL_BACKTEST_VOLATILITY_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved volatility forecasts plot: %s", C.VOL_BACKTEST_VOLATILITY_PLOT)
