"""Utility functions for GARCH data visualization.

Contains helper functions for data preparation and plotting operations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_logger


def prepare_x_axis(dates: np.ndarray | pd.Series | None, length: int) -> np.ndarray:
    """Prepare x-axis values from dates or indices.

    Args:
        dates: Optional date array/series
        length: Length of data

    Returns:
        X-axis values array
    """
    if dates is not None:
        try:
            dates_clean = pd.to_datetime(dates)  # type: ignore[arg-type]
            return np.asarray(dates_clean.values, dtype=object)
        except Exception:
            return np.arange(length, dtype=float)
    return np.arange(length, dtype=float)


def plot_returns_panel(ax: Any, x_vals: np.ndarray, returns: np.ndarray) -> None:
    """Plot returns panel.

    Args:
        ax: Matplotlib axis
        x_vals: X-axis values
        returns: Returns array
    """
    ax.plot(x_vals, returns, linewidth=0.5, alpha=0.7, color="blue")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax.set_title("Rendements (returns)")
    ax.set_ylabel("Rendement")
    ax.grid(True, alpha=0.3)


def plot_absolute_returns_panel(ax: Any, x_vals: np.ndarray, abs_returns: np.ndarray) -> None:
    """Plot absolute returns panel.

    Args:
        ax: Matplotlib axis
        x_vals: X-axis values
        abs_returns: Absolute returns array
    """
    ax.plot(x_vals, abs_returns, linewidth=0.5, alpha=0.7, color="orange")
    ax.set_title("Rendements absolus (|returns|)")
    ax.set_ylabel("|Rendement|")
    ax.grid(True, alpha=0.3)


def plot_squared_returns_panel(
    ax: Any, x_vals: np.ndarray, squared_returns: np.ndarray, has_dates: bool
) -> None:
    """Plot squared returns panel.

    Args:
        ax: Matplotlib axis
        x_vals: X-axis values
        squared_returns: Squared returns array
        has_dates: Whether dates are available
    """
    ax.plot(x_vals, squared_returns, linewidth=0.5, alpha=0.7, color="red")
    ax.set_title("Rendements au carré (returns²)")
    ax.set_ylabel("Rendement²")
    ax.set_xlabel("Date" if has_dates else "Observation")
    ax.grid(True, alpha=0.3)


def prepare_test_dataframe(df: pd.DataFrame) -> pd.DataFrame | None:
    """Prepare test split DataFrame for returns plotting.

    Args:
        df: Input DataFrame

    Returns:
        Test DataFrame or None if data unavailable
    """
    logger = get_logger(__name__)

    if "split" not in df.columns:
        logger.warning("Dataset missing 'split' column, using all data for returns plot")
        df_test = df.copy()
    else:
        df_test = df.loc[df["split"] == "test"].copy()
        if len(df_test) == 0:
            logger.warning("Test split is empty, using all data for returns plot")
            df_test = df.copy()

    # Ensure weighted_return exists
    if "weighted_return" not in df_test.columns:
        if "weighted_log_return" in df_test.columns:
            log_returns = np.asarray(df_test["weighted_log_return"].values, dtype=float)
            # Protect against overflow: clip extreme values
            log_returns_clipped = np.clip(log_returns, -10.0, 10.0)
            df_test["weighted_return"] = np.expm1(log_returns_clipped)  # type: ignore[index]
        else:
            logger.warning(
                "Neither 'weighted_return' nor 'weighted_log_return' found, skipping returns plot"
            )
            return None

    return df_test


def extract_dates_from_dataframe(df_test: pd.DataFrame) -> pd.Series | None:
    """Extract dates column from DataFrame if available.

    Args:
        df_test: DataFrame to extract dates from

    Returns:
        Dates Series or None
    """
    if "date" not in df_test.columns:
        return None
    return df_test["date"]  # type: ignore[assignment]
