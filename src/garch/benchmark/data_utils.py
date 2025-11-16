"""Data utilities for volatility backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_residual_column(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (e_all, dates_all, test_mask) from df using best available column.

    Prefers 'arima_residual_return'; falls back to 'weighted_log_return' for
    baselines-only scenarios in tests.

    Args:
        df: DataFrame with date, split, and residual/return columns.

    Returns:
        Tuple of (residuals array, dates array, test mask array).

    Raises:
        ValueError: If required columns are missing.
    """
    if "date" not in df.columns or "split" not in df.columns:
        msg = "DataFrame must contain 'date' and 'split' columns"
        raise ValueError(msg)
    data = df.sort_values("date").reset_index(drop=True)
    dates_series = pd.to_datetime(data["date"])
    dates = np.array(dates_series, dtype="datetime64[ns]")
    test_mask = np.array(data["split"].astype(str) == "test", dtype=bool)
    col = (
        "arima_residual_return"
        if "arima_residual_return" in data.columns
        else "weighted_log_return"
    )
    if col not in data.columns:
        msg = "Neither 'arima_residual_return' nor 'weighted_log_return' found in DataFrame"
        raise ValueError(msg)
    e_series = pd.to_numeric(data[col], errors="coerce")
    e_all = np.array(e_series, dtype=np.float64)
    return e_all, dates, test_mask


def pos_test_valid(e_all: np.ndarray, test_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return positions for valid series and test indices mapped onto it.

    Args:
        e_all: All residuals array.
        test_mask: Boolean mask for test observations.

    Returns:
        Tuple of (valid mask, test positions in valid array).
    """
    valid = np.isfinite(e_all)
    idx_all = np.arange(e_all.size)
    idx_valid = idx_all[valid]
    idx_test = idx_all[test_mask]
    idx_test_valid = np.intersect1d(idx_valid, idx_test, assume_unique=False)
    pos_in_valid = np.full(e_all.size, -1, dtype=int)
    pos_in_valid[idx_valid] = np.arange(idx_valid.size)
    pos_test = pos_in_valid[idx_test_valid]
    pos_test.sort()
    return valid, pos_test
