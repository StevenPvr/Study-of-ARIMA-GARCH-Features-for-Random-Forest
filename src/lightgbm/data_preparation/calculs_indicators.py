"""Technical indicators calculation for LightGBM data preparation.

This module now focuses on lightweight, leakage-safe indicators requested for
the LightGBM pipeline. Legacy indicators (SMA, EMA, MACD, ROC) are kept
only if still imported elsewhere, but the data-prep flow uses the custom
feature builder below.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.constants import (
    LIGHTGBM_TURNOVER_MA_WINDOW,
    LIGHTGBM_VOL_MA_LONG_WINDOW,
    LIGHTGBM_VOL_MA_SHORT_WINDOW,
)
from src.utils import get_logger

logger = get_logger(__name__)


def add_volume_features(tdf: pd.DataFrame, vol: pd.Series) -> pd.DataFrame:
    """Add log volume and normalized variants."""
    tdf["log_volume"] = cast(pd.Series, np.log1p(vol.astype(float)))
    log_vol = cast(pd.Series, tdf["log_volume"])
    log_vol_ma_5 = cast(
        pd.Series,
        log_vol.rolling(
            LIGHTGBM_VOL_MA_SHORT_WINDOW, min_periods=LIGHTGBM_VOL_MA_SHORT_WINDOW
        ).mean(),
    )
    log_vol_ma_20 = cast(
        pd.Series,
        log_vol.rolling(
            LIGHTGBM_VOL_MA_LONG_WINDOW, min_periods=LIGHTGBM_VOL_MA_LONG_WINDOW
        ).mean(),
    )
    log_vol_std_20 = cast(
        pd.Series,
        log_vol.rolling(LIGHTGBM_VOL_MA_LONG_WINDOW, min_periods=LIGHTGBM_VOL_MA_LONG_WINDOW).std(),
    )
    tdf["log_volume_rel_ma_5"] = log_vol - log_vol_ma_5
    tdf["log_volume_zscore_20"] = (log_vol - log_vol_ma_20) / log_vol_std_20.replace(0.0, pd.NA)
    return tdf


def add_return_features(tdf: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Add return features and transformations."""
    tdf["log_return"] = cast(pd.Series, np.log(close / close.shift(1)))
    tdf["abs_ret"] = cast(pd.Series, tdf["log_return"].abs())
    tdf["ret_sq"] = cast(pd.Series, tdf["log_return"] ** 2)
    return tdf


def add_calendar_features(tdf: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and seasonality features."""
    dt = cast(pd.Series, pd.to_datetime(tdf["date"]))
    tdf["day_of_week"] = dt.dt.dayofweek
    tdf["month"] = dt.dt.month
    tdf["is_month_end"] = dt.dt.is_month_end.astype(int)
    tdf["day_in_month_norm"] = dt.dt.day.astype(float) / 31.0
    return tdf


def add_turnover_features(tdf: pd.DataFrame, vol: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Add volume-price combined features."""
    turnover = cast(pd.Series, vol.astype(float) * close.astype(float))
    log_turnover = cast(pd.Series, np.log1p(turnover))
    tdf["log_turnover"] = log_turnover
    tdf["turnover_rel_ma_5"] = log_turnover - cast(
        pd.Series,
        log_turnover.rolling(
            LIGHTGBM_TURNOVER_MA_WINDOW, min_periods=LIGHTGBM_TURNOVER_MA_WINDOW
        ).mean(),
    )
    return tdf


def add_obv_feature(tdf: pd.DataFrame, vol: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Add On-Balance Volume (OBV) feature."""
    price_diff = cast(pd.Series, close.diff())
    direction = cast(pd.Series, np.sign(price_diff)).fillna(0.0)
    obv = cast(pd.Series, (direction * vol.astype(float)).cumsum())
    tdf["obv"] = obv
    return tdf


def add_atr_feature(
    tdf: pd.DataFrame,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """Add Average True Range (ATR) feature.

    ATR is calculated using a 14-period rolling average of True Range.
    True Range is the maximum of:
    - High - Low
    - |High - Previous Close|
    - |Low - Previous Close|

    Args:
        tdf: DataFrame for a single ticker.
        high: High price series.
        low: Low price series.
        close: Close price series.

    Returns:
        DataFrame with 'atr' column added.
    """
    high_float = cast(pd.Series, high.astype(float))
    low_float = cast(pd.Series, low.astype(float))
    close_float = cast(pd.Series, close.astype(float))

    # Calculate True Range
    tr1 = high_float - low_float
    tr2 = cast(pd.Series, (high_float - close_float.shift(1)).abs())
    tr3 = cast(pd.Series, (low_float - close_float.shift(1)).abs())

    true_range = cast(pd.Series, pd.concat([tr1, tr2, tr3], axis=1).max(axis=1))

    # Calculate ATR as 14-period rolling average of True Range
    atr_period = 14
    atr = cast(
        pd.Series,
        true_range.rolling(window=atr_period, min_periods=atr_period).mean(),
    )
    tdf["atr"] = atr
    return tdf


def compute_ticker_features(tdf: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for a single ticker.

    DEPRECATED: This function is kept for backward compatibility but
    _compute_ticker_features_vectorized should be used instead for performance.

    Args:
        tdf: DataFrame for a single ticker.

    Returns:
        DataFrame with all features computed.
    """
    tdf = tdf.copy()
    close = cast(pd.Series, tdf["close"])
    vol = cast(pd.Series, tdf["volume"])

    tdf = add_volume_features(tdf, vol)
    tdf = add_return_features(tdf, close)
    tdf = add_calendar_features(tdf)
    tdf = add_turnover_features(tdf, vol, close)
    tdf = add_obv_feature(tdf, vol, close)

    # Add ATR if high and low columns are available
    if "high" in tdf.columns and "low" in tdf.columns:
        high = cast(pd.Series, tdf["high"])
        low = cast(pd.Series, tdf["low"])
        tdf = add_atr_feature(tdf, high, low, close)

    return tdf


def _compute_ticker_features_vectorized(df: pd.DataFrame, ticker_col: str) -> pd.DataFrame:
    """Compute all ticker features using optimized groupby operations.

    This optimized version computes all features in a single groupby.apply() pass
    per ticker, reducing the number of groupby operations from ~27,000 to ~499.

    Args:
        df: DataFrame with ticker-level data, sorted by ticker and date.
        ticker_col: Name of the ticker column.

    Returns:
        DataFrame with all features computed.
    """
    out = df.copy()

    # Compute all features in a single groupby apply operation per ticker
    def _compute_all_features(group):
        group = group.copy()
        close = group["close"]
        vol = group["volume"]

        # Add volume features
        group = _add_volume_features_single_group(group, vol)

        # Add return features
        group["log_return"] = np.log(close / close.shift(1))
        group["abs_ret"] = group["log_return"].abs()
        group["ret_sq"] = group["log_return"] ** 2

        # Add turnover features
        turnover = vol.astype(float) * close.astype(float)
        log_turnover = np.log1p(turnover)
        group["log_turnover"] = log_turnover
        turnover_ma_5 = log_turnover.rolling(
            LIGHTGBM_TURNOVER_MA_WINDOW, min_periods=LIGHTGBM_TURNOVER_MA_WINDOW
        ).mean()
        group["turnover_rel_ma_5"] = log_turnover - turnover_ma_5

        # Add OBV feature
        close_diff = close.diff()
        direction = np.where(close_diff > 0, 1, np.where(close_diff < 0, -1, 0))
        group["obv"] = (direction * vol).cumsum()

        # Add ATR if high and low columns are available
        if "high" in group.columns and "low" in group.columns:
            high = group["high"]
            low = group["low"]
            high_low = high - low
            high_close = (high - close.shift(1)).abs()
            low_close = (low - close.shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            group["atr"] = tr.rolling(14, min_periods=1).mean()

        return group

    # Suppress pandas FutureWarning about groupby apply
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="DataFrameGroupBy.apply operated on the grouping columns",
            category=FutureWarning,
        )
        # Type ignore needed due to pandas type stubs not recognizing include_groups parameter
        grouped = out.groupby(ticker_col, group_keys=False)
        out = grouped.apply(_compute_all_features, include_groups=True)  # type: ignore

    # Add calendar features (no grouping needed)
    out = add_calendar_features(out)

    # Ensure DataFrame remains sorted by ticker and date for lag calculations
    out = out.sort_values([ticker_col, "date"]).reset_index(drop=True)

    # Rename ticker column to 'ticker' for consistency with downstream code
    out = out.rename(columns={ticker_col: "ticker"})

    return out


def _add_volume_features_single_group(df: pd.DataFrame, vol: pd.Series) -> pd.DataFrame:
    """Add volume features for a single ticker group."""
    df["log_volume"] = np.log1p(vol.astype(float))
    log_vol = df["log_volume"]

    log_vol_ma_5 = log_vol.rolling(
        LIGHTGBM_VOL_MA_SHORT_WINDOW, min_periods=LIGHTGBM_VOL_MA_SHORT_WINDOW
    ).mean()
    log_vol_ma_20 = log_vol.rolling(
        LIGHTGBM_VOL_MA_LONG_WINDOW, min_periods=LIGHTGBM_VOL_MA_LONG_WINDOW
    ).mean()
    log_vol_std_20 = log_vol.rolling(
        LIGHTGBM_VOL_MA_LONG_WINDOW, min_periods=LIGHTGBM_VOL_MA_LONG_WINDOW
    ).std()

    df["log_volume_rel_ma_5"] = log_vol - log_vol_ma_5
    df["log_volume_zscore_20"] = (log_vol - log_vol_ma_20) / log_vol_std_20.replace(0.0, pd.NA)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Deprecated placeholder for aggregated data.

    The RF pipeline now uses per-ticker custom indicators only.
    """
    return df.copy()


def add_custom_ml_indicators_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ML indicators per ticker without leakage.

    Features include volume, returns, calendar, turnover, and OBV indicators.
    All rolling operations use trailing windows to prevent look-ahead bias.

    OPTIMIZED VERSION: Uses vectorized operations instead of groupby.apply()
    for much better performance on large datasets.

    Args:
        df: Input ticker-level DataFrame with date, ticker, close, volume columns.

    Returns:
        DataFrame with new features appended.

    Raises:
        ValueError: If required columns are missing.
    """
    # Check for ticker column (could be 'ticker' or 'tickers')
    if "ticker" in df.columns:
        ticker_col_name = "ticker"
    elif "tickers" in df.columns:
        ticker_col_name = "tickers"
    else:
        raise ValueError("Neither 'ticker' nor 'tickers' column found in DataFrame")
    required = {"date", ticker_col_name, "close", "volume"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out = out.sort_values([ticker_col_name, "date"]).reset_index(drop=True)

    # Compute features using vectorized operations for better performance
    out = _compute_ticker_features_vectorized(out, ticker_col_name)

    logger.info("Custom ML indicators computed for ticker-level data (optimized)")
    return out
