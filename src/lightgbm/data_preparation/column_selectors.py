"""Reusable column selection functions for dataset creation.

This module consolidates all column selection logic to follow DRY principles.
Each function is composable and can be combined to create different dataset variants.
"""

from __future__ import annotations

import pandas as pd

from src.constants import LIGHTGBM_LAG_WINDOWS
from src.lightgbm.data_preparation.lag_features import get_base_columns
from src.lightgbm.data_preparation.target_creation import get_target_column_name


def select_metadata_columns(df: pd.DataFrame) -> list[str]:
    """Select metadata columns (date, split, ticker/tickers, ticker_id).

    Args:
        df: DataFrame to check for columns.

    Returns:
        List of metadata column names present in dataframe.
    """
    base_cols = ["date", "split"]
    keep_cols: list[str] = [col for col in base_cols if col in df.columns]

    # Handle both 'ticker' and 'tickers' column names
    if "ticker" in df.columns:
        keep_cols.append("ticker")
    if "tickers" in df.columns:
        keep_cols.append("tickers")
    if "ticker_id" in df.columns:
        keep_cols.append("ticker_id")

    return keep_cols


def select_target_columns(df: pd.DataFrame, include_lags: bool) -> list[str]:
    """Select target column with optional lags.

    Args:
        df: DataFrame with target column.
        include_lags: If True, include lag features.

    Returns:
        List of target column names (with optional lags).
    """
    keep_cols: list[str] = []
    return_col = get_target_column_name(df)

    if return_col in df.columns:
        keep_cols.append(return_col)

    if include_lags:
        lag_cols = [col for col in df.columns if col.startswith(f"{return_col}_lag_")]
        keep_cols.extend(lag_cols)

    return keep_cols


def select_garch_insight_columns(df: pd.DataFrame, include_lags: bool) -> list[str]:
    """Select ARIMA-GARCH insight columns with optional lags.

    Includes both ARIMA insights (sarima_pred, sarima_resid) and GARCH insights
    (sigma2_garch, sigma_garch, std_resid_garch).

    Args:
        df: DataFrame with potential ARIMA-GARCH insight columns.
        include_lags: If True, include lag features.

    Returns:
        List of ARIMA-GARCH insight column names (with optional lags).
    """
    keep_cols: list[str] = []
    insight_cols = [
        "sarima_pred",
        "sarima_resid",
        "sigma2_garch",
        "sigma_garch",
        "std_resid_garch",
    ]

    for insight_col in insight_cols:
        if insight_col in df.columns:
            keep_cols.append(insight_col)

        if include_lags:
            lag_cols = [
                f"{insight_col}_lag_{lag}"
                for lag in LIGHTGBM_LAG_WINDOWS
                if f"{insight_col}_lag_{lag}" in df.columns
            ]
            keep_cols.extend(lag_cols)

    return keep_cols


def select_base_feature_columns(df: pd.DataFrame, include_lags: bool) -> list[str]:
    """Select base feature columns (log_volatility) with optional lags.

    Args:
        df: DataFrame to check for columns.
        include_lags: If True, include lag features.

    Returns:
        List of base feature column names (with optional lags).
    """
    base_cols = get_base_columns(include_lags)
    return [col for col in base_cols if col in df.columns]


def select_technical_indicator_columns(df: pd.DataFrame, include_lags: bool) -> list[str]:
    """Select technical indicator columns with optional lags.

    Args:
        df: DataFrame to check for columns.
        include_lags: If True, include lag features.

    Returns:
        List of technical indicator column names (with optional lags).
    """
    from src.lightgbm.data_preparation.column_selection import get_technical_indicator_columns

    tech_cols = get_technical_indicator_columns(include_lags)
    return [col for col in tech_cols if col in df.columns]


def select_calendar_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select non-lagged calendar feature columns.

    Calendar features are discrete/time-indexed signals and should be included
    without creating lagged versions. This selector returns only the calendar
    columns that are present in the DataFrame.

    Args:
        df: DataFrame to check for columns.

    Returns:
        List of calendar column names present in dataframe.
    """
    from src.constants import LIGHTGBM_CALENDAR_FEATURE_COLUMNS

    return [col for col in LIGHTGBM_CALENDAR_FEATURE_COLUMNS if col in df.columns]


def select_close_column(df: pd.DataFrame) -> list[str]:
    """Select 'close' column if present.

    Args:
        df: DataFrame to check for columns.

    Returns:
        List containing 'close' if present, empty list otherwise.
    """
    return ["close"] if "close" in df.columns else []


def select_sigma_plus_base_columns(df: pd.DataFrame, include_lags: bool) -> list[str]:
    """Select columns for sigma-plus-base dataset.

    Combines metadata + target + GARCH insights + base features + close.

    Args:
        df: DataFrame to select columns from.
        include_lags: If True, include lag features.

    Returns:
        List of column names for sigma-plus-base dataset.
    """
    keep_cols: list[str] = []

    keep_cols.extend(select_metadata_columns(df))
    keep_cols.extend(select_target_columns(df, include_lags))
    keep_cols.extend(select_garch_insight_columns(df, include_lags))
    keep_cols.extend(select_base_feature_columns(df, include_lags))
    keep_cols.extend(select_close_column(df))

    # Remove duplicates while preserving order
    return list(dict.fromkeys(keep_cols))


def select_log_volatility_only_columns(df: pd.DataFrame, include_lags: bool) -> list[str]:
    """Select columns for log-volatility-only dataset.

    Combines metadata + target + base features (no GARCH insights) + close.

    Args:
        df: DataFrame to select columns from.
        include_lags: If True, include lag features.

    Returns:
        List of column names for log-volatility-only dataset.
    """
    keep_cols: list[str] = []

    keep_cols.extend(select_metadata_columns(df))
    keep_cols.extend(select_target_columns(df, include_lags))
    keep_cols.extend(select_base_feature_columns(df, include_lags))
    keep_cols.extend(select_close_column(df))

    return keep_cols


def select_technical_indicators_dataset_columns(df: pd.DataFrame, include_lags: bool) -> list[str]:
    """Select columns for technical indicators dataset.

    Combines metadata + target + technical indicators + close.

    Args:
        df: DataFrame to select columns from.
        include_lags: If True, include lag features.

    Returns:
        List of column names for technical indicators dataset.
    """
    keep_cols: list[str] = []

    keep_cols.extend(select_metadata_columns(df))
    keep_cols.extend(select_target_columns(df, include_lags))
    keep_cols.extend(select_technical_indicator_columns(df, include_lags))
    # Explicitly include calendar features (no lags)
    keep_cols.extend(select_calendar_feature_columns(df))
    keep_cols.extend(select_close_column(df))

    # Remove duplicates while preserving order
    return list(dict.fromkeys(keep_cols))


def get_sigma_plus_base_feature_columns_for_lag(df: pd.DataFrame) -> list[str]:
    """Get feature columns to apply lags for sigma-plus-base dataset.

    Returns:
        List of column names to create lags for.
    """
    from src.constants import LIGHTGBM_BASE_FEATURE_COLUMNS

    feature_columns = [
        "sigma2_garch",
        "sigma_garch",
        "std_resid_garch",
        *LIGHTGBM_BASE_FEATURE_COLUMNS,
    ]

    if "sarima_resid" in df.columns:
        feature_columns.append("sarima_resid")
    if "sarima_pred" in df.columns:
        feature_columns.append("sarima_pred")

    return feature_columns
