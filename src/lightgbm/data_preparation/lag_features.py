"""Lag feature creation functions."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.constants import LIGHTGBM_LAG_WINDOWS
from src.lightgbm.data_preparation.validation import (
    validate_data_sorted_by_date,
    validate_lag_value,
)
from src.utils import get_logger

logger = get_logger(__name__)


def add_single_lag_feature(df: pd.DataFrame, column: str, lag: int) -> pd.DataFrame:
    """Add a single lag feature to dataframe.

    Args:
        df: Input dataframe.
        column: Column name to lag.
        lag: Number of periods to shift.

    Returns:
        DataFrame with added lag feature.
    """
    lag_column_name = f"{column}_lag_{lag}"
    # CRITICAL: Use positive lag to shift data backward (past values)
    # This ensures features at date t use only data from t-lag, preventing look-ahead bias
    # Handle ticker-level data by grouping
    if "ticker" in df.columns:
        df[lag_column_name] = df.groupby("ticker")[column].shift(lag)
    else:
        df[lag_column_name] = df[column].shift(lag)
    return df


def log_missing_columns(feature_columns: Sequence[str], df: pd.DataFrame) -> None:
    """Log missing columns for lagging.

    Args:
        feature_columns: Column names to check.
        df: DataFrame to check columns against.
    """
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        logger.debug("Columns skipped for lagging because they are missing: %s", missing_columns)


def add_lags_for_column(df: pd.DataFrame, column: str, lag_windows: Sequence[int]) -> pd.DataFrame:
    """Add lag features for a single column.

    OPTIMIZED VERSION: Uses vectorized operations to add all lags at once.

    Args:
        df: Input dataframe.
        column: Column name to lag.
        lag_windows: Lag windows to apply.

    Returns:
        DataFrame with lag features added for the column.
    """
    # Validate all lag values first
    for lag in lag_windows:
        validate_lag_value(lag)

    # Use optimized vectorized approach for all lags at once
    df = _add_lags_for_column_vectorized(df, column, lag_windows)
    return df


def _add_lags_for_column_vectorized(
    df: pd.DataFrame, column: str, lag_windows: Sequence[int]
) -> pd.DataFrame:
    """Add lag features for a single column using vectorized operations.

    This optimized version uses groupby.shift() which is much more efficient
    than groupby.apply() for lag operations.

    Args:
        df: Input dataframe with ticker column.
        column: Column name to lag.
        lag_windows: Lag windows to apply.

    Returns:
        DataFrame with lag features added for the column.
    """
    if column not in df.columns:
        return df

    df_copy = df.copy()

    if "ticker" in df_copy.columns:
        # Use groupby.shift() which is vectorized and efficient
        for lag in lag_windows:
            lag_column_name = f"{column}_lag_{lag}"
            df_copy[lag_column_name] = df_copy.groupby("ticker")[column].shift(lag)
    else:
        # No ticker grouping needed
        for lag in lag_windows:
            lag_column_name = f"{column}_lag_{lag}"
            df_copy[lag_column_name] = df_copy[column].shift(lag)

    return df_copy


def add_lag_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    lag_windows: Sequence[int],
) -> pd.DataFrame:
    """Return a dataframe with lagged versions of selected columns.

    DATA LEAKAGE PREVENTION:
    - Uses shift(lag) with positive lag values to access past data only
    - Features at date t use only data from t-lag (past values)
    - The split column is NEVER used in lag computation
    - Lags are computed on the full dataset (train+test) after target/split shift
    - Per-ticker grouping ensures no cross-ticker leakage

    Args:
        df: Input dataframe sorted by time (must be sorted by date, or by ticker+date
            for ticker-level data). This is critical to prevent data leakage.
        feature_columns: Column names to lag if present in the dataframe.
        lag_windows: Positive integers indicating how many periods to shift backward.

    Returns:
        DataFrame including the original data and the requested lag features.

    Raises:
        ValueError: If DataFrame is not properly sorted by date.
    """
    logger.info(
        "Adding lag features for %d columns with windows %s",
        len(feature_columns),
        lag_windows,
    )

    # Validate that data is sorted to prevent data leakage
    validate_data_sorted_by_date(df, "add_lag_features")

    log_missing_columns(feature_columns, df)

    # Work on a copy to avoid modifying the input DataFrame
    df_with_lags = df.copy()

    # Add all lag features in a single pass to minimize DataFrame copies
    for column in feature_columns:
        if column not in df_with_lags.columns:
            continue
        df_with_lags = _add_lags_for_column_vectorized(df_with_lags, column, lag_windows)

    return df_with_lags


def get_base_columns(include_lags: bool) -> list[str]:
    """Return base feature columns (log_volatility) with optional lags."""
    from src.constants import LIGHTGBM_BASE_FEATURE_COLUMNS

    base_cols = list(LIGHTGBM_BASE_FEATURE_COLUMNS)
    if not include_lags:
        return base_cols

    lagged = [f"{col}_lag_{lag}" for col in base_cols for lag in LIGHTGBM_LAG_WINDOWS]
    return base_cols + lagged
