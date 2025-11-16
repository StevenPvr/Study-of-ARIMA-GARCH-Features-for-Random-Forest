"""Column selection and filtering functions for dataset creation."""

from __future__ import annotations

import pandas as pd

from src.constants import LIGHTGBM_ARIMA_GARCH_INSIGHT_COLUMNS, LIGHTGBM_LAG_WINDOWS
from src.utils import get_logger

logger = get_logger(__name__)


def remove_missing_values(
    df: pd.DataFrame,
    subset: list[str] | None = None,
) -> pd.DataFrame:
    """Remove rows with missing values on a selected subset of columns.

    Meta columns (``date``, ``ticker``, ``ticker_id``, ``split``) are never
    used to trigger row removal.

    Args:
        df: DataFrame that may contain missing values.
        subset: Optional list of columns on which to enforce non-NaN values.
            If None, all non-meta columns are used.

    Returns:
        Cleaned DataFrame without missing values on the selected subset.
    """
    initial_rows = len(df)
    meta_cols: set[str] = {"date", "ticker", "ticker_id", "split"}

    if subset is None:
        subset = [c for c in df.columns if c not in meta_cols]
    else:
        subset = [c for c in subset if c not in meta_cols]

    if not subset:
        logger.info("No subset columns provided for dropna; returning original dataframe")
        return df.reset_index(drop=True).copy()

    logger.info(
        "Removing rows with missing values from %d columns (ignoring meta columns)",
        len(subset),
    )
    df_clean = df.dropna(subset=subset).reset_index(drop=True).copy()
    removed_rows = initial_rows - len(df_clean)
    logger.info(
        "Removed %d rows with missing values (from %d to %d)",
        removed_rows,
        initial_rows,
        len(df_clean),
    )
    return df_clean


def get_base_insight_columns(df: pd.DataFrame) -> list[str]:
    """Get base ARIMA-GARCH insight columns present in dataframe.

    Args:
        df: DataFrame with potential insight columns.

    Returns:
        List of base insight column names present in dataframe.
    """
    return [col for col in LIGHTGBM_ARIMA_GARCH_INSIGHT_COLUMNS if col in df.columns]


def get_lagged_insight_columns(df: pd.DataFrame) -> list[str]:
    """Get lagged ARIMA-GARCH insight columns present in dataframe.

    Args:
        df: DataFrame with potential lagged insight columns.

    Returns:
        List of lagged insight column names present in dataframe.
    """
    lagged_cols: list[str] = []
    for col in LIGHTGBM_ARIMA_GARCH_INSIGHT_COLUMNS:
        lagged_cols.extend(
            [f"{col}_lag_{lag}" for lag in LIGHTGBM_LAG_WINDOWS if f"{col}_lag_{lag}" in df.columns]
        )
    return lagged_cols


def get_insight_columns_to_drop(df: pd.DataFrame) -> list[str]:
    """Get list of insight columns to drop from dataframe.

    Args:
        df: DataFrame to check for columns.

    Returns:
        List of insight column names to drop.
    """
    base_cols = get_base_insight_columns(df)
    lagged_cols = get_lagged_insight_columns(df)
    return base_cols + lagged_cols


def get_sigma2_columns_to_drop(df: pd.DataFrame) -> list[str]:
    """Get list of sigma2_garch columns (and lags) to drop from dataframe.

    Args:
        df: DataFrame to check for columns.

    Returns:
        List of sigma2_garch column names to drop.
    """
    columns_to_drop: list[str] = []
    if "sigma2_garch" in df.columns:
        columns_to_drop.append("sigma2_garch")
    for lag in LIGHTGBM_LAG_WINDOWS:
        col_name = f"sigma2_garch_lag_{lag}"
        if col_name in df.columns:
            columns_to_drop.append(col_name)
    return columns_to_drop


def get_non_observable_columns_to_drop(df: pd.DataFrame) -> list[str]:
    """Get list of columns that are not observable at prediction time.

    These columns should be dropped before model training to avoid data leakage.

    Args:
        df: DataFrame to check for columns.

    Returns:
        List of non-observable column names to drop.
    """
    non_observable_columns = [
        "sigma2_garch_true",
        "sigma_garch_true",
        "std_resid_garch_true",
        "sigma2_egarch_raw",  # EGARCH raw output, not used in LightGBM datasets
        # sarima_pred and sarima_resid are considered observable
        # (produced using past-only model predictions)
        # and must be kept per requirements.
        "target_date",
    ]
    return [col for col in non_observable_columns if col in df.columns]


def remove_closing_column(df: pd.DataFrame) -> pd.DataFrame:
    """Remove 'close' column if present."""
    if "close" in df.columns:
        df = df.drop(columns=["close"])
        logger.info("Removed 'close' column from dataframe")
    return df


def get_technical_indicator_columns(include_lags: bool) -> list[str]:
    """Return list of technical indicator columns (with optional lags)."""
    from src.constants import LIGHTGBM_TECHNICAL_FEATURE_COLUMNS

    base_cols = list(LIGHTGBM_TECHNICAL_FEATURE_COLUMNS)
    if not include_lags:
        return base_cols

    lagged_cols = [
        f"{indicator}_lag_{lag}" for indicator in base_cols for lag in LIGHTGBM_LAG_WINDOWS
    ]
    return base_cols + lagged_cols
