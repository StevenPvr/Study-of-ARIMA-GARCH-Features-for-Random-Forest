"""Individual dataset builder functions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import (
    LIGHTGBM_BASE_FEATURE_COLUMNS,
    LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE,
    LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE,
    LIGHTGBM_DATASET_TECHNICAL_INDICATORS_FILE,
    LIGHTGBM_LAG_WINDOWS,
    LIGHTGBM_TECHNICAL_FEATURE_COLUMNS,
)
from src.lightgbm.data_preparation.calculs_indicators import add_custom_ml_indicators_per_ticker
from src.lightgbm.data_preparation.column_selection import (
    get_non_observable_columns_to_drop,
    get_technical_indicator_columns,
)
from src.lightgbm.data_preparation.column_selectors import (
    get_sigma_plus_base_feature_columns_for_lag,
    select_log_volatility_only_columns,
    select_sigma_plus_base_columns,
    select_technical_indicators_dataset_columns,
)
from src.lightgbm.data_preparation.data_loading import ensure_ticker_id_column
from src.lightgbm.data_preparation.dataset_creation import prepare_datasets
from src.lightgbm.data_preparation.dataset_pipeline import (
    drop_rows_with_missing_features,
    prepare_base_dataframe,
)
from src.lightgbm.data_preparation.lag_features import add_lag_features, get_base_columns
from src.lightgbm.data_preparation.target_creation import create_target_columns
from src.utils import get_logger, save_parquet_and_csv

logger = get_logger(__name__)


def _process_lags(df: pd.DataFrame, feature_columns: list[str], include_lags: bool) -> pd.DataFrame:
    """Add lag features if requested.

    Args:
        df: DataFrame with features.
        feature_columns: Columns to create lags for.
        include_lags: If True, add lag features.

    Returns:
        DataFrame with optional lags added.
    """
    if not include_lags:
        return df

    return add_lag_features(df, feature_columns=feature_columns, lag_windows=LIGHTGBM_LAG_WINDOWS)


def _clean_non_observable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-observable columns from dataset.

    Args:
        df: DataFrame to clean.

    Returns:
        DataFrame without non-observable columns.
    """
    non_observable_columns = get_non_observable_columns_to_drop(df)
    columns_to_remove = [col for col in non_observable_columns if col in df.columns]

    if columns_to_remove:
        logger.info("Dropping non-observable columns: %s", columns_to_remove)
        return df.drop(columns=columns_to_remove)

    return df


def _add_indicators_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add custom indicators and create target columns.

    Args:
        df: Prepared DataFrame.

    Returns:
        DataFrame with indicators and target columns.
    """
    if "ticker" in df.columns:
        logger.info("Calculating custom ML indicators per ticker")
        df_with_indicators = add_custom_ml_indicators_per_ticker(df)
    else:
        logger.info("Aggregated data detected; skipping custom indicators")
        df_with_indicators = df.copy()

    return create_target_columns(df_with_indicators)


def create_dataset_sigma_plus_base(
    df: pd.DataFrame | None = None,
    output_path: Path | None = None,
    *,
    include_lags: bool = True,
) -> pd.DataFrame:
    """Create dataset combining GARCH insights with base features (log_volatility).

    Args:
        df: Ticker data DataFrame. If None, loads from ticker-level data.
        output_path: Path to save dataset. If None, uses LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE.
        include_lags: If True, include lag features.

    Returns:
        Dataset with sigma insights + base features.
    """
    df_prepared = prepare_base_dataframe(df, prefer_insights=False)
    df_with_features = _add_indicators_and_target(df_prepared)

    feature_columns = get_sigma_plus_base_feature_columns_for_lag(df_with_features)
    df_with_lags = _process_lags(df_with_features, feature_columns, include_lags)
    df_with_lags = ensure_ticker_id_column(df_with_lags)
    df_with_lags = _clean_non_observable_columns(df_with_lags)

    keep_cols = select_sigma_plus_base_columns(df_with_lags, include_lags)
    keep_cols = [c for c in keep_cols if c in df_with_lags.columns]
    df_selected = drop_rows_with_missing_features(df_with_lags, keep_cols)

    if output_path is None:
        output_path = LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE.with_suffix(".parquet")

    if "ticker" in df_selected.columns and "tickers" not in df_selected.columns:
        df_selected = df_selected.rename(columns={"ticker": "tickers"})
    save_parquet_and_csv(df_selected, output_path)
    logger.info(
        "Saved sigma-plus-base dataset: %s (%d rows, %d columns)",
        output_path,
        len(df_selected),
        len(df_selected.columns),
    )

    return df_selected


def create_dataset_log_volatility_only(
    df: pd.DataFrame | None = None,
    output_path: Path | None = None,
    *,
    include_lags: bool = True,
) -> pd.DataFrame:
    """Create dataset with only base features (log_volatility) and its lags.

    Args:
        df: Ticker data DataFrame. If None, loads from ticker-level data.
        output_path: Path to save dataset. If None, uses LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE.
        include_lags: If True, include lag features.

    Returns:
        Dataset with only base features.
    """
    df_prepared = prepare_base_dataframe(df, prefer_insights=False)
    df_with_features = _add_indicators_and_target(df_prepared)

    df_with_lags = _process_lags(
        df_with_features, list(LIGHTGBM_BASE_FEATURE_COLUMNS), include_lags
    )
    df_with_lags = ensure_ticker_id_column(df_with_lags)
    df_with_lags = _clean_non_observable_columns(df_with_lags)

    keep_cols = select_log_volatility_only_columns(df_with_lags, include_lags)
    keep_cols = [c for c in keep_cols if c in df_with_lags.columns]
    df_base = drop_rows_with_missing_features(df_with_lags, keep_cols)

    if output_path is None:
        output_path = LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE.with_suffix(".parquet")

    if "ticker" in df_base.columns and "tickers" not in df_base.columns:
        df_base = df_base.rename(columns={"ticker": "tickers"})
    save_parquet_and_csv(df_base, output_path)
    logger.info(
        "Saved base features dataset: %s (%d rows, %d columns)",
        output_path,
        len(df_base),
        len(df_base.columns),
    )

    return df_base


def create_dataset_technical_indicators(
    df: pd.DataFrame | None = None,
    output_path: Path | None = None,
    *,
    include_lags: bool = True,
) -> pd.DataFrame:
    """Create dataset with technical indicators and their lags.

    Args:
        df: Ticker data DataFrame. If None, loads from ticker-level data.
        output_path: Path to save dataset. If None, uses LIGHTGBM_DATASET_TECHNICAL_INDICATORS_FILE.
        include_lags: If True, include lag features.

    Returns:
        Dataset with technical indicators.
    """
    df_prepared = prepare_base_dataframe(df, prefer_insights=False)
    df_with_features = _add_indicators_and_target(df_prepared)

    df_with_lags = _process_lags(
        df_with_features, list(LIGHTGBM_TECHNICAL_FEATURE_COLUMNS), include_lags
    )
    df_with_lags = _clean_non_observable_columns(df_with_lags)

    keep_cols = select_technical_indicators_dataset_columns(df_with_lags, include_lags)
    keep_cols = list(dict.fromkeys(keep_cols))
    df_selected = drop_rows_with_missing_features(df_with_lags, keep_cols)

    if output_path is None:
        output_path = LIGHTGBM_DATASET_TECHNICAL_INDICATORS_FILE.with_suffix(".parquet")

    if "ticker" in df_selected.columns and "tickers" not in df_selected.columns:
        df_selected = df_selected.rename(columns={"ticker": "tickers"})
    save_parquet_and_csv(df_selected, output_path)
    logger.info(
        "Saved technical-indicators dataset: %s (%d rows, %d columns)",
        output_path,
        len(df_selected),
        len(df_selected.columns),
    )

    return df_selected


def ensure_log_volatility_only_dataset(*, include_lags: bool = True) -> Path:
    """Ensure the log-volatility-only dataset exists; create it if missing.

    If the dataset doesn't exist, calls prepare_datasets() to create all datasets
    at once, ensuring they all have the same number of rows.

    Args:
        include_lags: If True, include lag features when available.

    Returns:
        Path to the log-volatility-only dataset.
    """
    dataset_path = LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE.with_suffix(".parquet")

    # Check if complete dataset exists - if not, create all datasets at once
    from src.constants import LIGHTGBM_DATASET_COMPLETE_FILE

    complete_path = LIGHTGBM_DATASET_COMPLETE_FILE.with_suffix(".parquet")
    if not complete_path.exists():
        logger.info(
            "Complete dataset not found. Creating all datasets at once to ensure consistency."
        )
        prepare_datasets()
        return dataset_path

    if dataset_path.exists():
        if include_lags:
            df_existing = pd.read_parquet(dataset_path, engine="pyarrow")
            expected_cols = get_base_columns(include_lags=True)
            missing_cols = [col for col in expected_cols if col not in df_existing.columns]
            if missing_cols:
                logger.info(
                    "Log-volatility-only dataset missing columns %s. Recreating all datasets.",
                    missing_cols,
                )
                prepare_datasets()
        return dataset_path

    # Dataset doesn't exist - create all datasets at once for consistency
    logger.info("Log-volatility-only dataset not found. Creating all datasets at once.")
    prepare_datasets()
    return dataset_path


def ensure_technical_indicators_dataset(*, include_lags: bool = True) -> Path:
    """Ensure the technical indicators dataset exists; create it if missing.

    If the dataset doesn't exist, calls prepare_datasets() to create all datasets
    at once, ensuring they all have the same number of rows.

    Args:
        include_lags: If True, include lag features when available.

    Returns:
        Path to the technical indicators dataset.
    """
    dataset_path = LIGHTGBM_DATASET_TECHNICAL_INDICATORS_FILE.with_suffix(".parquet")

    # Check if complete dataset exists - if not, create all datasets at once
    from src.constants import LIGHTGBM_DATASET_COMPLETE_FILE

    complete_path = LIGHTGBM_DATASET_COMPLETE_FILE.with_suffix(".parquet")
    if not complete_path.exists():
        logger.info(
            "Complete dataset not found. Creating all datasets at once to ensure consistency."
        )
        prepare_datasets()
        return dataset_path

    if dataset_path.exists():
        if include_lags:
            df_existing = pd.read_parquet(dataset_path, engine="pyarrow")
            expected_cols = get_technical_indicator_columns(include_lags=True)
            missing_cols = [col for col in expected_cols if col not in df_existing.columns]
            if missing_cols:
                logger.info(
                    "Technical indicators dataset missing columns %s. Recreating all datasets.",
                    missing_cols,
                )
                prepare_datasets()
        return dataset_path

    # Dataset doesn't exist - create all datasets at once for consistency
    logger.info("Technical indicators dataset not found. Creating all datasets at once.")
    prepare_datasets()
    return dataset_path
