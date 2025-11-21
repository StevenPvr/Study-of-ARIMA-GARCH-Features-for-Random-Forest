"""Spearman correlation calculation and visualization for LightGBM datasets."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Set non-interactive backend before importing pyplot
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.constants import (
    LIGHTGBM_CORRELATION_PLOTS_DIR,
    LIGHTGBM_DATASET_COMPLETE_FILE,
    LIGHTGBM_DATASET_INSIGHTS_ONLY_FILE,
    LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE,
    LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE,
    LIGHTGBM_DATASET_TECHNICAL_ONLY_NO_TARGET_LAGS_FILE,
    LIGHTGBM_DATASET_TECHNICAL_PLUS_INSIGHTS_NO_TARGET_LAGS_FILE,
    LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE,
)
from typing import NamedTuple, cast
from src.utils import (
    ensure_output_dir,
    get_logger,
    get_parquet_path,
    load_csv_file,
    load_parquet_file,
)

logger = get_logger(__name__)


class DatasetConfig(NamedTuple):
    """Configuration for a LightGBM dataset."""

    name: str
    default_path: Path
    plot_filename: str
    display_name: str


# Configuration for all datasets to avoid repetition
DATASET_CONFIGS = [
    DatasetConfig(
        name="complete",
        default_path=LIGHTGBM_DATASET_COMPLETE_FILE,
        plot_filename="lightgbm_correlation_complete.png",
        display_name="Complete Dataset",
    ),
    DatasetConfig(
        name="without_insights",
        default_path=LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE,
        plot_filename="lightgbm_correlation_without_insights.png",
        display_name="Dataset Without Insights",
    ),
    DatasetConfig(
        name="log_volatility_only",
        default_path=LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE,
        plot_filename="lightgbm_correlation_log_volatility_only.png",
        display_name="Log Volatility Only Dataset",
    ),
    DatasetConfig(
        name="sigma_plus_base",
        default_path=LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE,
        plot_filename="lightgbm_correlation_sigma_plus_base.png",
        display_name="Sigma Plus Base Dataset",
    ),
    DatasetConfig(
        name="insights_only",
        default_path=LIGHTGBM_DATASET_INSIGHTS_ONLY_FILE,
        plot_filename="lightgbm_correlation_insights_only.png",
        display_name="Insights Only Dataset",
    ),
    DatasetConfig(
        name="technical_only_no_target_lags",
        default_path=LIGHTGBM_DATASET_TECHNICAL_ONLY_NO_TARGET_LAGS_FILE,
        plot_filename="lightgbm_correlation_technical_only_no_target_lags.png",
        display_name="Technical Only No Target Lags Dataset",
    ),
    DatasetConfig(
        name="technical_plus_insights_no_target_lags",
        default_path=LIGHTGBM_DATASET_TECHNICAL_PLUS_INSIGHTS_NO_TARGET_LAGS_FILE,
        plot_filename="lightgbm_correlation_technical_plus_insights_no_target_lags.png",
        display_name="Technical Plus Insights No Target Lags Dataset",
    ),
]


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load a LightGBM dataset.

    Tries to load Parquet file first, falls back to CSV if Parquet doesn't exist.

    Args:
        file_path: Path to the dataset file (CSV or Parquet).

    Returns:
        DataFrame with the dataset.

    Raises:
        FileNotFoundError: If neither Parquet nor CSV file exists.
    """
    parquet_path = get_parquet_path(file_path)

    df = load_parquet_file(parquet_path)
    if df is not None:
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df

    if file_path.exists():
        df = load_csv_file(file_path)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df

    raise FileNotFoundError(
        f"Dataset file not found: {file_path} (tried {parquet_path} and {file_path})"
    )


def calculate_spearman_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Spearman rank correlation matrix.

    Excludes non-numeric columns (date, split) and calculates correlation
    only on numeric features.

    Args:
        df: DataFrame with numeric and non-numeric columns.

    Returns:
        Correlation matrix as DataFrame.

    Raises:
        ValueError: If DataFrame is empty or has no numeric columns.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    # Select only numeric columns (exclude date, split, etc.)
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    if not numeric_cols:
        raise ValueError("No numeric columns found in DataFrame")

    logger.info(f"Calculating Spearman correlation for {len(numeric_cols)} numeric columns")
    numeric_df = df[numeric_cols]
    # DataFrame.corr() can be called without 'other' parameter to compute correlation matrix
    corr_matrix = numeric_df.corr(method="spearman")  # type: ignore[call-arg]

    logger.info("Spearman correlation matrix calculated successfully")
    return corr_matrix


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    output_path: Path,
    dataset_name: str,
    figsize: tuple[int, int] = (12, 10),
) -> None:
    """Create and save a heatmap visualization of the correlation matrix.

    Args:
        corr_matrix: Correlation matrix DataFrame.
        output_path: Path to save the plot.
        dataset_name: Name of the dataset (for logging).
        figsize: Figure size (width, height) in inches.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,  # type: ignore
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
    )
    plt.title(label=f"Spearman Correlation Matrix - {dataset_name}", fontsize=14, fontweight="bold")  # type: ignore
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  # type: ignore
    plt.close()
    logger.info(f"Saved correlation plot for {dataset_name}: {output_path}")


def _resolve_path(path: Path | None, default: Path) -> Path:
    """Resolve a single path to default if None.

    Args:
        path: Path or None.
        default: Default path to use if path is None.

    Returns:
        Resolved path.
    """
    return path if path is not None else default


def _resolve_dataset_paths(
    complete_dataset_path: Path | None,
    without_insights_dataset_path: Path | None,
    log_volatility_only_dataset_path: Path | None,
    sigma_plus_base_dataset_path: Path | None,
    insights_only_dataset_path: Path | None,
    technical_only_no_target_lags_dataset_path: Path | None,
    technical_plus_insights_no_target_lags_dataset_path: Path | None,
    output_dir: Path | None,
) -> list[Path]:
    """Resolve dataset paths to default values if None.

    Args:
        complete_dataset_path: Path to complete dataset or None.
        without_insights_dataset_path: Path to dataset without insights or None.
        log_volatility_only_dataset_path: Path to log-volatility-only dataset or None.
        sigma_plus_base_dataset_path: Path to sigma plus base dataset or None.
        insights_only_dataset_path: Path to insights-only dataset or None.
        technical_only_no_target_lags_dataset_path: Path to technical-only dataset
            without target lags or None.
        technical_plus_insights_no_target_lags_dataset_path: Path to technical-plus-insights
            dataset without target lags or None.
        output_dir: Output directory or None.

    Returns:
        List of resolved paths in same order as DATASET_CONFIGS, plus output_dir at end.
    """
    custom_paths = [
        complete_dataset_path,
        without_insights_dataset_path,
        log_volatility_only_dataset_path,
        sigma_plus_base_dataset_path,
        insights_only_dataset_path,
        technical_only_no_target_lags_dataset_path,
        technical_plus_insights_no_target_lags_dataset_path,
    ]

    resolved_paths = [
        _resolve_path(custom_path, config.default_path)
        for custom_path, config in zip(custom_paths, DATASET_CONFIGS, strict=False)
    ]
    resolved_paths.append(_resolve_path(output_dir, LIGHTGBM_CORRELATION_PLOTS_DIR))

    return resolved_paths


def _load_all_datasets(dataset_paths: list[Path]) -> list[pd.DataFrame]:
    """Load all LightGBM datasets.

    Args:
        dataset_paths: List of paths to datasets in same order as DATASET_CONFIGS.

    Returns:
        List of loaded DataFrames in same order as dataset_paths.

    Raises:
        FileNotFoundError: If any dataset file doesn't exist.
    """
    return [load_dataset(path) for path in dataset_paths]


def _calculate_all_correlations(dataframes: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Calculate Spearman correlations for all datasets.

    Args:
        dataframes: List of DataFrames in same order as DATASET_CONFIGS.

    Returns:
        List of correlation matrices in same order as dataframes.

    Raises:
        ValueError: If any dataset is empty or has no numeric columns.
    """
    return [calculate_spearman_correlation(df) for df in dataframes]


def _save_all_correlation_plots(correlations: list[pd.DataFrame], output_dir: Path) -> None:
    """Create and save correlation plots for all datasets.

    Args:
        correlations: List of correlation matrices in same order as DATASET_CONFIGS.
        output_dir: Directory to save plots.
    """
    for correlation, config in zip(correlations, DATASET_CONFIGS, strict=False):
        output_path = output_dir / config.plot_filename
        ensure_output_dir(output_path)
        plot_correlation_matrix(correlation, output_path, config.display_name)


def _process_correlations_workflow(
    dataset_paths: list[Path], output_dir: Path
) -> list[pd.DataFrame]:
    """Process the complete correlation workflow: load, calculate, and save plots.

    Args:
        dataset_paths: List of paths to datasets.
        output_dir: Directory to save correlation plots.

    Returns:
        List of correlation matrices.

    Raises:
        FileNotFoundError: If dataset files don't exist.
        ValueError: If datasets are empty or have no numeric columns.
    """
    dataframes = _load_all_datasets(dataset_paths)
    correlations = _calculate_all_correlations(dataframes)
    _save_all_correlation_plots(correlations, output_dir)
    return correlations


def _prepare_correlation_parameters(
    complete_dataset_path: Path | None,
    without_insights_dataset_path: Path | None,
    log_volatility_only_dataset_path: Path | None,
    sigma_plus_base_dataset_path: Path | None,
    insights_only_dataset_path: Path | None,
    technical_only_no_target_lags_dataset_path: Path | None,
    technical_plus_insights_no_target_lags_dataset_path: Path | None,
    output_dir: Path | None,
) -> tuple[list[Path], Path]:
    """Prepare and resolve all parameters for correlation computation.

    Args:
        complete_dataset_path: Path to complete dataset or None.
        without_insights_dataset_path: Path to dataset without insights or None.
        log_volatility_only_dataset_path: Path to log-volatility-only dataset or None.
        sigma_plus_base_dataset_path: Path to sigma plus base dataset or None.
        insights_only_dataset_path: Path to insights-only dataset or None.
        technical_only_no_target_lags_dataset_path: Path to technical-only dataset
            without target lags or None.
        technical_plus_insights_no_target_lags_dataset_path: Path to technical-plus-insights
            dataset without target lags or None.
        output_dir: Output directory or None.

    Returns:
        Tuple of (dataset_paths, output_dir_resolved).
    """
    resolved_paths = _resolve_dataset_paths(
        complete_dataset_path,
        without_insights_dataset_path,
        log_volatility_only_dataset_path,
        sigma_plus_base_dataset_path,
        insights_only_dataset_path,
        technical_only_no_target_lags_dataset_path,
        technical_plus_insights_no_target_lags_dataset_path,
        output_dir,
    )

    dataset_paths = resolved_paths[:-1]  # All except output_dir
    output_dir_resolved = resolved_paths[-1]

    return dataset_paths, output_dir_resolved


def compute_correlations(
    complete_dataset_path: Path | None = None,
    without_insights_dataset_path: Path | None = None,
    log_volatility_only_dataset_path: Path | None = None,
    sigma_plus_base_dataset_path: Path | None = None,
    insights_only_dataset_path: Path | None = None,
    technical_only_no_target_lags_dataset_path: Path | None = None,
    technical_plus_insights_no_target_lags_dataset_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Compute Spearman correlations for all LightGBM datasets.

    Args:
        complete_dataset_path: Path to complete dataset. If None, uses default.
        without_insights_dataset_path: Path to dataset without insights.
            If None, uses default.
        log_volatility_only_dataset_path: Path to log-volatility-only dataset.
            If None, uses default.
        sigma_plus_base_dataset_path: Path to sigma plus base dataset.
            If None, uses default.
        insights_only_dataset_path: Path to insights-only dataset.
            If None, uses default.
        technical_only_no_target_lags_dataset_path: Path to technical-only dataset
            without target lags. If None, uses default.
        technical_plus_insights_no_target_lags_dataset_path: Path to technical-plus-insights
            dataset without target lags. If None, uses default.
        output_dir: Directory to save correlation plots.
            If None, uses LIGHTGBM_CORRELATION_PLOTS_DIR.

    Returns:
        Tuple of correlation matrices for all datasets.

    Raises:
        FileNotFoundError: If dataset files don't exist.
        ValueError: If datasets are empty or have no numeric columns.
    """
    dataset_paths, output_dir_resolved = _prepare_correlation_parameters(
        complete_dataset_path,
        without_insights_dataset_path,
        log_volatility_only_dataset_path,
        sigma_plus_base_dataset_path,
        insights_only_dataset_path,
        technical_only_no_target_lags_dataset_path,
        technical_plus_insights_no_target_lags_dataset_path,
        output_dir,
    )

    correlations = _process_correlations_workflow(dataset_paths, output_dir_resolved)
    # Return as properly typed tuple for backward compatibility
    return cast(
        tuple[
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
        ],
        tuple(correlations),
    )
