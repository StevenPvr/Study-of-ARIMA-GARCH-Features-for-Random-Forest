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
    LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE,
)
from src.utils import (
    ensure_output_dir,
    get_logger,
    get_parquet_path,
    load_csv_file,
    load_parquet_file,
)

logger = get_logger(__name__)


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
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
    )
    plt.title(f"Spearman Correlation Matrix - {dataset_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
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
    output_dir: Path | None,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Resolve dataset paths to default values if None.

    Args:
        complete_dataset_path: Path to complete dataset or None.
        without_insights_dataset_path: Path to dataset without insights or None.
        log_volatility_only_dataset_path: Path to log-volatility-only dataset or None.
        sigma_plus_base_dataset_path: Path to sigma plus base dataset or None.
        insights_only_dataset_path: Path to insights-only dataset or None.
        output_dir: Output directory or None.

    Returns:
        Tuple of resolved paths (complete, without, log_volatility, sigma,
        insights_only, output_dir).
    """
    return (
        _resolve_path(complete_dataset_path, LIGHTGBM_DATASET_COMPLETE_FILE),
        _resolve_path(without_insights_dataset_path, LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE),
        _resolve_path(log_volatility_only_dataset_path, LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE),
        _resolve_path(sigma_plus_base_dataset_path, LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE),
        _resolve_path(insights_only_dataset_path, LIGHTGBM_DATASET_INSIGHTS_ONLY_FILE),
        _resolve_path(output_dir, LIGHTGBM_CORRELATION_PLOTS_DIR),
    )


def _load_all_datasets(
    complete_path: Path,
    without_path: Path,
    log_volatility_path: Path,
    sigma_path: Path,
    insights_only_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all LightGBM datasets.

    Args:
        complete_path: Path to complete dataset.
        without_path: Path to dataset without insights.
        log_volatility_path: Path to log-volatility-only dataset.
        sigma_path: Path to sigma plus base dataset.
        insights_only_path: Path to insights-only dataset.

    Returns:
        Tuple of loaded DataFrames (complete, without, log_volatility, sigma, insights_only).

    Raises:
        FileNotFoundError: If any dataset file doesn't exist.
    """
    df_complete = load_dataset(complete_path)
    df_without = load_dataset(without_path)
    df_log_volatility_only = load_dataset(log_volatility_path)
    df_sigma_plus_base = load_dataset(sigma_path)
    df_insights_only = load_dataset(insights_only_path)
    return df_complete, df_without, df_log_volatility_only, df_sigma_plus_base, df_insights_only


def _calculate_all_correlations(
    df_complete: pd.DataFrame,
    df_without: pd.DataFrame,
    df_log_volatility_only: pd.DataFrame,
    df_sigma_plus_base: pd.DataFrame,
    df_insights_only: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate Spearman correlations for all datasets.

    Args:
        df_complete: Complete dataset DataFrame.
        df_without: Dataset without insights DataFrame.
        df_log_volatility_only: Log-volatility-only dataset DataFrame.
        df_sigma_plus_base: Sigma plus base dataset DataFrame.
        df_insights_only: Insights-only dataset DataFrame.

    Returns:
        Tuple of correlation matrices (complete, without, log_volatility, sigma, insights_only).

    Raises:
        ValueError: If any dataset is empty or has no numeric columns.
    """
    corr_complete = calculate_spearman_correlation(df_complete)
    corr_without = calculate_spearman_correlation(df_without)
    corr_log_volatility_only = calculate_spearman_correlation(df_log_volatility_only)
    corr_sigma_plus_base = calculate_spearman_correlation(df_sigma_plus_base)
    corr_insights_only = calculate_spearman_correlation(df_insights_only)
    return (
        corr_complete,
        corr_without,
        corr_log_volatility_only,
        corr_sigma_plus_base,
        corr_insights_only,
    )


def _save_all_correlation_plots(
    corr_complete: pd.DataFrame,
    corr_without: pd.DataFrame,
    corr_log_volatility_only: pd.DataFrame,
    corr_sigma_plus_base: pd.DataFrame,
    corr_insights_only: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create and save correlation plots for all datasets.

    Args:
        corr_complete: Complete dataset correlation matrix.
        corr_without: Without insights correlation matrix.
        corr_log_volatility_only: Log-volatility-only correlation matrix.
        corr_sigma_plus_base: Sigma plus base correlation matrix.
        corr_insights_only: Insights-only correlation matrix.
        output_dir: Directory to save plots.
    """
    output_complete = output_dir / "lightgbm_correlation_complete.png"
    output_without = output_dir / "lightgbm_correlation_without_insights.png"
    output_log_volatility_only = output_dir / "lightgbm_correlation_log_volatility_only.png"
    output_sigma_plus_base = output_dir / "lightgbm_correlation_sigma_plus_base.png"
    output_insights_only = output_dir / "lightgbm_correlation_insights_only.png"

    ensure_output_dir(output_complete)
    plot_correlation_matrix(corr_complete, output_complete, "Complete Dataset")
    plot_correlation_matrix(corr_without, output_without, "Dataset Without Insights")
    plot_correlation_matrix(
        corr_log_volatility_only, output_log_volatility_only, "Log Volatility Only Dataset"
    )
    plot_correlation_matrix(corr_sigma_plus_base, output_sigma_plus_base, "Sigma Plus Base Dataset")
    plot_correlation_matrix(corr_insights_only, output_insights_only, "Insights Only Dataset")


def compute_correlations(
    complete_dataset_path: Path | None = None,
    without_insights_dataset_path: Path | None = None,
    log_volatility_only_dataset_path: Path | None = None,
    sigma_plus_base_dataset_path: Path | None = None,
    insights_only_dataset_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute Spearman correlations and create visualizations for all LightGBM datasets.

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
        output_dir: Directory to save correlation plots.
            If None, uses LIGHTGBM_CORRELATION_PLOTS_DIR.

    Returns:
        Tuple of (complete_corr, without_insights_corr, log_volatility_only_corr,
        sigma_plus_base_corr, insights_only_corr).

    Raises:
        FileNotFoundError: If dataset files don't exist.
        ValueError: If datasets are empty or have no numeric columns.
    """
    # Resolve paths to defaults if None
    (
        complete_path,
        without_path,
        log_volatility_path,
        sigma_path,
        insights_only_path,
        output_dir_resolved,
    ) = _resolve_dataset_paths(
        complete_dataset_path,
        without_insights_dataset_path,
        log_volatility_only_dataset_path,
        sigma_plus_base_dataset_path,
        insights_only_dataset_path,
        output_dir,
    )

    # Load all datasets
    df_complete, df_without, df_log_volatility_only, df_sigma_plus_base, df_insights_only = (
        _load_all_datasets(
            complete_path, without_path, log_volatility_path, sigma_path, insights_only_path
        )
    )

    # Calculate correlations
    (
        corr_complete,
        corr_without,
        corr_log_volatility_only,
        corr_sigma_plus_base,
        corr_insights_only,
    ) = _calculate_all_correlations(
        df_complete, df_without, df_log_volatility_only, df_sigma_plus_base, df_insights_only
    )

    # Create and save plots
    _save_all_correlation_plots(
        corr_complete,
        corr_without,
        corr_log_volatility_only,
        corr_sigma_plus_base,
        corr_insights_only,
        output_dir_resolved,
    )

    return (
        corr_complete,
        corr_without,
        corr_log_volatility_only,
        corr_sigma_plus_base,
        corr_insights_only,
    )
