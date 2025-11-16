"""CLI for visualizing returns and squared/absolute returns.

Generates plots for volatility clustering and autocorrelation of returns and squared returns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.constants import GARCH_ACF_LAGS_DEFAULT
from src.garch.garch_data_visualisation.plots import (
    plot_residuals_distribution,
    plot_returns_autocorrelation,
    save_returns_and_squared_plots,
)
from src.garch.garch_data_visualisation.utils import (
    extract_dates_from_dataframe,
    prepare_test_dataframe,
)
from src.garch.structure_garch.utils import load_garch_dataset
from src.path import GARCH_DATA_VISU_PLOTS_DIR, GARCH_DATASET_FILE, GARCH_RETURNS_CLUSTERING_PLOT
from src.utils import get_logger

logger = get_logger(__name__)


def _generate_returns_plot(df: pd.DataFrame) -> None:
    """Generate returns clustering plot.

    Args:
        df: DataFrame with returns data

    Raises:
        ValueError: If test dataframe is None or missing required columns
        ValueError: If no finite returns found
    """
    df_test = prepare_test_dataframe(df)
    if df_test is None:
        raise ValueError("Failed to prepare test dataframe for returns plot")
    if "weighted_return" not in df_test.columns:
        raise ValueError("DataFrame missing 'weighted_return' column")

    returns_array = df_test["weighted_return"].to_numpy().astype(float)
    # Filter out non-finite values
    returns_finite = returns_array[np.isfinite(returns_array)]
    if returns_finite.size == 0:
        raise ValueError("No finite returns found in test dataframe")

    try:
        dates_param = extract_dates_from_dataframe(df_test)
        save_returns_and_squared_plots(
            returns_array,
            dates=dates_param,
            outdir=GARCH_DATA_VISU_PLOTS_DIR,
            filename=GARCH_RETURNS_CLUSTERING_PLOT.name,
        )
    except Exception as e:
        logger.error(f"Failed to generate returns clustering plot: {e}")
        raise


def _generate_autocorrelation_plot(df: pd.DataFrame) -> None:
    """Generate autocorrelation plots for returns and squared returns.

    Args:
        df: DataFrame with returns data

    Raises:
        ValueError: If test dataframe is None or missing required columns
        ValueError: If no finite returns found
    """
    df_test = prepare_test_dataframe(df)
    if df_test is None:
        raise ValueError("Failed to prepare test dataframe for autocorrelation plot")
    if "weighted_return" not in df_test.columns:
        raise ValueError("DataFrame missing 'weighted_return' column")

    returns_array = df_test["weighted_return"].to_numpy().astype(float)
    # Filter out non-finite values
    returns_finite = returns_array[np.isfinite(returns_array)]
    if returns_finite.size == 0:
        raise ValueError("No finite returns found in test dataframe")

    try:
        plot_returns_autocorrelation(
            returns_array,
            lags=GARCH_ACF_LAGS_DEFAULT,
            outdir=GARCH_DATA_VISU_PLOTS_DIR,
            filename="garch_returns_autocorrelation.png",
        )
    except Exception as e:
        logger.error(f"Failed to generate autocorrelation plot: {e}")
        raise


def _generate_residuals_distribution_plot(df: pd.DataFrame) -> None:
    """Generate distribution plot for ARIMA residuals.

    Args:
        df: DataFrame with ARIMA residuals

    Raises:
        ValueError: If dataframe is missing required columns
        ValueError: If no finite residuals found
    """
    if "sarima_resid" not in df.columns:
        raise ValueError("DataFrame missing 'sarima_resid' column")

    residuals_array = df["sarima_resid"].to_numpy().astype(float)
    # Filter out non-finite values
    residuals_finite = residuals_array[np.isfinite(residuals_array)]
    if residuals_finite.size == 0:
        raise ValueError("No finite residuals found in dataframe")

    try:
        plot_residuals_distribution(
            residuals_array,
            outdir=GARCH_DATA_VISU_PLOTS_DIR,
            filename="arima_residuals_distribution.png",
        )
    except Exception as e:
        logger.error(f"Failed to generate residuals distribution plot: {e}")
        raise


def main() -> None:
    """Create returns and squared returns visualizations for test split."""
    logger.info("=" * 60)
    logger.info(
        "GARCH VISUALIZATION (returns, |returns|, returns^2, ACF(returns), "
        "ACF(returns^2), ARIMA residuals distribution)"
    )
    logger.info("=" * 60)

    try:
        df = load_garch_dataset(str(GARCH_DATASET_FILE))
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load GARCH dataset: {e}")
        raise

    # Generate returns clustering plot (point 3)
    _generate_returns_plot(df)

    # Generate autocorrelation plots (point 4)
    _generate_autocorrelation_plot(df)

    # Generate ARIMA residuals distribution plot
    _generate_residuals_distribution_plot(df)

    logger.info("Saved visualization plots to: %s", GARCH_DATA_VISU_PLOTS_DIR)


if __name__ == "__main__":
    main()
