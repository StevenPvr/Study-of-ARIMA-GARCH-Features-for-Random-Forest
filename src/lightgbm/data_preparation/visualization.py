"""Visualization utilities for LightGBM data preparation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.constants import (
    PLOT_ALPHA_DEFAULT,
    PLOT_DPI,
    PLOT_FONTSIZE_LABEL,
    PLOT_FONTSIZE_TITLE,
    PLOTS_DIR,
)
from src.utils import get_logger
from src.visualization import save_plot_wrapper

logger = get_logger(__name__)


def plot_log_volatility_distribution(
    df: pd.DataFrame,
    *,
    output_dir: Path | None = None,
) -> None:
    """Plot the distribution of log_volatility column.

    Args:
        df: DataFrame containing the log_volatility column.
        output_dir: Directory to save the plot. If None, uses default plots directory.
    """
    if output_dir is None:
        output_dir = PLOTS_DIR / "lightgbm" / "data_preparation"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    if "log_volatility" not in df.columns:
        logger.warning("log_volatility column not found in DataFrame, skipping plot")
        return

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Histogram with KDE
    sns.histplot(
        data=df,
        x="log_volatility",
        kde=True,
        ax=ax1,
        alpha=PLOT_ALPHA_DEFAULT,
        color="skyblue",
    )
    ax1.set_title(
        "Distribution de log_volatility (Histogramme + KDE)", fontsize=PLOT_FONTSIZE_TITLE
    )
    ax1.set_xlabel("log_volatility", fontsize=PLOT_FONTSIZE_LABEL)
    ax1.set_ylabel("Fréquence", fontsize=PLOT_FONTSIZE_LABEL)
    ax1.grid(True, alpha=0.3)

    # Box plot
    sns.boxplot(
        data=df,
        y="log_volatility",
        ax=ax2,
        color="lightcoral",
        width=0.3,
    )
    ax2.set_title("Distribution de log_volatility (Box Plot)", fontsize=PLOT_FONTSIZE_TITLE)
    ax2.set_ylabel("log_volatility", fontsize=PLOT_FONTSIZE_LABEL)
    ax2.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    output_file = output_dir / "log_volatility_distribution.png"
    save_plot_wrapper(str(output_file), dpi=PLOT_DPI)

    logger.info("Saved log_volatility distribution plot to: %s", output_file)
