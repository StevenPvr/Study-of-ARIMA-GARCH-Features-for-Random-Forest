"""Plotting utilities for visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from src.constants import (
    PLOT_ALPHA_DEFAULT,
    PLOT_ALPHA_FILL,
    PLOT_ALPHA_LIGHT,
    PLOT_ALPHA_MEDIUM,
    PLOT_FIGURE_SIZE_SEASONAL_DAILY,
    PLOT_FIGURE_SIZE_SEASONAL_FULL,
    PLOT_FIGURE_SIZE_SEASONAL_YEAR,
    PLOT_FONTSIZE_AXIS,
    PLOT_FONTSIZE_LABEL,
    PLOT_FONTSIZE_SUBTITLE,
    PLOT_FONTSIZE_TEXT,
    PLOT_FONTSIZE_TITLE,
    PLOT_LINEWIDTH_BOLD,
    PLOT_LINEWIDTH_DEFAULT,
    PLOT_LINEWIDTH_MEDIUM,
    PLOT_LINEWIDTH_THIN,
    STATIONARITY_SEPARATOR_LENGTH,
    STATIONARITY_TEXT_BOX_X,
    STATIONARITY_TEXT_BOX_Y,
)
from src.utils import get_logger
from src.visualization import (
    add_grid,
    add_legend,
    add_zero_line,
    create_standard_figure,
    format_date_axis,
    format_seasonal_axis_daily,
    save_plot_wrapper,
)

logger = get_logger(__name__)


def _format_date_axis(ax: Axes) -> None:
    """Format x-axis dates for better readability.

    Wrapper for backward compatibility.

    Args:
        ax: Matplotlib axes to format.
    """
    format_date_axis(ax)


def _save_plot(output_file: str) -> None:
    """Save the current plot to file.

    Wrapper for backward compatibility.

    Args:
        output_file: Path to save the plot.
    """
    save_plot_wrapper(output_file)


def _format_seasonal_axis_daily(ax: Axes) -> None:
    """Format x-axis for daily seasonal plots (1 year period).

    Wrapper for backward compatibility.

    Args:
        ax: Matplotlib axes to format.
    """
    format_seasonal_axis_daily(ax)


def _plot_seasonal_component_only(
    seasonal: pd.Series, *, title: str, full_period: bool = False
) -> None:
    """Plot a single seasonal component line with basic styling.

    Args:
        seasonal: Seasonal component series to plot.
        title: Plot title.
        full_period: If True, format for full period (10 years) with weekly resampling.
    """
    if full_period:
        _, ax = create_standard_figure(figsize=PLOT_FIGURE_SIZE_SEASONAL_FULL)
        ax.plot(
            seasonal.index, seasonal, linewidth=PLOT_LINEWIDTH_DEFAULT, alpha=PLOT_ALPHA_DEFAULT
        )
        ax.set_xlabel("Date", fontsize=PLOT_FONTSIZE_LABEL)
        ax.set_ylabel("Seasonal", fontsize=PLOT_FONTSIZE_LABEL)
        ax.set_title(title, fontsize=PLOT_FONTSIZE_TITLE)
        _format_date_axis(ax)
    else:
        _, ax = create_standard_figure(figsize=PLOT_FIGURE_SIZE_SEASONAL_YEAR)
        ax.plot(seasonal.index, seasonal, linewidth=PLOT_LINEWIDTH_MEDIUM)
        ax.set_xlabel("Date")
        ax.set_ylabel("Seasonal")
        ax.set_title(title, fontsize=PLOT_FONTSIZE_SUBTITLE)
    add_zero_line(ax, linewidth=PLOT_LINEWIDTH_THIN)
    add_grid(ax, alpha=PLOT_ALPHA_LIGHT, linestyle="--")
    plt.tight_layout()


def plot_seasonal_daily_long_period(seasonal: pd.Series, *, title: str) -> None:
    """Plot seasonal component for daily data over period (typically 1 year).

    Uses a standard figure size optimized for 1 year of daily data.

    Args:
        seasonal: Seasonal component series to plot.
        title: Plot title.
    """
    _, ax = create_standard_figure(figsize=PLOT_FIGURE_SIZE_SEASONAL_DAILY)
    ax.plot(seasonal.index, seasonal, linewidth=PLOT_LINEWIDTH_DEFAULT, alpha=PLOT_ALPHA_MEDIUM)
    add_zero_line(ax, linewidth=PLOT_LINEWIDTH_THIN)
    ax.set_title(title, fontsize=PLOT_FONTSIZE_TITLE)
    ax.set_xlabel("Date", fontsize=PLOT_FONTSIZE_LABEL)
    ax.set_ylabel("Seasonal Component", fontsize=PLOT_FONTSIZE_LABEL)
    add_grid(ax, alpha=PLOT_ALPHA_LIGHT, linestyle="--")
    _format_seasonal_axis_daily(ax)
    plt.tight_layout()


def plot_stationarity_series_with_bands(
    ax: Axes,
    weighted_series: pd.Series,
    rolling_mean: pd.Series,
    rolling_std: pd.Series,
    rolling_window: int,
) -> None:
    """Plot original series with rolling mean and std bands.

    Args:
        ax: Matplotlib axes to plot on.
        weighted_series: Original time series.
        rolling_mean: Rolling mean series.
        rolling_std: Rolling standard deviation series.
        rolling_window: Window size for rolling statistics.
    """
    ax.plot(
        weighted_series.index,
        weighted_series,
        label="Série originale",
        linewidth=PLOT_LINEWIDTH_THIN,
        alpha=PLOT_ALPHA_MEDIUM,
        color="blue",
    )
    ax.plot(
        rolling_mean.index,
        rolling_mean,
        label=f"Moyenne mobile (window={rolling_window})",
        linewidth=PLOT_LINEWIDTH_BOLD,
        color="red",
    )
    ax.fill_between(
        rolling_mean.index,
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        alpha=PLOT_ALPHA_FILL,
        color="red",
        label=f"±1 écart-type (window={rolling_window})",
    )
    add_zero_line(ax, linewidth=PLOT_LINEWIDTH_THIN, alpha=0.5)
    ax.set_title(
        "Série temporelle avec statistiques mobiles",
        fontsize=PLOT_FONTSIZE_SUBTITLE,
        fontweight="bold",
    )
    ax.set_ylabel("Rendement logarithmique pondéré", fontsize=PLOT_FONTSIZE_AXIS)
    add_legend(ax)
    add_grid(ax, alpha=PLOT_ALPHA_LIGHT, linestyle="--")
    _format_date_axis(ax)


def plot_stationarity_rolling_mean(
    ax: Axes,
    rolling_mean: pd.Series,
    global_mean: float,
) -> None:
    """Plot rolling mean over time with global mean reference.

    Args:
        ax: Matplotlib axes to plot on.
        rolling_mean: Rolling mean series.
        global_mean: Global mean value.
    """
    ax.plot(rolling_mean.index, rolling_mean, linewidth=PLOT_LINEWIDTH_BOLD, color="red")
    ax.axhline(
        global_mean,
        color="green",
        linewidth=PLOT_LINEWIDTH_BOLD,
        linestyle="--",
        label=f"Moyenne globale: {global_mean:.6f}",
    )
    ax.set_title(
        "Évolution de la moyenne mobile", fontsize=PLOT_FONTSIZE_SUBTITLE, fontweight="bold"
    )
    ax.set_ylabel("Moyenne mobile", fontsize=PLOT_FONTSIZE_AXIS)
    add_legend(ax)
    add_grid(ax, alpha=PLOT_ALPHA_LIGHT, linestyle="--")
    _format_date_axis(ax)


def plot_stationarity_rolling_std(
    ax: Axes,
    rolling_std: pd.Series,
    global_std: float,
) -> None:
    """Plot rolling std over time with global std reference.

    Args:
        ax: Matplotlib axes to plot on.
        rolling_std: Rolling standard deviation series.
        global_std: Global standard deviation value.
    """
    ax.plot(rolling_std.index, rolling_std, linewidth=PLOT_LINEWIDTH_BOLD, color="orange")
    ax.axhline(
        global_std,
        color="green",
        linewidth=PLOT_LINEWIDTH_BOLD,
        linestyle="--",
        label=f"Écart-type global: {global_std:.6f}",
    )
    ax.set_title(
        "Évolution de l'écart-type mobile", fontsize=PLOT_FONTSIZE_SUBTITLE, fontweight="bold"
    )
    ax.set_ylabel("Écart-type mobile", fontsize=PLOT_FONTSIZE_AXIS)
    ax.set_xlabel("Date", fontsize=PLOT_FONTSIZE_AXIS)
    add_legend(ax)
    add_grid(ax, alpha=PLOT_ALPHA_LIGHT, linestyle="--")
    _format_date_axis(ax)


def format_stationarity_test_text(
    alpha: float,
    adf_stat: float,
    adf_pval: float,
    adf_lags: int,
    kpss_stat: float,
    kpss_pval: float,
    kpss_lags: int,
    overall_verdict: str,
) -> str:
    """Format stationarity test results as text.

    Args:
        alpha: Significance level.
        adf_stat: ADF test statistic.
        adf_pval: ADF test p-value.
        adf_lags: ADF test lags.
        kpss_stat: KPSS test statistic.
        kpss_pval: KPSS test p-value.
        kpss_lags: KPSS test lags.
        overall_verdict: Overall stationarity verdict.

    Returns:
        Formatted text string with test results.
    """
    adf_verdict = "Stationnaire" if adf_pval < alpha else "Non stationnaire"
    kpss_verdict = (
        "Stationnaire" if kpss_pval > alpha or (kpss_pval != kpss_pval) else "Non stationnaire"
    )

    return (
        f"Résultats des tests de stationnarité (α = {alpha})\n"
        f"{'=' * STATIONARITY_SEPARATOR_LENGTH}\n"
        f"Verdict global: {overall_verdict}\n"
        f"\n"
        f"Test ADF (Augmented Dickey-Fuller):\n"
        f"  Statistique: {adf_stat:.6f}\n"
        f"  p-value: {adf_pval:.6f}\n"
        f"  Lags: {adf_lags}\n"
        f"  Verdict: {adf_verdict} (p-value {'<' if adf_pval < alpha else '≥'} {alpha})\n"
        f"\n"
        f"Test KPSS (Kwiatkowski-Phillips-Schmidt-Shin):\n"
        f"  Statistique: {kpss_stat:.6f}\n"
        f"  p-value: {kpss_pval:.6f}\n"
        f"  Lags: {kpss_lags}\n"
        f"  Verdict: {kpss_verdict} "
        f"(p-value {'>' if kpss_pval > alpha or (kpss_pval != kpss_pval) else '≤'} {alpha})"
    )


def add_stationarity_text_box(ax: Axes, test_text: str) -> None:
    """Add stationarity test results as text box to axes.

    Args:
        ax: Matplotlib axes to add text box to.
        test_text: Formatted test results text.
    """
    ax.text(
        STATIONARITY_TEXT_BOX_X,
        STATIONARITY_TEXT_BOX_Y,
        test_text,
        transform=ax.transAxes,
        fontsize=PLOT_FONTSIZE_TEXT,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=PLOT_ALPHA_DEFAULT, edgecolor="black"),
        family="monospace",
    )
