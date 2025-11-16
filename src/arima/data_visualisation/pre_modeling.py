"""Pre-modeling visualizations for ARIMA analysis.

This module contains all visualizations used before model fitting:
- Time series plots
- Stationarity analysis
- Seasonality decomposition
- ACF/PACF analysis
- Distribution of log returns
"""

from __future__ import annotations

from typing import cast

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.arima.stationnarity_check.stationnarity_check import (
    StationarityReport,
    evaluate_stationarity,
)
from src.constants import (
    ACF_PACF_CONFIDENCE_ALPHA,
    ACF_PACF_DEFAULT_LAGS,
    ACF_PACF_MIN_LAGS,
    DISTRIBUTION_HISTOGRAM_BINS_DEFAULT,
    PLOT_ALPHA_DEFAULT,
    PLOT_ALPHA_LIGHT,
    PLOT_COLOR_NORMAL_FIT,
    PLOT_COLOR_RESIDUAL,
    PLOT_COLOR_SERIES_ORIGINAL,
    PLOT_FIGURE_SIZE_ACF_PACF,
    PLOT_FIGURE_SIZE_DEFAULT,
    PLOT_FIGURE_SIZE_STATIONARITY,
    PLOT_FIGURE_SIZE_WEIGHTED_SERIES,
    PLOT_FONTSIZE_AXIS,
    PLOT_FONTSIZE_LABEL,
    PLOT_FONTSIZE_SUBTITLE,
    PLOT_FONTSIZE_TITLE,
    PLOT_LINEWIDTH_DEFAULT,
    PLOT_TEXTBOX_STYLE_DEFAULT,
    SEASONAL_DEFAULT_MODEL,
    SEASONAL_DEFAULT_PERIOD_DAILY,
    SEASONAL_DEFAULT_PERIOD_MONTHLY,
    SEASONAL_DEFAULT_PERIOD_WEEKLY,
    SEASONAL_DEFAULT_YEARS,
    SEASONAL_MIN_PERIODS,
    SEASONAL_RESAMPLE_FREQ_BUSINESS,
    SEASONAL_RESAMPLE_FREQ_MONTHLY,
    SEASONAL_RESAMPLE_FREQ_WEEKLY,
    STATIONARITY_ALPHA_DEFAULT,
    STATIONARITY_ROLLING_WINDOW_DEFAULT,
    STATIONARITY_SUPTITLE_Y,
)
from src.utils import get_logger
from src.visualization import (
    add_grid,
    add_zero_line,
    create_standard_figure,
    plot_histogram_with_normal_overlay,
)

from .data_loading import load_and_validate_data, load_series_for_year
from .plotting import (
    _format_date_axis,
    _plot_seasonal_component_only,
    _save_plot,
    add_stationarity_text_box,
    format_stationarity_test_text,
    plot_seasonal_daily_long_period,
    plot_stationarity_rolling_mean,
    plot_stationarity_rolling_std,
    plot_stationarity_series_with_bands,
)
from .seasonal import (
    _decompose_seasonal_component,
    _infer_seasonal_period,
    _maybe_resample,
)
from .validation import validate_minimum_periods, validate_seasonal_params

logger = get_logger(__name__)


def plot_weighted_series(
    data_file: str,
    output_file: str,
) -> None:
    """Plot the time series of weighted log-returns.

    Args:
        data_file: Path to weighted log-returns CSV file.
        output_file: Path to save the plot.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If required column is missing or data is empty.
    """
    logger.info("Loading weighted log-returns data")
    aggregated_returns = load_and_validate_data(data_file, "weighted_log_return")
    weighted_series = aggregated_returns["weighted_log_return"].dropna()

    # Resample to weekly for readability
    weighted_series_resampled = weighted_series.resample(SEASONAL_RESAMPLE_FREQ_WEEKLY).mean()

    _, ax = create_standard_figure(figsize=PLOT_FIGURE_SIZE_WEIGHTED_SERIES)
    ax.plot(
        weighted_series_resampled.index,
        weighted_series_resampled,
        linewidth=PLOT_LINEWIDTH_DEFAULT,
        alpha=PLOT_ALPHA_DEFAULT,
        color=PLOT_COLOR_SERIES_ORIGINAL,
    )
    add_zero_line(ax)
    ax.set_title(
        "Rendements logarithmiques pondérés du portefeuille (10 ans)",
        fontsize=PLOT_FONTSIZE_TITLE,
    )
    ax.set_ylabel("Log-return", fontsize=PLOT_FONTSIZE_LABEL)
    ax.set_xlabel("Date", fontsize=PLOT_FONTSIZE_LABEL)
    add_grid(ax, alpha=PLOT_ALPHA_LIGHT, linestyle="--")

    _format_date_axis(ax)
    plt.tight_layout()
    _save_plot(output_file)


def plot_log_returns_distribution(
    data_file: str,
    output_file: str,
    bins: int = DISTRIBUTION_HISTOGRAM_BINS_DEFAULT,
) -> None:
    """Plot histogram of log returns with fitted normal distribution overlay.

    Args:
        data_file: Path to weighted log-returns CSV file.
        output_file: Path to save the plot.
        bins: Number of histogram bins.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If required column is missing or data is empty.
    """
    logger.info("Loading weighted log-returns data for distribution plot")
    aggregated_returns = load_and_validate_data(data_file, "weighted_log_return")
    returns = aggregated_returns["weighted_log_return"].dropna()

    if returns.empty:
        msg = "No valid data found in weighted_log_return column"
        raise ValueError(msg)

    _, ax = create_standard_figure(figsize=PLOT_FIGURE_SIZE_DEFAULT)

    mean, std = plot_histogram_with_normal_overlay(
        ax,
        returns,
        bins=bins,
        show_mean_line=True,
        hist_color=PLOT_COLOR_RESIDUAL,
        fit_color=PLOT_COLOR_NORMAL_FIT,
    )

    # Add statistics
    skewness = float(stats.skew(returns))
    kurtosis = float(stats.kurtosis(returns))

    stats_text = (
        f"N = {len(returns):,}\n"
        f"Mean = {mean:.6f}\n"
        f"Std Dev = {std:.6f}\n"
        f"Skewness = {skewness:.4f}\n"
        f"Kurtosis = {kurtosis:.4f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=PLOT_TEXTBOX_STYLE_DEFAULT,
    )

    ax.set_title(
        "Distribution des rendements logarithmiques pondérés",
        fontsize=PLOT_FONTSIZE_TITLE,
        fontweight="bold",
    )
    ax.set_xlabel("Log-return", fontsize=PLOT_FONTSIZE_LABEL)
    ax.set_ylabel("Density", fontsize=PLOT_FONTSIZE_LABEL)
    ax.legend(loc="upper right", fontsize=10)
    add_grid(ax, alpha=PLOT_ALPHA_LIGHT, linestyle="--")

    plt.tight_layout()
    _save_plot(output_file)
    logger.info(
        "Distribution plot saved: mean=%.6f, std=%.6f, skew=%.4f, kurtosis=%.4f",
        mean,
        std,
        skewness,
        kurtosis,
    )


def plot_acf_pacf(
    data_file: str,
    output_file: str,
    lags: int = ACF_PACF_DEFAULT_LAGS,
) -> None:
    """Plot autocorrelation (ACF) and partial autocorrelation (PACF) functions.

    Args:
        data_file: Path to weighted log-returns CSV file.
        output_file: Path to save the plot.
        lags: Number of lags to display (excluding lag 0).

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If lags is invalid or required column is missing.
    """
    if lags <= 0:
        msg = f"lags must be positive, got {lags}"
        raise ValueError(msg)

    logger.info("Loading weighted log-returns data for ACF/PACF")
    aggregated_returns = load_and_validate_data(data_file, "weighted_log_return")
    weighted_series = aggregated_returns["weighted_log_return"].dropna()

    if len(weighted_series) < lags + 1:
        logger.warning(
            f"Series length ({len(weighted_series)}) < requested lags ({lags}). "
            f"Using {len(weighted_series) - 1}"
        )
        lags = max(ACF_PACF_MIN_LAGS, len(weighted_series) - 1)

    _, axes = create_standard_figure(n_rows=1, n_cols=2, figsize=PLOT_FIGURE_SIZE_ACF_PACF)

    # ACF
    plot_acf(
        weighted_series,
        lags=lags,
        ax=axes[0],
        zero=False,
        alpha=ACF_PACF_CONFIDENCE_ALPHA,
    )
    axes[0].set_title(
        "Fonction d'autocorrélation (ACF)",
        fontsize=PLOT_FONTSIZE_SUBTITLE,
        fontweight="bold",
    )
    axes[0].set_xlabel("Lag", fontsize=PLOT_FONTSIZE_AXIS)
    axes[0].set_ylabel("Autocorrelation", fontsize=PLOT_FONTSIZE_AXIS)
    axes[0].grid(alpha=PLOT_ALPHA_LIGHT, linestyle="--")

    # PACF
    plot_pacf(
        weighted_series,
        lags=lags,
        ax=axes[1],
        method="ywm",
        zero=False,
        alpha=ACF_PACF_CONFIDENCE_ALPHA,
    )
    axes[1].set_title(
        "Fonction d'autocorrélation partielle (PACF)",
        fontsize=PLOT_FONTSIZE_SUBTITLE,
        fontweight="bold",
    )
    axes[1].set_xlabel("Lag", fontsize=PLOT_FONTSIZE_AXIS)
    axes[1].set_ylabel("Partial Autocorrelation", fontsize=PLOT_FONTSIZE_AXIS)
    axes[1].grid(alpha=PLOT_ALPHA_LIGHT, linestyle="--")

    plt.tight_layout()
    _save_plot(output_file)


def _validate_stationarity_params(rolling_window: int, alpha: float) -> None:
    """Validate stationarity plotting parameters."""
    if rolling_window <= 0:
        msg = f"rolling_window must be positive, got {rolling_window}"
        raise ValueError(msg)

    if not 0 < alpha < 1:
        msg = f"alpha must be in (0, 1), got {alpha}"
        raise ValueError(msg)


def _load_stationarity_data(
    data_file: str,
    rolling_window: int,
) -> tuple[pd.Series, int]:
    """Load and prepare data for stationarity analysis."""
    logger.info("Loading weighted log-returns data for stationarity analysis")
    aggregated_returns = load_and_validate_data(data_file, "weighted_log_return")
    weighted_series_raw = aggregated_returns["weighted_log_return"].dropna()

    if not isinstance(weighted_series_raw, pd.Series):
        msg = f"Expected Series, got {type(weighted_series_raw)}"
        raise ValueError(msg)

    weighted_series = cast(pd.Series, weighted_series_raw)

    if len(weighted_series) < rolling_window:
        logger.warning(
            f"Series length ({len(weighted_series)}) < rolling_window ({rolling_window}). "
            f"Using {len(weighted_series)}"
        )
        rolling_window = len(weighted_series)

    return weighted_series, rolling_window


def _create_stationarity_plots(
    fig: matplotlib.figure.Figure,
    axes: np.ndarray,
    weighted_series: pd.Series,
    rolling_mean: pd.Series,
    rolling_std: pd.Series,
    rolling_window: int,
    stationarity_report: StationarityReport,
    alpha: float,
) -> None:
    """Create all stationarity subplots and annotations."""
    plot_stationarity_series_with_bands(
        axes[0], weighted_series, rolling_mean, rolling_std, rolling_window
    )
    plot_stationarity_rolling_mean(axes[1], rolling_mean, float(weighted_series.mean()))
    plot_stationarity_rolling_std(axes[2], rolling_std, float(weighted_series.std()))

    adf_res = stationarity_report.adf
    kpss_res = stationarity_report.kpss
    overall_verdict = "STATIONNAIRE" if stationarity_report.stationary else "NON STATIONNAIRE"
    adf_lags = adf_res["lags"]
    kpss_lags = kpss_res["lags"]
    if adf_lags is None or kpss_lags is None:
        msg = "Stationarity test lags cannot be None"
        raise ValueError(msg)
    test_text = format_stationarity_test_text(
        alpha,
        adf_res["statistic"],
        adf_res["p_value"],
        adf_lags,
        kpss_res["statistic"],
        kpss_res["p_value"],
        kpss_lags,
        overall_verdict,
    )
    add_stationarity_text_box(axes[0], test_text)

    verdict_color = "green" if stationarity_report.stationary else "red"
    fig.suptitle(
        f"Analyse de stationnarité - Verdict: {overall_verdict}",
        fontsize=PLOT_FONTSIZE_TITLE,
        fontweight="bold",
        color=verdict_color,
        y=STATIONARITY_SUPTITLE_Y,
    )


def plot_stationarity(
    data_file: str,
    output_file: str,
    rolling_window: int = STATIONARITY_ROLLING_WINDOW_DEFAULT,
    alpha: float = STATIONARITY_ALPHA_DEFAULT,
) -> None:
    """Plot stationarity visualization with rolling statistics and test results.

    Args:
        data_file: Path to weighted log-returns CSV file.
        output_file: Path to save the plot.
        rolling_window: Window size for rolling statistics.
        alpha: Significance level for stationarity tests.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If rolling_window is invalid, alpha is not in (0, 1),
            or required column is missing.
    """
    _validate_stationarity_params(rolling_window, alpha)
    weighted_series, adjusted_window = _load_stationarity_data(data_file, rolling_window)

    rolling_mean = cast(
        pd.Series, weighted_series.rolling(window=adjusted_window, center=False).mean()
    )
    rolling_std = cast(
        pd.Series, weighted_series.rolling(window=adjusted_window, center=False).std()
    )

    logger.info("Running ADF and KPSS tests for stationarity")
    stationarity_report = evaluate_stationarity(weighted_series, alpha=alpha)

    fig, axes = create_standard_figure(n_rows=3, n_cols=1, figsize=PLOT_FIGURE_SIZE_STATIONARITY)
    _create_stationarity_plots(
        fig,
        axes,
        weighted_series,
        rolling_mean,
        rolling_std,
        adjusted_window,
        stationarity_report,
        alpha,
    )

    plt.tight_layout()
    _save_plot(output_file)

    logger.info(
        "Stationarity analysis completed: stationary=%s (ADF p-value=%.6f, KPSS p-value=%.6f)",
        stationarity_report.stationary,
        stationarity_report.adf["p_value"],
        stationarity_report.kpss["p_value"],
    )


def plot_seasonality_for_year(
    year: int,
    *,
    data_file: str,
    output_file: str,
    period: int | None = None,
    model: str = SEASONAL_DEFAULT_MODEL,
    column: str = "weighted_log_return",
    resample_to: str | None = SEASONAL_RESAMPLE_FREQ_BUSINESS,
) -> None:
    """Plot ONLY the seasonal component for a given calendar year."""
    data_path = str(data_file)
    out_path = str(output_file)

    base_series = load_series_for_year(year=year, data_file=data_path, column=column)
    series = _maybe_resample(base_series, resample_to=resample_to)
    eff_period = _infer_seasonal_period(resample_to=resample_to, override=period)
    seasonal = _decompose_seasonal_component(series, model=model, period=eff_period)
    _plot_seasonal_component_only(
        seasonal,
        title=f"Seasonal component - {year} (model={model}, period={eff_period})",
        full_period=False,
    )
    _save_plot(out_path)


def _filter_series_to_recent_years(series: pd.Series, years: int) -> pd.Series:
    """Filter series to the last N years of data."""
    if series.empty:
        msg = "Cannot filter empty series"
        raise ValueError(msg)

    end_date = cast(pd.Timestamp, pd.to_datetime(series.index.max()))
    start_date = end_date - pd.DateOffset(years=years)
    filtered = series.loc[start_date:end_date]

    logger.info(
        "Filtered to %d observations from %s to %s",
        len(filtered),
        start_date.date(),
        end_date.date(),
    )
    return filtered


def plot_seasonality_full_period(
    *,
    data_file: str,
    output_file: str,
    period: int | None = None,
    model: str = "additive",
    column: str = "weighted_log_return",
) -> None:
    """Plot the seasonal component for the full period (10 years) with weekly resampling."""
    validate_seasonal_params(model, period)

    logger.info("Loading data for seasonal component (full period)")
    dataframe = load_and_validate_data(data_file, column)
    base_series = dataframe[column].dropna()

    series = base_series.resample(SEASONAL_RESAMPLE_FREQ_WEEKLY).mean().dropna()
    eff_period = period if period is not None else SEASONAL_DEFAULT_PERIOD_WEEKLY

    seasonal = _decompose_seasonal_component(series, model=model, period=eff_period)
    _plot_seasonal_component_only(
        seasonal,
        title=f"Seasonal component - Full period (model={model}, period={eff_period})",
        full_period=True,
    )
    _save_plot(output_file)


def plot_seasonality_daily(
    *,
    data_file: str,
    output_file: str,
    period: int = SEASONAL_DEFAULT_PERIOD_DAILY,
    model: str = SEASONAL_DEFAULT_MODEL,
    column: str = "weighted_log_return",
    years: int = SEASONAL_DEFAULT_YEARS,
) -> None:
    """Plot seasonal component for daily data (weekly seasonality - 5 business days)."""
    validate_seasonal_params(model, period)

    logger.info(
        "Loading daily data for weekly seasonal component (period=%d, last %d years)",
        period,
        years,
    )
    dataframe = load_and_validate_data(data_file, column)
    series = dataframe[column].dropna()
    if not isinstance(series, pd.Series):
        msg = f"Column '{column}' did not return a Series"
        raise ValueError(msg)

    series = _filter_series_to_recent_years(series, years)
    validate_minimum_periods(series, period, min_periods=SEASONAL_MIN_PERIODS)

    seasonal = _decompose_seasonal_component(series, model=model, period=period)
    title = (
        f"Seasonal component - Daily data "
        f"(last {years} year{'s' if years > 1 else ''}, "
        f"weekly pattern, period={period} days, model={model})"
    )
    plot_seasonal_daily_long_period(seasonal, title=title)
    _save_plot(output_file)


def plot_seasonality_monthly(
    *,
    data_file: str,
    output_file: str,
    period: int = SEASONAL_DEFAULT_PERIOD_MONTHLY,
    model: str = SEASONAL_DEFAULT_MODEL,
    column: str = "weighted_log_return",
) -> None:
    """Plot seasonal component for monthly data (annual seasonality - 12 months)."""
    validate_seasonal_params(model, period)

    logger.info("Loading data for monthly seasonal component (period=%d)", period)
    dataframe = load_and_validate_data(data_file, column)
    base_series = dataframe[column].dropna()

    series = base_series.resample(SEASONAL_RESAMPLE_FREQ_MONTHLY).mean().dropna()
    validate_minimum_periods(series, period, min_periods=SEASONAL_MIN_PERIODS)

    seasonal = _decompose_seasonal_component(series, model=model, period=period)
    _plot_seasonal_component_only(
        seasonal,
        title=(
            f"Seasonal component - Monthly data "
            f"(annual pattern, period={period} months, model={model})"
        ),
        full_period=True,
    )
    _save_plot(output_file)
