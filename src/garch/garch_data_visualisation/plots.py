"""GARCH data visualization and exploration module.

Implements methodology for GARCH data visualization:
3. Visualize returns and their squared/absolute values (volatility clustering)
4. Calculate autocorrelation of returns and squared returns
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

from src.constants import (
    GARCH_ACF_LAGS_DEFAULT,
    GARCH_DATA_VISU_FIGURE_SIZE_AUTOCORR,
    GARCH_DATA_VISU_FIGURE_SIZE_RETURNS,
    GARCH_DATA_VISU_PLOTS_DIR,
    PLOT_FONTSIZE_TITLE,
)
from src.utils import get_logger
from src.visualization import create_figure_canvas, prepare_temporal_axis, save_canvas

from .utils import plot_absolute_returns_panel, plot_returns_panel, plot_squared_returns_panel

logger = get_logger(__name__)


def _clean_returns(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clean returns and compute absolute and squared returns.

    Args:
        returns: Raw returns array

    Returns:
        Tuple of (cleaned_returns, absolute_returns, squared_returns)

    Raises:
        ValueError: If returns is None or empty after cleaning
    """
    if returns is None:
        raise ValueError("returns cannot be None")

    returns_clean = np.asarray(returns, dtype=float)
    returns_clean = returns_clean[np.isfinite(returns_clean)]

    if returns_clean.size == 0:
        raise ValueError("No finite returns found after cleaning")

    abs_returns = np.abs(returns_clean)
    squared_returns = returns_clean**2

    return returns_clean, abs_returns, squared_returns


def _create_returns_figure() -> tuple[Any, Any, list]:
    """Create figure with 3 subplots for returns visualization.

    Returns:
        Tuple of (figure, canvas, axes_list)
    """
    fig, canvas, axes = create_figure_canvas(
        GARCH_DATA_VISU_FIGURE_SIZE_RETURNS, n_rows=3, n_cols=1
    )
    return fig, canvas, axes


def _plot_returns_panels(
    axes: list,
    x_vals: np.ndarray,
    returns: np.ndarray,
    abs_returns: np.ndarray,
    squared_returns: np.ndarray,
    has_dates: bool,
) -> None:
    """Plot all three returns panels.

    Args:
        axes: List of matplotlib axes
        x_vals: X-axis values
        returns: Cleaned returns
        abs_returns: Absolute returns
        squared_returns: Squared returns
        has_dates: Whether dates are available
    """
    plot_returns_panel(axes[0], x_vals, returns)
    plot_absolute_returns_panel(axes[1], x_vals, abs_returns)
    plot_squared_returns_panel(axes[2], x_vals, squared_returns, has_dates)


def _save_returns_figure(fig: Any, canvas: Any, out_path: Path, title: str) -> None:
    """Save returns figure to file.

    Args:
        fig: Matplotlib figure
        canvas: Matplotlib canvas
        out_path: Output file path
        title: Figure title
    """
    fig.suptitle(title, fontsize=PLOT_FONTSIZE_TITLE, fontweight="bold")
    save_canvas(canvas, out_path, format="png")
    logger.info(f"Saved returns clustering plot: {out_path}")


def save_returns_and_squared_plots(
    returns: np.ndarray,
    *,
    dates: np.ndarray | pd.Series | None = None,
    outdir: Path | str = GARCH_DATA_VISU_PLOTS_DIR,
    filename: str = "garch_returns_clustering.png",
) -> Path:
    """Visualize returns, absolute returns, and squared returns for volatility clustering.

    Creates a 3-panel plot: returns time series, absolute returns, squared returns.

    Args:
        returns: Array of returns (arithmetic returns)
        dates: Optional date array/series for x-axis
        outdir: Output directory for plot
        filename: Output filename

    Returns:
        Path to saved plot

    Raises:
        ValueError: If returns array is empty or invalid
    """
    out_path = Path(outdir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    returns_clean, abs_returns, squared_returns = _clean_returns(returns)

    if returns_clean.size == 0:
        raise ValueError("Cannot create plot: no finite returns after cleaning")

    fig, canvas, axes = _create_returns_figure()
    x_vals = prepare_temporal_axis(dates, len(returns_clean))
    _plot_returns_panels(
        axes, x_vals, returns_clean, abs_returns, squared_returns, dates is not None
    )

    _save_returns_figure(fig, canvas, out_path, "Visualisation du clustering de volatilité")
    return out_path


def _validate_autocorrelation_inputs(returns: np.ndarray | None, lags: int) -> None:
    """Validate inputs for autocorrelation plotting.

    Args:
        returns: Returns array
        lags: Number of lags

    Raises:
        ValueError: If inputs are invalid
    """
    if returns is None:
        raise ValueError("returns cannot be None")
    if lags < 1:
        raise ValueError(f"lags must be positive, got {lags}")


def _prepare_autocorrelation_data(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prepare cleaned returns and squared returns for ACF plotting.

    Args:
        returns: Raw returns array

    Returns:
        Tuple of (cleaned_returns, squared_returns)
    """
    returns_clean = np.asarray(returns, dtype=float)
    returns_clean = returns_clean[np.isfinite(returns_clean)]
    squared_returns = returns_clean**2
    return returns_clean, squared_returns


def _create_autocorrelation_figure() -> tuple[Any, Any, list]:
    """Create figure with 2 subplots for autocorrelation visualization.

    Returns:
        Tuple of (figure, canvas, axes_list)
    """
    fig, canvas, axes = create_figure_canvas(
        GARCH_DATA_VISU_FIGURE_SIZE_AUTOCORR, n_rows=1, n_cols=2
    )
    return fig, canvas, axes


def _plot_acf_panels(
    axes: list, returns: np.ndarray, squared_returns: np.ndarray, lags: int
) -> None:
    """Plot ACF for returns and squared returns.

    Args:
        axes: List of matplotlib axes
        returns: Cleaned returns
        squared_returns: Squared returns
        lags: Number of lags for ACF
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        plot_acf(returns, lags=lags, ax=axes[0], zero=False)
        plot_acf(squared_returns, lags=lags, ax=axes[1], zero=False)


def _configure_acf_axes(axes: list) -> None:
    """Configure axis labels and titles for ACF plots.

    Args:
        axes: List of matplotlib axes
    """
    axes[0].set_title("ACF des rendements")
    axes[0].set_xlabel("Décalage (lag)")
    axes[0].set_ylabel("Autocorrélation")

    axes[1].set_title("ACF des rendements au carré")
    axes[1].set_xlabel("Décalage (lag)")
    axes[1].set_ylabel("Autocorrélation")


def _save_autocorr_figure(fig: Any, canvas: Any, out_path: Path, title: str) -> None:
    """Save autocorrelation figure to file.

    Args:
        fig: Matplotlib figure
        canvas: Matplotlib canvas
        out_path: Output file path
        title: Figure title
    """
    fig.suptitle(title, fontsize=PLOT_FONTSIZE_TITLE, fontweight="bold")
    save_canvas(canvas, out_path, format="png")
    logger.info(f"Saved returns autocorrelation plot: {out_path}")


def plot_returns_autocorrelation(
    returns: np.ndarray,
    *,
    lags: int = GARCH_ACF_LAGS_DEFAULT,
    outdir: Path | str = GARCH_DATA_VISU_PLOTS_DIR,
    filename: str = "garch_returns_autocorrelation.png",
) -> Path:
    """Plot autocorrelation of returns and squared returns side by side.

    Args:
        returns: Returns array
        lags: Number of lags for ACF (must be positive)
        outdir: Output directory
        filename: Output filename

    Returns:
        Path to saved plot

    Raises:
        ValueError: If returns is None, lags is invalid, or insufficient data
    """
    _validate_autocorrelation_inputs(returns, lags)
    out_path = Path(outdir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    returns_clean, squared_returns = _prepare_autocorrelation_data(returns)

    if returns_clean.size < 3:
        raise ValueError(
            f"Insufficient data for ACF plot: need at least 3 observations, "
            f"got {returns_clean.size}"
        )

    fig, canvas, axes = _create_autocorrelation_figure()
    _plot_acf_panels(axes, returns_clean, squared_returns, lags)
    _configure_acf_axes(axes)
    _save_autocorr_figure(
        fig, canvas, out_path, "Autocorrélation des rendements et des rendements au carré"
    )
    return out_path


def _clean_residuals(residuals: np.ndarray) -> np.ndarray:
    """Clean residuals array by removing non-finite values.

    Args:
        residuals: Raw residuals array

    Returns:
        Cleaned residuals array

    Raises:
        ValueError: If residuals is None or empty after cleaning
    """
    if residuals is None:
        raise ValueError("residuals cannot be None")

    residuals_clean = np.asarray(residuals, dtype=float)
    residuals_clean = residuals_clean[np.isfinite(residuals_clean)]

    if residuals_clean.size == 0:
        raise ValueError("No finite residuals found after cleaning")

    return residuals_clean


def _create_distribution_figure() -> tuple[Any, Any, Any]:
    """Create figure for residuals distribution visualization.

    Returns:
        Tuple of (figure, canvas, axis)
    """
    fig, canvas, axes = create_figure_canvas((12, 6), n_rows=1, n_cols=1)
    return fig, canvas, axes[0] if isinstance(axes, list) else axes


def _plot_histogram_with_normal(ax: Any, residuals: np.ndarray) -> None:
    """Plot histogram of residuals with normal distribution overlay.

    Args:
        ax: Matplotlib axis
        residuals: Cleaned residuals array
    """
    import scipy.stats as stats

    # Plot histogram
    n, bins, patches = ax.hist(
        residuals, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="black"
    )

    # Overlay normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2, label="Distribution normale")

    ax.set_xlabel("Résidus ARIMA")
    ax.set_ylabel("Densité")
    ax.set_title("Distribution des résidus ARIMA")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _save_distribution_figure(fig: Any, canvas: Any, out_path: Path, title: str) -> None:
    """Save distribution figure to file.

    Args:
        fig: Matplotlib figure
        canvas: Matplotlib canvas
        out_path: Output file path
        title: Figure title
    """
    fig.suptitle(title, fontsize=PLOT_FONTSIZE_TITLE, fontweight="bold")
    save_canvas(canvas, out_path, format="png")
    logger.info(f"Saved residuals distribution plot: {out_path}")


def plot_residuals_distribution(
    residuals: np.ndarray,
    *,
    outdir: Path | str = GARCH_DATA_VISU_PLOTS_DIR,
    filename: str = "arima_residuals_distribution.png",
) -> Path:
    """Plot distribution of ARIMA residuals with normal distribution overlay.

    Creates a histogram of residuals with a fitted normal distribution curve.

    Args:
        residuals: Array of ARIMA residuals
        outdir: Output directory for plot
        filename: Output filename

    Returns:
        Path to saved plot

    Raises:
        ValueError: If residuals array is empty or invalid
    """
    out_path = Path(outdir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    residuals_clean = _clean_residuals(residuals)

    if residuals_clean.size == 0:
        raise ValueError("Cannot create plot: no finite residuals after cleaning")

    fig, canvas, ax = _create_distribution_figure()
    _plot_histogram_with_normal(ax, residuals_clean)
    _save_distribution_figure(fig, canvas, out_path, "Distribution des résidus ARIMA")

    return out_path
