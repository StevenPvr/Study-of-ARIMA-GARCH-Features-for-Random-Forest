"""Post-estimation diagnostics for GARCH models.

Implements methodology for verifying GARCH model adequacy:
1. Verify standardized residuals εt/σt behave as centered white noise
2. Verify squared standardized residuals show no significant autocorrelation
   (ACF/PACF plots + Ljung-Box tests)
3. Verify distribution adequacy for zt (Normal or Student-t)
   (graphical diagnostics + normality tests)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.constants import (
    GARCH_ACF_LAGS_DEFAULT,
    GARCH_DIAGNOSTICS_PLOTS_DIR,
    GARCH_LJUNG_BOX_LAGS_DEFAULT,
    GARCH_STD_EPSILON,
)
from src.garch.garch_diagnostic.utils import (
    compute_acf_pacf_data,
    compute_ljung_box_statistics,
    compute_qq_data,
    create_figure_canvas,
    create_residual_plots_figure,
    plot_acf_subplot,
    plot_pacf_subplot,
    plot_qq_scatter,
    prepare_output_path,
    prepare_residual_data,
    prepare_standardized_residuals_for_plotting,
    save_figure_or_placeholder,
    standardize_residuals,
    write_placeholder_png,
)
from src.utils import get_logger

logger = get_logger(__name__)


def compute_ljung_box_on_std_squared(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    lags: int = GARCH_LJUNG_BOX_LAGS_DEFAULT,
    dist: str = "normal",
    nu: float | None = None,
) -> dict[str, list[int] | list[float]]:
    """Compute Ljung-Box test on standardized squared residuals (z²).

    Verifies that squared standardized residuals show no significant autocorrelation.
    If the model captures volatility correctly, z² should be uncorrelated.
    """
    z = standardize_residuals(residuals, garch_params, dist=dist, nu=nu)
    y = (z**2) - np.mean(z**2)
    return compute_ljung_box_statistics(y, lags)


def compute_ljung_box_on_std(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    lags: int = GARCH_LJUNG_BOX_LAGS_DEFAULT,
    dist: str = "normal",
    nu: float | None = None,
) -> dict[str, list[int] | list[float]]:
    """Compute Ljung-Box test on standardized residuals (z).

    Tests for white noise behavior: standardized residuals should show
    no significant autocorrelation if the model captures volatility correctly.
    """
    z = standardize_residuals(residuals, garch_params, dist=dist, nu=nu)
    return compute_ljung_box_statistics(z, lags)


def save_acf_pacf_std_squared_plots(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    lags: int = GARCH_ACF_LAGS_DEFAULT,
    outdir: str | Path = GARCH_DIAGNOSTICS_PLOTS_DIR,
    filename: str | None = None,
    dist: str = "normal",
    nu: float | None = None,
) -> Path:
    """Save ACF and PACF plots of standardized squared residuals (z²).

    Verifies that squared standardized residuals show no significant autocorrelation.
    ACF/PACF should be near zero for all lags if volatility is correctly captured.
    """
    z = standardize_residuals(residuals, garch_params, dist=dist, nu=nu)
    y = z**2 - np.mean(z**2)
    acf, pacf_vals, se = compute_acf_pacf_data(y, lags)
    out_path = prepare_output_path(outdir, filename, "garch_std_squared_acf_pacf")

    try:
        fig, canvas = create_figure_canvas(figsize=(10, 6))
        ax1, ax2 = fig.subplots(2, 1)
        plot_acf_subplot(ax1, acf, se, "ACF of standardized squared residuals (z^2)", lags)
        plot_pacf_subplot(ax2, pacf_vals, se, "PACF of standardized squared residuals (z^2)", lags)
        fig.suptitle("ACF/PACF of standardized squared residuals")
        save_figure_or_placeholder(canvas, out_path, "Saved ACF/PACF(z^2) plots")
    except ImportError:  # pragma: no cover - matplotlib optional
        write_placeholder_png(out_path)
        logger.warning("Matplotlib unavailable; wrote placeholder PNG: %s", out_path)
    return out_path


def save_acf_pacf_std_plots(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    lags: int = GARCH_ACF_LAGS_DEFAULT,
    outdir: str | Path = GARCH_DIAGNOSTICS_PLOTS_DIR,
    filename: str | None = None,
    dist: str = "normal",
    nu: float | None = None,
) -> Path:
    """Save ACF and PACF plots of standardized residuals (z).

    Verifies that standardized residuals εt/σt behave as centered white noise.
    For white noise, ACF/PACF should be near zero for all lags.
    """
    z = standardize_residuals(residuals, garch_params, dist=dist, nu=nu)
    zc = z - np.mean(z)
    acf, pacf_vals, se = compute_acf_pacf_data(zc, lags)
    out_path = prepare_output_path(outdir, filename, "garch_std_acf_pacf")

    try:
        fig, canvas = create_figure_canvas(figsize=(10, 6))
        ax1, ax2 = fig.subplots(2, 1)
        plot_acf_subplot(ax1, acf, se, "ACF of standardized residuals (z)", lags)
        plot_pacf_subplot(ax2, pacf_vals, se, "PACF of standardized residuals (z)", lags)
        fig.suptitle("ACF/PACF of standardized residuals")
        save_figure_or_placeholder(canvas, out_path, "Saved ACF/PACF(z) plots")
    except ImportError:  # pragma: no cover - matplotlib optional
        write_placeholder_png(out_path)
        logger.warning("Matplotlib unavailable; wrote placeholder PNG: %s", out_path)
    return out_path


def compute_distribution_diagnostics(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    dist: str,
    nu: float | None = None,
) -> dict[str, float | str | None]:
    """Compute distribution diagnostics for standardized residuals.

    Verifies adequacy of chosen distribution (Normal or Student-t) for zt:
    - Skewness and kurtosis
    - Jarque-Bera test (normality)
    - Kolmogorov-Smirnov test (distribution fit)

    Args:
        residuals: Raw residuals εt from mean model.
        garch_params: GARCH parameter dictionary.
        dist: Distribution name ('normal' or 'student').
        nu: Degrees of freedom for Student-t (if applicable).

    Returns:
        Dictionary with diagnostic statistics and test results.
    """
    from scipy.stats import jarque_bera, kstest, norm, t  # type: ignore

    z = standardize_residuals(residuals, garch_params, dist=dist, nu=nu)
    zc = (z - np.mean(z)) / (np.std(z) + GARCH_STD_EPSILON)
    skew = float(np.nanmean(zc**3))
    kurt = float(np.nanmean(zc**4))
    jb = jarque_bera(z)
    jb_stat = float(jb.statistic)  # type: ignore[attr-defined]
    jb_p = float(jb.pvalue)  # type: ignore[attr-defined]
    if dist.lower() == "student" and nu is not None and nu > 2:
        ks = kstest(z, lambda x: t.cdf(x, df=nu))
        used = "student"
    else:
        ks = kstest(z, lambda x: norm.cdf(x))
        used = "normal"
    return {
        "dist": used,
        "nu": float(nu) if nu is not None else None,
        "skewness": skew,
        "kurtosis": kurt,
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pvalue": jb_p,
        "ks_stat": float(ks.statistic),  # type: ignore[attr-defined]
        "ks_pvalue": float(ks.pvalue),  # type: ignore[attr-defined]
    }


def save_qq_plot_std_residuals(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    dist: str,
    nu: float | None = None,
    outdir: str | Path = GARCH_DIAGNOSTICS_PLOTS_DIR,
    filename: str | None = None,
) -> Path:
    """Save QQ-plot of standardized residuals vs chosen distribution.

    Graphical diagnostic to verify distribution adequacy:
    - Points should lie along the diagonal line if distribution is appropriate
    - Deviations indicate distribution misspecification
    """
    z = standardize_residuals(residuals, garch_params, dist=dist, nu=nu)
    theo_q, z_sorted, title = compute_qq_data(z, dist, nu)
    out_path = prepare_output_path(outdir, filename, "garch_std_residuals_qq")

    try:
        fig, canvas = create_figure_canvas(figsize=(6, 6))
        ax = fig.subplots(1, 1)
        plot_qq_scatter(ax, theo_q, z_sorted, title)
        save_figure_or_placeholder(canvas, out_path, "Saved QQ plot")
    except ImportError:  # pragma: no cover - matplotlib optional
        write_placeholder_png(out_path)
        logger.warning("Matplotlib unavailable; wrote placeholder PNG: %s", out_path)
    return out_path


def save_residual_plots(
    resid_train: np.ndarray,
    resid_test: np.ndarray,
    *,
    garch_params: dict[str, float] | None = None,
    outdir: str | Path = GARCH_DIAGNOSTICS_PLOTS_DIR,
    filename: str | None = None,
    dist: str = "normal",
    nu: float | None = None,
) -> Path:
    """Save plots of residuals and standardized residuals (if params provided)."""
    train, test, n_train = prepare_residual_data(resid_train, resid_test)
    all_res = np.concatenate([train, test]) if test.size else train
    z_all = prepare_standardized_residuals_for_plotting(all_res, garch_params, dist, nu)
    out_path = prepare_output_path(outdir, filename, "garch_residuals")

    try:
        _, canvas = create_residual_plots_figure(train, test, z_all, n_train)
        save_figure_or_placeholder(canvas, out_path, "Saved residual plots")
    except ImportError:  # pragma: no cover - matplotlib optional
        write_placeholder_png(out_path)
        logger.warning("Matplotlib unavailable; wrote placeholder PNG: %s", out_path)
    return out_path


__all__ = [
    "save_acf_pacf_std_squared_plots",
    "compute_ljung_box_on_std_squared",
    "compute_ljung_box_on_std",
    "save_acf_pacf_std_plots",
    "compute_distribution_diagnostics",
    "save_qq_plot_std_residuals",
    "save_residual_plots",
]
