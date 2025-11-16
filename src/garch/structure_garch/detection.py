"""Identification and pre-diagnostics for ARCH/GARCH.

Implements methodology for detecting conditional heteroskedasticity:
1. Extract residuals εt from SARIMA model (mean model)
2. Test for ARCH effect using Lagrange Multiplier test (ARCH-LM)
3. Inspect autocorrelation of squared residuals

A significant autocorrelation structure in squared residuals indicates
that a GARCH model is relevant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from src.constants import (
    GARCH_ACF_LAGS_DEFAULT,
    GARCH_DEFAULT_ALPHA,
    GARCH_LM_LAGS_DEFAULT,
    GARCH_PLOT_Z_CONF,
)
from src.garch.structure_garch.utils import (
    compute_arch_lm_test,
    compute_squared_acf,
    prepare_plot_series,
    render_with_matplotlib,
    resolve_out_path,
    safe_import_matplotlib,
    verify_or_fallback,
)
from src.utils import ensure_output_dir, get_logger, write_placeholder_file

logger = get_logger(__name__)


def detect_heteroskedasticity(
    residuals: np.ndarray,
    *,
    lags: int = GARCH_LM_LAGS_DEFAULT,
    acf_lags: int = GARCH_ACF_LAGS_DEFAULT,
    alpha: float = GARCH_DEFAULT_ALPHA,
) -> dict[str, object]:
    """Detect conditional heteroskedasticity (ARCH/GARCH effect).

    Runs:
    1. ARCH-LM test (Lagrange Multiplier test) for ARCH effect
    2. ACF of squared residuals to inspect autocorrelation structure

    Args:
        residuals: Residual series εt from mean model (SARIMA).
        lags: Lags for ARCH-LM test.
        acf_lags: Max lag for ACF(ε^2).
        alpha: Significance level for ARCH-LM test.

    Returns:
        Dict summarizing test statistics and boolean flags.
    """
    lm = compute_arch_lm_test(residuals, lags=lags)
    acf_sq = compute_squared_acf(residuals, nlags=acf_lags)
    n = int(np.asarray(residuals, dtype=float).size)
    acf_sig = GARCH_PLOT_Z_CONF / np.sqrt(max(1, n))

    arch_present = bool(np.isfinite(lm["p_value"]) and lm["p_value"] < alpha)
    acf_significant = bool(np.any(np.abs(acf_sq) > acf_sig))

    return {
        "arch_lm": lm,
        "acf_squared": acf_sq.tolist(),
        "acf_significance_level": float(acf_sig),
        "arch_effect_present": arch_present,
        "acf_significant": acf_significant,
    }


def _prepare_plot_data(
    residuals: np.ndarray, acf_lags: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Prepare data for diagnostics plot.

    Args:
        residuals: Residual series εt from mean model (SARIMA).
        acf_lags: Max lag for ACF(ε^2).

    Returns:
        Tuple of (finite_residuals, acf_squared, confidence_level).
    """
    x, acf_sq, conf = prepare_plot_series(residuals=residuals, acf_lags=acf_lags)
    return x, acf_sq, conf


def _resolve_and_ensure_output_path(out_path: Path | None) -> Path:
    """Resolve output path and ensure parent directory exists.

    Args:
        out_path: Optional output path.

    Returns:
        Resolved output path.
    """
    resolved_path = resolve_out_path(out_path)
    ensure_output_dir(resolved_path)
    return resolved_path


def _save_diagnostics_plot(
    x: np.ndarray,
    acf_sq: np.ndarray,
    conf: float,
    acf_lags: int,
    out_path: Path,
) -> None:
    """Save diagnostics plot using matplotlib or placeholder.

    Args:
        x: Finite residual series.
        acf_sq: ACF of squared residuals.
        conf: Confidence level for significance bands.
        acf_lags: Maximum lag for ACF plot.
        out_path: Output file path.
    """
    Figure, FigureCanvas, have_matplotlib = safe_import_matplotlib()

    if have_matplotlib:
        render_with_matplotlib(
            Figure=cast(Any, Figure),
            FigureCanvas=cast(Any, FigureCanvas),
            x=x,
            acf_sq=acf_sq,
            conf=conf,
            acf_lags=acf_lags,
            out_path=out_path,
        )
    else:
        write_placeholder_file(out_path)

    verify_or_fallback(out_path)


def plot_arch_diagnostics(
    residuals: np.ndarray,
    *,
    acf_lags: int = GARCH_ACF_LAGS_DEFAULT,
    out_path: Path | None = None,
) -> Path:
    """Create and save ARCH/GARCH diagnostic plot.

    Visualizes:
    - Residuals εt time series
    - ACF of squared residuals ε_t^2

    Saved to plots/garch/structure/ by default.

    Args:
        residuals: Residual series εt from mean model (SARIMA).
        acf_lags: Max lag for ACF(ε^2).
        out_path: Optional output path. Defaults to GARCH_STRUCTURE_PLOT.

    Returns:
        Path to saved plot.
    """
    x, acf_sq, conf = _prepare_plot_data(residuals, acf_lags)
    resolved_path = _resolve_and_ensure_output_path(out_path)
    _save_diagnostics_plot(x, acf_sq, conf, acf_lags, resolved_path)
    logger.info("Saved heteroskedasticity plot: %s", resolved_path)
    return resolved_path


__all__ = [
    "detect_heteroskedasticity",
    "plot_arch_diagnostics",
]
