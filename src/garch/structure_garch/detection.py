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
    GARCH_LM_LAGS_DEFAULT,
    GARCH_PLOT_Z_CONF,
)
from src.garch.structure_garch.utils import (
    compute_arch_lm_test,
    compute_squared_acf,
    ensure_output_dir,
    prepare_plot_series,
    render_with_matplotlib,
    resolve_out_path,
    safe_import_matplotlib,
    verify_or_fallback,
    write_placeholder_file,
)
from src.utils import get_logger

logger = get_logger(__name__)


def detect_heteroskedasticity(
    residuals: np.ndarray,
    *,
    lags: int = GARCH_LM_LAGS_DEFAULT,
    acf_lags: int = GARCH_ACF_LAGS_DEFAULT,
    alpha: float = 0.05,
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

    arch_present = np.isfinite(lm["p_value"]) and lm["p_value"] < alpha
    acf_significant = bool(np.any(np.abs(acf_sq) > acf_sig))

    return {
        "arch_lm": lm,
        "acf_squared": acf_sq.tolist(),
        "acf_significance_level": float(acf_sig),
        "arch_effect_present": arch_present,
        "acf_significant": acf_significant,
    }


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
    Figure, FigureCanvas, have_matplotlib = safe_import_matplotlib()
    x, acf_sq, conf = prepare_plot_series(residuals=residuals, acf_lags=acf_lags)
    out_path = resolve_out_path(out_path)
    ensure_output_dir(out_path)

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
    logger.info("Saved heteroskedasticity plot: %s", out_path)
    return out_path


__all__ = [
    "detect_heteroskedasticity",
    "plot_arch_diagnostics",
]
