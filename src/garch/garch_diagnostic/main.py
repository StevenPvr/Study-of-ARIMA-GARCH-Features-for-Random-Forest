"""CLI for post-estimation diagnostics of GARCH models.

Implements methodology for verifying GARCH model adequacy:
1. Verify standardized residuals εt/σt behave as centered white noise
2. Verify squared standardized residuals show no significant autocorrelation
   (ACF/PACF plots + Ljung-Box tests)
3. Verify distribution adequacy for zt (Normal or Student-t)
   (graphical diagnostics + normality tests)

Results saved to results/garch/diagnostic/
Plots saved to plots/garch/diagnostics/
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from src.constants import (
    GARCH_ACF_LAGS_DEFAULT,
    GARCH_DIAGNOSTIC_DIR,
    GARCH_DIAGNOSTICS_PLOTS_DIR,
    GARCH_DISTRIBUTION_DIAGNOSTICS_FILE,
    GARCH_LJUNG_BOX_LAGS_DEFAULT,
    GARCH_LJUNGBOX_FILE,
    GARCH_STD_ACF_PACF_PLOT,
    GARCH_STD_QQ_PLOT,
    GARCH_STD_SQUARED_ACF_PACF_PLOT,
)
from src.garch.garch_diagnostic.diagnostics import (
    compute_distribution_diagnostics,
    compute_ljung_box_on_std,
    compute_ljung_box_on_std_squared,
    save_acf_pacf_std_plots,
    save_acf_pacf_std_squared_plots,
    save_qq_plot_std_residuals,
)
from src.garch.garch_diagnostic.utils import load_data_and_params
from src.utils import get_logger

logger = get_logger(__name__)


def _generate_acf_pacf_plots(
    resid_test: np.ndarray,
    best_params: dict,
    dist: str | None,
    nu: float | None,
) -> None:
    """Generate and save ACF/PACF plots for standardized residuals.

    Verifies:
    1. Standardized residuals (z) behave as centered white noise
    2. Squared standardized residuals (z²) show no significant autocorrelation

    Args:
        resid_test: Test residuals array εt.
        best_params: Best GARCH parameters dictionary.
        dist: Distribution name (normal or student).
        nu: Degrees of freedom for Student-t distribution (if applicable).
    """
    try:
        save_acf_pacf_std_squared_plots(
            resid_test,
            best_params,
            lags=GARCH_ACF_LAGS_DEFAULT,
            outdir=GARCH_DIAGNOSTICS_PLOTS_DIR,
            filename=GARCH_STD_SQUARED_ACF_PACF_PLOT.name,
            dist=(dist or "normal"),
            nu=nu,
        )
        save_acf_pacf_std_plots(
            resid_test,
            best_params,
            lags=GARCH_ACF_LAGS_DEFAULT,
            outdir=GARCH_DIAGNOSTICS_PLOTS_DIR,
            filename=GARCH_STD_ACF_PACF_PLOT.name,
            dist=(dist or "normal"),
            nu=nu,
        )
        logger.info("Saved ACF/PACF plots to %s", GARCH_DIAGNOSTICS_PLOTS_DIR)
    except Exception as ex:
        logger.warning("ACF/PACF plotting failed: %s", ex)


def _run_ljung_box_tests(
    resid_test: np.ndarray,
    best_params: dict,
    dist: str | None,
    nu: float | None,
) -> None:
    """Run Ljung-Box tests on standardized and squared standardized residuals.

    Tests for white noise behavior:
    - z should show no autocorrelation (white noise)
    - z² should show no autocorrelation (volatility correctly captured)

    Results saved to results/garch/diagnostic/

    Args:
        resid_test: Test residuals array εt.
        best_params: Best GARCH parameters dictionary.
        dist: Distribution name (normal or student).
        nu: Degrees of freedom for Student-t distribution (if applicable).
    """
    try:
        lb2 = compute_ljung_box_on_std_squared(
            resid_test,
            best_params,
            lags=GARCH_LJUNG_BOX_LAGS_DEFAULT,
            dist=(dist or "normal"),
            nu=nu,
        )
        GARCH_DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
        with Path(GARCH_LJUNGBOX_FILE).open("w", encoding="utf-8") as f:
            json.dump(lb2, f, indent=2)
        logger.info("Saved Ljung-Box(z^2) to: %s", GARCH_LJUNGBOX_FILE)
    except Exception as ex:
        logger.warning("Ljung-Box(z^2) failed: %s", ex)
    try:
        compute_ljung_box_on_std(
            resid_test,
            best_params,
            lags=GARCH_LJUNG_BOX_LAGS_DEFAULT,
            dist=(dist or "normal"),
            nu=nu,
        )
    except Exception as ex:
        logger.warning("Ljung-Box(z) failed: %s", ex)


def _run_distribution_diagnostics(
    resid_test: np.ndarray,
    best_params: dict,
    dist: str | None,
) -> None:
    """Run distribution diagnostics and generate QQ plot.

    Verifies adequacy of chosen distribution (Normal or Student-t) for zt:
    - Graphical diagnostics: QQ plot
    - Numerical tests: Jarque-Bera, Kolmogorov-Smirnov

    Results saved to results/garch/diagnostic/
    Plots saved to plots/garch/diagnostics/

    Args:
        resid_test: Test residuals array εt.
        best_params: Best GARCH parameters dictionary.
        dist: Distribution name (normal or student).
    """
    try:
        nu_val = best_params.get("nu")
        nu = float(nu_val) if nu_val is not None else None  # type: ignore[arg-type]
        diag = compute_distribution_diagnostics(
            resid_test, best_params, dist=(dist or "normal"), nu=nu
        )
        save_qq_plot_std_residuals(
            resid_test,
            best_params,
            dist=(dist or "normal"),
            nu=nu,
            outdir=GARCH_DIAGNOSTICS_PLOTS_DIR,
            filename=GARCH_STD_QQ_PLOT.name,
        )
        GARCH_DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
        with Path(GARCH_DISTRIBUTION_DIAGNOSTICS_FILE).open("w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)
        logger.info("Saved distribution diagnostics to: %s", GARCH_DISTRIBUTION_DIAGNOSTICS_FILE)
        logger.info("Distribution diagnostics: %s", diag)
    except Exception as ex:
        logger.warning("Distribution diagnostics failed: %s", ex)


def main() -> None:
    """Run post-estimation GARCH diagnostics.

    Methodology:
    1. Verify standardized residuals εt/σt behave as centered white noise
    2. Verify squared standardized residuals show no significant autocorrelation
       (ACF/PACF plots + Ljung-Box tests)
    3. Verify distribution adequacy for zt (Normal or Student-t)
       (graphical diagnostics + normality tests)

    Results saved to results/garch/diagnostic/
    Plots saved to plots/garch/diagnostics/
    """
    logger.info("=" * 60)
    logger.info("GARCH DIAGNOSTICS (post-estimation)")
    logger.info("=" * 60)

    result = load_data_and_params()
    if result is None:
        return
    resid_test, dist, best, nu = result

    _generate_acf_pacf_plots(resid_test, best, dist, nu)
    _run_ljung_box_tests(resid_test, best, dist, nu)
    _run_distribution_diagnostics(resid_test, best, dist)


if __name__ == "__main__":
    main()
