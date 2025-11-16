"""CLI for GARCH identification (ARCH/GARCH detection).

Implements methodology for detecting conditional heteroskedasticity:
1. Extract residuals εt from SARIMA model (mean model)
2. Test for ARCH effect using Lagrange Multiplier test (ARCH-LM)
3. Inspect autocorrelation of squared residuals

Results saved to results/garch/structure/
Plots saved to plots/garch/structure/
"""

from __future__ import annotations

import json

import numpy as np

from src.utils import setup_project_path

# Ensure project root
setup_project_path()

from src.constants import (
    GARCH_ACF_LAGS_DEFAULT,
    GARCH_DATASET_FILE,
    GARCH_DEFAULT_ALPHA,
    GARCH_DIAGNOSTICS_FILE,
    GARCH_LM_LAGS_DEFAULT,
    GARCH_STRUCTURE_DIR,
)
from src.garch.structure_garch.detection import detect_heteroskedasticity, plot_arch_diagnostics
from src.garch.structure_garch.utils import load_garch_dataset, prepare_residuals
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run ARCH/GARCH identification diagnostics.

    Methodology:
    1. Extract residuals εt from SARIMA model (mean model)
    2. Test for ARCH effect using Lagrange Multiplier test (ARCH-LM)
    3. Inspect autocorrelation of squared residuals

    Results saved to results/garch/structure/
    Plots saved to plots/garch/structure/
    """
    logger.info("=" * 60)
    logger.info("GARCH IDENTIFICATION (ARCH-LM + ACF(e^2))")
    logger.info("=" * 60)

    df = load_garch_dataset(str(GARCH_DATASET_FILE))
    resid_test = prepare_residuals(df, use_test_only=True)
    resid_test = resid_test[np.isfinite(resid_test)]

    results = detect_heteroskedasticity(
        resid_test,
        lags=GARCH_LM_LAGS_DEFAULT,
        acf_lags=GARCH_ACF_LAGS_DEFAULT,
        alpha=GARCH_DEFAULT_ALPHA,
    )

    out = {
        "source": str(GARCH_DATASET_FILE),
        "diagnostics": results,
        "n_test": int(resid_test.size),
    }
    GARCH_STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
    with GARCH_DIAGNOSTICS_FILE.open("w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved identification diagnostics: %s", GARCH_DIAGNOSTICS_FILE)

    # Save a compact plot to visualize heteroscedasticity
    plot_arch_diagnostics(resid_test, acf_lags=GARCH_ACF_LAGS_DEFAULT)


if __name__ == "__main__":
    main()
