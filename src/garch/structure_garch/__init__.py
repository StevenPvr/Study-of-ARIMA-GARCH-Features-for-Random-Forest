"""GARCH structure detection module.

Provides utilities for ARCH/GARCH identification and diagnostics:
- Loading GARCH datasets
- Computing ACF of returns and squared returns
- Running Engle's ARCH-LM test
- Detecting heteroskedasticity
- Generating diagnostic plots
"""

from __future__ import annotations

from src.garch.structure_garch.detection import (
    detect_heteroskedasticity,
    plot_arch_diagnostics,
)
from src.garch.structure_garch.utils import (
    chi2_sf,
    compute_acf,
    compute_arch_lm_test,
    compute_squared_acf,
    load_garch_dataset,
    prepare_residuals,
)

__all__ = [
    # Main detection functions
    "detect_heteroskedasticity",
    "plot_arch_diagnostics",
    # Utility functions
    "compute_acf",
    "compute_arch_lm_test",
    "compute_squared_acf",
    "chi2_sf",
    "load_garch_dataset",
    "prepare_residuals",
]
