"""Compatibility module for GARCH diagnostics utilities.

Re-exports functions from specialized modules for backward compatibility.
"""

from __future__ import annotations

# Re-export from specialized modules
from src.garch.garch_diagnostic.autocorrelation import (
    autocorr,
    compute_autocorr_denominator,
    compute_autocorr_lag,
    pacf_compute_lag,
    pacf_from_autocorr,
    pacf_init_first_lag,
)
from src.garch.garch_diagnostic.data_loading import (
    check_converged_params,
    choose_best_params,
    extract_nu_from_params,
    load_and_prepare_residuals,
    load_data_and_params,
    load_estimation_file,
    try_legacy_format_params,
    try_new_format_params,
)
from src.garch.garch_diagnostic.plotting import (
    compute_acf_pacf_data,
    compute_qq_data,
    create_figure_canvas,
    create_residual_plots_figure,
    plot_acf_subplot,
    plot_pacf_subplot,
    plot_qq_scatter,
    plot_raw_residuals,
    plot_standardized_residuals,
    prepare_output_path,
    prepare_residual_data,
    save_figure_or_placeholder,
    write_placeholder_png,
)
from src.garch.garch_diagnostic.standardization import (
    compute_standardized_residuals_for_plot,
    prepare_standardized_residuals_for_plotting,
    standardize_residuals,
)
from src.garch.garch_diagnostic.statistical_tests import compute_ljung_box_statistics

__all__ = [
    # Autocorrelation
    "autocorr",
    "compute_autocorr_denominator",
    "compute_autocorr_lag",
    "pacf_compute_lag",
    "pacf_from_autocorr",
    "pacf_init_first_lag",
    # Data loading
    "check_converged_params",
    "choose_best_params",
    "extract_nu_from_params",
    "load_and_prepare_residuals",
    "load_data_and_params",
    "load_estimation_file",
    "try_legacy_format_params",
    "try_new_format_params",
    # Plotting
    "compute_acf_pacf_data",
    "compute_qq_data",
    "create_figure_canvas",
    "create_residual_plots_figure",
    "plot_acf_subplot",
    "plot_pacf_subplot",
    "plot_qq_scatter",
    "plot_raw_residuals",
    "plot_standardized_residuals",
    "prepare_output_path",
    "prepare_residual_data",
    "save_figure_or_placeholder",
    "write_placeholder_png",
    # Standardization
    "compute_standardized_residuals_for_plot",
    "prepare_standardized_residuals_for_plotting",
    "standardize_residuals",
    # Statistical tests
    "compute_ljung_box_statistics",
]
