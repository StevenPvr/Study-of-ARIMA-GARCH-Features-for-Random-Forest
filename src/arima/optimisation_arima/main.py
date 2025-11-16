"""CLI for SARIMA optimization module."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to Python path for direct execution.
# This must be done before importing src modules.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


from src.arima.optimisation_arima.optimisation_arima import (  # noqa: E402
    load_train_data,
    optimize_sarima_models,
)
from src.constants import (  # noqa: E402
    SARIMA_DEFAULT_SEASONAL_PERIOD,
    SARIMA_DEFAULT_VALUE_COLUMN,
    SARIMA_OPTIMIZATION_N_SPLITS,
    SARIMA_OPTIMIZATION_N_TRIALS,
    SARIMA_REFIT_EVERY_DEFAULT,
)
from src.path import SARIMA_ARTIFACTS_DIR, WEIGHTED_LOG_RETURNS_SPLIT_FILE  # noqa: E402


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Add data-related command-line arguments.

    Args:
        parser: ArgumentParser to add arguments to.
    """
    parser.add_argument(
        "--csv",
        type=Path,
        default=WEIGHTED_LOG_RETURNS_SPLIT_FILE,
        help=f"Path to data file (default: {WEIGHTED_LOG_RETURNS_SPLIT_FILE}).",
    )
    parser.add_argument(
        "--value-col",
        type=str,
        default=SARIMA_DEFAULT_VALUE_COLUMN,
        help=f"Numeric column to model (default: {SARIMA_DEFAULT_VALUE_COLUMN}).",
    )
    parser.add_argument("--date-col", type=str, default=None, help="Optional date column.")


def _add_optimization_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optimization-related command-line arguments.

    Args:
        parser: ArgumentParser to add arguments to.
    """
    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=SARIMA_DEFAULT_SEASONAL_PERIOD,
        help=(
            f"SARIMA seasonal period (deprecated: S is now optimized, "
            f"default: {SARIMA_DEFAULT_SEASONAL_PERIOD})."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=SARIMA_OPTIMIZATION_N_TRIALS,
        help=f"Optuna trials per criterion (default: {SARIMA_OPTIMIZATION_N_TRIALS}).",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=SARIMA_OPTIMIZATION_N_SPLITS,
        help=f"Walk-forward CV splits (default: {SARIMA_OPTIMIZATION_N_SPLITS}).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help=(
            "Test window size per split (number of observations). "
            "If not provided, computed automatically as 20%% of training data "
            "divided by number of splits."
        ),
    )
    parser.add_argument(
        "--refit-every",
        type=int,
        default=SARIMA_REFIT_EVERY_DEFAULT,
        help=f"Refit frequency (default: {SARIMA_REFIT_EVERY_DEFAULT}).",
    )
    parser.add_argument("--n-jobs", type=int, default=1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for SARIMA optimization.

    Returns:
        Namespace object with parsed arguments.
    """
    p = argparse.ArgumentParser(description="Optimize SARIMA via Optuna or grid search.")
    _add_data_arguments(p)
    _add_optimization_arguments(p)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=SARIMA_ARTIFACTS_DIR,
        help=f"Output directory (default: {SARIMA_ARTIFACTS_DIR}).",
    )
    return p.parse_args()


def _format_model_info(label: str, model: Dict[str, Any], primary: str, secondary: str) -> str:
    """Format model information for display.

    Args:
        label: Model label (e.g., "Best AIC model").
        model: Model dictionary with params and metrics.
        primary: Primary metric name (e.g., "aic").
        secondary: Secondary metric name (e.g., "bic").

    Returns:
        Formatted string with model information.
    """
    params = model.get("params")
    primary_val = model.get(primary, float("nan"))
    secondary_val = model.get(secondary, float("nan"))
    return (
        f"{label}: params={params} | {primary.upper()}={primary_val:.6f} | "
        f"{secondary.upper()}={secondary_val:.6f}"
    )


def print_best_models(
    best_aic: Dict[str, Any], best_bic: Optional[Dict[str, Any]], out_dir: Path
) -> None:
    """Print best AIC and BIC models.

    Args:
        best_aic: Dictionary with best AIC model parameters and metrics.
        best_bic: Dictionary with best BIC model parameters and metrics, or None if BIC
        optimization is disabled.
        out_dir: Output directory path.
    """
    separator_length = 80
    print("=" * separator_length)
    print(_format_model_info("Best AIC model", best_aic, "aic", "bic"))
    if best_bic is not None:
        print(_format_model_info("Best BIC model", best_bic, "bic", "aic"))
    print("Results saved to:", out_dir)
    print("=" * separator_length)


def main() -> None:
    """Main CLI entry point for SARIMA optimization.

    Loads data, runs optimization, and prints results.
    """
    args = parse_args()

    train = load_train_data(
        csv_path=args.csv,
        value_col=args.value_col,
        date_col=args.date_col,
    )

    _, best_aic, best_bic = optimize_sarima_models(
        train_series=train,
        test_series=None,
        n_trials=args.trials,
        n_jobs=args.n_jobs,
        backtest_n_splits=args.splits,
        backtest_test_size=args.test_size,
        backtest_refit_every=args.refit_every,
        out_dir=args.out_dir,
    )

    print_best_models(best_aic, best_bic, args.out_dir)


if __name__ == "__main__":
    main()
