"""CLI entry to run volatility backtest and save outputs."""

from __future__ import annotations


import argparse
import json
import sys

from src.utils import setup_project_path

# Ensure project root
setup_project_path()

import numpy as np

from src.constants import VOL_EWMA_LAMBDA_DEFAULT, VOL_ROLLING_WINDOW_DEFAULT
from src.garch.benchmark.bench_volatility import (
    run_benchmark_section4,
    run_vol_backtest,
    save_vol_backtest_outputs,
)
from src.garch.structure_garch.utils import load_garch_dataset
from src.path import BENCHMARK_RESULTS_DIR
from src.utils import get_logger

logger = get_logger(__name__)


def _run_backtest(args: argparse.Namespace) -> None:
    """Execute volatility backtest with parsed arguments."""
    logger.info(
        "Vol backtest: ewma=%.3f, window=%d, refit_every=%d",
        args.ewma_lambda,
        args.window,
        args.refit_every,
    )
    data_frame = load_garch_dataset()  # uses constants path
    forecasts, metrics = run_vol_backtest(
        data_frame,
        ewma_lambda=float(args.ewma_lambda),
        rolling_window=int(args.window),
        refit_every=int(args.refit_every),
    )
    save_vol_backtest_outputs(forecasts, metrics)
    logger.info("Volatility backtest completed successfully")


def _run_benchmark_section4(args: argparse.Namespace) -> None:
    """Execute benchmark comparison according to Section 4 methodology."""
    logger.info(
        "Benchmark Section 4: ewma=%.3f, window=%d",
        args.ewma_lambda,
        args.window,
    )
    alphas = [float(x) for x in args.alphas.split(",")] if args.alphas else [0.01, 0.05]

    try:
        results = run_benchmark_section4(
            ewma_lambda=float(args.ewma_lambda),
            rolling_window=int(args.window),
            alphas=alphas,
        )

        # Save results
        BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_file = BENCHMARK_RESULTS_DIR / "benchmark_section4.json"
        with output_file.open("w") as f:
            json.dump(results, f, indent=2)

        logger.info("Benchmark comparison completed successfully")
        logger.info("Results saved to: %s", output_file)
        logger.info("Comparison table:")
        for row in results.get("comparison_table", []):
            model_name = row.get("Model", "Unknown")
            qlike = row.get("QLIKE", float("nan"))
            mse_var = row.get("MSE_var", float("nan"))
            mae_var = row.get("MAE_var", float("nan"))

            # VaR metrics
            var_1_hit = row.get("VaR_1%_hit_rate", float("nan"))
            var_5_hit = row.get("VaR_5%_hit_rate", float("nan"))

            var_1_hit_pct = var_1_hit * 100 if not np.isnan(var_1_hit) else float("nan")
            var_5_hit_pct = var_5_hit * 100 if not np.isnan(var_5_hit) else float("nan")
            logger.info(
                (
                    "  %s: QLIKE=%.4f, MSE_var=%.6f, MAE_var=%.6f, "
                    "VaR_1%%_hit=%.2f%%, VaR_5%%_hit=%.2f%%"
                ),
                model_name,
                qlike,
                mse_var,
                mae_var,
                var_1_hit_pct,
                var_5_hit_pct,
            )
    except FileNotFoundError as e:
        logger.error("Required file not found: %s", e)
        raise
    except ValueError as e:
        logger.error("Invalid parameter: %s", e)
        raise


def main() -> None:
    """CLI entry point for volatility backtest."""
    parser = argparse.ArgumentParser(
        description="Volatility backtest: EGARCH vs baselines (Section 4)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["section4", "legacy"],
        default="section4",
        help=(
            "Benchmark method: 'section4' (new methodology) or 'legacy' "
            "(old method). Defaults to 'section4'."
        ),
    )
    parser.add_argument("--ewma-lambda", type=float, default=VOL_EWMA_LAMBDA_DEFAULT)
    parser.add_argument("--window", type=int, default=VOL_ROLLING_WINDOW_DEFAULT)
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.01,0.05",
        help="Comma-separated VaR alpha levels (default: 0.01,0.05)",
    )
    parser.add_argument(
        "--refit-every",
        type=int,
        default=20,
        help="GARCH refit frequency (in test observations, legacy only). Defaults to 20.",
    )
    args = parser.parse_args()

    try:
        if args.method == "section4":
            _run_benchmark_section4(args)
        else:
            _run_backtest(args)
    except FileNotFoundError as e:
        logger.error("Data file not found: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Invalid parameter: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during volatility backtest: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
