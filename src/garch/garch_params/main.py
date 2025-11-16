"""CLI for optimizing EGARCH hyperparameters via Optuna.

Performs walk-forward cross-validation on TRAIN data only,
minimizing QLIKE out-of-sample to find best hyperparameters:
- EGARCH orders: o, p ∈ {1,2}
- Distribution: normal, Student-t, Skew-t
- Refit frequency: 5, 10, 20
- Window type: expanding, rolling
- Rolling window size: 500, 1000 (if applicable)

Results saved to results/garch/optimization/
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.constants import GARCH_DATASET_FILE, GARCH_OPTIMIZATION_RESULTS_FILE
from src.garch.garch_params.data import load_and_prepare_data
from src.garch.garch_params.estimation_batch import run_batch_estimation_and_save
from src.garch.garch_params.optimization import (
    optimize_egarch_hyperparameters,
    save_optimization_results,
)
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Optimize EGARCH hyperparameters via Optuna with walk-forward CV.

    Methodology:
    - Runs batch estimation first (required preparation step)
    - Uses TRAIN data only (no test data leakage)
    - Walk-forward cross-validation with 30% burn-in
    - Minimizes QLIKE out-of-sample
    - Searches over: orders (o,p), distribution, refit_freq, window_type, window_size

    Results saved to results/garch/optimization/
    """
    logger.info("=" * 60)
    logger.info("EGARCH HYPERPARAMETER OPTIMIZATION")
    logger.info("Method: Optuna + Walk-forward CV on TRAIN")
    logger.info("Objective: Minimize QLIKE out-of-sample")
    logger.info("=" * 60)

    # Load dataset for batch estimation
    df = pd.read_csv(GARCH_DATASET_FILE, parse_dates=["date"])

    # Stage 1: Batch estimation (required before optimization)
    logger.info("")
    logger.info("-" * 60)
    logger.info("STAGE 1: Batch Estimation (preparation step)")
    logger.info("-" * 60)
    run_batch_estimation_and_save(df, o=1, p=1)
    logger.info("")

    # Stage 2: Hyperparameter optimization
    logger.info("-" * 60)
    logger.info("STAGE 2: Hyperparameter Optimization")
    logger.info("-" * 60)
    resid_train, resid_test = load_and_prepare_data()

    logger.info("Training data: %d observations", resid_train.size)
    logger.info("Test data: %d observations (not used in optimization)", resid_test.size)

    # Optimize hyperparameters
    results = optimize_egarch_hyperparameters(resid_train)

    # Add metadata
    results["source"] = str(GARCH_DATASET_FILE)
    results["methodology"] = "Optuna optimization with walk-forward CV on TRAIN"
    results["n_obs_train"] = int(resid_train.size)
    results["n_obs_test"] = int(resid_test.size)

    # Save results
    save_optimization_results(results, GARCH_OPTIMIZATION_RESULTS_FILE)

    logger.info("=" * 60)
    logger.info("Optimization completed successfully")
    logger.info("Best hyperparameters saved to: %s", GARCH_OPTIMIZATION_RESULTS_FILE)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
