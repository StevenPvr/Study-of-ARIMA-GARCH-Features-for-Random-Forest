"""Optuna-based hyperparameter optimization for EGARCH models.

This module implements:
- Optuna study creation and configuration
- Hyperparameter suggestion strategy
- Objective function for QLIKE minimization
- Results saving and reporting
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import optuna

from src.constants import (
    DEFAULT_RANDOM_STATE,
    GARCH_OPTIMIZATION_DISTRIBUTIONS,
    GARCH_OPTIMIZATION_N_SPLITS,
    GARCH_OPTIMIZATION_N_TRIALS,
    GARCH_OPTIMIZATION_RESULTS_FILE,
    GARCH_OPTIMIZATION_ROLLING_WINDOW_SIZES,
    GARCH_OPTIMIZATION_WINDOW_TYPES,
)
from src.garch.garch_params.optimization.cross_validation import walk_forward_cv
from src.utils import get_logger, save_json_pretty

logger = get_logger(__name__)


def _create_optuna_study() -> optuna.Study:
    """Create Optuna study for minimization.

    Uses RandomSampler for small discrete search spaces (144 total combinations).
    RandomSampler is more appropriate than TPE for:
    - Small discrete search spaces
    - Categorical parameters
    - Ensuring uniform exploration

    With seed fixed, RandomSampler is still reproducible but explores
    the space more uniformly than TPE.

    Returns:
        Configured Optuna study.
    """
    # Total combinations: o(2) * p(2) * dist(3) * refit(3) * window(2) * size(2) = 144
    # RandomSampler is better suited for small discrete spaces than TPE
    # TPE is optimized for continuous spaces and may not explore discrete spaces well
    sampler = optuna.samplers.RandomSampler(seed=DEFAULT_RANDOM_STATE)
    return optuna.create_study(direction="minimize", sampler=sampler)


def _suggest_hyperparameters(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest hyperparameters for Optuna trial.

    Args:
        trial: Optuna trial object.

    Returns:
        Dictionary with suggested hyperparameters.
    """
    o = trial.suggest_int("o", 1, 2)
    p = trial.suggest_int("p", 1, 2)
    dist = trial.suggest_categorical("distribution", GARCH_OPTIMIZATION_DISTRIBUTIONS)
    # Test refit frequencies from 5 to 50 to find optimal balance
    # between model freshness and computational cost
    refit_freq = trial.suggest_int("refit_freq", 5, 60, step=5)
    window_type = trial.suggest_categorical("window_type", GARCH_OPTIMIZATION_WINDOW_TYPES)

    window_size = None
    if window_type == "rolling":
        window_size = trial.suggest_categorical(
            "window_size", GARCH_OPTIMIZATION_ROLLING_WINDOW_SIZES
        )

    return {
        "o": o,
        "p": p,
        "distribution": dist,
        "refit_freq": refit_freq,
        "window_type": window_type,
        "window_size": window_size,
    }


def optuna_objective(
    trial: optuna.Trial,
    resid_train: Any,
) -> float:
    """Optuna objective function for EGARCH hyperparameter optimization.

    Minimizes mean QLIKE from walk-forward cross-validation on TRAIN.

    Args:
        trial: Optuna trial object.
        resid_train: Training residuals (TRAIN split only).

    Returns:
        Mean QLIKE loss (to minimize).

    Raises:
        optuna.TrialPruned: If trial should be pruned.
    """
    params = _suggest_hyperparameters(trial)

    # Log trial parameters for debugging duplicate trials
    param_str = (
        f"o={params['o']}, p={params['p']}, dist={params['distribution']}, "
        f"refit_freq={params['refit_freq']}, window_type={params['window_type']}"
    )
    if params.get("window_size") is not None:
        param_str += f", window_size={params['window_size']}"
    logger.debug("Trial %d: %s", trial.number, param_str)

    try:
        mean_qlike = walk_forward_cv(
            resid_train,
            o=params["o"],
            p=params["p"],
            dist=params["distribution"],
            n_splits=GARCH_OPTIMIZATION_N_SPLITS,
            window_type=params["window_type"],
            window_size=params.get("window_size"),
            refit_freq=params["refit_freq"],
        )

        if not np.isfinite(mean_qlike):
            raise optuna.TrialPruned()

        logger.debug("Trial %d completed: QLIKE=%.6f", trial.number, mean_qlike)
        return float(mean_qlike)

    except Exception as ex:
        logger.debug("Trial %d failed: %s", trial.number, ex)
        raise optuna.TrialPruned() from ex


def optimize_egarch_hyperparameters(
    resid_train: Any,
    *,
    n_trials: int | None = None,
) -> dict[str, Any]:
    """Optimize EGARCH hyperparameters using Optuna.

    Performs walk-forward cross-validation on TRAIN data only,
    minimizing QLIKE out-of-sample.

    Args:
        resid_train: Training residuals (TRAIN split only).
        n_trials: Number of Optuna trials. Defaults to GARCH_OPTIMIZATION_N_TRIALS.

    Returns:
        Dictionary with best hyperparameters and optimization results.

    Raises:
        RuntimeError: If optimization fails or no valid trials.
    """
    if n_trials is None:
        n_trials = GARCH_OPTIMIZATION_N_TRIALS

    # Ensure reproducibility
    np.random.seed(DEFAULT_RANDOM_STATE)

    study = _create_optuna_study()

    logger.info("Starting EGARCH hyperparameter optimization with %d trials", n_trials)
    logger.info("Using walk-forward CV on TRAIN data only")

    try:
        study.optimize(
            lambda trial: optuna_objective(trial, resid_train),
            n_trials=n_trials,
            show_progress_bar=True,
        )
    except Exception as ex:
        msg = f"Optuna optimization failed: {ex}"
        logger.error(msg)
        raise RuntimeError(msg) from ex

    if len(study.trials) == 0:
        msg = "No completed trials in Optuna study"
        raise RuntimeError(msg)

    best_trial = study.best_trial

    if best_trial.value is None:
        msg = "Best trial has no value"
        raise RuntimeError(msg)

    if not np.isfinite(best_trial.value):
        msg = "Best trial has invalid QLIKE value"
        raise RuntimeError(msg)

    best_params = {
        "o": best_trial.params["o"],
        "p": best_trial.params["p"],
        "distribution": best_trial.params["distribution"],
        "refit_freq": best_trial.params["refit_freq"],
        "window_type": best_trial.params["window_type"],
    }

    if best_params["window_type"] == "rolling":
        best_params["window_size"] = best_trial.params["window_size"]

    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    result = {
        "best_params": best_params,
        "best_qlike": float(best_trial.value),
        "n_trials": len(study.trials),
        "n_complete_trials": n_complete,
    }

    logger.info("Optimization completed: best QLIKE=%.6f", result["best_qlike"])
    logger.info("Best hyperparameters: %s", best_params)

    return result


def save_optimization_results(
    results: dict[str, Any],
    output_file: Any | None = None,
) -> None:
    """Save optimization results to JSON file.

    Args:
        results: Optimization results dictionary.
        output_file: Output file path. Defaults to GARCH_OPTIMIZATION_RESULTS_FILE.
    """
    if output_file is None:
        output_file = GARCH_OPTIMIZATION_RESULTS_FILE

    output_path = Path(output_file)

    save_json_pretty(results, output_path)
    logger.info("Saved optimization results: %s", output_path)
