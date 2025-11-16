"""End-to-end SARIMA optimization (Optuna/grid), diagnostics and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd

from src.constants import (
    DEFAULT_RANDOM_STATE,
    SARIMA_LJUNGBOX_PENALTY_WEIGHT,
    SARIMA_LJUNGBOX_P_VALUE_THRESHOLD,
    SARIMA_OPTIMIZATION_N_SPLITS,
    SARIMA_OPTIMIZATION_N_TRIALS,
    SARIMA_OPTIMIZATION_TEST_SIZE,
    SARIMA_OPTIMIZATION_TOP_K_RESULTS,
    SARIMA_REFIT_EVERY_DEFAULT,
)
from src.path import (
    SARIMA_BEST_MODELS_FILE,
    SARIMA_OPTIMIZATION_RESULTS_FILE,
)
from src.utils import get_logger, get_parquet_path, load_parquet_file, validate_required_columns

from .objective import NormalizationStatsCallback, objective_aic
from .results_processing import pick_best, save_results, to_dataframe
from .validation import compute_test_size_from_ratio, validate_backtest_config, validate_series

logger = get_logger(__name__)


def _extract_and_sort_train_data(
    df: pd.DataFrame, value_col: str, date_col: Optional[str]
) -> pd.Series:
    """Extract training data from DataFrame and optionally sort by date.

    Args:
        df: DataFrame with 'split' column.
        value_col: Name of the numeric column to extract.
        date_col: Optional name of date column for sorting.

    Returns:
        Training series sorted by date if date_col provided.

    Raises:
        ValueError: If no train data found.
    """
    train_df = df[df["split"] == "train"].copy()
    if train_df.empty:
        msg = "No 'train' data found in split column"
        raise ValueError(msg)
    if date_col and date_col in train_df.columns:
        train_df[date_col] = pd.to_datetime(train_df[date_col], utc=False)
        date_col_str: str = date_col
        train_df = train_df.sort_values(by=[date_col_str])  # type: ignore[arg-type]
    y = pd.Series(train_df[value_col].astype(float)).reset_index(drop=True)
    validate_series("train_series", y)
    return y


def load_train_data(
    csv_path: Path,
    value_col: str,
    date_col: Optional[str] = None,
) -> pd.Series:
    """Load training data from Parquet file with split column.

    Loads a Parquet file and returns only the training split.
    The file must contain a 'split' column with 'train' and 'test' values.
    Only 'train' data is returned for optimization (no test data leakage).

    Args:
        csv_path: Path to data file (Parquet file will be loaded from this path
            with .parquet extension).
        value_col: Name of the numeric column to model.
        date_col: Optional name of date column for sorting (if None, no sorting).

    Returns:
        Training series (only 'train' split from the data).

    Raises:
        FileNotFoundError: If Parquet file does not exist.
        ValueError: If path exists but is not a file, required columns are missing,
            or no train data found.
    """
    parquet_path = get_parquet_path(csv_path)
    if not parquet_path.exists():
        msg = f"Parquet file not found: {parquet_path}"
        raise FileNotFoundError(msg)
    if not parquet_path.is_file():
        msg = f"Path is not a file: {parquet_path}"
        raise ValueError(msg)

    logger.info(f"Loading training dataset from {parquet_path}")
    df = load_parquet_file(parquet_path)
    if df is None or df.empty:
        msg = f"Failed to load data from {parquet_path}"
        raise ValueError(msg)
    validate_required_columns(df, [value_col, "split"], df_name="Data file")
    return _extract_and_sort_train_data(df, value_col, date_col)


def _prepare_optuna(seed: int = DEFAULT_RANDOM_STATE) -> optuna.Study:
    """Create and configure an Optuna study for minimization.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Configured Optuna study with TPE sampler.
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    return optuna.create_study(direction="minimize", sampler=sampler)


def _extract_results_from_trials(
    study: optuna.Study, top_k: int = SARIMA_OPTIMIZATION_TOP_K_RESULTS
) -> List[Dict]:
    """Extract complete evaluation results from top Optuna trials.

    Results are already stored in trial user_attrs during optimization,
    so no re-evaluation is needed.

    Args:
        study: Optuna study with completed trials.
        top_k: Number of top trials to extract (default: SARIMA_OPTIMIZATION_TOP_K_RESULTS).

    Returns:
        List of evaluation result dictionaries from top trials.
    """
    trials = sorted(
        study.trials, key=lambda tr: tr.value if tr.value is not None else float("inf")
    )[: min(top_k, len(study.trials))]
    results: List[Dict] = []
    for tr in trials:
        p = tr.params
        params_dict = {
            "p": int(p.get("p", 0)),
            "d": int(p.get("d", 0)),
            "q": int(p.get("q", 0)),
            "P": int(p.get("P", 0)),
            "D": int(p.get("D", 0)),
            "Q": int(p.get("Q", 0)),
            "s": int(p.get("s", 1)),
            "trend": str(p.get("trend", "n")),
            "refit_every": int(p.get("refit_every", SARIMA_REFIT_EVERY_DEFAULT)),
        }
        # Extract results from trial user_attrs (stored during optimization)
        if "error" in tr.user_attrs:
            results.append({"params": params_dict, "error": tr.user_attrs["error"]})
        else:
            result_dict: Dict[str, Any] = {
                "params": params_dict,
                "aic": tr.user_attrs.get("aic", float("inf")),
                "bic": tr.user_attrs.get("bic", float("inf")),
                "composite_score": tr.user_attrs.get("composite_score", float("inf")),
                "lb_stat": tr.user_attrs.get("lb_stat", float("nan")),
                "lb_pvalue": tr.user_attrs.get("lb_pvalue", float("nan")),
            }
            # Include validation metrics if available
            if "val_rmse" in tr.user_attrs:
                result_dict["val_rmse"] = tr.user_attrs.get("val_rmse", float("inf"))
                result_dict["val_mae"] = tr.user_attrs.get("val_mae", float("inf"))
                result_dict["val_mean_error"] = tr.user_attrs.get("val_mean_error", 0.0)
            results.append(result_dict)
    return results


def _run_optuna(
    train: pd.Series,
    n_trials: int,
    criterion: str = "aic",
    seed: int = DEFAULT_RANDOM_STATE,
    backtest_cfg: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    """Run Optuna optimization and extract results from trials.

    Performs hyperparameter search using Optuna. Complete evaluation results
    (AIC, BIC, Ljung-Box test, validation metrics) are stored in trial user_attrs
    during optimization, so no re-evaluation is needed.

    When backtest_cfg is provided, uses z-score normalization for composite scores
    to properly balance information criterion and validation RMSE (which are on
    different scales).

    Args:
        train: Training time series data.
        n_trials: Number of Optuna trials to run.
        criterion: Optimization criterion ("aic" or "bic", default: "aic").
        seed: Random seed for reproducibility.
        backtest_cfg: Optional backtest configuration for validation metrics.
            If provided, optimization uses composite score (criterion + validation)
            with z-score normalization.

    Returns:
        List of evaluation result dictionaries for top candidates.
    """
    study = _prepare_optuna(seed)
    # Only AIC is supported - optimal for forecasting applications
    if criterion.lower() != "aic":
        raise ValueError(
            f"Only 'aic' criterion is supported (got '{criterion}'). "
            "BIC was removed as AIC is theoretically optimal for forecasting. "
            "AIC minimizes one-step-ahead prediction error, while BIC minimizes description length."
        )
    obj = objective_aic
    logger.debug("Using AIC objective function for optimization")

    if backtest_cfg:
        logger.info(
            f"Using validation-aware optimization with z-score normalization. "
            f"Backtest config: {backtest_cfg}"
        )

    # Create callback for collecting normalization statistics
    stats_callback = NormalizationStatsCallback(criterion=criterion)

    logger.info(f"Starting {criterion.upper()} optimization with {n_trials} trials...")

    try:
        # Use callback to collect stats during optimization
        study.optimize(
            lambda t: obj(t, train, backtest_cfg, stats_callback),
            n_trials=n_trials,
            callbacks=[stats_callback],
        )
        logger.info(
            f"{criterion.upper()} optimization completed - " f"Best value: {study.best_value:.4f}"
        )

        # Log normalization statistics if available
        if backtest_cfg:
            criterion_stats, rmse_stats = stats_callback.get_stats()
            if criterion_stats is not None and rmse_stats is not None:
                logger.info(
                    f"Normalization stats - {criterion.upper()}: mean={criterion_stats[0]:.4f}, "
                    f"std={criterion_stats[1]:.4f} | "
                    f"RMSE: mean={rmse_stats[0]:.6f}, std={rmse_stats[1]:.6f}"
                )
            else:
                logger.warning("Insufficient data for normalization statistics")

    except Exception as e:
        logger.error(f"Optuna optimization failed for {criterion}: {e}")
        raise
    logger.info(f"Extracting results from top trials for {criterion} optimization")
    results = _extract_results_from_trials(study)
    if not results:
        logger.warning(f"No valid results found for {criterion} optimization")
        return []
    logger.info(f"Extracted {len(results)} top candidates for {criterion}")
    return results


def _validate_optimization_inputs(
    train_series: pd.Series,
    test_series: Optional[pd.Series],
    backtest_cfg: Optional[Dict[str, int]],
    search: str,
) -> None:
    """Validate inputs for SARIMA optimization.

    Args:
        train_series: Training time series data.
        test_series: Optional test series.
        backtest_cfg: Optional backtest configuration dict (not used in optimization).
        search: Search method (must be "optuna").

    Raises:
        ValueError: If validation fails.
    """
    validate_series("train_series", train_series)
    if test_series is not None:
        validate_series("test_series", test_series)
    if backtest_cfg:
        # refit_every is now part of SarimaParams, not backtest_cfg
        # Use default refit_every value for validation
        from src.constants import SARIMA_REFIT_EVERY_DEFAULT

        validate_backtest_config(
            backtest_cfg["n_splits"],
            backtest_cfg["test_size"],
            backtest_cfg.get("refit_every", SARIMA_REFIT_EVERY_DEFAULT),
            len(train_series),
        )
    if search.lower() != "optuna":
        raise ValueError("Only 'optuna' is supported for parameter 'search'.")


def _run_aic_optimization(
    train_series: pd.Series,
    n_trials: int,
    random_state: int,
    backtest_cfg: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    """Run AIC optimization.

    AIC (Akaike Information Criterion) is used exclusively for SARIMA optimization
    because it is theoretically optimal for minimizing one-step-ahead prediction error,
    which is the primary objective in financial forecasting.

    BIC (Bayesian Information Criterion) was removed because:
    - BIC minimizes description length, not prediction error
    - BIC penalizes model complexity more heavily than AIC
    - For forecasting (vs model identification), AIC is asymptotically optimal

    Args:
        train_series: Training time series data.
        n_trials: Number of Optuna trials.
        random_state: Random seed for reproducibility.
        backtest_cfg: Optional backtest configuration for validation metrics.

    Returns:
        List of evaluation results from AIC optimization.
    """
    logger.info("Starting AIC optimization (%d trials)", n_trials)

    # Run AIC optimization
    logger.info("Running AIC optimization...")
    results_aic = _run_optuna(
        train_series,
        n_trials=n_trials,
        criterion="aic",
        seed=random_state,
        backtest_cfg=backtest_cfg,
    )
    logger.info("AIC optimization completed")

    return results_aic


def _validate_backtest_parameters(
    backtest_cfg: Optional[Dict[str, int]],
    backtest_n_splits: Optional[int],
    backtest_test_size: Optional[int],
    backtest_refit_every: Optional[int],
) -> None:
    """Validate backtest parameter combinations for conflicts.

    Args:
        backtest_cfg: Deprecated parameter, kept for backward compatibility.
        backtest_n_splits: Number of temporal backtest splits.
        backtest_test_size: Absolute size of each test window.
        backtest_refit_every: Refit frequency for temporal backtesting.

    Raises:
        ValueError: If parameters conflict.
    """
    if backtest_cfg is not None and any(
        value is not None for value in (backtest_n_splits, backtest_test_size, backtest_refit_every)
    ):
        msg = (
            "Provide either 'backtest_cfg' or the explicit backtest parameters "
            "('backtest_n_splits', 'backtest_test_size', 'backtest_refit_every'), not both."
        )
        raise ValueError(msg)


def _validate_required_backtest_params(
    backtest_n_splits: Optional[int],
    backtest_refit_every: Optional[int],
) -> None:
    """Validate that required backtest parameters are provided.

    Args:
        backtest_n_splits: Number of temporal backtest splits.
        backtest_refit_every: Refit frequency for temporal backtesting.

    Raises:
        ValueError: If required parameters are missing.
    """
    if backtest_n_splits is None or backtest_refit_every is None:
        test_size_pct = SARIMA_OPTIMIZATION_TEST_SIZE * 100
        msg = (
            "Backtest parameters are required for validation-aware optimization. "
            "Please provide at minimum: "
            f"'backtest_n_splits' (recommended: {SARIMA_OPTIMIZATION_N_SPLITS}), "
            f"'backtest_refit_every' (recommended: {SARIMA_REFIT_EVERY_DEFAULT}). "
            f"'backtest_test_size' is optional and defaults to {test_size_pct:.0f}% "
            "of training data divided by n_splits."
        )
        raise ValueError(msg)


def _compute_test_size(
    backtest_test_size: Optional[int],
    train_series: pd.Series,
    n_splits: int,
) -> int:
    """Compute test size for backtesting.

    Args:
        backtest_test_size: Explicit test size if provided.
        train_series: Training time series data.
        n_splits: Number of backtest splits.

    Returns:
        Computed or explicit test size.
    """
    if backtest_test_size is None:
        computed_test_size = compute_test_size_from_ratio(
            train_len=len(train_series),
            test_size_ratio=SARIMA_OPTIMIZATION_TEST_SIZE,
            n_splits=n_splits,
        )
        logger.info(
            f"Computed test_size={computed_test_size} from ratio "
            f"{SARIMA_OPTIMIZATION_TEST_SIZE:.2f} with train_len={len(train_series)} "
            f"and n_splits={n_splits}"
        )
        return computed_test_size
    else:
        final_test_size = int(backtest_test_size)
        logger.info(f"Using explicit test_size={final_test_size}")
        return final_test_size


def _compute_backtest_configuration(
    backtest_n_splits: Optional[int],
    backtest_test_size: Optional[int],
    backtest_refit_every: Optional[int],
    train_series: pd.Series,
) -> Dict[str, int]:
    """Compute backtest configuration with validation and defaults.

    Args:
        backtest_n_splits: REQUIRED. Number of temporal backtest splits.
        backtest_test_size: Optional. Absolute size of each test window.
        backtest_refit_every: REQUIRED. Refit frequency for temporal backtesting.
        train_series: Training time series data for size calculations.

    Returns:
        Configured backtest parameters dictionary.

    Raises:
        ValueError: If required parameters are missing.
    """
    _validate_required_backtest_params(backtest_n_splits, backtest_refit_every)
    # At this point, we know these are not None due to validation above
    assert backtest_n_splits is not None
    assert backtest_refit_every is not None
    n_splits = backtest_n_splits
    refit_every = backtest_refit_every
    final_test_size = _compute_test_size(backtest_test_size, train_series, n_splits)
    assert isinstance(final_test_size, int)  # Should always be int

    backtest_cfg = {
        "n_splits": n_splits,
        "test_size": final_test_size,
        "refit_every": refit_every,
    }
    logger.info(
        f"Validation configuration: "
        f"n_splits={n_splits}, "
        f"test_size={final_test_size}, "
        f"refit_every={refit_every}"
    )
    return backtest_cfg


def _analyze_ljung_box_statistics(df: pd.DataFrame) -> None:
    """Analyze and log Ljung-Box test statistics from optimization results.

    Args:
        df: DataFrame containing optimization results with lb_pvalue column.
    """
    # Summarize Ljung–Box p-values distribution to assist λ calibration
    if "lb_pvalue" in df.columns and not df["lb_pvalue"].isna().all():
        lb = df["lb_pvalue"].astype(float)
        lb_valid = lb[np.isfinite(lb)]
        if not lb_valid.empty:
            n = int(lb_valid.shape[0])
            below = int((lb_valid < SARIMA_LJUNGBOX_P_VALUE_THRESHOLD).sum())
            frac = (below / n) * 100.0
            lb_mean = float(lb_valid.mean())
            lb_min = float(lb_valid.min())
            lb_max = float(lb_valid.max())
            logger.info(
                "Ljung-Box p-values summary | n=%d, below_thresh=%d (%.1f%%), "
                "mean=%.4f, min=%.4f, max=%.4f | threshold=%.2f, penalty=%.2f",
                n,
                below,
                frac,
                lb_mean,
                lb_min,
                lb_max,
                SARIMA_LJUNGBOX_P_VALUE_THRESHOLD,
                SARIMA_LJUNGBOX_PENALTY_WEIGHT,
            )


def _select_and_log_best_model(df: pd.DataFrame) -> Dict:
    """Select best model and log results.

    Args:
        df: DataFrame containing optimization results.

    Returns:
        Best model dictionary.
    """
    # Select best model based on AIC
    best_aic, _ = pick_best(df)  # pick_best returns (best_aic, best_bic)
    # We only use best_aic; best_bic is None/ignored

    logger.info(
        "Best model: params=%s | AIC=%.6f | BIC=%.6f (BIC for reference only)",
        best_aic.get("params"),
        float(best_aic.get("aic", float("nan"))),
        float(best_aic.get("bic", float("nan"))),  # BIC computed but not optimized
    )
    return best_aic


def _save_optimization_results(
    df: pd.DataFrame,
    best_model: Dict,
    out_dir: Optional[Path],
) -> None:
    """Save optimization results to disk.

    Args:
        df: DataFrame containing optimization results.
        best_model: Best model dictionary.
        out_dir: Optional output directory for saving results.
            If None, no saving is performed.
    """
    if out_dir is None:
        return  # Don't save results when out_dir is explicitly None (for testing)

    save_results(
        df,
        best_model,
        None,  # best_bic set to None (no longer optimized)
        out_dir,
        best_models_file=SARIMA_BEST_MODELS_FILE,
        results_file=SARIMA_OPTIMIZATION_RESULTS_FILE,
    )


def optimize_sarima_models(
    train_series: pd.Series,
    test_series: Optional[pd.Series] = None,
    *,
    search: str = "optuna",  # kept for backward compat; only "optuna" is supported
    n_trials: int = SARIMA_OPTIMIZATION_N_TRIALS,
    n_jobs: int | None = None,
    backtest_cfg: Optional[Dict[str, int]] = None,  # Deprecated, kept for backward compatibility
    backtest_n_splits: Optional[int] = None,
    backtest_test_size: Optional[int] = None,
    backtest_refit_every: Optional[int] = None,
    out_dir: Optional[Path] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict, Optional[Dict]]:
    """Main entry point for SARIMA model optimization using Optuna with validation.

    Performs hyperparameter search using Optuna with walk-forward cross-validation,
    computes diagnostics, and saves results to disk.

    **Important**: This function runs AIC optimization only (BIC removed for forecasting).
    Uses composite score combining AIC with validation RMSE from walk-forward CV.

    **Validation Parameters Required**: This function requires explicit validation
    parameters. You must provide backtest_n_splits and backtest_refit_every.
    The backtest_test_size parameter is optional:
    - backtest_n_splits (recommended: 5)
    - backtest_test_size (optional, default: computed as 20% of train_series length)
    - backtest_refit_every (recommended: 20)

    Args:
        train_series: Training time series data.
        test_series: Optional test series (currently unused, reserved for future use).
        search: Only "optuna" is supported. Other values raise ValueError.
        n_trials: Number of Optuna trials (default: 100).
        n_jobs: Deprecated parameter, kept for backward compatibility (ignored).
        backtest_cfg: Deprecated parameter, kept for backward compatibility (ignored).
        backtest_n_splits: REQUIRED. Number of temporal backtest splits for validation.
            Recommended value: 5 (SARIMA_OPTIMIZATION_N_SPLITS).
        backtest_test_size: Optional. Absolute size of each test window in walk-forward CV.
            If not provided, computed as 20% of train_series length divided by n_splits
            (SARIMA_OPTIMIZATION_TEST_SIZE = 0.2). If provided, used directly as
            absolute number of observations per split.
        backtest_refit_every: REQUIRED. Refit frequency for temporal backtesting.
            Recommended value: 20 (SARIMA_REFIT_EVERY_DEFAULT).
        out_dir: Optional output directory for saving results.
        random_state: Random seed for reproducibility (default: 42).

    Returns:
        Tuple of (results_dataframe, best_model_dict, None).
        The third element is None since BIC optimization was removed.

    Raises:
        ValueError: If backtest parameters are missing, if validation fails,
            or if search method is not "optuna".
    """
    # Set random seed for reproducibility (numpy, Optuna handles its own seed)
    np.random.seed(random_state)

    # Handle deprecated parameters
    if n_jobs not in (None, 1):
        logger.warning("Parameter n_jobs=%s is deprecated and ignored.", n_jobs)

    # Validate parameter combinations
    _validate_backtest_parameters(
        backtest_cfg, backtest_n_splits, backtest_test_size, backtest_refit_every
    )

    # Configure backtest parameters if not provided
    if backtest_cfg is None:
        backtest_cfg = _compute_backtest_configuration(
            backtest_n_splits, backtest_test_size, backtest_refit_every, train_series
        )

    # Validate all inputs
    _validate_optimization_inputs(train_series, test_series, backtest_cfg, search)

    # Run AIC optimization (BIC removed - AIC is optimal for forecasting)
    results = _run_aic_optimization(train_series, n_trials, random_state, backtest_cfg)
    logger.info("Collected %d AIC candidates", len(results))

    # Process results
    df = to_dataframe(results)
    _analyze_ljung_box_statistics(df)
    best_model = _select_and_log_best_model(df)
    _save_optimization_results(df, best_model, out_dir)

    # Return format: (df, best_model, None) since BIC optimization was removed
    return df, best_model, None
