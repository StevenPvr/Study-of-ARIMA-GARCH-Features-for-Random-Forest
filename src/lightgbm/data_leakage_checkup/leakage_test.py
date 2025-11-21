"""Target shuffle test for data leakage detection.

This test trains a model with shuffled target to detect potential data leakage.
If the model achieves high R² with shuffled target, it indicates leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import lightgbm as lgb
from src.constants import (
    DEFAULT_RANDOM_STATE,
    LEAKAGE_R2_THRESHOLD,
    LIGHTGBM_COMPLETE_MODEL_NAME,
    LIGHTGBM_DATASET_COMPLETE_FILE,
    LIGHTGBM_DATASET_COMPLETE_MODEL_NAME,
    LIGHTGBM_EVAL_RESULTS_FILE,
    LIGHTGBM_OPTIMIZATION_RESULTS_FILE,
    TEST_SPLIT_LABEL,
    TRAIN_SPLIT_LABEL,
)
from src.utils import get_logger
from src.utils.transforms import extract_features_and_target

logger = get_logger(__name__)


def _load_json_results(
    results_file: Path, key_name: str, results_type: str = "results"
) -> dict[str, Any]:
    """Load JSON results file and extract specific data.

    Args:
        results_file: Path to the JSON results file.
        key_name: Key to extract from the results.
        results_type: Type of results for error messages.

    Returns:
        Dictionary with the extracted data.

    Raises:
        FileNotFoundError: If results file doesn't exist.
        KeyError: If key_name not found in results.
    """
    if not results_file.exists():
        raise FileNotFoundError(f"{results_type} not found: {results_file}")

    with open(results_file) as f:
        results = json.load(f)

    if key_name not in results:
        raise KeyError(f"{key_name} not found in {results_type}")

    logger.info(f"Loaded {results_type} for {key_name}")
    return results[key_name]


def _load_optimization_params(
    model_name: str = LIGHTGBM_DATASET_COMPLETE_MODEL_NAME,
) -> dict[str, Any]:
    """Load best hyperparameters from optimization results.

    Args:
        model_name: Name of the model configuration.

    Returns:
        Dictionary of best hyperparameters.

    Raises:
        FileNotFoundError: If optimization results file doesn't exist.
        KeyError: If model_name not found in results.
    """
    model_results = _load_json_results(
        LIGHTGBM_OPTIMIZATION_RESULTS_FILE, model_name, "optimization results"
    )
    return model_results["best_params"]


def _load_dataset(
    dataset_path: Path = LIGHTGBM_DATASET_COMPLETE_FILE,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load and split dataset into train and test.

    Args:
        dataset_path: Path to dataset file.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test).
    """
    if dataset_path.suffix == ".parquet":
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix == ".csv":
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path.suffix}")

    # Split by 'split' column
    df_train = cast(pd.DataFrame, df[df["split"] == TRAIN_SPLIT_LABEL].copy())
    df_test = cast(pd.DataFrame, df[df["split"] == TEST_SPLIT_LABEL].copy())

    # Remove metadata columns before extracting features
    from src.utils.transforms import remove_metadata_columns

    df_train_clean = remove_metadata_columns(df_train)
    df_test_clean = remove_metadata_columns(df_test)

    # Extract features and target
    X_train, y_train = extract_features_and_target(df_train_clean)
    X_test, y_test = extract_features_and_target(df_test_clean)

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logger.info(f"Features: {len(X_train.columns)}")

    return X_train, y_train, X_test, y_test


def _shuffle_target(
    y_train: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.Series:
    """Shuffle target while preserving the same index.

    Args:
        y_train: Training target.
        random_state: Random seed for reproducibility.

    Returns:
        Shuffled target with original index.
    """
    y_shuffled = y_train.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    # Preserve original index for alignment
    y_shuffled.index = y_train.index
    logger.info(f"Shuffled {len(y_shuffled)} target values (random_state={random_state})")
    return y_shuffled


def _train_model_with_shuffled_target(
    X_train: pd.DataFrame,
    y_train_shuffled: pd.Series,
    best_params: dict[str, Any],
    random_state: int = DEFAULT_RANDOM_STATE,
) -> lgb.LGBMRegressor:
    """Train LightGBM model with shuffled target.

    Args:
        X_train: Training features.
        y_train_shuffled: Shuffled training target.
        best_params: Best hyperparameters from optimization.
        random_state: Random seed for reproducibility.

    Returns:
        Trained LightGBM model.
    """
    logger.info("Training LightGBM with shuffled target...")

    # Reset index for alignment
    X_train_reset = X_train.reset_index(drop=True)
    y_train_reset = y_train_shuffled.reset_index(drop=True)

    model = lgb.LGBMRegressor(
        **best_params,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(X_train_reset, y_train_reset)
    logger.info("Training completed")

    return model


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics.

    Args:
        y_true: True target values.
        y_pred: Predicted values.

    Returns:
        Dictionary with MAE, MSE, RMSE, R² metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def _load_original_metrics() -> dict[str, float]:
    """Load original model metrics for comparison.

    Returns:
        Dictionary with original test metrics.

    Raises:
        FileNotFoundError: If evaluation results file doesn't exist.
        KeyError: If lightgbm_complete not found in results.
    """
    model_results = _load_json_results(
        LIGHTGBM_EVAL_RESULTS_FILE, LIGHTGBM_COMPLETE_MODEL_NAME, "evaluation results"
    )
    return model_results["test_metrics"]


def _log_test_header() -> None:
    """Log the test header."""
    logger.info("=" * 70)
    logger.info("DATA LEAKAGE TEST: Target Shuffle")
    logger.info("=" * 70)


def _log_metrics_comparison(
    shuffled_metrics: dict[str, float],
    original_metrics: dict[str, float],
) -> None:
    """Log shuffled and original model metrics for comparison."""
    logger.info("Shuffled Model Metrics:")
    logger.info(f"  MAE:  {shuffled_metrics['mae']:.6f}")
    logger.info(f"  MSE:  {shuffled_metrics['mse']:.6f}")
    logger.info(f"  RMSE: {shuffled_metrics['rmse']:.6f}")
    logger.info(f"  R²:   {shuffled_metrics['r2']:.6f}")

    logger.info("\nOriginal Model Metrics (for comparison):")
    logger.info(f"  MAE:  {original_metrics['mae']:.6f}")
    logger.info(f"  MSE:  {original_metrics['mse']:.6f}")
    logger.info(f"  RMSE: {original_metrics['rmse']:.6f}")
    logger.info(f"  R²:   {original_metrics['r2']:.6f}")


def _compute_delta_metrics(
    shuffled_metrics: dict[str, float],
    original_metrics: dict[str, float],
) -> dict[str, float]:
    """Compute difference between shuffled and original metrics."""
    return {
        "delta_mae": shuffled_metrics["mae"] - original_metrics["mae"],
        "delta_mse": shuffled_metrics["mse"] - original_metrics["mse"],
        "delta_rmse": shuffled_metrics["rmse"] - original_metrics["rmse"],
        "delta_r2": shuffled_metrics["r2"] - original_metrics["r2"],
    }


def _detect_leakage(r2_score: float) -> bool:
    """Detect data leakage based on R² threshold."""
    return r2_score > LEAKAGE_R2_THRESHOLD


def _log_leakage_result(
    leakage_detected: bool,
    r2_score: float,
) -> str:
    """Log leakage detection result and return interpretation."""
    logger.info("\n" + "=" * 70)
    logger.info("LEAKAGE DETECTION RESULT")
    logger.info("=" * 70)

    if leakage_detected:
        logger.warning(
            f"⚠️  POTENTIAL DATA LEAKAGE DETECTED! R² = {r2_score:.4f} > " f"{LEAKAGE_R2_THRESHOLD}"
        )
        logger.warning("The model achieves significant predictive power even with shuffled target.")
        logger.warning("This suggests that target information may be leaking into features.")
        interpretation = f"LEAKAGE DETECTED: R²={r2_score:.4f} > {LEAKAGE_R2_THRESHOLD}"
    else:
        logger.info(f"✓ No leakage detected. R² = {r2_score:.4f} ≤ {LEAKAGE_R2_THRESHOLD}")
        logger.info("The model performs poorly with shuffled target, as expected.")
        logger.info("Features do not contain leaked target information.")
        interpretation = f"NO LEAKAGE: R²={r2_score:.4f} ≤ {LEAKAGE_R2_THRESHOLD}"

    return interpretation


def run_target_shuffle_test(
    dataset_path: Path = LIGHTGBM_DATASET_COMPLETE_FILE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Run target shuffle test to detect data leakage.

    This is the most powerful leakage test:
    - Keep the same features X
    - Shuffle the target y_train randomly
    - Re-train the model with same hyperparameters
    - Evaluate on original y_test

    Expected result if NO leakage:
    - R² ≈ 0 or negative
    - MAE/MSE/RMSE much worse than original model

    Red flag if leakage:
    - R² >> 0.1 with shuffled target
    - Metrics close to original model

    Args:
        dataset_path: Path to dataset file (default: complete dataset).
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with test results including:
        - shuffled_metrics: Metrics with shuffled target
        - original_metrics: Original model metrics
        - delta_metrics: Difference between original and shuffled
        - leakage_detected: Boolean flag
        - interpretation: Text interpretation
    """
    _log_test_header()

    # Load hyperparameters and dataset
    best_params = _load_optimization_params(LIGHTGBM_DATASET_COMPLETE_MODEL_NAME)
    X_train, y_train, X_test, y_test = _load_dataset(dataset_path)

    # Shuffle target and train model
    y_train_shuffled = _shuffle_target(y_train, random_state=random_state)
    model_shuffled = _train_model_with_shuffled_target(
        X_train,
        y_train_shuffled,
        best_params,
        random_state=random_state,
    )

    # Predict and compute metrics
    y_pred_shuffled = np.asarray(model_shuffled.predict(X_test))
    shuffled_metrics = _compute_metrics(y_test, y_pred_shuffled)
    original_metrics = _load_original_metrics()

    # Log comparison
    _log_metrics_comparison(shuffled_metrics, original_metrics)

    # Compute results
    delta_metrics = _compute_delta_metrics(shuffled_metrics, original_metrics)
    leakage_detected = _detect_leakage(shuffled_metrics["r2"])
    interpretation = _log_leakage_result(leakage_detected, shuffled_metrics["r2"])

    return {
        "test_name": "target_shuffle",
        "shuffled_metrics": shuffled_metrics,
        "original_metrics": original_metrics,
        "delta_metrics": delta_metrics,
        "leakage_detected": leakage_detected,
        "r2_threshold": LEAKAGE_R2_THRESHOLD,
        "interpretation": interpretation,
        "random_state": random_state,
        "test_size": len(X_test),
        "train_size": len(X_train),
        "n_features": len(X_train.columns),
    }
