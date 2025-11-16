"""Results processing utilities for SARIMA optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, cast

import pandas as pd

from src.constants import SARIMA_LJUNGBOX_P_VALUE_THRESHOLD
from src.path import SARIMA_BEST_MODELS_FILE, SARIMA_OPTIMIZATION_RESULTS_FILE
from src.utils import ensure_output_dir, get_logger, save_json_pretty

logger = get_logger(__name__)


def to_dataframe(results: Iterable[dict]) -> pd.DataFrame:
    """Convert evaluation results to a structured DataFrame.

    Flattens nested 'params' dictionary into columns prefixed with 'param_'
    and orders columns with metrics first, then parameters, then other fields.

    Args:
        results: Iterable of result dictionaries from model evaluation.

    Returns:
        DataFrame with ordered columns: metrics, parameters, other fields.
    """
    df = pd.DataFrame(results).copy()

    if df.empty:
        return df

    params_column = "params"
    param_cols: list[str] = []

    if params_column in df.columns:
        normalized = pd.json_normalize(
            df[params_column].apply(lambda x: x if isinstance(x, dict) else {}).tolist(),
        ).add_prefix("param_")
        param_cols = list(normalized.columns)
        df = df.drop(columns=[params_column]).join(normalized)

    front = ["aic", "bic", "lb_stat", "lb_pvalue", "error"]
    front_cols = [c for c in front if c in df.columns]
    other_cols = [c for c in df.columns if c not in front_cols + param_cols]

    ordered_cols = front_cols + param_cols + other_cols
    return df.loc[:, ordered_cols]


def _filter_successful_models(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out models with errors.

    Args:
        df: DataFrame with evaluation results.

    Returns:
        DataFrame with only successful models.

    Raises:
        RuntimeError: If no successful models found.
    """
    df_ok = df.copy()
    if "error" in df_ok.columns:
        mask = df_ok["error"].isna()
        df_ok = df_ok.loc[mask].copy()
    if df_ok.empty:
        raise RuntimeError("No successful models.")
    return df_ok


def _filter_by_ljungbox(df: pd.DataFrame) -> pd.DataFrame:
    """Filter models by Ljung-Box test.

    Args:
        df: DataFrame with evaluation results.

    Returns:
        Filtered DataFrame (or original if no models pass).
    """
    if "lb_pvalue" not in df.columns:
        return df

    n_before = len(df)
    lb_passing = df["lb_pvalue"] >= SARIMA_LJUNGBOX_P_VALUE_THRESHOLD
    n_passing = lb_passing.sum()

    if n_passing > 0:
        df_filtered = df.loc[lb_passing].copy()
        logger.info(
            f"Filtered by Ljung-Box test (p >= {SARIMA_LJUNGBOX_P_VALUE_THRESHOLD}): "
            f"{n_passing}/{n_before} models passed"
        )
        return df_filtered

    logger.warning(
        f"No models passed Ljung-Box test (p >= {SARIMA_LJUNGBOX_P_VALUE_THRESHOLD}). "
        "Using all models."
    )
    return df


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    """Convert DataFrame row to model dictionary.

    Args:
        row: DataFrame row with model results.

    Returns:
        Dictionary with params and metrics.
    """
    params = {
        str(k).replace("param_", ""): (str(row[k]) if k == "param_trend" else int(row[k]))
        for k in row.index
        if isinstance(k, str) and k.startswith("param_")
    }
    result: dict[str, Any] = {
        "params": params,
        "aic": float(row["aic"]),
        "bic": float(row["bic"]),
    }
    # Include optional metrics
    for metric in ["val_rmse", "val_mae", "lb_pvalue"]:
        if metric in row.index:
            result[metric] = float(row[metric])
    return result


def pick_best(df: object) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select best models based on composite_score (z-score) and BIC with Ljung-Box filtering.

    The best model is selected using the composite_score (normalized z-score)
    that combines AIC and validation RMSE during optimization.

    Args:
        df: DataFrame with evaluation results.

    Returns:
        Tuple of (best_composite_score_dict, best_bic_dict).

    Raises:
        RuntimeError: If no successful models found.
        TypeError: If df is not a DataFrame.
        ValueError: If composite_score column is missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    df_typed: pd.DataFrame = cast(pd.DataFrame, df)

    # Check for successful models first (this raises RuntimeError if empty)
    df_ok = _filter_successful_models(df_typed)

    # Only check for required columns if we have successful models
    if "composite_score" not in df_ok.columns:
        raise ValueError("DataFrame must contain 'composite_score' column for z-score selection.")

    if "bic" not in df_ok.columns:
        logger.warning("BIC column not found")
    df_ok = _filter_by_ljungbox(df_ok)

    best_aic_idx = df_ok["composite_score"].astype(float).idxmin()
    # For BIC, still use BIC if available, otherwise use composite_score
    bic_column = "bic" if "bic" in df_ok.columns else "composite_score"
    best_bic_idx = df_ok[bic_column].astype(float).idxmin()

    best_aic_row = cast(pd.Series, df_ok.loc[best_aic_idx])
    best_bic_row = cast(pd.Series, df_ok.loc[best_bic_idx])

    # Log validation metrics and selection criteria if available
    logger.info("Best model selected using composite_score (z-score) minimization")
    if "val_rmse" in df_ok.columns:
        logger.info(f"Best AIC model validation RMSE: {best_aic_row.get('val_rmse', 'N/A')}")
        logger.info(f"Best BIC model validation RMSE: {best_bic_row.get('val_rmse', 'N/A')}")

    return _row_to_dict(best_aic_row), _row_to_dict(best_bic_row)


def determine_sort_columns(df: pd.DataFrame, criterion: str) -> list[str]:
    """Determine sort columns for DataFrame based on available metrics.

    Returns a list of column names to use for sorting, prioritizing:
    1. lb_reject_5pct (if available) - prefer models that don't reject Ljung-Box test
    2. The specified criterion (aic or bic)

    Args:
        df: DataFrame with model evaluation results.
        criterion: Criterion to use for sorting ('aic' or 'bic').

    Returns:
        List of column names for sorting.
    """
    sort_cols: list[str] = []

    # Prioritize models that don't reject Ljung-Box test
    if "lb_reject_5pct" in df.columns:
        sort_cols.append("lb_reject_5pct")

    # Always include the criterion
    sort_cols.append(criterion)

    return sort_cols


def build_best_model_dict(row: pd.Series) -> dict[str, Any]:
    """Build a model dictionary from a pandas Series row.

    Extracts SARIMA parameters and metrics from a DataFrame row and constructs
    a dictionary with all parameters plus a formatted params string.

    Args:
        row: Pandas Series row containing SARIMA parameters and metrics.

    Returns:
        Dictionary with all parameters (p, d, q, P, D, Q, s, aic, bic) and params string.

    Raises:
        ValueError: If required SARIMA parameters are missing.
    """
    required_params = ["p", "d", "q", "P", "D", "Q", "s"]
    missing_params = [p for p in required_params if p not in row.index]
    if missing_params:
        msg = f"Missing required SARIMA parameters: {missing_params}"
        raise ValueError(msg)

    model_dict: dict[str, Any] = {
        "p": int(row["p"]),
        "d": int(row["d"]),
        "q": int(row["q"]),
        "P": int(row["P"]),
        "D": int(row["D"]),
        "Q": int(row["Q"]),
        "s": int(row["s"]),
    }

    # Add metrics if present
    if "aic" in row.index:
        model_dict["aic"] = float(row["aic"])
    if "bic" in row.index:
        model_dict["bic"] = float(row["bic"])

    # Add params string
    model_dict["params"] = (
        f"SARIMA({model_dict['p']},{model_dict['d']},{model_dict['q']})"
        f"({model_dict['P']},{model_dict['D']},{model_dict['Q']})[{model_dict['s']}]"
    )

    # Add any other columns from the row
    for key in row.index:
        if key not in model_dict:
            model_dict[str(key)] = row[key]

    return model_dict


def _resolve_path(file_path: Path, out_dir: Path) -> Path:
    """Resolve file path relative to output directory.

    Args:
        file_path: File path (absolute or relative).
        out_dir: Output directory.

    Returns:
        Resolved absolute path.
    """
    if file_path.is_absolute():
        return file_path
    return out_dir / file_path


def save_results(
    df: pd.DataFrame,
    best_aic: dict,
    best_bic: Optional[dict],
    out_dir: Path,
    *,
    best_models_file: Path = SARIMA_BEST_MODELS_FILE,
    results_file: Path = SARIMA_OPTIMIZATION_RESULTS_FILE,
) -> None:
    """Save optimization results to CSV and JSON files.

    Args:
        df: DataFrame with all evaluation results.
        best_aic: Dictionary with best AIC model parameters and metrics.
        best_bic: Dictionary with best BIC model parameters and metrics, or None.
        out_dir: Output directory path (created if it doesn't exist).
        best_models_file: Target path for best models JSON file.
        results_file: Target path for optimization results CSV file.

    Raises:
        ValueError: If df is empty.
    """
    if df.empty:
        msg = "Optimization results DataFrame is empty"
        raise ValueError(msg)

    ensure_output_dir(out_dir)

    results_path = _resolve_path(results_file, out_dir)
    ensure_output_dir(results_path.parent)
    df.to_csv(results_path, index=False)
    logger.info(f"Saved optimization table to CSV: {results_path}")

    best_models_path = _resolve_path(best_models_file, out_dir)
    best_models_data = {"best_aic": best_aic, "best_bic": best_bic}
    save_json_pretty(best_models_data, best_models_path)
    logger.info(f"Saved best-models JSON: {best_models_path}")
