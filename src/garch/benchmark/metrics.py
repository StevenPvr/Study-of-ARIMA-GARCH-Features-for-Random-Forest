"""Metrics computation for volatility forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.garch.garch_eval.metrics import mse_mae_variance, qlike_loss


def r2_variance(e: np.ndarray, s2: np.ndarray) -> float:
    """Compute R² between realized e^2 and forecast sigma^2.

    Args:
        e: Realized residuals.
        s2: Forecast variance.

    Returns:
        R-squared coefficient (higher is better, max 1.0).
    """
    e = np.asarray(e, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    m = np.isfinite(e) & np.isfinite(s2)
    if not np.any(m) or m.sum() < 2:
        return float("nan")
    y = e[m] ** 2
    f = s2[m]
    ss_res = float(np.sum((y - f) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def compute_model_metrics(e_test: np.ndarray, s2: np.ndarray) -> dict[str, float]:
    """Compute volatility forecast metrics for a single model.

    Args:
        e_test: Test residuals.
        s2: Forecast variance.

    Returns:
        Dictionary with qlike, mse, mae, rmse, and r2.
    """
    mse_mae = mse_mae_variance(e_test, s2)
    mse_val = mse_mae.get("mse", float("nan"))
    mae_val = mse_mae.get("mae", float("nan"))
    rmse_val = float(np.sqrt(mse_val)) if np.isfinite(mse_val) else float("nan")
    return {
        "qlike": qlike_loss(e_test, s2),
        "mse": mse_val,
        "mae": mae_val,
        "rmse": rmse_val,
        "r2": r2_variance(e_test, s2),
    }


def rank_models(metrics: dict[str, dict[str, float]]) -> list[tuple[str, float, str]]:
    """Rank models by QLIKE score.

    Args:
        metrics: Dictionary with model metrics.

    Returns:
        List of (model_name, qlike_score, rank) tuples, sorted by QLIKE.
    """
    model_scores = []
    for model_name, model_metrics in metrics.items():
        if model_name == "n_test":
            continue
        qlike_score = model_metrics.get("qlike", float("inf"))
        if not np.isnan(qlike_score):
            model_scores.append((model_name, qlike_score))

    model_scores.sort(key=lambda x: x[1])  # Sort by QLIKE (lower is better)
    return [(name, score, f"#{i+1}") for i, (name, score) in enumerate(model_scores)]


def _build_ranking_dict(
    rankings: list[tuple[str, float, str]],
) -> dict[str, dict[str, str | float]]:
    """Build ranking dictionary from rankings list.

    Args:
        rankings: List of (model_name, qlike_score, rank) tuples.

    Returns:
        Ranking dictionary with rank and qlike for each model.
    """
    return {name: {"rank": rank, "qlike": qlike_score} for name, qlike_score, rank in rankings}


def compute_metrics(
    e_test: np.ndarray,
    s2_garch: np.ndarray,
    s2_ewma: np.ndarray,
    s2_roll_var: np.ndarray,
    s2_roll_std: np.ndarray,
    s2_arch1: np.ndarray,
    s2_har3: np.ndarray,
) -> dict[str, Any]:
    """Compute volatility forecast metrics for all models.

    Args:
        e_test: Test residuals.
        s2_garch: GARCH variance forecasts.
        s2_ewma: EWMA variance forecasts.
        s2_roll_var: Rolling variance forecasts.
        s2_roll_std: Rolling std forecasts.
        s2_arch1: ARCH(1) variance forecasts.
        s2_har3: HAR(3) variance forecasts.

    Returns:
        Metrics dictionary with QLIKE, MSE, MAE, RMSE, R², and rankings.
    """
    models_metrics = {
        "arima_garch": compute_model_metrics(e_test, s2_garch),
        "ewma": compute_model_metrics(e_test, s2_ewma),
        "roll_var": compute_model_metrics(e_test, s2_roll_var),
        "roll_std": compute_model_metrics(e_test, s2_roll_std),
        "arch1": compute_model_metrics(e_test, s2_arch1),
        "har3": compute_model_metrics(e_test, s2_har3),
    }
    rankings = rank_models(models_metrics)
    ranking_dict = _build_ranking_dict(rankings)
    return {
        "n_test": int(e_test.size),
        **models_metrics,
        "rankings": ranking_dict,
    }
