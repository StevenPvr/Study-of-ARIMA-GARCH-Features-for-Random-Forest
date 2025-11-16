"""CLI entry point for SARIMA evaluation.

Keeps `src.*` imports intact as requested.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

# Add project root to Python path for direct execution.
# This must be done before importing src modules.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


from src.arima.evaluation_arima.evaluation_arima import (  # type: ignore
    backtest_full_series,
    evaluate_model,
    ljung_box_on_residuals,
    plot_residuals_acf_with_ljungbox,
    run_all_normality_tests,
    save_evaluation_results,
    save_ljung_box_results,
)
from src.arima.evaluation_arima.save_data_for_garch import (  # type: ignore
    regenerate_garch_dataset_from_rolling_predictions,
    save_garch_dataset,
)
from src.arima.evaluation_arima.utils import detect_value_column  # type: ignore
from src.constants import SARIMA_LJUNGBOX_LAGS_DEFAULT, SARIMA_REFIT_EVERY_DEFAULT  # type: ignore
from src.path import ARIMA_RESULTS_DIR, WEIGHTED_LOG_RETURNS_SPLIT_FILE  # type: ignore
from src.utils import (  # type: ignore
    ensure_output_dir,
    get_logger,
    load_csv_file,
    save_json_pretty,
)

logger = get_logger(__name__)


def _load_split_df() -> pd.DataFrame:
    path = Path(WEIGHTED_LOG_RETURNS_SPLIT_FILE)
    df = load_csv_file(path)
    if "date" not in df.columns:
        raise ValueError("Split file must contain a 'date' column.")
    return df


def _split_train_test(df: pd.DataFrame, value_col: str) -> tuple[pd.Series, pd.Series]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if "split" in df.columns:
        if set(df["split"].unique()) >= {"train", "test"}:
            train_df = df[df["split"] == "train"].set_index("date")
            test_df = df[df["split"] == "test"].set_index("date")
            train: pd.Series = train_df[value_col]  # type: ignore[assignment]
            test: pd.Series = test_df[value_col]  # type: ignore[assignment]
            return train, test

    if "is_test" in df.columns:
        mask = df["is_test"].astype(bool)
        train_df = df[~mask].set_index("date")
        test_df = df[mask].set_index("date")
        train_series: pd.Series = train_df[value_col]  # type: ignore[assignment]
        test_series: pd.Series = test_df[value_col]  # type: ignore[assignment]
        return train_series, test_series

    # No implicit fallback allowed: explicit split markers required
    raise ValueError(
        "No explicit split markers found in dataframe. "
        "Expected 'split' column with values {'train','test'} or a boolean 'is_test' column."
    )


def _load_trained_model_and_orders() -> tuple[Any, tuple[int, int, int], tuple[int, int, int, int]]:
    """Load a trained ARIMA model and extract (order, seasonal_order).

    Raises explicit errors instead of silently falling back to defaults.
    """
    try:
        from src.arima.training_arima.training_arima import load_trained_model  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Failed to import ARIMA training loader 'load_trained_model'.") from exc

    fitted_model, model_info = load_trained_model()

    order: tuple[int, int, int] | None = None
    seasonal_order: tuple[int, int, int, int] | None = None

    # Prefer explicit metadata
    if isinstance(model_info, dict):
        maybe_order = model_info.get("order")
        maybe_seasonal = model_info.get("seasonal_order")
        if maybe_order is not None:
            order = tuple(maybe_order)  # type: ignore[arg-type]
        if maybe_seasonal is not None:
            seasonal_order = tuple(maybe_seasonal)  # type: ignore[arg-type]

    # Fallback extraction from fitted model attributes is not allowed; enforce explicit presence
    if order is None or seasonal_order is None:
        raise ValueError(
            "Trained model metadata must include 'order' and 'seasonal_order'. "
            "Extraction from fitted model attributes is not allowed."
        )

    return fitted_model, order, seasonal_order


def _run_residual_diagnostics(residuals: Sequence[float], lags: int) -> None:
    """Generate residual diagnostics (ACF plot + Ljung–Box results)."""
    try:
        plot_residuals_acf_with_ljungbox(residuals, lags=lags)
        lb_result = ljung_box_on_residuals(residuals, lags=lags)
        save_ljung_box_results(lb_result)
    except Exception as exc:  # pragma: no cover
        logger.warning("Diagnostics could not be generated: %s", exc)


def _save_normality_tests(residuals: Sequence[float]) -> None:
    """Run and persist residual normality tests."""
    logger.info("=" * 60)
    logger.info("RUNNING NORMALITY TESTS ON RESIDUALS")
    logger.info("=" * 60)
    try:
        results = run_all_normality_tests(residuals)
        logger.info(
            "Jarque-Bera: statistic=%.4f, p-value=%.4f",
            results["jarque_bera"]["statistic"],
            results["jarque_bera"]["p_value"],
        )
        logger.info(
            "Shapiro-Wilk: statistic=%.4f, p-value=%.4f",
            results["shapiro_wilk"]["statistic"],
            results["shapiro_wilk"]["p_value"],
        )
        logger.info(
            "Anderson-Darling: statistic=%.4f",
            results["anderson_darling"]["statistic"],
        )
        output = Path(ARIMA_RESULTS_DIR) / "evaluation" / "normality_tests.json"
        ensure_output_dir(output.parent)
        save_json_pretty(results, output)
        logger.info("Saved normality test results → %s", output)
    except Exception as exc:  # pragma: no cover
        logger.warning("Normality tests could not be generated: %s", exc)


def _compute_backtest_metrics(history: pd.DataFrame) -> dict[str, Any]:
    """Compute evaluation metrics from backtest history.

    Args:
        history: DataFrame with y_true and y_pred columns.

    Returns:
        Dict with MSE, RMSE, MAE metrics.
    """
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mse = mean_squared_error(history["y_true"], history["y_pred"])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(history["y_true"], history["y_pred"])

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "n_observations": len(history),
    }


def _save_backtest_results(
    history: pd.DataFrame,
    metrics: dict[str, Any],
    refit_every: int,
) -> None:
    """Save backtest results and summary to disk.

    Args:
        history: DataFrame with backtest results.
        metrics: Dict with computed metrics.
        refit_every: Refit interval used.
    """
    output_dir = Path(ARIMA_RESULTS_DIR) / "evaluation"
    ensure_output_dir(output_dir)

    if not history.empty:
        backtest_file = output_dir / "full_series_backtest_residuals.csv"
        history.to_csv(backtest_file, index=False)
        logger.info("Saved full series backtest outputs → %s", backtest_file)

    summary = {**metrics, "refit_every": refit_every}
    save_json_pretty(summary, output_dir / "full_series_backtest_summary.json")


def _run_full_series_backtest(
    train: pd.Series,
    test: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> pd.DataFrame | None:
    """Execute backtest on full series (train+test) and persist outputs.

    This backtest generates residuals for all dates (train+test) needed by GARCH,
    using a rolling forecast with periodic refits every 20 days.
    """
    logger.info("=" * 60)
    logger.info("RUNNING FULL SERIES BACKTEST (train+test)")
    try:
        history = backtest_full_series(
            train_series=train,
            test_series=test,
            order=order,
            seasonal_order=seasonal_order,
            refit_every=SARIMA_REFIT_EVERY_DEFAULT,
            verbose=True,
        )

        metrics = _compute_backtest_metrics(history)

        logger.info(
            "Backtest metrics: MSE=%.6f | RMSE=%.6f | MAE=%.6f",
            metrics["mse"],
            metrics["rmse"],
            metrics["mae"],
        )

        _save_backtest_results(history, metrics, SARIMA_REFIT_EVERY_DEFAULT)
        return history

    except Exception as exc:  # pragma: no cover
        logger.warning("Full series backtest could not be completed: %s", exc)
        logger.exception(exc)
        return None


def _build_garch_outputs(
    results: dict[str, Any],
    fitted_model: Any | None,
    backtest_residuals: pd.DataFrame | None,
) -> None:
    """Persist datasets required by the GARCH pipeline."""
    logger.info("=" * 60)
    logger.info("BUILDING GARCH DATASET")
    logger.info("=" * 60)
    try:
        save_garch_dataset(
            results,
            fitted_model=fitted_model,
            backtest_residuals=backtest_residuals,
        )
        return
    except Exception as exc:  # pragma: no cover
        logger.warning("GARCH dataset generation failed: %s", exc)
    try:
        logger.info("Attempting to regenerate GARCH dataset from rolling_predictions.csv…")
        regenerate_garch_dataset_from_rolling_predictions()
    except Exception as exc:  # pragma: no cover
        logger.warning("GARCH dataset regeneration also failed: %s", exc)


def main() -> None:
    logger.info("=" * 60)
    logger.info("SARIMA evaluation starting…")

    df = _load_split_df()
    value_col = detect_value_column(df)
    train, test = _split_train_test(df, value_col=value_col)

    fitted_model, order, seasonal_order = _load_trained_model_and_orders()

    results = evaluate_model(
        train_series=train,
        test_series=test,
        order=order,
        seasonal_order=seasonal_order,
        refit_every=SARIMA_REFIT_EVERY_DEFAULT,
        verbose=True,
    )

    save_evaluation_results(results)

    residuals = results.get("residuals")
    if residuals is None:
        raise KeyError("Evaluation results must include 'residuals'.")

    _run_residual_diagnostics(residuals, int(SARIMA_LJUNGBOX_LAGS_DEFAULT))
    _save_normality_tests(residuals)
    backtest_history = _run_full_series_backtest(train, test, order, seasonal_order)
    _build_garch_outputs(results, fitted_model, backtest_history)

    logger.info("=" * 60)
    logger.info("SARIMA evaluation complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
