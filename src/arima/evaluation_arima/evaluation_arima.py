"""SARIMA model evaluation module (publication-grade, DRY/KISS).

This module intentionally keeps `src.*` imports intact. It does not
introduce new package-level dependencies beyond statsmodels/sklearn/matplotlib.

Exposed functions:
- rolling_forecast
- evaluate_model
- calculate_metrics
- ljung_box_on_residuals
- plot_residuals_acf_with_ljungbox
- save_evaluation_results
- save_ljung_box_results
- walk_forward_backtest

Note: compute_residuals is imported from src.utils for consistency
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import pandas as pd

# Third-party (optional imports are guarded at runtime)
try:
    import matplotlib

    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception as _e:  # pragma: no cover
    SARIMAX = None  # type: ignore
    plot_acf = None  # type: ignore
    acorr_ljungbox = None  # type: ignore
    plt = None  # type: ignore

from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.constants import (  # type: ignore
    DEFAULT_PLACEHOLDER_DATE,
    LJUNGBOX_SIGNIFICANCE_LEVEL,
    PLOT_DPI_EVALUATION,
    PLOT_FIGURE_SIZE_DEFAULT,
    PLOT_TEXT_POSITION_X,
    PLOT_TEXT_POSITION_Y,
    ROLLING_FORECAST_PROGRESS_INTERVAL,
    SARIMA_HISTORY_COLUMNS,
    SARIMA_LJUNGBOX_LAGS_DEFAULT,
    SARIMA_MIN_TRAIN_SIZE_FOR_RESIDUALS,
    SARIMA_REFIT_EVERY_DEFAULT,
)
from src.path import (  # type: ignore
    LJUNGBOX_RESIDUALS_SARIMA_FILE,
    ROLLING_PREDICTIONS_SARIMA_FILE,
    ROLLING_VALIDATION_METRICS_SARIMA_FILE,
    SARIMA_RESIDUALS_LJUNGBOX_PLOT,
)

# Keep project imports as-is (do NOT remove per user's instruction)
from src.utils import (  # type: ignore
    compute_residuals,
    format_dates_to_string,
    get_logger,
    save_json_pretty,
    suppress_statsmodels_warnings,
    validate_temporal_order_series,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class SarimaSpec:
    """Container for SARIMA hyper-parameters."""

    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)
    trend: str = "c"  # Default trend parameter


# ----------------------------- Utilities ----------------------------- #


def _append_point_to_train_series(
    y_train: pd.Series,
    ts_idx: Any,
    y_true: float,
) -> pd.Series:
    """Append a new point to training series while preserving frequency."""
    train_freq = y_train.index.freq if isinstance(y_train.index, pd.DatetimeIndex) else None
    if train_freq is not None:
        new_idx = pd.date_range(start=ts_idx, periods=1, freq=train_freq)
        new_point = pd.Series([y_true], index=new_idx, name=y_train.name)
    else:
        new_point = pd.Series([y_true], index=[ts_idx], name=y_train.name)
    y_train = pd.concat([y_train, new_point])
    # Ensure frequency is preserved after concat by recreating index if needed
    if isinstance(y_train.index, pd.DatetimeIndex) and y_train.index.freq is None:
        dt_idx = cast(pd.DatetimeIndex, y_train.index)
        inferred = pd.infer_freq(dt_idx) or train_freq or "D"
        y_train.index = pd.date_range(start=dt_idx[0], periods=len(dt_idx), freq=inferred)
    return y_train


def _as_series(x: Iterable[float] | pd.Series, name: str = "y") -> pd.Series:
    """Coerce to pandas Series."""
    if isinstance(x, pd.Series):
        s = x.copy()
        if s.name is None:
            s.name = name
        return s
    s = pd.Series(list(x), name=name)
    return s


def _ensure_datetime_index(s: pd.Series) -> pd.Series:
    """Ensure the index is datetime-like with frequency; if not, fabricate sequential dates.

    - If index is a DatetimeIndex, ensure it has a frequency (infer if missing).
    - Else, create daily dates with frequency "D".
    """
    if isinstance(s.index, pd.DatetimeIndex):
        s2 = s.copy()
        dt_index = cast(pd.DatetimeIndex, s2.index)
        # Ensure frequency is set to avoid statsmodels warnings
        if dt_index.freq is None:
            # Try to infer frequency from the index
            try:
                inferred_freq = pd.infer_freq(dt_index)
                if inferred_freq:
                    # Recreate index with inferred frequency
                    s2.index = pd.date_range(
                        start=dt_index[0], periods=len(dt_index), freq=inferred_freq
                    )
                else:
                    # Default to daily frequency if inference fails
                    s2.index = pd.date_range(start=dt_index[0], periods=len(dt_index), freq="D")
            except Exception as e:
                logger.warning(f"Failed to infer frequency, using daily: {e}")
                s2.index = pd.date_range(start=dt_index[0], periods=len(dt_index), freq="D")
        return s2
    # fabricate deterministic daily index for plotting and merges
    start = pd.Timestamp(DEFAULT_PLACEHOLDER_DATE)
    idx = pd.date_range(start=start, periods=len(s), freq="D")
    s2 = s.copy()
    s2.index = idx
    return s2


def _fit_sarima(y: pd.Series, spec: SarimaSpec) -> Any:
    """Fit SARIMA and return fitted model."""
    if SARIMAX is None:  # pragma: no cover
        raise RuntimeError("statsmodels is required for SARIMAX.")
    model = SARIMAX(
        y,
        order=spec.order,
        seasonal_order=spec.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        trend=spec.trend,
    )
    return model.fit(disp=False)


def _make_one_step_forecast(fitted_model: Any, y_train: pd.Series, step: int) -> float:
    """Make a one-step forecast, with fallback to last observed value."""
    try:
        forecast_result = fitted_model.forecast(steps=1)  # type: ignore
        return float(
            forecast_result.iloc[0] if hasattr(forecast_result, "iloc") else forecast_result[0]
        )
    except Exception as e:
        logger.warning(f"Forecast failed at step {step+1}, using last observed value: {e}")
        return float(y_train.iloc[-1])


# ----------------------------- Core API ----------------------------- #


def _setup_rolling_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    trend: str = "c",
) -> tuple[pd.Series, pd.Series, SarimaSpec]:
    """Setup rolling forecast: filter warnings, prepare series, validate order."""
    suppress_statsmodels_warnings()

    y_train = _ensure_datetime_index(_as_series(train_series, "y_train"))
    y_test = _ensure_datetime_index(_as_series(test_series, "y_test"))
    validate_temporal_order_series(y_train, y_test, function_name="rolling_forecast")

    spec = SarimaSpec(order=order, seasonal_order=seasonal_order, trend=trend)
    return y_train, y_test, spec


def _execute_rolling_forecast_loop(
    y_train: pd.Series,
    y_test: pd.Series,
    spec: SarimaSpec,
    refit_every: int,
    verbose: bool,
) -> tuple[list[float], list[float]]:
    """Execute the main rolling forecast loop."""
    preds: list[float] = []
    actuals: list[float] = []
    fitted_model = None

    for i, (ts_idx, y_true) in enumerate(zip(y_test.index, y_test.values, strict=False)):
        if i == 0 or (refit_every > 0 and i % refit_every == 0) or fitted_model is None:
            fitted_model = _fit_sarima(y_train, spec)
            if verbose:
                logger.info(f"Refit at step {i+1}/{len(y_test)} (train n={len(y_train)})")
        pred = _make_one_step_forecast(fitted_model, y_train, i)
        preds.append(pred)
        actuals.append(float(y_true))
        y_train = _append_point_to_train_series(y_train, ts_idx, y_true)
        if verbose and (i + 1) % ROLLING_FORECAST_PROGRESS_INTERVAL == 0:
            logger.info(f"Rolling forecast progress: {i+1}/{len(y_test)}")
    return preds, actuals


def rolling_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    refit_every: int = SARIMA_REFIT_EVERY_DEFAULT,
    verbose: bool = False,
    trend: str = "c",
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling one-step-ahead forecast with periodic refits.

    For t in test horizon:
      - If t % refit_every == 0: refit SARIMA on all available data up to t-1
      - Else: reuse previous fitted model and update via append (fast)

    Args:
        train_series: Training data series.
        test_series: Test data series.
        order: ARIMA order (p, d, q).
        seasonal_order: Seasonal ARIMA order (P, D, Q, s).
        refit_every: Number of steps between model refits (default: 20).
        verbose: Whether to log progress (default: False).
        trend: Trend component ("n", "c", "t", "ct"). Default: "c".

    Returns
    -------
    preds : np.ndarray
        Forecasts aligned with test_series.
    actuals : np.ndarray
        Actual values from test_series.
    """
    y_train, y_test, spec = _setup_rolling_forecast(
        train_series, test_series, order, seasonal_order, trend
    )
    preds, actuals = _execute_rolling_forecast_loop(y_train, y_test, spec, refit_every, verbose)
    return np.array(preds, dtype=float), np.array(actuals, dtype=float)


def _create_non_nan_mask(y_true_s: pd.Series, y_pred_s: pd.Series) -> pd.Series:
    """Create a boolean mask for non-NaN values in both series."""
    return y_true_s.notna() & y_pred_s.notna()  # type: ignore[return-value]


def calculate_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Compute standard regression metrics (MSE, RMSE, MAE)."""
    y_true_s = _as_series(y_true, "y_true")
    y_pred_s = _as_series(y_pred, "y_pred")

    # Filter out NaN values
    mask = _create_non_nan_mask(y_true_s, y_pred_s)
    y_true_clean = y_true_s[mask]
    y_pred_clean = y_pred_s[mask]

    # Check if all predictions are NaN
    if len(y_pred_clean) == 0:
        raise RuntimeError("All predictions are NaN")

    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    return {"MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae)}


# Note: compute_residuals is now imported from src.utils for consistency


def ljung_box_on_residuals(residuals: Iterable[float], lags: int) -> dict[str, Any]:
    """Run Ljung–Box test on residuals and return dict with statistics and p-values."""
    if acorr_ljungbox is None:  # pragma: no cover
        raise RuntimeError("statsmodels is required for Ljung–Box.")
    res = np.asarray(list(residuals), dtype=float)
    res = res[np.isfinite(res)]
    lags_list = list(range(1, lags + 1))
    lb = acorr_ljungbox(res, lags=lags_list, return_df=True)
    # Extract statistics for all lags
    q_stat = lb["lb_stat"].values.tolist()
    p_value = lb["lb_pvalue"].values.tolist()
    reject_5pct = (lb["lb_pvalue"] < LJUNGBOX_SIGNIFICANCE_LEVEL).values.tolist()
    out = {
        "lags": lags_list,
        "q_stat": q_stat,
        "p_value": p_value,
        "reject_5pct": reject_5pct,
        "n": int(res.size),
    }
    return out


def _compute_jarque_bera_statistics(res: np.ndarray) -> dict[str, float]:
    """Compute Jarque-Bera test statistics and moments.

    Args:
        res: Cleaned residuals array (finite values only).

    Returns:
        Dict with statistic, p_value, skewness, kurtosis, n.
    """
    jb_result = stats.jarque_bera(res)
    jb_stat_val = cast(float, jb_result[0])
    p_val_val = cast(float, jb_result[1])
    jb_stat = float(jb_stat_val)
    p_val = float(p_val_val)
    skew = float(stats.skew(res))
    kurt = float(stats.kurtosis(res, fisher=True))  # Excess kurtosis (0 for normal)

    return {
        "statistic": jb_stat,
        "p_value": p_val,
        "skewness": skew,
        "kurtosis": kurt,
        "n": int(res.size),
    }


def jarque_bera_test(residuals: Iterable[float]) -> dict[str, float]:
    """Jarque-Bera test for normality of residuals.

    Tests whether residuals have skewness and kurtosis matching a normal distribution.

    H0: Residuals are normally distributed
    H1: Residuals are not normally distributed

    Args:
        residuals: Residual series from ARIMA model.

    Returns:
        Dict with statistic, p_value, skewness, kurtosis, n.

    Note:
        - p-value > 0.05 suggests residuals are normally distributed
        - p-value < 0.05 suggests deviation from normality
    """
    res = np.asarray(list(residuals), dtype=float)
    res = res[np.isfinite(res)]

    if res.size < 3:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "skewness": float("nan"),
            "kurtosis": float("nan"),
            "n": int(res.size),
        }

    return _compute_jarque_bera_statistics(res)


def _compute_shapiro_wilk_statistics(res: np.ndarray) -> dict[str, float]:
    """Compute Shapiro-Wilk test statistics.

    Args:
        res: Cleaned residuals array (finite values only).

    Returns:
        Dict with statistic, p_value, n.
    """
    if res.size > 5000:
        logger.warning(
            "Shapiro-Wilk test: sample size %d exceeds 5000. Consider using Jarque-Bera instead.",
            res.size,
        )

    sw_stat, p_val = stats.shapiro(res)

    return {
        "statistic": float(sw_stat),
        "p_value": float(p_val),
        "n": int(res.size),
    }


def shapiro_wilk_test(residuals: Iterable[float]) -> dict[str, float]:
    """Shapiro-Wilk test for normality of residuals.

    More powerful than Jarque-Bera for small samples (n < 2000).
    Tests whether residuals come from a normal distribution.

    H0: Residuals are normally distributed
    H1: Residuals are not normally distributed

    Args:
        residuals: Residual series from ARIMA model.

    Returns:
        Dict with statistic, p_value, n.

    Note:
        - p-value > 0.05 suggests residuals are normally distributed
        - p-value < 0.05 suggests deviation from normality
        - Test is most powerful for n between 3 and 5000
    """
    res = np.asarray(list(residuals), dtype=float)
    res = res[np.isfinite(res)]

    if res.size < 3:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n": int(res.size),
        }

    return _compute_shapiro_wilk_statistics(res)


def _build_critical_values_dict(
    significance_levels: np.ndarray, critical_values: np.ndarray
) -> dict[str, float]:
    """Build critical values dictionary from Anderson-Darling result.

    Args:
        significance_levels: Array of significance levels.
        critical_values: Array of critical values.

    Returns:
        Dict mapping significance level strings to critical values.
    """
    crit_dict = {}
    for sig_level, crit_val in zip(significance_levels, critical_values, strict=False):
        crit_dict[f"{sig_level}%"] = float(crit_val)
    return crit_dict


def _compute_anderson_darling_statistics(res: np.ndarray) -> dict[str, Any]:
    """Compute Anderson-Darling test statistics.

    Args:
        res: Cleaned residuals array (finite values only).

    Returns:
        Dict with statistic, critical_values, significance_levels, n.
    """
    result = stats.anderson(res, dist="norm")

    significance_levels = getattr(result, "significance_level", np.array([]))
    critical_values = getattr(result, "critical_values", np.array([]))
    statistic = getattr(result, "statistic", float("nan"))

    crit_dict = _build_critical_values_dict(significance_levels, critical_values)

    return {
        "statistic": float(statistic),
        "critical_values": crit_dict,
        "significance_levels": (
            significance_levels.tolist() if hasattr(significance_levels, "tolist") else []
        ),
        "n": int(res.size),
    }


def anderson_darling_test(residuals: Iterable[float]) -> dict[str, Any]:
    """Anderson-Darling test for normality of residuals.

    Tests whether residuals come from a normal distribution.
    More sensitive to tails than Jarque-Bera or Shapiro-Wilk.

    H0: Residuals are normally distributed
    H1: Residuals are not normally distributed

    Args:
        residuals: Residual series from ARIMA model.

    Returns:
        Dict with statistic, critical_values, significance_levels, n.

    Note:
        - If statistic < critical_value at a given significance level,
          do not reject H0 at that level
        - If statistic > critical_value, reject H0
        - Critical values provided for [15%, 10%, 5%, 2.5%, 1%]
    """
    res = np.asarray(list(residuals), dtype=float)
    res = res[np.isfinite(res)]

    if res.size < 3:
        return {
            "statistic": float("nan"),
            "critical_values": {},
            "significance_levels": [],
            "n": int(res.size),
        }

    return _compute_anderson_darling_statistics(res)


def run_all_normality_tests(residuals: Iterable[float]) -> dict[str, dict[str, Any]]:
    """Run all three normality tests on residuals.

    Convenience function that runs:
    1. Jarque-Bera test (good for large samples, based on moments)
    2. Shapiro-Wilk test (powerful for small/medium samples)
    3. Anderson-Darling test (sensitive to tail behavior)

    Args:
        residuals: Residual series from ARIMA model.

    Returns:
        Dict with keys: jarque_bera, shapiro_wilk, anderson_darling.
        Each value is a dict with test statistics and results.

    Note:
        Consistent rejection across all tests provides strong evidence
        against normality assumption.
    """
    return {
        "jarque_bera": jarque_bera_test(residuals),
        "shapiro_wilk": shapiro_wilk_test(residuals),
        "anderson_darling": anderson_darling_test(residuals),
    }


def plot_residuals_acf_with_ljungbox(
    residuals: Iterable[float],
    lags: int | None = None,
    out_path: Path | None = None,
) -> Path:
    """Plot ACF of residuals and annotate Ljung–Box p-value."""
    if plt is None or plot_acf is None:  # pragma: no cover
        raise RuntimeError("matplotlib/statsmodels required for plotting.")

    res = np.asarray(list(residuals), dtype=float)
    res = res[np.isfinite(res)]
    lags = int(lags or SARIMA_LJUNGBOX_LAGS_DEFAULT)

    if out_path is None:
        out_path = Path(SARIMA_RESIDUALS_LJUNGBOX_PLOT)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=PLOT_FIGURE_SIZE_DEFAULT)
    ax = fig.add_subplot(111)
    plot_acf(res, lags=lags, ax=ax, zero=False)  # Exclude lag 0 (always 1.0)
    ax.set_title(f"Residuals ACF (lags={lags})")

    lb = ljung_box_on_residuals(res, lags=lags)
    # Use the last p-value (cumulative test up to max lag)
    p_value = lb["p_value"][-1] if lb["p_value"] else np.nan
    txt = f"Ljung–Box (lag={lags}) p-value={p_value:.4f}, n={lb['n']}"
    ax.text(PLOT_TEXT_POSITION_X, PLOT_TEXT_POSITION_Y, txt, transform=ax.transAxes, va="top")

    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI_EVALUATION)
    plt.close(fig)
    logger.info(f"Saved residuals ACF + Ljung–Box: {out_path}")
    return out_path


def _ensure_path(path_like: Path | str | PathLike[str] | Any) -> Path:
    """Convert arbitrary path-like objects (including mocks) to :class:`Path`."""
    if isinstance(path_like, Path):
        return path_like
    if isinstance(path_like, str):
        return Path(path_like)
    try:
        return Path(path_like)
    except TypeError:
        return cast(Path, path_like)


def _prepare_predictions_dataframe(results: dict[str, Any]) -> pd.DataFrame:
    """Prepare predictions dataframe from results dict.

    Args:
        results: Evaluation results dictionary.

    Returns:
        DataFrame with date, y_true, y_pred, residual columns.
    """
    y_true = results.get("y_true", results.get("actuals", []))
    y_pred = results.get("y_pred", results.get("predictions", []))
    residuals = results.get("residuals", [])
    if not residuals and y_true and y_pred:
        residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)

    df = pd.DataFrame(
        {
            "date": format_dates_to_string(results["dates"]),
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
            "residual": np.asarray(residuals, dtype=float),
        }
    )
    return df


def _prepare_metrics_payload(results: dict[str, Any], n_test: int) -> dict[str, Any]:
    """Prepare metrics payload for JSON serialization.

    Args:
        results: Evaluation results dictionary.
        n_test: Number of test observations.

    Returns:
        Metrics payload dictionary.
    """
    metrics = results.get("metrics", {})
    payload = {
        "order": list(results.get("order", [])),
        "seasonal_order": list(results.get("seasonal_order", [])),
        "refit_every": int(results.get("refit_every", SARIMA_REFIT_EVERY_DEFAULT)),
        "metrics": metrics,
        "n_test": n_test,
    }
    return payload


def save_evaluation_results(results: dict[str, Any]) -> tuple[Path, Path]:
    """Persist rolling predictions and metrics to disk using constants paths."""
    preds_file = _ensure_path(ROLLING_PREDICTIONS_SARIMA_FILE)
    metrics_file = _ensure_path(ROLLING_VALIDATION_METRICS_SARIMA_FILE)

    preds_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    df = _prepare_predictions_dataframe(results)
    df.to_csv(preds_file, index=False)

    payload = _prepare_metrics_payload(results, len(df))
    save_json_pretty(payload, metrics_file)

    logger.info(f"Saved predictions → {preds_file}")
    logger.info(f"Saved metrics → {metrics_file}")
    return preds_file, metrics_file


def save_ljung_box_results(lb_result: dict[str, Any]) -> Path:
    """Persist Ljung–Box results as JSON."""
    path = Path(LJUNGBOX_RESIDUALS_SARIMA_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json_pretty(lb_result, path)
    logger.info(f"Saved Ljung–Box results → {path}")
    return path


def _setup_backtest_configuration(
    train_series: pd.Series,
    test_series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    trend: str,
) -> tuple[pd.Series, list[Any], SarimaSpec, int]:
    """Setup configuration for full series backtest."""
    full_series = pd.concat([train_series, test_series]).sort_index()

    if isinstance(full_series.index, pd.DatetimeIndex):
        original_dates = full_series.index.tolist()
    else:
        original_dates = pd.to_datetime(full_series.index).tolist()

    y_full = _ensure_datetime_index(_as_series(full_series, "y_full"))
    spec = SarimaSpec(order=order, seasonal_order=seasonal_order, trend=trend)
    suppress_statsmodels_warnings()

    min_train_size = min(SARIMA_MIN_TRAIN_SIZE_FOR_RESIDUALS, len(y_full) // 4)
    if min_train_size < SARIMA_MIN_TRAIN_SIZE_FOR_RESIDUALS // 2:
        min_train_size = min(SARIMA_MIN_TRAIN_SIZE_FOR_RESIDUALS // 2, len(y_full) // 2)

    return y_full, original_dates, spec, min_train_size


def _run_backtest_loop(
    y_full: pd.Series,
    spec: SarimaSpec,
    min_train_size: int,
    refit_every: int,
    verbose: bool,
) -> tuple[list[float], list[float]]:
    """Execute backtest loop and return predictions and actuals."""
    preds: list[float] = []
    actuals: list[float] = []
    fitted_model = None

    for i in range(min_train_size, len(y_full)):
        y_train_current = y_full.iloc[:i]
        y_true = float(y_full.iloc[i])

        needs_refit = (
            i == min_train_size
            or (refit_every > 0 and (i - min_train_size) % refit_every == 0)
            or fitted_model is None
        )
        if needs_refit:
            fitted_model = _fit_sarima(y_train_current, spec)
            if verbose:
                logger.info(
                    f"Backtest refit at step {i}/{len(y_full)} (train n={len(y_train_current)})"
                )

        pred = _make_one_step_forecast(fitted_model, y_train_current, i)
        preds.append(pred)
        actuals.append(y_true)

        if verbose and (i + 1 - min_train_size) % ROLLING_FORECAST_PROGRESS_INTERVAL == 0:
            logger.info(f"Backtest progress: {i + 1}/{len(y_full)}")

    return preds, actuals


def backtest_full_series(
    train_series: pd.Series,
    test_series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    refit_every: int = SARIMA_REFIT_EVERY_DEFAULT,
    verbose: bool = False,
    trend: str = "c",
) -> pd.DataFrame:
    """Backtest on full series (train+test) with periodic refits to generate residuals for GARCH.

    This function generates predictions and residuals for the entire dataset (train + test)
    by doing a rolling forecast with periodic refits. This ensures we have residuals
    for all dates needed by the GARCH model.

    Args:
        train_series: Training data series.
        test_series: Test data series.
        order: ARIMA order (p, d, q).
        seasonal_order: Seasonal ARIMA order (P, D, Q, s).
        refit_every: Number of steps between model refits.
        verbose: Whether to log progress.
        trend: Trend component ("n", "c", "t", "ct"). Default: "c".

    Returns:
        DataFrame with columns: date, y_true, y_pred, sarima_resid for all dates.
    """
    y_full, original_dates, spec, min_train_size = _setup_backtest_configuration(
        train_series, test_series, order, seasonal_order, trend
    )

    preds, actuals = _run_backtest_loop(y_full, spec, min_train_size, refit_every, verbose)

    # Compute residuals
    residuals = np.array(actuals) - np.array(preds)

    # Format dates for the forecasted portion
    forecast_dates = original_dates[min_train_size:]

    # Create DataFrame
    history = pd.DataFrame(
        {
            "date": format_dates_to_string(forecast_dates),
            "y_true": actuals,
            "y_pred": preds,
            "sarima_resid": residuals,
        }
    )

    return history


def evaluate_model(
    train_series: pd.Series,
    test_series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    refit_every: int = SARIMA_REFIT_EVERY_DEFAULT,
    verbose: bool = False,
    model_info: dict[str, Any] | None = None,
    trend: str = "c",
) -> dict[str, Any]:
    """End-to-end evaluation producing predictions, metrics, residuals.

    Args:
        train_series: Training data series.
        test_series: Test data series.
        order: ARIMA order (p, d, q).
        seasonal_order: Seasonal ARIMA order (P, D, Q, s).
        refit_every: Number of steps between model refits (default: 20).
        verbose: Whether to log progress (default: False).
        model_info: Optional model metadata dictionary.
        trend: Trend component ("n", "c", "t", "ct"). Default: "c".

    Returns:
        Dictionary with evaluation results including metrics and predictions.
    """
    # Preserve original dates from test_series BEFORE any transformations
    # This ensures we keep trading dates (without weekends/holidays) instead of consecutive dates
    if isinstance(test_series.index, pd.DatetimeIndex):
        original_dates = test_series.index.tolist()
    else:
        # If not datetime, convert to datetime for consistency
        original_dates = pd.to_datetime(test_series.index).tolist()

    preds, actuals = rolling_forecast(
        train_series=train_series,
        test_series=test_series,
        order=order,
        seasonal_order=seasonal_order,
        refit_every=refit_every,
        verbose=verbose,
        trend=trend,
    )
    metrics = calculate_metrics(actuals, preds)
    residuals = compute_residuals(actuals, preds)

    # Get model name from model_info if provided
    model_name = "ARIMA"
    if model_info and "params" in model_info:
        model_name = str(model_info["params"])

    results = {
        "model": model_name,
        "order": order,  # Keep as tuple to match test expectations
        "metrics": metrics,
        "predictions": preds.tolist(),
        "actuals": actuals.tolist(),
        "dates": original_dates,  # Use preserved original dates (trading days only)
        # Legacy keys for compatibility with save_garch_dataset and diagnostics
        "y_true": actuals.tolist(),
        "y_pred": preds.tolist(),
        "residuals": residuals.tolist(),
    }
    return results


def _create_split_result_row(
    k: int,
    train: pd.Series,
    test: pd.Series,
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Create a result row for a single split."""
    return {
        "split": k + 1,
        "train_start": train.index[0] if len(train) > 0 else pd.NaT,
        "train_end": train.index[-1] if len(train) > 0 else pd.NaT,
        "validation_start": test.index[0],
        "validation_end": test.index[-1],
        "MSE": float(metrics.get("MSE", np.nan)),
        "RMSE": float(metrics.get("RMSE", np.nan)),
        "MAE": float(metrics.get("MAE", np.nan)),
    }


def _process_single_split(
    k: int,
    train: pd.Series,
    test: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    refit_every: int,
    verbose: bool,
    trend: str = "c",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Process a single split and return result row and per-step residuals."""
    preds, actuals = rolling_forecast(
        train_series=train,
        test_series=test,
        order=order,
        seasonal_order=seasonal_order,
        refit_every=refit_every,
        verbose=verbose,
        trend=trend,
    )
    metrics = calculate_metrics(actuals, preds)
    residuals = compute_residuals(actuals, preds)
    split_dates = format_dates_to_string(test.index.to_series())  # type: ignore[arg-type]
    history = pd.DataFrame(
        {
            "split": k + 1,
            "date": split_dates,
            "y_true": actuals,
            "y_pred": preds,
            "sarima_resid": residuals,
        }
    )
    return _create_split_result_row(k, train, test, metrics), history


def _compute_summary_stats(
    mse_list: list[float],
    rmse_list: list[float],
    mae_list: list[float],
) -> dict[str, float]:
    """Compute summary statistics across splits."""
    return {
        "mse_mean": float(np.nanmean(mse_list)),
        "mse_std": float(np.nanstd(mse_list, ddof=0)),
        "rmse_mean": float(np.nanmean(rmse_list)),
        "rmse_std": float(np.nanstd(rmse_list, ddof=0)),
        "mae_mean": float(np.nanmean(mae_list)),
        "mae_std": float(np.nanstd(mae_list, ddof=0)),
    }


def _validate_walk_forward_params(
    n_splits: int,
    test_size: int,
    series_len: int,
) -> None:
    """Validate walk-forward backtest parameters."""
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1.")
    if test_size <= 0 or test_size >= series_len:
        raise ValueError("test_size must be > 0 and < len(series).")
    total_needed = n_splits * test_size
    if total_needed >= series_len:
        raise ValueError("Series too short for requested walk-forward configuration.")


def _validate_basic_params(
    series: pd.Series,
    test_size: int | None,
) -> tuple[pd.Series, int]:
    """Validate basic parameters and return processed series and test size."""
    s = _ensure_datetime_index(_as_series(series, "y"))
    total_len = len(s)

    if total_len < 2:
        raise ValueError("Series too short for walk-forward backtest.")

    if test_size is None:
        raise ValueError("test_size must be provided explicitly (no fallback).")

    effective_test_size = int(test_size)
    if effective_test_size <= 0:
        raise ValueError("test_size must be positive.")

    return s, effective_test_size


def _compute_initial_train_size(
    total_len: int,
    n_splits: int | None,
    test_size: int,
    initial_train_size: int | None,
) -> int:
    """Compute initial_train_size if not provided."""
    if initial_train_size is None:
        if n_splits is None:
            raise ValueError(
                "initial_train_size must be provided when n_splits is not specified (no fallback)."
            )
        initial_train_size = max(
            total_len - int(n_splits) * test_size,
            test_size,
        )
    return initial_train_size


def _validate_initial_train_size(initial_train_size: int, total_len: int) -> None:
    """Validate initial_train_size parameter."""
    if initial_train_size <= 0 or initial_train_size >= total_len:
        raise ValueError(
            f"initial_train_size must be between 1 and len(series)-1 "
            f"(got {initial_train_size}, len={total_len})"
        )


def _compute_effective_splits(
    total_len: int,
    initial_train_size: int,
    n_splits: int | None,
    test_size: int,
) -> tuple[int, int]:
    """Compute effective number of splits and test size."""
    max_available = total_len - initial_train_size
    if max_available <= 0:
        raise ValueError("Series too short for requested walk-forward configuration.")

    effective_n_splits = n_splits if n_splits is not None else max_available // test_size
    effective_test_size = test_size

    if effective_n_splits <= 0:
        effective_n_splits = max_available // effective_test_size
    if effective_n_splits <= 0:
        effective_n_splits = 1
        effective_test_size = max_available

    # Adjust if we need more points than available
    required_points = effective_n_splits * effective_test_size
    if required_points > max_available:
        effective_n_splits = max_available // effective_test_size
        if effective_n_splits == 0:
            effective_n_splits = 1
            effective_test_size = max_available

    return effective_n_splits, effective_test_size


def _configure_walk_forward_params(
    series: pd.Series,
    n_splits: int | None,
    test_size: int | None,
    initial_train_size: int | None,
) -> tuple[int, int, int]:
    """Configure and validate walk-forward parameters."""
    s, effective_test_size = _validate_basic_params(series, test_size)
    total_len = len(s)

    initial_train_size = _compute_initial_train_size(
        total_len, n_splits, effective_test_size, initial_train_size
    )
    _validate_initial_train_size(initial_train_size, total_len)

    effective_n_splits, final_test_size = _compute_effective_splits(
        total_len, initial_train_size, n_splits, effective_test_size
    )

    return initial_train_size, effective_n_splits, final_test_size


def walk_forward_backtest(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    n_splits: int | None = None,
    test_size: int | None = None,
    refit_every: int = SARIMA_REFIT_EVERY_DEFAULT,
    verbose: bool = False,
    *,
    initial_train_size: int | None = None,
    trend: str = "c",
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Walk-forward backtest over n_splits contiguous test windows.

    Args:
        series: Full time series data.
        order: ARIMA order (p, d, q).
        seasonal_order: Seasonal ARIMA order (P, D, Q, s).
        n_splits: Number of temporal splits (optional).
        test_size: Size of each test window (optional).
        refit_every: Steps between model refits (default: 20).
        verbose: Whether to log progress (default: False).
        initial_train_size: Initial training window size (optional).
        trend: Trend component ("n", "c", "t", "ct"). Default: "c".

    Returns
    -------
    split_df : pd.DataFrame
        One row per split with temporal boundaries and metrics (MSE, RMSE, MAE).
    summary : dict[str, float]
        Aggregated statistics across splits: mse_mean/std, rmse_mean/std, mae_mean/std.
    history : pd.DataFrame
        Per-step walk-forward predictions with columns ['split', 'date', 'y_true',
        'y_pred', 'sarima_resid'].
    """
    initial_train_size, effective_n_splits, effective_test_size = _configure_walk_forward_params(
        series, n_splits, test_size, initial_train_size
    )

    s = _ensure_datetime_index(_as_series(series, "y"))
    total_len = len(s)

    _validate_walk_forward_params(effective_n_splits, effective_test_size, total_len)

    rows: list[dict[str, Any]] = []
    mse_list: list[float] = []
    rmse_list: list[float] = []
    mae_list: list[float] = []
    history_frames: list[pd.DataFrame] = []

    for k in range(effective_n_splits):
        test_start = initial_train_size + k * effective_test_size
        if test_start >= total_len:
            break
        test_end = min(test_start + effective_test_size, total_len)
        train = s.iloc[:test_start]
        test = s.iloc[test_start:test_end]
        if test.empty:
            continue
        row, history = _process_single_split(
            k,
            train,
            test,
            order,
            seasonal_order,
            refit_every,
            verbose,
            trend,
        )
        rows.append(row)
        history_frames.append(history)
        mse_list.append(row["MSE"])
        rmse_list.append(row["RMSE"])
        mae_list.append(row["MAE"])

    split_df = pd.DataFrame(rows)
    summary = _compute_summary_stats(mse_list, rmse_list, mae_list)
    history_df = (
        pd.concat(history_frames, ignore_index=True)
        if history_frames
        else pd.DataFrame(columns=pd.Index(SARIMA_HISTORY_COLUMNS))
    )
    return split_df, summary, history_df
