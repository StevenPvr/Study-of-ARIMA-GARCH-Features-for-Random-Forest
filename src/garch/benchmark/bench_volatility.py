"""Volatility backtest orchestrator: EGARCH vs simple baselines.

This module prepares a fair comparison for volatility forecasting
(not directional decisions) by computing one-step-ahead variance paths
for multiple models on the same test split:
- ARIMA-EGARCH (rolling, periodic refits)
- EWMA
- GARCH(1,1)
- Rolling variance/std (historical volatility)
- Naive model (constant variance)
- ARCH(1)
- HAR(3) on squared returns

Implements Section 4 of the methodology:
- 4.1: Baselines (EWMA, GARCH(1,1), historical volatility, naive)
- 4.2: Evaluation (QLIKE, MSE_var, MAE_var, VaR backtests, MZ regression)
- 4.3: Comparison table

Inputs are expected to include either 'arima_residual_return' (preferred)
or 'weighted_log_return'. Only the residual column is used for EGARCH.
"""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import pandas as pd

import src.constants as C
from src.constants import DEFAULT_RANDOM_STATE
from src.garch.garch_eval.helpers import chi2_sf, load_best_model
from src.garch.garch_eval.metrics import (
    build_var_series,
    christoffersen_ind_test,
    kupiec_pof_test,
    mincer_zarnowitz,
    mse_mae_variance,
    qlike_loss,
)
from src.garch.structure_garch.utils import load_garch_dataset
from src.garch.training_garch.predictions_io import load_garch_forecasts
from src.path import GARCH_FORECASTS_FILE
from src.utils import get_logger

from .data_utils import pos_test_valid, select_residual_column
from .forecasts import compute_baseline_forecasts
from .metrics import compute_metrics
from .plotting import plot_volatility_forecasts
from .validation import validate_backtest_params

logger = get_logger(__name__)


# Exposed for tests to monkeypatch (keeps CI fast when needed)
def run_rolling_garch_from_artifacts(
    *,
    df: pd.DataFrame | None = None,
    refit_every: int,
    window: str,
    window_size: int | None,
    dist_preference: str,
    keep_nu_between_refits: bool,
    var_alphas: list[float] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute or load EGARCH rolling forecasts from artifacts or DataFrame."""
    _validate_refit_every(refit_every)
    if df is not None:
        return _compute_forecasts_from_dataframe(
            df,
            refit_every=refit_every,
            window=window,
            window_size=window_size,
            dist_preference=dist_preference,
            keep_nu_between_refits=keep_nu_between_refits,
            var_alphas=var_alphas,
        )
    return _load_forecasts_from_artifacts(
        refit_every=refit_every,
        window=window,
        window_size=window_size,
        dist_preference=dist_preference,
        keep_nu_between_refits=keep_nu_between_refits,
        var_alphas=var_alphas,
    )


def _validate_refit_every(refit_every: int) -> None:
    """Validate refit frequency parameter."""
    if refit_every <= 0:
        msg = "refit_every must be a positive integer"
        raise ValueError(msg)


def _compute_forecasts_from_dataframe(
    df: pd.DataFrame,
    *,
    refit_every: int,
    window: str,
    window_size: int | None,
    dist_preference: str,
    keep_nu_between_refits: bool,
    var_alphas: list[float] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute lightweight variance forecasts directly from a DataFrame."""
    residual_series = _extract_residual_series(df)
    sigma2 = _compute_variance_forecasts(residual_series, refit_every)
    forecasts = _build_garch_forecast_frame(df, sigma2)
    metrics = _build_dataframe_metrics(
        df,
        sigma2,
        refit_every=refit_every,
        window=window,
        window_size=window_size,
        dist_preference=dist_preference,
        keep_nu_between_refits=keep_nu_between_refits,
        var_alphas=var_alphas,
    )
    return forecasts, metrics


def _load_forecasts_from_artifacts(
    *,
    refit_every: int,
    window: str,
    window_size: int | None,
    dist_preference: str,
    keep_nu_between_refits: bool,
    var_alphas: list[float] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load forecasts and metadata from persisted artifacts."""
    forecasts, metadata = load_garch_forecasts()
    if "sigma2_forecast" not in forecasts.columns and "sigma_forecast" not in forecasts.columns:
        msg = "Loaded GARCH forecasts must contain 'sigma2_forecast' or " "'sigma_forecast' column"
        raise ValueError(msg)

    metrics = metadata.copy() if metadata is not None else {}
    metrics.setdefault("source", "artifacts")
    metrics.setdefault("refit_frequency", refit_every)
    metrics.setdefault("window", window)
    metrics.setdefault("window_size", window_size)
    metrics.setdefault("dist_preference", dist_preference)
    metrics.setdefault("keep_nu_between_refits", bool(keep_nu_between_refits))
    metrics.setdefault("var_alphas", list(var_alphas) if var_alphas else [])
    return forecasts, metrics


def _extract_residual_series(df: pd.DataFrame) -> pd.Series:
    """Extract residual series from DataFrame with strong validation."""
    candidate_columns = [
        "arima_residual_return",
        "sarima_resid",
        "weighted_log_return",
    ]
    for col in candidate_columns:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if not isinstance(series, pd.Series):
                msg = f"Residual column '{col}' must be a pandas Series"
                raise TypeError(msg)
            if series.notna().any():
                return series
            msg = f"Column '{col}' contains no valid residual values"
            raise ValueError(msg)
    msg = "Input DataFrame must contain at least one residual column among " f"{candidate_columns}"
    raise ValueError(msg)


def _compute_variance_forecasts(residuals: pd.Series, refit_every: int) -> np.ndarray:
    """Compute simple rolling variance forecasts as a lightweight fallback."""
    window = max(refit_every, 5)
    rolling_var = residuals.rolling(window=window, min_periods=1).var(ddof=1)
    if not isinstance(rolling_var, pd.Series):
        msg = "Rolling variance computation must return a pandas Series"
        raise TypeError(msg)
    rolling_var = cast(pd.Series, rolling_var)
    valid_variances = rolling_var.dropna()
    if not valid_variances.empty:
        first_valid = float(valid_variances.iloc[0])
    else:
        # Compute variance with ddof=1 (sample variance)
        residuals_array = residuals.to_numpy()
        first_valid = float(np.var(residuals_array, ddof=1))  # type: ignore
    if not np.isfinite(first_valid) or first_valid <= 0.0:
        # Fallback to mean of squared residuals if variance is invalid
        residuals_array = residuals.to_numpy()
        first_valid = float(np.mean(np.square(residuals_array)))
    rolling_var = rolling_var.fillna(first_valid)
    rolling_var = rolling_var.clip(lower=1e-12)
    return rolling_var.to_numpy(dtype=float)


def _build_garch_forecast_frame(df: pd.DataFrame, sigma2: np.ndarray) -> pd.DataFrame:
    """Assemble forecasts DataFrame aligned with input DataFrame order."""
    if "date" not in df.columns:
        msg = "Input DataFrame must contain 'date' column"
        raise ValueError(msg)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"], errors="coerce"),
            "sigma2_forecast": sigma2,
        }
    )
    date_nulls = out["date"].isna()
    has_invalid_dates = cast(bool, date_nulls.any())
    if has_invalid_dates:
        msg = "Found invalid dates in input DataFrame"
        raise ValueError(msg)
    if "split" in df.columns:
        out["split"] = df["split"].to_numpy()
    return out


def _build_dataframe_metrics(
    df: pd.DataFrame,
    sigma2: np.ndarray,
    *,
    refit_every: int,
    window: str,
    window_size: int | None,
    dist_preference: str,
    keep_nu_between_refits: bool,
    var_alphas: list[float] | None,
) -> dict[str, Any]:
    """Build metadata dictionary for DataFrame-based forecasts."""
    n_obs = int(df.shape[0])
    refit_count = max((n_obs - 1) // max(refit_every, 1), 0)
    metrics: dict[str, Any] = {
        "source": "dataframe",
        "n_observations": n_obs,
        "refit_frequency": int(refit_every),
        "refit_count": int(refit_count),
        "window": window,
        "window_size": window_size,
        "dist_preference": dist_preference,
        "keep_nu_between_refits": bool(keep_nu_between_refits),
        "var_alphas": list(var_alphas) if var_alphas else [],
        "variance_min": float(np.min(sigma2)),
        "variance_max": float(np.max(sigma2)),
        "variance_mean": float(np.mean(sigma2)),
    }
    return metrics


def _merge_outputs(
    dates_test: np.ndarray,
    e_test: np.ndarray,
    s2_garch: np.ndarray,
    s2_ewma: np.ndarray,
    s2_roll_var: np.ndarray,
    s2_roll_std: np.ndarray,
    s2_arch1: np.ndarray,
    s2_har3: np.ndarray,
) -> pd.DataFrame:
    """Merge all forecasts into a single DataFrame.

    Args:
        dates_test: Test dates array.
        e_test: Test residuals.
        s2_garch: GARCH variance forecasts.
        s2_ewma: EWMA variance forecasts.
        s2_roll_var: Rolling variance forecasts.
        s2_roll_std: Rolling std forecasts.
        s2_arch1: ARCH(1) variance forecasts.
        s2_har3: HAR(3) variance forecasts.

    Returns:
        DataFrame with all forecasts.
    """
    return pd.DataFrame(
        {
            "date": dates_test,
            "e": e_test,
            "sigma2_forecast": s2_garch,
            "s2_arima_garch": s2_garch,
            "s2_ewma": s2_ewma,
            "s2_roll_var": s2_roll_var,
            "s2_roll_std": s2_roll_std,
            "s2_arch1": s2_arch1,
            "s2_har3": s2_har3,
        }
    )


def _run_garch_forecasts(
    df: pd.DataFrame | None,
    var_alphas: list[float] | None,
    refit_every: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run GARCH forecasts from DataFrame or artifacts.

    Args:
        df: DataFrame with residuals, or None to use artifacts.
        var_alphas: VaR alphas for GARCH runner.
        refit_every: Refit frequency (in test observations).

    Returns:
        Tuple of (forecasts DataFrame, metrics dict with dist/nu info).
    """
    runner_kwargs: dict[str, Any] = {
        "refit_every": refit_every,
        "window": C.GARCH_REFIT_WINDOW_DEFAULT,
        "window_size": C.GARCH_REFIT_WINDOW_SIZE_DEFAULT,
        "dist_preference": "auto",
        "keep_nu_between_refits": True,
        "var_alphas": var_alphas,
    }
    if df is not None:
        runner_kwargs["df"] = df
    fore_garch, metr_garch = run_rolling_garch_from_artifacts(**runner_kwargs)
    return fore_garch, metr_garch


def _align_garch_forecasts(
    s2_garch: np.ndarray,
    target_size: int,
) -> np.ndarray:
    """Align GARCH forecasts to target size.

    Args:
        s2_garch: GARCH variance forecasts.
        target_size: Target array size.

    Returns:
        Aligned GARCH forecasts array.
    """
    if s2_garch.size == target_size:
        return s2_garch
    if s2_garch.size >= target_size:
        return s2_garch[:target_size]
    pad = np.full(target_size - s2_garch.size, float("nan"))
    return np.concatenate([s2_garch, pad])


def _prepare_residuals_data(
    df: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare residuals data for backtest.

    Args:
        df: Optional input DataFrame.

    Returns:
        Tuple of (e_valid, dates_valid, e_test, dates_test, pos_test).

    Raises:
        ValueError: If no valid test observations found.
    """
    if df is None:
        logger.info(
            "No DataFrame provided; loading default GARCH dataset: %s", C.GARCH_DATASET_FILE
        )
        data_for_residuals = load_garch_dataset()
    else:
        data_for_residuals = df

    e_all, dates_all, test_mask = select_residual_column(data_for_residuals)
    valid, pos_test = pos_test_valid(e_all, test_mask)
    if pos_test.size == 0:
        msg = "No valid test observations found"
        raise ValueError(msg)

    e_valid = e_all[valid]
    dates_valid = dates_all[valid]
    e_test = e_valid[pos_test]
    dates_test = dates_valid[pos_test]

    return e_valid, dates_valid, e_test, dates_test, pos_test


def _extract_garch_variance(fore_garch: pd.DataFrame) -> np.ndarray:
    """Extract variance from GARCH forecasts DataFrame.

    Args:
        fore_garch: GARCH forecasts DataFrame.

    Returns:
        GARCH variance array.

    Raises:
        ValueError: If required columns are missing.
    """
    if "sigma_forecast" in fore_garch.columns:
        sigma_garch_raw = np.asarray(
            fore_garch.loc[:, "sigma_forecast"].to_numpy(), dtype=np.float64
        )
        return sigma_garch_raw**2
    elif "sigma2_forecast" in fore_garch.columns:
        return np.asarray(fore_garch.loc[:, "sigma2_forecast"].to_numpy(), dtype=np.float64)
    else:
        msg = "GARCH forecasts DataFrame must contain either 'sigma_forecast' or 'sigma2_forecast'"
        raise ValueError(msg)


def _compute_all_forecasts(
    e_valid: np.ndarray,
    pos_test: np.ndarray,
    fore_garch: pd.DataFrame,
    dates_test: np.ndarray,
    ewma_lambda: float,
    rolling_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute all volatility forecasts.

    Args:
        e_valid: Valid residuals array.
        pos_test: Test positions in valid array.
        fore_garch: GARCH forecasts DataFrame.
        dates_test: Test dates array.
        ewma_lambda: EWMA decay factor.
        rolling_window: Rolling window size.

    Returns:
        Tuple of (s2_garch, s2_ewma, s2_roll_var, s2_roll_std, s2_arch1, s2_har3).
    """
    s2_ewma, s2_garch11, s2_roll_var, s2_naive, s2_arch1, s2_har3 = compute_baseline_forecasts(
        e_valid, pos_test, ewma_lambda, rolling_window
    )
    # Keep s2_roll_std for backward compatibility (same as s2_roll_var)
    s2_roll_std = s2_roll_var.copy()

    s2_garch_raw = _extract_garch_variance(fore_garch)
    s2_garch = _align_garch_forecasts(s2_garch_raw, dates_test.size)

    return s2_garch, s2_ewma, s2_roll_var, s2_roll_std, s2_arch1, s2_har3


def _assemble_backtest_results(
    dates_test: np.ndarray,
    e_test: np.ndarray,
    s2_garch: np.ndarray,
    s2_ewma: np.ndarray,
    s2_roll_var: np.ndarray,
    s2_roll_std: np.ndarray,
    s2_arch1: np.ndarray,
    s2_har3: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble forecasts DataFrame and compute metrics.

    Args:
        dates_test: Test dates array.
        e_test: Test residuals.
        s2_garch: GARCH variance forecasts.
        s2_ewma: EWMA variance forecasts.
        s2_roll_var: Rolling variance forecasts.
        s2_roll_std: Rolling std forecasts.
        s2_arch1: ARCH(1) variance forecasts.
        s2_har3: HAR(3) variance forecasts.

    Returns:
        Tuple of (forecasts DataFrame, metrics dictionary).
    """
    forecasts = _merge_outputs(
        dates_test, e_test, s2_garch, s2_ewma, s2_roll_var, s2_roll_std, s2_arch1, s2_har3
    )
    metrics = compute_metrics(
        e_test, s2_garch, s2_ewma, s2_roll_var, s2_roll_std, s2_arch1, s2_har3
    )
    return forecasts, metrics


def run_vol_backtest(
    df: pd.DataFrame | None = None,
    *,
    ewma_lambda: float | None = None,
    rolling_window: int | None = None,
    var_alphas: list[float] | None = None,
    refit_every: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run volatility backtest: EGARCH vs baselines on same test split.

    Args:
        df: Optional input DataFrame. If provided, uses it for residuals and
            passes it to the EGARCH runner. If None, automatically loads the
            default GARCH dataset from `C.GARCH_DATASET_FILE` to extract
            residuals, and runs EGARCH from artifacts.
        ewma_lambda: EWMA decay factor (lambda).
        rolling_window: Window size for rolling baselines.
        var_alphas: VaR alphas; passed to the EGARCH runner.
        refit_every: GARCH refit frequency (in test observations). Defaults to 20.

    Returns:
        Tuple of (forecasts DataFrame, metrics dictionary).

    Raises:
        ValueError: If parameters are invalid or data is insufficient.
    """
    ewma_lambda, rolling_window = validate_backtest_params(ewma_lambda, rolling_window, var_alphas)
    fore_garch, _ = _run_garch_forecasts(df, var_alphas, refit_every=refit_every)
    e_valid, dates_valid, e_test, dates_test, pos_test = _prepare_residuals_data(df)
    s2_garch, s2_ewma, s2_roll_var, s2_roll_std, s2_arch1, s2_har3 = _compute_all_forecasts(
        e_valid, pos_test, fore_garch, dates_test, ewma_lambda, rolling_window
    )
    return _assemble_backtest_results(
        dates_test, e_test, s2_garch, s2_ewma, s2_roll_var, s2_roll_std, s2_arch1, s2_har3
    )


def save_vol_backtest_outputs(forecasts: pd.DataFrame, metrics: dict[str, Any]) -> None:
    """Persist volatility backtest forecasts, metrics, and plots to disk.

    Creates directories as needed under constants paths.

    Args:
        forecasts: Forecasts DataFrame.
        metrics: Metrics dictionary.
    """
    C.VOL_BACKTEST_FORECASTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(C.VOL_BACKTEST_FORECASTS_FILE, index=False)
    C.VOL_BACKTEST_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with C.VOL_BACKTEST_METRICS_FILE.open("w") as f:
        json.dump(metrics, f, indent=2)

    dates = pd.to_datetime(forecasts["date"]).to_numpy()
    e_test = forecasts["e"].to_numpy()
    plot_volatility_forecasts(dates, e_test, forecasts)

    logger.info(
        "Saved volatility backtest outputs: %s, %s, %s",
        C.VOL_BACKTEST_FORECASTS_FILE,
        C.VOL_BACKTEST_METRICS_FILE,
        C.VOL_BACKTEST_VOLATILITY_PLOT,
    )


def _compute_model_comprehensive_metrics(
    e_test: np.ndarray,
    RV_test: np.ndarray,
    sigma2_forecast: np.ndarray,
    model_name: str,
    alphas: list[float],
    *,
    dist: str | None = None,
    nu: float | None = None,
    lambda_skew: float | None = None,
) -> dict[str, Any]:
    """Compute comprehensive metrics for a single model.

    Args:
        e_test: Test residuals.
        RV_test: Realized variance (e_test^2).
        sigma2_forecast: Variance forecasts.
        model_name: Model name.
        alphas: VaR alpha levels.
        dist: Distribution type for VaR calculation (default: 'normal' for baselines).
        nu: Degrees of freedom for Student-t/Skew-t (default: None).
        lambda_skew: Skewness parameter for Skew-t (default: None).

    Returns:
        Dictionary with all metrics.
    """
    # Filter valid data
    valid_mask = (
        np.isfinite(e_test)
        & np.isfinite(RV_test)
        & np.isfinite(sigma2_forecast)
        & (sigma2_forecast > 0)
    )
    if not np.any(valid_mask):
        return {
            "qlike": float("nan"),
            "mse_var": float("nan"),
            "mae_var": float("nan"),
            "var_backtests": {},
            "mz": {},
        }

    e_valid = e_test[valid_mask]
    RV_valid = RV_test[valid_mask]
    sigma2_valid = sigma2_forecast[valid_mask]

    # Variance metrics (using RV vs sigma2)
    variance_losses = mse_mae_variance(RV_valid, sigma2_valid)
    qlike_val = qlike_loss(RV_valid, sigma2_valid)

    # MZ regression
    mz_results = mincer_zarnowitz(e_valid, sigma2_valid)

    # VaR backtests
    # For EGARCH: use the distribution from trained model (Skew-t, Student-t, or Normal)
    # For baselines: use normal distribution
    var_backtests: dict[str, dict[str, float]] = {}
    for alpha in alphas:
        if dist and dist.lower() in ("student", "skewt"):
            # Use distribution-specific quantile for EGARCH
            var_t = build_var_series(sigma2_valid, alpha, dist, nu, lambda_skew)
        else:
            # Use normal distribution for baselines
            var_t = build_var_series(sigma2_valid, alpha, "normal", None, None)

        # Detection: violation when e_valid < var_t (both are negative, var_t is threshold)
        hits = (e_valid < var_t).astype(int)
        kup = kupiec_pof_test(hits, alpha)
        ind = christoffersen_ind_test(hits)
        var_backtests[str(alpha)] = {
            "n": kup["n"],
            "violations": kup["x"],
            "hit_rate": kup["hit_rate"],
            "lr_uc": kup["lr_uc"],
            "p_uc": kup["p_value"],
            "lr_ind": ind["lr_ind"],
            "p_ind": ind["p_value"],
            "lr_cc": float(kup["lr_uc"] + ind["lr_ind"]),
            "p_cc": chi2_sf(float(kup["lr_uc"] + ind["lr_ind"]), df=2),
        }

    return {
        "qlike": qlike_val,
        "mse_var": variance_losses["mse"],
        "mae_var": variance_losses["mae"],
        "var_backtests": var_backtests,
        "mz": {
            "intercept": mz_results.get("intercept", float("nan")),
            "slope": mz_results.get("slope", float("nan")),
            "r2": mz_results.get("r2", float("nan")),
        },
    }


def _validate_and_set_defaults(
    ewma_lambda: float | None,
    rolling_window: int | None,
    alphas: list[float] | None,
) -> tuple[float, int, list[float]]:
    """Validate parameters and set defaults.

    Args:
        ewma_lambda: EWMA decay factor or None for default.
        rolling_window: Rolling window size or None for default.
        alphas: VaR alpha levels or None for default.

    Returns:
        Tuple of (ewma_lambda, rolling_window, alphas) with defaults applied.
    """
    # Set defaults
    if ewma_lambda is None:
        ewma_lambda = C.VOL_EWMA_LAMBDA_DEFAULT
    if rolling_window is None:
        rolling_window = C.VOL_ROLLING_WINDOW_DEFAULT
    if alphas is None:
        alphas = [0.01, 0.05]

    logger.info("Starting benchmark comparison (Section 4 methodology)")
    logger.info("EWMA lambda: %.3f, Rolling window: %d", ewma_lambda, rolling_window)

    return ewma_lambda, rolling_window, alphas


def _load_egarch_forecasts_and_distribution() -> (
    tuple[pd.DataFrame, str, float | None, float | None]
):
    """Load EGARCH forecasts and distribution parameters.

    Returns:
        Tuple of (egarch_forecasts, distribution, nu, lambda_skew).

    Raises:
        FileNotFoundError: If EGARCH forecasts file is missing.
    """
    if not GARCH_FORECASTS_FILE.exists():
        msg = f"EGARCH forecasts not found at {GARCH_FORECASTS_FILE}. Run evaluation first."
        raise FileNotFoundError(msg)

    egarch_forecasts = pd.read_csv(GARCH_FORECASTS_FILE, parse_dates=["date"])  # type: ignore[arg-type]
    logger.info("Loaded EGARCH forecasts: %d observations", len(egarch_forecasts))

    # Load distribution parameters from best model
    egarch_dist = "normal"
    egarch_nu = None
    egarch_lambda_skew = None
    try:
        _, _, egarch_dist, egarch_nu, egarch_lambda_skew = load_best_model()
        logger.info(
            "Loaded best EGARCH model distribution: %s (nu=%s, lambda=%s)",
            egarch_dist,
            egarch_nu,
            egarch_lambda_skew,
        )
    except Exception as ex:
        logger.warning(
            "Failed to load best EGARCH model for distribution info: %s. Using normal.", ex
        )

    return egarch_forecasts, egarch_dist, egarch_nu, egarch_lambda_skew


def _prepare_test_data_from_egarch_forecasts(
    egarch_forecasts: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract and validate test data from EGARCH forecasts.

    Args:
        egarch_forecasts: EGARCH forecasts DataFrame.

    Returns:
        Tuple of (resid_test, RV_test, sigma2_egarch, dates_test).

    Raises:
        ValueError: If EGARCH variance contains invalid values.
    """
    resid_test = egarch_forecasts["resid"].to_numpy(dtype=float)
    RV_test = egarch_forecasts["RV"].to_numpy(dtype=float)
    sigma2_egarch = egarch_forecasts["sigma2_egarch_raw"].to_numpy(dtype=float)
    dates_test = egarch_forecasts["date"].to_numpy()

    # Validate EGARCH variance
    if np.any(sigma2_egarch > 0.1):
        logger.warning(
            "EGARCH variance values seem unusually large (max=%.6f). "
            "Ensure 'sigma2_egarch_raw' contains variance (σ²), not standard deviation (σ).",
            np.nanmax(sigma2_egarch),
        )
    if np.any(sigma2_egarch < 0):
        msg = "EGARCH variance contains negative values. Invalid variance."
        raise ValueError(msg)

    return resid_test, RV_test, sigma2_egarch, dates_test


def _prepare_full_residuals_data() -> tuple[np.ndarray, np.ndarray]:
    """Load and prepare full residuals data (train + test) for baseline computation.

    Returns:
        Tuple of (e_valid, pos_test).

    Raises:
        ValueError: If residual column is not found.
    """
    df = load_garch_dataset()
    df_sorted = df.sort_values("date").reset_index(drop=True)

    # Select residual column
    col = (
        "arima_residual_return"
        if "arima_residual_return" in df_sorted.columns
        else "weighted_log_return"
    )
    if col not in df_sorted.columns:
        msg = f"Column '{col}' not found in dataset"
        raise ValueError(msg)

    # Extract and filter train data
    df_train = df_sorted.loc[df_sorted["split"] == "train"].copy()
    df_train = df_train.sort_values("date").reset_index(drop=True)
    resid_train_series = pd.to_numeric(df_train[col], errors="coerce")
    resid_train_full = np.asarray(resid_train_series, dtype=np.float64)
    resid_train_filtered = resid_train_full[np.isfinite(resid_train_full)]

    # Extract and filter test data
    df_test = df_sorted.loc[df_sorted["split"] == "test"].copy()
    df_test = df_test.sort_values("date").reset_index(drop=True)
    resid_test_series = pd.to_numeric(df_test[col], errors="coerce")
    resid_test_full = np.asarray(resid_test_series, dtype=np.float64)
    resid_test_filtered = resid_test_full[np.isfinite(resid_test_full)]

    # Combine train and test
    e_valid = np.concatenate([resid_train_filtered, resid_test_filtered])
    train_size = resid_train_filtered.size
    pos_test = np.arange(train_size, train_size + resid_test_filtered.size)

    return e_valid, pos_test


def _align_baseline_forecasts_to_egarch(
    n_egarch: int,
    s2_ewma_all: np.ndarray,
    s2_garch11_all: np.ndarray,
    s2_roll_var_all: np.ndarray,
    s2_naive_all: np.ndarray,
    s2_arch1_all: np.ndarray,
    s2_har3_all: np.ndarray,
    resid_test: np.ndarray,
    RV_test: np.ndarray,
    sigma2_egarch: np.ndarray,
    dates_test: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Align baseline forecasts to EGARCH forecasts length.

    Args:
        n_egarch: Length of EGARCH forecasts.
        s2_ewma_all: EWMA forecasts.
        s2_garch11_all: GARCH(1,1) forecasts.
        s2_roll_var_all: Rolling variance forecasts.
        s2_naive_all: Naive forecasts.
        s2_arch1_all: ARCH(1) forecasts.
        s2_har3_all: HAR(3) forecasts.
        resid_test: Test residuals.
        RV_test: Realized variance.
        sigma2_egarch: EGARCH variance.
        dates_test: Test dates.

    Returns:
        Tuple of aligned arrays.
    """
    if s2_ewma_all.size != n_egarch:
        logger.warning(
            "Baseline forecasts length (%d) != EGARCH forecasts length (%d). "
            "Using first %d observations.",
            s2_ewma_all.size,
            n_egarch,
            min(n_egarch, s2_ewma_all.size),
        )
        min_len = min(
            n_egarch,
            s2_ewma_all.size,
            s2_garch11_all.size,
            s2_roll_var_all.size,
            s2_naive_all.size,
            s2_arch1_all.size,
            s2_har3_all.size,
        )
        return (
            s2_ewma_all[:min_len],
            s2_garch11_all[:min_len],
            s2_roll_var_all[:min_len],
            s2_naive_all[:min_len],
            s2_arch1_all[:min_len],
            s2_har3_all[:min_len],
            resid_test[:min_len],
            RV_test[:min_len],
            sigma2_egarch[:min_len],
            dates_test[:min_len],
        )

    return (
        s2_ewma_all,
        s2_garch11_all,
        s2_roll_var_all,
        s2_naive_all,
        s2_arch1_all,
        s2_har3_all,
        resid_test,
        RV_test,
        sigma2_egarch,
        dates_test,
    )


def _compute_all_models_metrics(
    e_test: np.ndarray,
    RV_test: np.ndarray,
    sigma2_egarch: np.ndarray,
    s2_ewma: np.ndarray,
    s2_garch11: np.ndarray,
    s2_roll_var: np.ndarray,
    s2_naive: np.ndarray,
    s2_arch1: np.ndarray,
    s2_har3: np.ndarray,
    alphas: list[float],
    egarch_dist: str,
    egarch_nu: float | None,
    egarch_lambda_skew: float | None,
) -> dict[str, dict[str, Any]]:
    """Compute comprehensive metrics for all models.

    Args:
        e_test: Test residuals.
        RV_test: Realized variance.
        sigma2_egarch: EGARCH variance forecasts.
        s2_ewma: EWMA forecasts.
        s2_garch11: GARCH(1,1) forecasts.
        s2_roll_var: Rolling variance forecasts.
        s2_naive: Naive forecasts.
        s2_arch1: ARCH(1) forecasts.
        s2_har3: HAR(3) forecasts.
        alphas: VaR alpha levels.
        egarch_dist: EGARCH distribution type.
        egarch_nu: EGARCH degrees of freedom.
        egarch_lambda_skew: EGARCH skewness parameter.

    Returns:
        Dictionary with metrics for all models.
    """
    models_metrics: dict[str, dict[str, Any]] = {}

    models_metrics["EGARCH_raw"] = _compute_model_comprehensive_metrics(
        e_test,
        RV_test,
        sigma2_egarch,
        "EGARCH_raw",
        alphas,
        dist=egarch_dist,
        nu=egarch_nu,
        lambda_skew=egarch_lambda_skew,
    )

    models_metrics["EWMA"] = _compute_model_comprehensive_metrics(
        e_test, RV_test, s2_ewma, "EWMA", alphas
    )
    models_metrics["GARCH(1,1)"] = _compute_model_comprehensive_metrics(
        e_test, RV_test, s2_garch11, "GARCH(1,1)", alphas
    )
    models_metrics["Historical_Volatility"] = _compute_model_comprehensive_metrics(
        e_test, RV_test, s2_roll_var, "Historical_Volatility", alphas
    )
    models_metrics["Naive"] = _compute_model_comprehensive_metrics(
        e_test, RV_test, s2_naive, "Naive", alphas
    )
    models_metrics["ARCH(1)"] = _compute_model_comprehensive_metrics(
        e_test, RV_test, s2_arch1, "ARCH(1)", alphas
    )
    models_metrics["HAR(3)"] = _compute_model_comprehensive_metrics(
        e_test, RV_test, s2_har3, "HAR(3)", alphas
    )

    return models_metrics


def _build_comparison_table(models_metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build comparison table from models metrics.

    Args:
        models_metrics: Dictionary with metrics for all models.

    Returns:
        List of dictionaries representing comparison table rows.
    """
    comparison_table: list[dict[str, Any]] = []

    for model_name, metrics in models_metrics.items():
        row: dict[str, Any] = {"Model": model_name}
        row["QLIKE"] = metrics.get("qlike", float("nan"))
        row["MSE_var"] = metrics.get("mse_var", float("nan"))
        row["MAE_var"] = metrics.get("mae_var", float("nan"))

        # VaR metrics for alpha=0.01
        var_001 = metrics.get("var_backtests", {}).get("0.01", {})
        row["VaR_1%_hit_rate"] = var_001.get("hit_rate", float("nan"))
        row["VaR_1%_p_UC"] = var_001.get("p_uc", float("nan"))
        row["VaR_1%_p_IND"] = var_001.get("p_ind", float("nan"))
        row["VaR_1%_p_CC"] = var_001.get("p_cc", float("nan"))

        # VaR metrics for alpha=0.05
        var_005 = metrics.get("var_backtests", {}).get("0.05", {})
        row["VaR_5%_hit_rate"] = var_005.get("hit_rate", float("nan"))
        row["VaR_5%_p_UC"] = var_005.get("p_uc", float("nan"))
        row["VaR_5%_p_IND"] = var_005.get("p_ind", float("nan"))
        row["VaR_5%_p_CC"] = var_005.get("p_cc", float("nan"))

        # MZ metrics
        mz = metrics.get("mz", {})
        row["MZ_intercept"] = mz.get("intercept", float("nan"))
        row["MZ_slope"] = mz.get("slope", float("nan"))
        row["MZ_R2"] = mz.get("r2", float("nan"))

        comparison_table.append(row)

    return comparison_table


def _create_forecasts_dataframe(
    dates_test: np.ndarray,
    e_test: np.ndarray,
    sigma2_egarch: np.ndarray,
    s2_ewma: np.ndarray,
    s2_arch1: np.ndarray,
    s2_har3: np.ndarray,
    s2_roll_var: np.ndarray,
) -> pd.DataFrame:
    """Create forecasts DataFrame for plotting.

    Args:
        dates_test: Test dates.
        e_test: Test residuals.
        sigma2_egarch: EGARCH variance forecasts.
        s2_ewma: EWMA forecasts.
        s2_arch1: ARCH(1) forecasts.
        s2_har3: HAR(3) forecasts.
        s2_roll_var: Rolling variance forecasts.

    Returns:
        DataFrame with all forecasts.
    """
    return pd.DataFrame(
        {
            "date": dates_test,
            "e": e_test,
            "s2_arima_garch": sigma2_egarch,
            "s2_ewma": s2_ewma,
            "s2_arch1": s2_arch1,
            "s2_har3": s2_har3,
            "s2_roll_var": s2_roll_var,
            "s2_roll_std": np.sqrt(s2_roll_var),
        }
    )


def _generate_volatility_plot(
    dates_test: np.ndarray, e_test: np.ndarray, forecasts_df: pd.DataFrame
) -> None:
    """Generate and save volatility forecasts comparison plot.

    Args:
        dates_test: Test dates array.
        e_test: Test residuals.
        forecasts_df: DataFrame with all forecasts.
    """
    dates_array = dates_test if isinstance(dates_test, np.ndarray) else dates_test.to_numpy()
    try:
        plot_volatility_forecasts(dates_array, e_test, forecasts_df)
        logger.info(
            "Saved volatility forecasts comparison plot: %s", C.VOL_BACKTEST_VOLATILITY_PLOT
        )
    except Exception as ex:
        logger.warning("Failed to generate volatility forecasts plot: %s", ex)


def run_benchmark_section4(
    *,
    ewma_lambda: float | None = None,
    rolling_window: int | None = None,
    alphas: list[float] | None = None,
) -> dict[str, Any]:
    """Run benchmark comparison according to Section 4 methodology.

    Implements Section 4:
    - 4.1: Baselines (EWMA, GARCH(1,1), historical volatility, naive)
    - 4.2: Evaluation (QLIKE, MSE_var, MAE_var, VaR backtests, MZ regression)
    - 4.3: Comparison table

    Args:
        ewma_lambda: EWMA decay factor. Defaults to VOL_EWMA_LAMBDA_DEFAULT.
        rolling_window: Rolling window size. Defaults to VOL_ROLLING_WINDOW_DEFAULT.
        alphas: VaR alpha levels. Defaults to [0.01, 0.05].

    Returns:
        Dictionary with comprehensive metrics for all models.

    Raises:
        FileNotFoundError: If EGARCH forecasts are missing.
        ValueError: If parameters are invalid or data is insufficient.
    """
    np.random.seed(DEFAULT_RANDOM_STATE)

    ewma_lambda, rolling_window, alphas = _validate_and_set_defaults(
        ewma_lambda, rolling_window, alphas
    )

    egarch_forecasts, egarch_dist, egarch_nu, egarch_lambda_skew = (
        _load_egarch_forecasts_and_distribution()
    )

    resid_test, RV_test, sigma2_egarch, dates_test = _prepare_test_data_from_egarch_forecasts(
        egarch_forecasts
    )

    e_valid, pos_test = _prepare_full_residuals_data()

    (
        s2_ewma_all,
        s2_garch11_all,
        s2_roll_var_all,
        s2_naive_all,
        s2_arch1_all,
        s2_har3_all,
    ) = compute_baseline_forecasts(e_valid, pos_test, ewma_lambda, rolling_window)

    (
        s2_ewma,
        s2_garch11,
        s2_roll_var,
        s2_naive,
        s2_arch1,
        s2_har3,
        resid_test,
        RV_test,
        sigma2_egarch,
        dates_test,
    ) = _align_baseline_forecasts_to_egarch(
        len(egarch_forecasts),
        s2_ewma_all,
        s2_garch11_all,
        s2_roll_var_all,
        s2_naive_all,
        s2_arch1_all,
        s2_har3_all,
        resid_test,
        RV_test,
        sigma2_egarch,
        dates_test,
    )

    models_metrics = _compute_all_models_metrics(
        resid_test,
        RV_test,
        sigma2_egarch,
        s2_ewma,
        s2_garch11,
        s2_roll_var,
        s2_naive,
        s2_arch1,
        s2_har3,
        alphas,
        egarch_dist,
        egarch_nu,
        egarch_lambda_skew,
    )

    comparison_table = _build_comparison_table(models_metrics)

    forecasts_df = _create_forecasts_dataframe(
        dates_test, resid_test, sigma2_egarch, s2_ewma, s2_arch1, s2_har3, s2_roll_var
    )

    _generate_volatility_plot(dates_test, resid_test, forecasts_df)

    result: dict[str, Any] = {
        "n_test": int(resid_test.size),
        "models": models_metrics,
        "comparison_table": comparison_table,
        "methodology": "Section 4: Benchmarks and comparison",
    }

    logger.info("Benchmark comparison completed: %d models evaluated", len(models_metrics))
    return result


__all__ = [
    "run_vol_backtest",
    "save_vol_backtest_outputs",
    "run_rolling_garch_from_artifacts",
    "run_benchmark_section4",
]
