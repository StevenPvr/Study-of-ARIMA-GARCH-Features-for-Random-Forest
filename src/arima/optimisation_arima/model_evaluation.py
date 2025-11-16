"""Model evaluation for SARIMA (fit, diagnostics, walk-forward backtest)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, cast

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.constants import (
    SARIMA_ADF_ALPHA_DEFAULT,
    SARIMA_LJUNGBOX_LAGS_DEFAULT,
    SARIMA_MIN_SERIES_LENGTH_DIFFERENCED,
    SARIMA_MIN_SERIES_LENGTH_STATIONARITY,
)
from src.utils import get_logger

logger = get_logger(__name__)

# Suppress common statsmodels warnings during optimization
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels.tsa.statespace.sarimax")
warnings.filterwarnings("ignore", message=".*Non-invertible starting MA parameters.*")
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed to converge.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")

# Type alias for fitted SARIMA model (SARIMAXResults)
FittedSARIMAModel = Any  # statsmodels.tsa.statespace.sarimax.SARIMAXResults


@dataclass(frozen=True)
class SarimaParams:
    """SARIMA model parameters.

    Attributes:
        p: AR order (non-negative integer).
        d: Differencing order (non-negative integer).
        q: MA order (non-negative integer).
        P: Seasonal AR order (non-negative integer).
        D: Seasonal differencing order (non-negative integer).
        Q: Seasonal MA order (non-negative integer).
        s: Seasonal period (non-negative integer).
        trend: Trend component ("n", "c", "t", or "ct").
        refit_every: Frequency of model refitting during walk-forward backtest (must be >= 1).
    """

    p: int
    d: int
    q: int
    P: int
    D: int
    Q: int
    s: int
    trend: str
    refit_every: int


def _check_series_length_for_validation(y: pd.Series) -> bool:
    """Check if series is long enough for stationarity validation.

    Args:
        y: Time series data.

    Returns:
        True if series is long enough for validation.
    """
    if len(y) < SARIMA_MIN_SERIES_LENGTH_STATIONARITY:
        logger.debug("Series too short for stationarity validation (n=%d)", len(y))
        return False
    return True


def _apply_differencing(y: pd.Series, d: int, d_seasonal: int, s: int) -> pd.Series:
    """Apply differencing to time series.

    Args:
        y: Time series data.
        d: Non-seasonal differencing order.
        d_seasonal: Seasonal differencing order.
        s: Seasonal period.

    Returns:
        Differenced series.
    """
    # Apply non-seasonal differencing
    diff_y = y.diff(d).dropna()

    # Apply seasonal differencing if specified
    if d_seasonal > 0 and s > 1:
        seasonal_diff = cast(pd.Series, diff_y.diff(s * d_seasonal))
        diff_y = seasonal_diff.dropna()  # type: ignore[unreachable]

    return diff_y


def _test_stationarity(y: pd.Series, alpha: float) -> tuple[float, float, Any]:
    """Test stationarity of time series using ADF test.

    Args:
        y: Time series data to test.
        alpha: Significance level for ADF test.

    Returns:
        Tuple of (p_value, test_statistic, critical_values).
    """
    adf_result = adfuller(y.dropna(), autolag="AIC")
    adf_pvalue = float(adf_result[1])
    adf_stat = float(adf_result[0])  # Ensure it's a float
    critical_values = adf_result[4] if len(adf_result) > 4 else "N/A"
    return adf_pvalue, adf_stat, critical_values


def _log_stationarity_result(
    p_value: float,
    alpha: float,
    test_stat: float,
    critical_values: Any,
    context: str,
) -> None:
    """Log the result of stationarity test.

    Args:
        p_value: ADF p-value.
        alpha: Significance level.
        test_stat: ADF test statistic.
        critical_values: Critical values from ADF test.
        context: Context description for logging.
    """
    if p_value > alpha:
        logger.warning(
            "Series may be non-stationary (ADF p-value=%.4f > %.2f) %s. "
            "Consider using d=1 for better model fit. "
            "Current ADF statistic: %.4f, critical values: %s",
            p_value,
            alpha,
            context,
            test_stat,
            critical_values,
        )
    else:
        logger.debug(
            "Series appears stationary (ADF p-value=%.4f <= %.2f) %s.",
            p_value,
            alpha,
            context,
        )


def _validate_differencing(
    y: pd.Series,
    d: int,
    d_seasonal: int = 0,
    s: int = 0,
    alpha: float = SARIMA_ADF_ALPHA_DEFAULT,
) -> None:
    """Validate that differencing order matches data stationarity properties.

    This function performs ADF (Augmented Dickey-Fuller) tests to verify that:
    1. If d=0, the series should be stationary (otherwise d should be increased)
    2. If d>0, the differenced series should be stationary

    Note: This is a diagnostic function that logs warnings but does not raise errors.
    The optimization process may still select models with suboptimal d values if they
    minimize the information criterion.

    Args:
        y: Time series data.
        d: Non-seasonal differencing order.
        d_seasonal: Seasonal differencing order (default: 0).
        s: Seasonal period (required if d_seasonal > 0).
        alpha: Significance level for ADF test.

    Warnings:
        Logs warnings if differencing order appears inappropriate based on ADF test.
    """
    if not _check_series_length_for_validation(y):
        return

    try:
        # Test 1: If d=0, check if series is stationary
        if d == 0 and d_seasonal == 0:
            adf_pvalue, adf_stat, critical_values = _test_stationarity(y, alpha)
            context = "but d=0 specified"
            _log_stationarity_result(adf_pvalue, alpha, adf_stat, critical_values, context)

        # Test 2: If d>0, check if differenced series is stationary
        elif d > 0:
            diff_y = _apply_differencing(y, d, d_seasonal, s)

            if len(diff_y) > SARIMA_MIN_SERIES_LENGTH_DIFFERENCED:
                adf_pvalue, adf_stat, critical_values = _test_stationarity(diff_y, alpha)
                context = f"after differencing (d={d}, D={d_seasonal}, s={s})"
                _log_stationarity_result(adf_pvalue, alpha, adf_stat, critical_values, context)
            else:
                logger.debug(
                    "Differenced series too short for validation (n=%d after differencing)",
                    len(diff_y),
                )

    except Exception as e:
        logger.debug("Stationarity validation failed: %s", str(e))


def _configure_warnings_for_fitting() -> None:
    """Configure warnings to suppress during SARIMA model fitting."""
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
    warnings.filterwarnings("ignore", message=".*Non-invertible starting MA parameters.*")
    warnings.filterwarnings(
        "ignore", message=".*Maximum Likelihood optimization failed to converge.*"
    )
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def _create_sarima_model(
    y: pd.Series,
    params: SarimaParams,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
) -> SARIMAX:
    """Create SARIMA model with given parameters.

    Args:
        y: Time series data to fit.
        params: SARIMA parameters.
        enforce_stationarity: Whether to enforce stationarity constraints.
        enforce_invertibility: Whether to enforce invertibility constraints.

    Returns:
        SARIMAX model instance.
    """
    return SARIMAX(
        y,
        order=(params.p, params.d, params.q),
        seasonal_order=(params.P, params.D, params.Q, params.s),
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        trend=params.trend,
    )


def _fit_model_with_powell(model: SARIMAX, disp: bool) -> FittedSARIMAModel:
    """Fit SARIMA model using Powell optimization method.

    Args:
        model: SARIMAX model to fit.
        disp: Whether to display optimization output.

    Returns:
        Fitted model results.

    Raises:
        RuntimeError: If Powell optimization fails.
    """
    try:
        return model.fit(method="powell", disp=disp, maxiter=200)
    except Exception as e:
        logger.error("Powell optimization method failed: %s", str(e))
        raise RuntimeError(
            "Powell optimization failed for SARIMA model. This ensures reproducibility "
            "but may require different parameter values."
        ) from e


def _check_model_convergence(res: FittedSARIMAModel, params: SarimaParams) -> None:
    """Check if fitted model converged and log warning if not.

    Args:
        res: Fitted model results.
        params: SARIMA parameters used for fitting.
    """
    if hasattr(res, "mle_retvals"):
        converged = res.mle_retvals.get("converged", False)  # type: ignore
        if not converged:
            logger.warning(
                "SARIMA model (p=%d,d=%d,q=%d)(P=%d,D=%d,Q=%d)[%d] did not converge. "
                "AIC/BIC values may be unreliable. Consider different parameter values.",
                params.p,
                params.d,
                params.q,
                params.P,
                params.D,
                params.Q,
                params.s,
            )


def _fit_sarima(
    y: pd.Series,
    params: SarimaParams,
    enforce_stationarity: bool = False,  # Disabled for stationary log-returns
    enforce_invertibility: bool = True,
    disp: bool = False,
    validate_stationarity: bool = True,
) -> FittedSARIMAModel:
    """Fit a SARIMA model with given parameters.

    Optionally validates that the differencing order matches data stationarity
    properties before fitting. This helps identify potential issues with model
    specification early in the process.

    Args:
        y: Time series data to fit.
        params: SARIMA parameters.
        enforce_stationarity: Whether to enforce stationarity constraints.
        enforce_invertibility: Whether to enforce invertibility constraints.
        disp: Whether to display optimization output.
        validate_stationarity: Whether to validate differencing order (default: True).
            Logs warnings if d appears inappropriate but does not prevent fitting.

    Returns:
        Fitted SARIMA model (SARIMAXResults object).
    """
    # Validate differencing order if requested
    if validate_stationarity:
        _validate_differencing(y, params.d, params.D, params.s)

    # Suppress warnings during model fitting
    with warnings.catch_warnings():
        _configure_warnings_for_fitting()

        model = _create_sarima_model(y, params, enforce_stationarity, enforce_invertibility)
        res = _fit_model_with_powell(model, disp)

    # Check convergence after fitting
    _check_model_convergence(res, params)

    return res


def ljung_box_on_residuals(
    residuals: np.ndarray, lags: int = SARIMA_LJUNGBOX_LAGS_DEFAULT
) -> Dict[str, float]:
    """Perform Ljung-Box test for residual autocorrelation.

    Args:
        residuals: Model residuals to test.
        lags: Number of lags to test.

    Returns:
        Dictionary with Ljung-Box statistic and p-value.
    """
    max_lags = max(1, len(residuals) // 10)
    effective_lags = min(lags, max_lags)
    lb = acorr_ljungbox(residuals, lags=[effective_lags], return_df=True)
    return {"lb_stat": float(lb["lb_stat"].iloc[0]), "lb_pvalue": float(lb["lb_pvalue"].iloc[0])}


def _validate_backtest_inputs(y: pd.Series, n_splits: int, test_size: int) -> int:
    """Validate backtest inputs and return training end index.

    Args:
        y: Time series data.
        n_splits: Number of splits for backtesting.
        test_size: Size of each test set.

    Returns:
        Training end index.

    Raises:
        ValueError: If inputs are invalid.
    """
    if len(y) == 0:
        raise ValueError("Time series is empty.")
    if n_splits <= 0:
        raise ValueError("n_splits must be positive.")
    if test_size <= 0:
        raise ValueError("test_size must be positive.")

    train_end = len(y) - n_splits * test_size
    if train_end <= 0:
        raise ValueError("Training segment is empty; reduce n_splits or test_size.")

    return train_end


def _prepare_time_series(y: pd.Series) -> pd.Series:
    """Prepare time series for backtesting.

    Args:
        y: Input time series.

    Returns:
        Prepared time series with RangeIndex.
    """
    y_prepared = y.copy()
    y_prepared.index = pd.RangeIndex(len(y_prepared))
    return y_prepared


def _predict_single_split(
    y: pd.Series,
    params: SarimaParams,
    test_start: int,
    test_end: int,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
) -> tuple[list[float], list[float]]:
    """Make predictions for a single backtest split.

    Args:
        y: Prepared time series data.
        params: SARIMA parameters.
        test_start: Start index of test split.
        test_end: End index of test split.
        enforce_stationarity: Whether to enforce stationarity constraints.
        enforce_invertibility: Whether to enforce invertibility constraints.

    Returns:
        Tuple of (predictions, actuals) for this split.

    Raises:
        ValueError: If insufficient training data.
    """
    predictions: list[float] = []
    actuals: list[float] = []

    for offset in range(0, test_end - test_start, max(1, params.refit_every)):
        block_start = test_start + offset
        block_end = min(block_start + params.refit_every, test_end)

        y_train = y.iloc[:block_start]
        if len(y_train) <= 0:
            raise ValueError("Insufficient training data before first validation block.")

        # Fit model for this block
        fitted = _fit_sarima(
            y_train,
            params,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )

        # Make predictions for this block
        predictions_block, actuals_block = _predict_block(y, fitted, block_start, block_end)
        predictions.extend(predictions_block)
        actuals.extend(actuals_block)

    return predictions, actuals


def _predict_block(
    y: pd.Series, fitted: Any, block_start: int, block_end: int
) -> tuple[list[float], list[float]]:
    """Make true walk-forward one-step-ahead predictions efficiently.

    Uses statsmodels' apply() method to compute all one-step-ahead forecasts
    for the block in a single vectorized call. This is much faster than
    calling forecast(1) + append() in a loop while maintaining true
    walk-forward behavior where each forecast uses only prior observations.

    Args:
        y: Time series data.
        fitted: Fitted SARIMA model.
        block_start: Start index of block.
        block_end: End index of block.

    Returns:
        Tuple of (predictions, actuals) for this block.
    """
    predictions: list[float] = []
    actuals: list[float] = []

    # Get the block of new observations to predict
    block_size = min(block_end - block_start, len(y) - block_start)
    if block_size <= 0:
        return predictions, actuals

    # Get new observations for this block (as array, not Series, to avoid index issues)
    new_obs = y.iloc[block_start : block_start + block_size].values

    # Apply the new observations to update filter state and get one-step-ahead forecasts
    # This is much faster than a loop of forecast(1) + append(refit=False)
    applied_results = fitted.apply(new_obs)

    # Extract one-step-ahead forecasts from the filter
    # forecasts is shape (1, n_obs) where each column is the forecast for that observation
    forecasts = applied_results.forecasts[0, :]

    # Store predictions and actuals
    for i in range(block_size):
        predictions.append(float(forecasts[i]))
        actuals.append(float(y.iloc[block_start + i]))

    return predictions, actuals


def _calculate_validation_metrics(
    predictions: list[float], actuals: list[float]
) -> Dict[str, float]:
    """Calculate validation metrics from predictions and actuals.

    Args:
        predictions: List of predicted values.
        actuals: List of actual values.

    Returns:
        Dictionary with validation metrics.
    """
    if len(predictions) == 0:
        return {"rmse": float("inf"), "mae": float("inf"), "mean_error": float("inf")}

    pred_array = np.array(predictions)
    actual_array = np.array(actuals)
    errors = actual_array - pred_array

    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    mean_error = float(np.mean(errors))

    return {"rmse": rmse, "mae": mae, "mean_error": mean_error}


def walk_forward_backtest(
    y: pd.Series,
    params: SarimaParams,
    n_splits: int,
    test_size: int,
    enforce_stationarity: bool = False,  # Disabled for stationary log-returns
    enforce_invertibility: bool = True,
) -> Dict[str, float]:
    """Perform strict time-aware walk-forward backtest with periodic refits.

    This function is used for validation during optimization and returns
    forecast accuracy metrics (RMSE, MAE) for model selection.

    Args:
        y: Full time series data (training data only - no test data leakage).
        params: SARIMA parameters to use (includes refit_every).
        n_splits: Number of time splits for backtesting.
        test_size: Size of each test set.
        enforce_stationarity: Whether to enforce stationarity constraints.
        enforce_invertibility: Whether to enforce invertibility constraints.

    Returns:
        Dictionary with validation metrics: rmse, mae, mean_error.

    Raises:
        ValueError: If training segment is empty or insufficient data.
    """
    # Validate inputs and prepare data
    train_end = _validate_backtest_inputs(y, n_splits, test_size)
    y_prepared = _prepare_time_series(y)

    # Collect all predictions and actuals across splits
    all_predictions: list[float] = []
    all_actuals: list[float] = []

    for split in range(n_splits):
        test_start = train_end + split * test_size
        test_end = test_start + test_size

        split_predictions, split_actuals = _predict_single_split(
            y_prepared,
            params,
            test_start,
            test_end,
            enforce_stationarity,
            enforce_invertibility,
        )
        all_predictions.extend(split_predictions)
        all_actuals.extend(split_actuals)

    # Calculate and return validation metrics
    return _calculate_validation_metrics(all_predictions, all_actuals)


def _extract_model_metrics(fitted: Any, params: SarimaParams) -> Dict[str, object]:
    """Extract metrics from fitted SARIMA model.

    Args:
        fitted: Fitted SARIMA model.
        params: SARIMA parameters used for fitting.

    Returns:
        Dictionary with model metrics (params, aic, bic, lb_stat, lb_pvalue).
    """
    aic = float(getattr(fitted, "aic", np.inf))
    bic = float(getattr(fitted, "bic", np.inf))
    resid = np.asarray(fitted.resid.dropna())
    diag = ljung_box_on_residuals(resid)
    return {
        "params": params.__dict__,
        "aic": aic,
        "bic": bic,
        "lb_stat": diag["lb_stat"],
        "lb_pvalue": diag["lb_pvalue"],
    }


def _add_validation_metrics(
    result: Dict[str, object],
    y: pd.Series,
    params: SarimaParams,
    backtest_cfg: Dict[str, int],
    enforce_stationarity: bool,
    enforce_invertibility: bool,
) -> None:
    """Add validation metrics to result dictionary.

    Args:
        result: Result dictionary to update.
        y: Time series data.
        params: SARIMA parameters.
        backtest_cfg: Backtest configuration.
        enforce_stationarity: Whether to enforce stationarity constraints.
        enforce_invertibility: Whether to enforce invertibility constraints.
    """
    validation_metrics = walk_forward_backtest(
        y,
        params,
        n_splits=backtest_cfg["n_splits"],
        test_size=backtest_cfg["test_size"],
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
    )
    result["val_rmse"] = validation_metrics.get("rmse", float("inf"))
    result["val_mae"] = validation_metrics.get("mae", float("inf"))
    result["val_mean_error"] = validation_metrics.get("mean_error", 0.0)


def evaluate_param_combination(
    y: pd.Series,
    params: SarimaParams,
    backtest_cfg: Optional[Dict[str, int]] = None,
    enforce_stationarity: bool = False,  # Disabled for stationary log-returns
    enforce_invertibility: bool = True,
) -> Dict[str, object]:
    """Evaluate a single SARIMA parameter combination.

    Args:
        y: Time series data.
        params: SARIMA parameters to evaluate.
        backtest_cfg: Optional backtest configuration for validation metrics.
        enforce_stationarity: Whether to enforce stationarity constraints.
        enforce_invertibility: Whether to enforce invertibility constraints.

    Returns:
        Dictionary with params, aic, bic, lb_stat, lb_pvalue, and optionally
        validation metrics. Returns error dict if fitting fails.
    """
    try:
        fitted = _fit_sarima(
            y,
            params,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )
        result = _extract_model_metrics(fitted, params)

        if backtest_cfg:
            _add_validation_metrics(
                result, y, params, backtest_cfg, enforce_stationarity, enforce_invertibility
            )

        return result
    except Exception as e:
        return {"params": params.__dict__, "error": str(e)}


def evaluate_param_grid(
    y: pd.Series,
    grid: Iterable[SarimaParams],
    enforce_stationarity: bool = False,  # Disabled for stationary log-returns
    enforce_invertibility: bool = True,
) -> List[Dict[str, object]]:
    """Evaluate multiple SARIMA parameter combinations sequentially.

    Used to re-evaluate top Optuna candidates with complete diagnostics.
    Backtest is not performed during re-evaluation (only model fitting and diagnostics).

    Args:
        y: Time series data.
        grid: Iterable of SARIMA parameter combinations to evaluate.
        enforce_stationarity: Whether to enforce stationarity constraints.
        enforce_invertibility: Whether to enforce invertibility constraints.

    Returns:
        List of evaluation result dictionaries, one per parameter combination.
    """
    grid_list = list(grid)
    if not grid_list:
        return []
    results: List[Dict[str, object]] = []
    for p in grid_list:
        # Backtest is disabled during re-evaluation (backtest_cfg=None)
        # Re-evaluation is only for getting complete diagnostics (AIC, BIC, Ljung-Box)
        results.append(
            evaluate_param_combination(
                y,
                p,
                backtest_cfg=None,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility,
            )
        )
    return results
