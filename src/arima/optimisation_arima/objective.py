"""Optuna objectives for SARIMA hyper-parameter search."""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd

from src.constants import (
    SARIMA_D_MAX,
    SARIMA_D_MIN,
    SARIMA_D_SEASONAL_MAX,
    SARIMA_D_SEASONAL_MIN,
    SARIMA_LJUNGBOX_P_VALUE_THRESHOLD,
    SARIMA_LJUNGBOX_PENALTY_WEIGHT,
    SARIMA_MIN_STATS_SAMPLES,
    SARIMA_NORMALIZATION_EPSILON,
    SARIMA_P_MAX,
    SARIMA_P_MIN,
    SARIMA_P_SEASONAL_MAX,
    SARIMA_P_SEASONAL_MIN,
    SARIMA_Q_MAX,
    SARIMA_Q_MIN,
    SARIMA_Q_SEASONAL_MAX,
    SARIMA_Q_SEASONAL_MIN,
    SARIMA_REFIT_EVERY_OPTIONS,
    SARIMA_S_ALLOWED_VALUES,
    SARIMA_TREND_ALLOWED_VALUES,
    SARIMA_VALIDATION_WEIGHT,
)
from src.utils import get_logger

from .model_evaluation import SarimaParams, evaluate_param_combination

logger = get_logger(__name__)


class NormalizationStatsCallback:
    """Optuna callback to track statistics for composite score normalization.

    This callback collects criterion (AIC/BIC) and validation RMSE values from
    completed trials to compute running mean and std for z-score normalization.

    Attributes:
        criterion: Name of the criterion being optimized ("aic" or "bic").
        criterion_values: List of criterion values from completed trials.
        rmse_values: List of validation RMSE values from completed trials.
    """

    def __init__(self, criterion: str = "aic") -> None:
        """Initialize the callback.

        Args:
            criterion: Criterion being optimized ("aic" or "bic").
        """
        self.criterion = criterion.lower()
        self.criterion_values: list[float] = []
        self.rmse_values: list[float] = []

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Collect statistics from completed trial.

        Args:
            study: Optuna study object.
            trial: Completed trial.
        """
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        # Extract criterion value from user attributes
        criterion_val = trial.user_attrs.get(self.criterion)
        if criterion_val is not None and isinstance(criterion_val, (int, float)):
            if not np.isinf(criterion_val):
                self.criterion_values.append(float(criterion_val))

        # Extract validation RMSE from user attributes
        val_rmse = trial.user_attrs.get("val_rmse")
        if val_rmse is not None and isinstance(val_rmse, (int, float)):
            if not np.isinf(val_rmse):
                self.rmse_values.append(float(val_rmse))

    def get_stats(self) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        """Compute current statistics from collected values.

        Returns:
            Tuple of ((criterion_mean, criterion_std), (rmse_mean, rmse_std)).
            Returns (None, None) if insufficient data.
        """
        criterion_stats = self._compute_stats(self.criterion_values)
        rmse_stats = self._compute_stats(self.rmse_values)
        return criterion_stats, rmse_stats

    def _compute_stats(self, values: list[float]) -> tuple[float, float] | None:
        """Compute mean and std from values if sufficient data available.

        Args:
            values: List of numeric values.

        Returns:
            Tuple of (mean, std) or None if insufficient data.
        """
        if len(values) < SARIMA_MIN_STATS_SAMPLES:
            return None
        return (float(np.mean(values)), float(np.std(values)))


def _suggest_params(trial: optuna.Trial) -> SarimaParams:
    """Suggest SARIMA parameters for an Optuna trial.

    Optimizes all hyperparameters including refit_every frequency, which controls
    how often the model is retrained during rolling forecast. Lower values (e.g., 10)
    mean more frequent refits (better adaptation but higher cost), while higher values
    (e.g., 50) mean less frequent refits (faster but potentially less adaptive).

    Args:
        trial: Optuna trial object.

    Returns:
        SarimaParams object with suggested parameters including trend and refit_every.
    """
    p = trial.suggest_int("p", SARIMA_P_MIN, SARIMA_P_MAX)
    d = trial.suggest_int("d", SARIMA_D_MIN, SARIMA_D_MAX)
    q = trial.suggest_int("q", SARIMA_Q_MIN, SARIMA_Q_MAX)
    s = int(trial.suggest_categorical("s", SARIMA_S_ALLOWED_VALUES))
    trend = str(trial.suggest_categorical("trend", SARIMA_TREND_ALLOWED_VALUES))
    if s > 1:
        P = trial.suggest_int("P", SARIMA_P_SEASONAL_MIN, SARIMA_P_SEASONAL_MAX)
        D = trial.suggest_int("D", SARIMA_D_SEASONAL_MIN, SARIMA_D_SEASONAL_MAX)
        Q = trial.suggest_int("Q", SARIMA_Q_SEASONAL_MIN, SARIMA_Q_SEASONAL_MAX)
    else:
        P = D = Q = 0

    # Optimize refit frequency (trade-off between adaptation and computational cost)
    refit_every = trial.suggest_categorical("refit_every", SARIMA_REFIT_EVERY_OPTIONS)

    return SarimaParams(p, d, q, P, D, Q, s, trend, refit_every)


def _store_evaluation_results(
    trial: optuna.Trial, res: dict, composite_score: float | None = None
) -> None:
    """Store evaluation results in trial user attributes.

    Args:
        trial: Optuna trial object.
        res: Evaluation results dictionary.
        composite_score: Final composite score (lower is better) used for optimization.
    """
    if "error" not in res:
        trial.set_user_attr("aic", res.get("aic", float("inf")))
        trial.set_user_attr("bic", res.get("bic", float("inf")))
        trial.set_user_attr("lb_stat", res.get("lb_stat", float("nan")))
        trial.set_user_attr("lb_pvalue", res.get("lb_pvalue", float("nan")))
        # Store composite score if provided (this is what Optuna minimizes)
        if composite_score is not None:
            trial.set_user_attr("composite_score", composite_score)
        # Store validation metrics if available
        if "val_rmse" in res:
            trial.set_user_attr("val_rmse", res.get("val_rmse", float("inf")))
            trial.set_user_attr("val_mae", res.get("val_mae", float("inf")))
            trial.set_user_attr("val_mean_error", res.get("val_mean_error", 0.0))
    else:
        trial.set_user_attr("error", res.get("error", "Unknown error"))


def _normalize_z_score(value: float, mean: float, std: float) -> float:
    """Normalize a value using z-score.

    Args:
        value: Value to normalize.
        mean: Mean for normalization.
        std: Standard deviation for normalization.

    Returns:
        Z-score normalized value.
    """
    return (value - mean) / (std + SARIMA_NORMALIZATION_EPSILON)


def _compute_normalized_composite(
    criterion_value: float,
    val_rmse: float,
    weight: float,
    criterion_stats: tuple[float, float],
    rmse_stats: tuple[float, float],
) -> float:
    """Compute normalized composite score.

    Args:
        criterion_value: AIC or BIC value.
        val_rmse: Validation RMSE.
        weight: Weight for validation RMSE.
        criterion_stats: (mean, std) for criterion normalization.
        rmse_stats: (mean, std) for RMSE normalization.

    Returns:
        Normalized composite score.
    """
    criterion_mean, criterion_std = criterion_stats
    rmse_mean, rmse_std = rmse_stats

    norm_criterion = _normalize_z_score(criterion_value, criterion_mean, criterion_std)
    norm_rmse = _normalize_z_score(val_rmse, rmse_mean, rmse_std)

    composite = (1.0 - weight) * norm_criterion + weight * norm_rmse
    logger.debug(
        f"Normalized composite: criterion={norm_criterion:.4f} (raw={criterion_value:.4f}), "
        f"rmse={norm_rmse:.4f} (raw={val_rmse:.4f}), composite={composite:.4f}"
    )
    return float(composite)


def _compute_composite_score(
    criterion_value: float,
    val_rmse: float | None,
    weight: float = SARIMA_VALIDATION_WEIGHT,
    criterion_stats: tuple[float, float] | None = None,
    rmse_stats: tuple[float, float] | None = None,
) -> float:
    """Compute composite score combining information criterion and validation RMSE.

    Args:
        criterion_value: AIC or BIC value.
        val_rmse: Validation RMSE. If None, only criterion is used.
        weight: Weight for validation RMSE (0-1).
        criterion_stats: Optional (mean, std) for criterion normalization.
        rmse_stats: Optional (mean, std) for RMSE normalization.

    Returns:
        Composite score (lower is better).
    """
    if val_rmse is None or not isinstance(val_rmse, (int, float)) or val_rmse == float("inf"):
        return float(criterion_value)

    if criterion_stats is not None and rmse_stats is not None:
        return _compute_normalized_composite(
            criterion_value, val_rmse, weight, criterion_stats, rmse_stats
        )

    # Fallback without normalization (NOT RECOMMENDED)
    logger.warning(
        "Computing composite score WITHOUT normalization. "
        "Criterion and RMSE are on different scales. "
        "Provide criterion_stats and rmse_stats for proper normalization."
    )
    return float((1.0 - weight) * criterion_value + weight * val_rmse)


def _log_trial_parameters(trial: optuna.Trial, params: SarimaParams) -> None:
    """Log the parameters suggested for this trial.

    Args:
        trial: Optuna trial object.
        params: SARIMA parameters for this trial.
    """
    logger.info(
        "Trial %d | SARIMA(%d,%d,%d)(%d,%d,%d)[%d], trend=%s, refit_every=%d",
        trial.number,
        params.p,
        params.d,
        params.q,
        params.P,
        params.D,
        params.Q,
        params.s,
        params.trend,
        params.refit_every,
    )


def _handle_evaluation_error(trial: optuna.Trial, res: dict) -> float | None:
    """Handle evaluation errors and return infinity if evaluation failed.

    Args:
        trial: Optuna trial object.
        res: Evaluation results dictionary.

    Returns:
        Infinity if error occurred, None otherwise.
    """
    if "error" in res:
        logger.warning("Trial %d | evaluation error: %s", trial.number, res["error"])
        return float("inf")
    return None


def _extract_criterion_value(res: dict, criterion: str) -> float | None:
    """Extract and validate the criterion value from evaluation results.

    Args:
        res: Evaluation results dictionary.
        criterion: Criterion name ("aic" or "bic").

    Returns:
        Valid criterion value or None if invalid.
    """
    criterion_value = res.get(criterion, float("inf"))
    if not isinstance(criterion_value, (int, float)) or not np.isfinite(criterion_value):
        return None
    return float(criterion_value)


def _compute_ljung_box_penalty(trial: optuna.Trial, res: dict) -> float:
    """Compute Ljung-Box penalty based on p-value.

    Args:
        trial: Optuna trial object.
        res: Evaluation results dictionary.

    Returns:
        Penalty value (0.0 if no penalty needed).
    """
    ljung_penalty = 0.0
    lb_pvalue_raw = res.get("lb_pvalue", None)
    if isinstance(lb_pvalue_raw, (int, float)):
        lb_pvalue = float(lb_pvalue_raw)
        if np.isfinite(lb_pvalue):
            if lb_pvalue < SARIMA_LJUNGBOX_P_VALUE_THRESHOLD:
                deficit = SARIMA_LJUNGBOX_P_VALUE_THRESHOLD - lb_pvalue
                ljung_penalty = float(SARIMA_LJUNGBOX_PENALTY_WEIGHT) * float(deficit)
                logger.info(
                    "Trial %d | Ljung-Box p=%.4f < %.4f → penalty=%.4f",
                    trial.number,
                    lb_pvalue,
                    SARIMA_LJUNGBOX_P_VALUE_THRESHOLD,
                    ljung_penalty,
                )
            else:
                # Explicitly log p-value even without penalty (traceability)
                logger.info(
                    "Trial %d | Ljung-Box p=%.4f ≥ %.4f → penalty=0.0000",
                    trial.number,
                    lb_pvalue,
                    SARIMA_LJUNGBOX_P_VALUE_THRESHOLD,
                )
    return ljung_penalty


def _extract_validation_metrics(res: dict) -> float | None:
    """Extract validation RMSE from evaluation results.

    Args:
        res: Evaluation results dictionary.

    Returns:
        Validation RMSE value or None if not available/invalid.
    """
    val_rmse_raw = res.get("val_rmse", float("inf"))
    if isinstance(val_rmse_raw, (int, float)):
        return float(val_rmse_raw)
    return None


def _bootstrap_normalization_stats(
    criterion_value: float,
    val_rmse: float | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bootstrap normalization statistics when insufficient data available.

    Args:
        criterion_value: Current criterion value.
        val_rmse: Current validation RMSE.

    Returns:
        Tuple of (criterion_stats, rmse_stats) for bootstrapping.
    """
    criterion_stats = (criterion_value, 0.0)
    rmse_anchor = float(val_rmse) if isinstance(val_rmse, float) else 0.0
    rmse_stats = (rmse_anchor, 0.0)
    return criterion_stats, rmse_stats


def _compute_validation_score(
    trial: optuna.Trial,
    criterion_value: float,
    val_rmse: float | None,
    stats_callback: NormalizationStatsCallback | None,
) -> float:
    """Compute composite score with validation metrics.

    Args:
        trial: Optuna trial object.
        criterion_value: Criterion value (AIC/BIC).
        val_rmse: Validation RMSE.
        stats_callback: Optional callback for normalization statistics.

    Returns:
        Composite score including validation.
    """
    criterion_stats, rmse_stats = (None, None)
    if stats_callback is not None:
        criterion_stats, rmse_stats = stats_callback.get_stats()

    # Bootstrap stats if insufficient data
    if criterion_stats is None or rmse_stats is None:
        criterion_stats, rmse_stats = _bootstrap_normalization_stats(criterion_value, val_rmse)
        logger.info(
            "Trial %d | Normalization bootstrap: criterion_mean=%.4f, rmse_mean=%.6f",
            trial.number,
            criterion_stats[0],
            rmse_stats[0],
        )

    return _compute_composite_score(
        criterion_value,
        val_rmse,
        weight=SARIMA_VALIDATION_WEIGHT,
        criterion_stats=criterion_stats,
        rmse_stats=rmse_stats,
    )


def _log_final_score(
    trial: optuna.Trial,
    criterion: str,
    score: float,
    base_score: float | None,
    ljung_penalty: float,
) -> None:
    """Log the final score for this trial.

    Args:
        trial: Optuna trial object.
        criterion: Criterion name ("aic" or "bic").
        score: Final composite score.
        base_score: Base score before Ljung-Box penalty (None for simple scores).
        ljung_penalty: Ljung-Box penalty applied.
    """
    if base_score is not None:
        logger.info(
            "Trial %d | composite=% .4f (base=%.4f, ljung_penalty=%.4f)",
            trial.number,
            score,
            base_score,
            ljung_penalty,
        )
    else:
        logger.info(
            "Trial %d | %s=%.4f, ljung_penalty=%.4f, total=%.4f",
            trial.number,
            criterion.upper(),
            score - ljung_penalty,  # Remove penalty to show base criterion
            ljung_penalty,
            score,
        )


def _evaluate_and_store(
    trial: optuna.Trial,
    train_series: pd.Series,
    criterion: str,
    backtest_cfg: dict[str, int] | None = None,
    stats_callback: NormalizationStatsCallback | None = None,
) -> float:
    """Generic objective for AIC/BIC with optional validation and Ljung–Box penalty.

    Uses a soft penalty on Ljung–Box p-value instead of hard rejection to keep
    informative trials while discouraging autocorrelated residuals. Returns +inf
    only when evaluation clearly fails (non-numeric/invalid criterion).

    Args:
        trial: Optuna trial object.
        train_series: Training time series data.
        criterion: Criterion to optimize ("aic" or "bic").
        backtest_cfg: Optional backtest configuration for validation metrics.
        stats_callback: Optional callback to accumulate stats for normalization.

    Returns:
        Composite objective value (lower is better).
    """
    params = _suggest_params(trial)
    _log_trial_parameters(trial, params)

    res = evaluate_param_combination(train_series, params, backtest_cfg=backtest_cfg)

    # Handle evaluation errors
    error_score = _handle_evaluation_error(trial, res)
    if error_score is not None:
        return error_score

    # Extract and validate criterion value
    criterion_value = _extract_criterion_value(res, criterion)
    if criterion_value is None:
        return float("inf")

    # Compute Ljung-Box penalty
    ljung_penalty = _compute_ljung_box_penalty(trial, res)

    # Compute final score
    if backtest_cfg is not None and "val_rmse" in res:
        val_rmse = _extract_validation_metrics(res)
        base_score = _compute_validation_score(trial, criterion_value, val_rmse, stats_callback)
        score = float(base_score) + float(ljung_penalty)
        _log_final_score(trial, criterion, score, base_score, ljung_penalty)
        # Store the composite score that was actually optimized
        _store_evaluation_results(trial, res, score)
        return float(score)

    # Simple score without validation
    score = criterion_value + ljung_penalty
    _log_final_score(trial, criterion, score, None, ljung_penalty)
    # Store the score that was actually optimized
    _store_evaluation_results(trial, res, score)
    return float(score)


def objective_aic(
    trial: optuna.Trial,
    train_series: pd.Series,
    backtest_cfg: dict[str, int] | None = None,
    stats_callback: NormalizationStatsCallback | None = None,
) -> float:
    """Optuna objective function minimizing AIC or composite score (AIC + validation).

    AIC (Akaike Information Criterion) is used as the sole optimization criterion because:
    1. AIC minimizes one-step-ahead prediction error (asymptotically optimal for forecasting)
    2. BIC minimizes description length (better for model selection, not forecasting)
    3. For financial time series forecasting, AIC is theoretically superior

    Reference: Burnham & Anderson (2004), "Multimodel Inference: Understanding AIC and BIC
    in Model Selection"

    Args:
        trial: Optuna trial object.
        train_series: Training time series data.
        backtest_cfg: Optional backtest configuration for validation metrics.
            If provided, minimizes composite score (AIC + validation RMSE).
        stats_callback: Optional callback for collecting normalization statistics.

    Returns:
        AIC value or composite score for the suggested parameters.
        Returns infinity if evaluation fails.
    """
    return _evaluate_and_store(trial, train_series, "aic", backtest_cfg, stats_callback)
