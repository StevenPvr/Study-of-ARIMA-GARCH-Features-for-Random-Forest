"""Model comparison and confidence set evaluation for volatility forecasting.

This module implements state-of-the-art model comparison techniques:
1. Systematic RV benchmark comparison
2. Diebold-Mariano testing for model comparison
3. Automatic MZ calibration
4. Model Confidence Set (MCS) implementation (Hansen et al. 2011)
5. Rolling window evaluation for stability analysis

Academic References:
    - Hansen, Lunde & Nason (2011): "The Model Confidence Set"
    - Diebold & Mariano (1995): "Comparing predictive accuracy"
    - Mincer & Zarnowitz (1969): "The evaluation of economic forecasts"

Author: Steven
Date: November 2024
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.constants import GARCH_EVALUATION_DIR, GARCH_FORECASTS_FILE
from src.garch.benchmark.realized_volatility import compute_realized_measures
from src.garch.benchmark.statistical_tests import diebold_mariano_test
from src.garch.garch_eval.metrics import (
    apply_mz_calibration,
    mincer_zarnowitz,
    mse_mae_variance,
    qlike_loss,
)
from src.utils import get_logger, load_dataframe, validate_file_exists

logger = get_logger(__name__)


class ModelComparisonEvaluator:
    """Model comparison evaluator with RV benchmark and statistical tests.

    Provides comprehensive tools for comparing volatility forecasting models:
    - Model Confidence Set (MCS) for identifying superior models
    - Diebold-Mariano tests for pairwise comparisons
    - Realized volatility benchmarking
    - Mincer-Zarnowitz calibration for bias correction
    - Rolling window stability analysis
    """

    def __init__(
        self,
        forecasts_df: pd.DataFrame,
        hloc_data: pd.DataFrame | None = None,
        apply_mz_cal: bool = True,
        rv_estimator: str = "yang_zhang",
    ):
        """Initialize model comparison evaluator.

        Args:
            forecasts_df: DataFrame with GARCH forecasts (date, resid, sigma2_egarch_raw)
            hloc_data: DataFrame with High, Low, Open, Close prices for RV computation
            apply_mz_cal: Whether to apply MZ calibration automatically
            rv_estimator: Which RV estimator to use ('yang_zhang', 'garman_klass', etc.)
        """
        self.forecasts = forecasts_df.copy()
        self.hloc_data = hloc_data
        self.apply_mz_cal = apply_mz_cal
        self.rv_estimator = rv_estimator

        # Compute realized variance if HLOC data provided
        if hloc_data is not None:
            self.realized_measures = self._compute_rv_measures()
        else:
            self.realized_measures = None

        # Apply MZ calibration if requested
        if apply_mz_cal:
            self._apply_mz_calibration()

    def _compute_rv_measures(self) -> pd.DataFrame | None:
        """Compute realized volatility measures from HLOC data."""
        if self.hloc_data is None:
            return None

        logger.info("Computing realized volatility measures...")
        rv_measures = compute_realized_measures(
            self.hloc_data,
            high_col="High",
            low_col="Low",
            close_col="Close",
            open_col="Open",
        )

        # Select the appropriate RV estimator
        rv_map = {
            "yang_zhang": "YangZhang",
            "garman_klass": "GarmanKlass",
            "rogers_satchell": "RogersSatchell",
            "parkinson": "Parkinson",
            "classical": "RV",
        }

        rv_col = rv_map.get(self.rv_estimator, "YangZhang")
        logger.info(f"Using {rv_col} as realized variance benchmark")

        return rv_measures

    def _apply_mz_calibration(self) -> None:
        """Apply Mincer-Zarnowitz calibration to forecasts."""
        if "resid" not in self.forecasts.columns:
            logger.warning("No residuals found, skipping MZ calibration")
            return

        # Compute MZ regression
        resid = self.forecasts["resid"].to_numpy()
        sigma2_raw = self.forecasts["sigma2_egarch_raw"].to_numpy()

        mz_results = mincer_zarnowitz(resid, sigma2_raw)
        intercept = mz_results["intercept"]
        slope = mz_results["slope"]

        logger.info(
            f"MZ Calibration: intercept={intercept:.6f} "
            f"(p={mz_results.get('p_intercept', np.nan):.4f}), "
            f"slope={slope:.4f} (p={mz_results.get('p_slope', np.nan):.4f})"
        )

        # Apply calibration (multiplicative only if intercept not significant)
        use_intercept = mz_results.get("p_intercept", 1.0) < 0.05

        self.forecasts["sigma2_calibrated"] = apply_mz_calibration(
            sigma2_raw,
            intercept,
            slope,
            use_intercept=use_intercept,
        )

        logger.info(
            f"Applied MZ calibration (use_intercept={use_intercept}). "
            f"Mean adjustment factor: {slope:.4f}"
        )

    def evaluate_vs_rv(self) -> Dict[str, Any]:
        """Evaluate forecasts against a realized-volatility benchmark."""
        sigma2_col = self._select_sigma_column()
        sigma2 = self.forecasts[sigma2_col].to_numpy()
        resid = self.forecasts["resid"].to_numpy()
        rv_proxy = self._build_rv_proxy()

        metrics = self._create_rv_summary(sigma2_col)
        metrics.update(self._compute_variance_metrics(resid, sigma2))
        self._add_naive_benchmark(metrics, rv_proxy, sigma2, resid)

        return metrics

    def _select_sigma_column(self) -> str:
        """Select calibrated variance column when available."""
        return "sigma2_calibrated" if "sigma2_calibrated" in self.forecasts else "sigma2_egarch_raw"

    def _build_rv_proxy(self) -> np.ndarray:
        """Return realized volatility proxy aligned with forecasts."""
        if self.realized_measures is None:
            logger.warning("No RV measures available, using squared residuals")
            return np.asarray(self.forecasts["resid"].to_numpy(), dtype=float) ** 2

        rv_map = {
            "yang_zhang": "YangZhang",
            "garman_klass": "GarmanKlass",
            "rogers_satchell": "RogersSatchell",
            "parkinson": "Parkinson",
            "classical": "RV",
        }
        rv_col = rv_map.get(self.rv_estimator, "YangZhang")
        return self._align_rv_with_forecasts(rv_col)

    def _create_rv_summary(self, sigma2_col: str) -> Dict[str, Any]:
        """Prepare basic RV evaluation summary."""
        return {
            "n_obs": len(self.forecasts),
            "forecast_column": sigma2_col,
            "rv_estimator": self.rv_estimator,
        }

    def _compute_variance_metrics(
        self,
        resid: np.ndarray,
        sigma2: np.ndarray,
    ) -> Dict[str, float]:
        """Compute QLIKE, MSE, and MAE metrics."""
        metrics: Dict[str, float] = {"qlike": qlike_loss(resid, sigma2)}
        metrics.update(mse_mae_variance(resid, sigma2))
        return metrics

    def _add_naive_benchmark(
        self,
        metrics: Dict[str, Any],
        rv_proxy: np.ndarray,
        sigma2: np.ndarray,
        resid: np.ndarray,
    ) -> None:
        """Augment metrics with naive RV benchmark and DM tests."""
        if rv_proxy.size <= 1:
            return

        naive_forecast = np.roll(rv_proxy, 1)[1:]
        resid_aligned = resid[1:]
        sigma2_h1 = sigma2[1:]

        valid_mask = (
            np.isfinite(naive_forecast)
            & np.isfinite(resid_aligned)
            & np.isfinite(sigma2_h1)
            & (naive_forecast > 0)
            & (sigma2_h1 > 0)
        )
        if not np.any(valid_mask):
            msg = "No valid observations for naive benchmark comparison"
            raise ValueError(msg)

        naive_forecast = naive_forecast[valid_mask]
        resid_aligned = resid_aligned[valid_mask]
        sigma2_h1 = sigma2_h1[valid_mask]

        metrics["naive_qlike"] = qlike_loss(resid_aligned, naive_forecast)
        naive_losses = mse_mae_variance(resid_aligned, naive_forecast)
        metrics["naive_mse"] = naive_losses["mse"]
        metrics["naive_mae"] = naive_losses["mae"]

        dm_qlike = diebold_mariano_test(
            resid_aligned,
            sigma2_h1,
            naive_forecast,
            loss_function="qlike",
        )
        metrics["dm_test_qlike"] = dm_qlike
        metrics["conclusion_qlike"] = self._derive_dm_conclusion(dm_qlike)

        dm_mse = diebold_mariano_test(
            resid_aligned,
            sigma2_h1,
            naive_forecast,
            loss_function="mse",
        )
        metrics["dm_test_mse"] = dm_mse

    def _derive_dm_conclusion(self, dm_result: Dict[str, Any]) -> str:
        """Interpret Diebold-Mariano outcome with strict validation."""
        p_value = self._extract_numeric_pvalue(dm_result)
        better_model = dm_result.get("better_model")

        if p_value < 0.05:
            if better_model == "model_1":
                return "GARCH significantly better than naive RV (p<0.05)"
            if better_model == "model_2":
                return "Naive RV significantly better than GARCH (p<0.05)"
            msg = f"Unexpected better_model label: {better_model!r}"
            raise ValueError(msg)

        return "No significant difference (p>=0.05)"

    def _extract_numeric_pvalue(self, dm_result: Dict[str, Any]) -> float:
        """Return validated numeric DM-test p-value."""
        p_value_raw = dm_result.get("p_value")
        if not isinstance(p_value_raw, (float, int)):
            msg = f"Unexpected DM p-value type: {type(p_value_raw)!r}"
            raise TypeError(msg)

        p_value = float(p_value_raw)
        if not np.isfinite(p_value):
            msg = f"Non-finite DM p-value encountered: {p_value}"
            raise ValueError(msg)
        if p_value < 0.0 or p_value > 1.0:
            msg = f"DM p-value outside [0, 1]: {p_value}"
            raise ValueError(msg)
        return p_value

    def _align_rv_with_forecasts(self, rv_column: str) -> np.ndarray:
        """Align RV measures with forecast dates."""
        if self.realized_measures is None:
            resid_float = np.asarray(self.forecasts["resid"].to_numpy(), dtype=float)
            return resid_float**2

        # This is a simplified alignment - in practice you'd need proper date matching
        rv_values = self.realized_measures[rv_column].to_numpy()

        # Handle length mismatch
        n_forecasts = len(self.forecasts)
        if len(rv_values) >= n_forecasts:
            return rv_values[-n_forecasts:]
        else:
            # Pad with NaN if not enough RV values
            return np.concatenate([np.full(n_forecasts - len(rv_values), np.nan), rv_values])

    def model_confidence_set(
        self,
        models: Dict[str, np.ndarray],
        alpha: float = 0.10,
        loss_function: str = "qlike",
    ) -> Dict[str, Any]:
        """Compute Hansen et al. (2011) Model Confidence Set.

        Args:
            models: Dictionary mapping model names to variance forecast arrays
            alpha: Significance level (default: 0.10 for 90% confidence)
            loss_function: Loss function to use ('qlike', 'mse', or 'mae')

        Returns:
            Dictionary with MCS results:
                - mcs_set: List of models in the confidence set
                - eliminated: List of eliminated models
                - p_values: P-values for each model
                - alpha: Significance level used
                - loss_function: Loss function used
                - n_models: Number of models compared
        """
        logger.info("Computing Model Confidence Set with %d models...", len(models))

        model_names = list(models.keys())
        losses = self._compute_model_losses(model_names, models, loss_function)
        loss_diff = self._build_loss_differentials(model_names, losses)
        mcs_set, eliminated, p_values = self._execute_mcs_procedure(
            model_names,
            loss_diff,
            alpha,
        )

        return {
            "mcs_set": mcs_set,
            "eliminated": eliminated,
            "p_values": p_values,
            "alpha": alpha,
            "loss_function": loss_function,
            "n_models": len(model_names),
        }

    def _compute_model_losses(
        self,
        model_names: List[str],
        models: Dict[str, np.ndarray],
        loss_function: str,
    ) -> Dict[str, np.ndarray]:
        """Compute loss series for every model."""
        resid = np.asarray(self.forecasts["resid"].to_numpy(), dtype=float)
        losses: Dict[str, np.ndarray] = {}

        for name in model_names:
            forecasts = np.asarray(models[name], dtype=float).ravel()
            if forecasts.size != resid.size:
                msg = f"Forecast length mismatch for {name}: {forecasts.size} vs {resid.size}"
                raise ValueError(msg)
            if not np.all(np.isfinite(forecasts)):
                msg = f"Non-finite forecasts encountered for model {name}"
                raise ValueError(msg)
            if np.any(forecasts <= 0):
                msg = f"Non-positive variance forecast encountered for model {name}"
                raise ValueError(msg)

            if loss_function == "qlike":
                losses[name] = np.log(forecasts) + (resid**2) / forecasts
            elif loss_function == "mse":
                losses[name] = (resid**2 - forecasts) ** 2
            elif loss_function == "mae":
                losses[name] = np.abs(resid**2 - forecasts)
            else:
                raise ValueError(f"Unknown loss function: {loss_function}")

        return losses

    def _build_loss_differentials(
        self,
        model_names: List[str],
        losses: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Construct pairwise loss differential tensor."""
        n_models = len(model_names)
        n_obs = losses[model_names[0]].size
        loss_diff = np.zeros((n_models, n_models, n_obs))

        for i, name_i in enumerate(model_names):
            for j, name_j in enumerate(model_names):
                loss_diff[i, j, :] = losses[name_i] - losses[name_j]

        return loss_diff

    def _execute_mcs_procedure(
        self,
        model_names: List[str],
        loss_diff: np.ndarray,
        alpha: float,
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        """Iteratively eliminate inferior models via MCS t-statistics."""
        mcs_set: List[str] = list(model_names)
        eliminated: List[str] = []
        p_values: Dict[str, float] = {}

        while len(mcs_set) > 1:
            t_stats = self._compute_t_statistics(mcs_set, model_names, loss_diff)
            if not t_stats:
                break

            worst_model = max(t_stats, key=lambda name: t_stats[name])
            worst_t = t_stats[worst_model]
            p_val = float(2 * (1 - stats.norm.cdf(abs(worst_t))))
            p_values[worst_model] = p_val

            if p_val < alpha:
                mcs_set.remove(worst_model)
                eliminated.append(worst_model)
                logger.info("Eliminated %s from MCS (p=%.4f)", worst_model, p_val)
            else:
                break

        return mcs_set, eliminated, p_values

    def _compute_t_statistics(
        self,
        mcs_set: List[str],
        model_names: List[str],
        loss_diff: np.ndarray,
    ) -> Dict[str, float]:
        """Compute MCS t-statistics for active models."""
        t_stats: Dict[str, float] = {}
        for name in mcs_set:
            idx_i = model_names.index(name)
            diffs = []
            for other in mcs_set:
                if other == name:
                    continue
                idx_j = model_names.index(other)
                diffs.append(loss_diff[idx_i, idx_j, :])

            if not diffs:
                continue

            d_bar = np.mean(diffs, axis=0)
            std = float(np.std(d_bar, ddof=1))
            if std <= 0 or not np.isfinite(std):
                msg = f"Cannot compute MCS t-statistic for {name}: std={std}"
                raise ValueError(msg)

            mean_diff = float(np.mean(d_bar))
            t_stats[name] = mean_diff / (std / np.sqrt(d_bar.size))

        return t_stats

    def rolling_window_evaluation(
        self,
        window_size: int = 252,
        step_size: int = 21,
    ) -> pd.DataFrame:
        """Perform rolling window evaluation for stability analysis.

        Args:
            window_size: Size of rolling window (default: 252 = 1 trading year)
            step_size: Step size between windows (default: 21 = 1 trading month)

        Returns:
            DataFrame with rolling window metrics (QLIKE, MSE, MAE, MZ stats)
        """
        logger.info("Rolling window evaluation: window=%d, step=%d", window_size, step_size)

        n_obs = len(self.forecasts)
        if n_obs < window_size:
            logger.warning(
                "Not enough data for rolling window (n=%d < window=%d)", n_obs, window_size
            )
            return pd.DataFrame()

        sigma2_col = self._select_sigma_column()
        results = [
            self._compute_window_metrics(start, window_size, sigma2_col)
            for start in range(0, n_obs - window_size + 1, step_size)
        ]

        results_df = pd.DataFrame(results)
        if results_df.empty:
            return results_df

        self._log_rolling_summary(results_df)
        return results_df

    def _compute_window_metrics(
        self,
        start: int,
        window_size: int,
        sigma2_col: str,
    ) -> Dict[str, float]:
        """Compute evaluation metrics for a single window."""
        end = start + window_size
        window_resid = self.forecasts["resid"].iloc[start:end].to_numpy()
        window_sigma2 = self.forecasts[sigma2_col].iloc[start:end].to_numpy()

        metrics: Dict[str, float] = {
            "window_start": float(start),
            "window_end": float(end),
            "qlike": qlike_loss(window_resid, window_sigma2),
        }
        metrics.update(mse_mae_variance(window_resid, window_sigma2))

        mz = mincer_zarnowitz(window_resid, window_sigma2)
        metrics["mz_intercept"] = mz["intercept"]
        metrics["mz_slope"] = mz["slope"]
        metrics["mz_r2"] = mz["r2"]
        return metrics

    def _log_rolling_summary(self, results_df: pd.DataFrame) -> None:
        """Log summary statistics for rolling evaluation."""
        logger.info("Rolling evaluation summary:")
        logger.info(
            "  QLIKE: mean=%.4f, std=%.4f",
            results_df["qlike"].mean(),
            results_df["qlike"].std(),
        )
        logger.info(
            "  MSE: mean=%.4e, std=%.4e",
            results_df["mse"].mean(),
            results_df["mse"].std(),
        )
        logger.info(
            "  MZ Slope: mean=%.4f, std=%.4f",
            results_df["mz_slope"].mean(),
            results_df["mz_slope"].std(),
        )


def run_advanced_evaluation(
    forecasts_path: Path | None = None,
    hloc_data_path: Path | None = None,
    output_dir: Path | None = None,
) -> Dict[str, Any]:
    """Run complete advanced model comparison evaluation pipeline.

    Args:
        forecasts_path: Path to forecasts CSV file
        hloc_data_path: Path to HLOC data CSV file (optional)
        output_dir: Directory to save results (default: GARCH_EVALUATION_DIR)

    Returns:
        Dictionary with evaluation results:
            - rv_benchmark: RV benchmark comparison results
            - model_confidence_set: MCS results
            - rolling_evaluation: Rolling window stability results
    """
    forecasts_path_resolved, output_dir_resolved = _resolve_advanced_paths(
        forecasts_path,
        output_dir,
    )

    logger.info("=" * 60)
    logger.info("ADVANCED MODEL COMPARISON EVALUATION")
    logger.info("=" * 60)

    forecasts_df = _load_forecasts(forecasts_path_resolved)
    hloc_data = _load_hloc_data(hloc_data_path)

    evaluator = ModelComparisonEvaluator(
        forecasts_df,
        hloc_data=hloc_data,
        apply_mz_cal=True,
        rv_estimator="yang_zhang",
    )

    results: Dict[str, Any] = {}
    results["rv_benchmark"] = _evaluate_realized_benchmark(evaluator)
    results["model_confidence_set"] = _compute_mcs_results(forecasts_df, evaluator)

    rolling_summary = _compute_rolling_summary(evaluator, output_dir_resolved)
    if rolling_summary:
        results["rolling_evaluation"] = rolling_summary

    _persist_advanced_results(results, output_dir_resolved)

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    return results


def _resolve_advanced_paths(
    forecasts_path: Path | None,
    output_dir: Path | None,
) -> Tuple[Path, Path]:
    """Resolve default paths for advanced evaluation."""
    forecasts = forecasts_path or GARCH_FORECASTS_FILE
    outputs = output_dir or GARCH_EVALUATION_DIR
    return forecasts, outputs


def _load_forecasts(forecasts_path: Path) -> pd.DataFrame:
    """Load forecasts CSV with strict validation."""
    validate_file_exists(forecasts_path, "Forecasts file")

    forecasts_df = load_dataframe(forecasts_path, date_columns=["date"], validate_not_empty=False)
    logger.info("Loaded %d forecasts from %s", len(forecasts_df), forecasts_path)
    return forecasts_df


def _load_hloc_data(hloc_data_path: Path | None) -> pd.DataFrame | None:
    """Load HLOC data if available."""
    if hloc_data_path is None:
        logger.info("No HLOC data path provided (skipping RV benchmark input)")
        return None
    if not hloc_data_path.exists():
        logger.info("HLOC data path does not exist (%s), skipping RV benchmark", hloc_data_path)
        return None

    hloc_data = load_dataframe(hloc_data_path, date_columns=["Date"], validate_not_empty=False)
    logger.info("Loaded HLOC data from %s", hloc_data_path)
    return hloc_data


def _evaluate_realized_benchmark(evaluator: ModelComparisonEvaluator) -> Dict[str, Any]:
    """Run realized volatility benchmark with logging."""
    logger.info("\n1. EVALUATION VS REALIZED VOLATILITY")
    logger.info("-" * 40)

    rv_results = evaluator.evaluate_vs_rv()
    logger.info("QLIKE: GARCH=%.4f", rv_results["qlike"])

    naive_key = "naive_qlike"
    if naive_key in rv_results:
        logger.info("       Naive RV=%.4f", rv_results[naive_key])

    dm_key = "dm_test_qlike"
    if dm_key in rv_results:
        dm_result = rv_results[dm_key]
        logger.info(
            "Diebold-Mariano: stat=%.3f, p=%.4f",
            dm_result["dm_statistic"],
            dm_result["p_value"],
        )
        logger.info("Conclusion: %s", rv_results["conclusion_qlike"])

    return rv_results


def _compute_mcs_results(
    forecasts_df: pd.DataFrame,
    evaluator: ModelComparisonEvaluator,
) -> Dict[str, Any]:
    """Compute and log Model Confidence Set results."""
    logger.info("\n2. MODEL CONFIDENCE SET")
    logger.info("-" * 40)

    models: Dict[str, np.ndarray] = {
        "EGARCH_raw": forecasts_df["sigma2_egarch_raw"].to_numpy(),
    }
    if "sigma2_calibrated" in evaluator.forecasts:
        models["EGARCH_calibrated"] = evaluator.forecasts["sigma2_calibrated"].to_numpy()

    if len(forecasts_df) > 1:
        rv_proxy = np.asarray(forecasts_df["resid"].to_numpy(), dtype=float) ** 2
        naive = np.roll(rv_proxy, 1)
        naive[0] = rv_proxy[0]
        models["Naive_RV"] = naive

    mcs_results = evaluator.model_confidence_set(models, alpha=0.10)
    logger.info("Models in 90%% MCS: %s", ", ".join(mcs_results["mcs_set"]))
    if mcs_results["eliminated"]:
        logger.info("Eliminated models: %s", ", ".join(mcs_results["eliminated"]))
    return mcs_results


def _compute_rolling_summary(
    evaluator: ModelComparisonEvaluator,
    output_dir: Path,
) -> Dict[str, float]:
    """Compute rolling window statistics and persist CSV."""
    logger.info("\n3. ROLLING WINDOW STABILITY")
    logger.info("-" * 40)

    rolling_df = evaluator.rolling_window_evaluation(window_size=252, step_size=21)
    if rolling_df.empty:
        return {}

    summary = {
        "n_windows": int(len(rolling_df)),
        "qlike_mean": float(rolling_df["qlike"].mean()),
        "qlike_std": float(rolling_df["qlike"].std()),
        "mz_slope_mean": float(rolling_df["mz_slope"].mean()),
        "mz_slope_std": float(rolling_df["mz_slope"].std()),
    }

    rolling_path = output_dir / "rolling_evaluation.csv"
    rolling_df.to_csv(rolling_path, index=False)
    logger.info("Saved rolling evaluation to %s", rolling_path)
    return summary


def _persist_advanced_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Persist advanced evaluation results to disk."""
    import json  # Local import for custom default handler

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "advanced_evaluation_results.json"

    with open(results_path, "w", encoding="utf-8") as file:
        json_results = json.dumps(
            results,
            indent=2,
            default=lambda x: float(x) if isinstance(x, np.number) else str(x),
        )
        file.write(json_results)

    logger.info("\nSaved complete results to %s", results_path)


__all__ = [
    "ModelComparisonEvaluator",
    "run_advanced_evaluation",
]
