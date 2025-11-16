"""Benchmark GARCH forecasts against realized volatility estimators.

This module compares GARCH model forecasts with realized volatility estimators
using High-Low-Open-Close (HLOC) price data. It implements model comparison
tests (Diebold-Mariano) and efficiency analysis.

Academic References:
    - Parkinson (1980): "The extreme value method for estimating the variance"
    - Garman-Klass (1980): "On the estimation of security price volatilities"
    - Rogers-Satchell (1991): "Estimating variance from high, low and closing prices"
    - Yang-Zhang (2000): "Drift-independent volatility estimation"
    - Diebold-Mariano (1995): "Comparing predictive accuracy"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from src.garch.benchmark.realized_volatility import compute_realized_measures
from src.garch.benchmark.statistical_tests import diebold_mariano_test
from src.garch.garch_eval.metrics import mse_mae_variance, qlike_loss
from src.path import GARCH_RESULTS_DIR
from src.utils import get_logger

logger = get_logger(__name__)


def load_hloc_data(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load High-Low-Open-Close price data for realized volatility estimation.

    Args:
        data_path: Path to HLOC data file. If None, uses default from constants.

    Returns:
        DataFrame with columns: Date, High, Low, Open, Close.

    Raises:
        FileNotFoundError: If HLOC data file not found.
        ValueError: If required columns are missing.
    """
    if data_path is None:
        # Try to find HLOC data in data directory
        from src.path import DATA_DIR

        data_path = Path(DATA_DIR) / "sp500_hloc.csv"

    path = Path(data_path)
    if not path.exists():
        msg = f"HLOC data file not found: {path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(path)

    # Check required columns (case-insensitive)
    required_cols = {"high", "low", "open", "close", "date"}
    df_cols_lower = {c.lower() for c in df.columns}

    if not required_cols.issubset(df_cols_lower):
        missing = required_cols - df_cols_lower
        msg = f"HLOC data missing required columns: {missing}"
        raise ValueError(msg)

    # Standardize column names
    col_mapping = {c: c.lower() for c in df.columns}
    df = df.rename(columns=col_mapping)

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    result = df[["date", "high", "low", "open", "close"]]
    return cast(pd.DataFrame, result)


def align_forecasts_with_hloc(
    garch_forecasts: pd.DataFrame,
    hloc_data: pd.DataFrame,
) -> pd.DataFrame:
    """Align GARCH forecasts with HLOC data by date (≤40 lines).

    Args:
        garch_forecasts: DataFrame with date, resid, RV, sigma2_egarch_raw.
        hloc_data: DataFrame with date, high, low, open, close.

    Returns:
        Merged DataFrame with both GARCH forecasts and HLOC prices.
    """
    # Ensure both have datetime index
    if "date" in garch_forecasts.columns:
        garch_forecasts = garch_forecasts.copy()
        garch_forecasts["date"] = pd.to_datetime(garch_forecasts["date"])

    if "date" in hloc_data.columns:
        hloc_data = hloc_data.copy()
        hloc_data["date"] = pd.to_datetime(hloc_data["date"])

    # Merge on date
    merged = pd.merge(
        garch_forecasts,
        hloc_data,
        on="date",
        how="inner",
        suffixes=("_garch", "_hloc"),
    )

    if merged.empty:
        logger.warning("No overlapping dates between GARCH forecasts and HLOC data")

    return merged


def compute_realized_volatility_benchmarks(
    aligned_data: pd.DataFrame,
) -> pd.DataFrame:
    """Compute all realized volatility estimators on aligned data (≤40 lines).

    Args:
        aligned_data: DataFrame with high, low, open, close columns.

    Returns:
        DataFrame with realized volatility measures added as columns.
    """
    # Compute realized measures
    rv_measures = compute_realized_measures(
        aligned_data,
        high_col="high",
        low_col="low",
        open_col="open",
        close_col="close",
    )

    # Merge back with aligned data
    result = aligned_data.copy()
    for col in rv_measures.columns:
        if col not in result.columns:
            result[col] = rv_measures[col]

    return result


def compare_garch_vs_realized_estimators(
    data: pd.DataFrame,
    *,
    garch_forecast_col: str = "sigma2_egarch_raw",
    realized_variance_col: str = "RV",
) -> dict[str, Any]:
    """Compare GARCH forecasts with realized volatility estimators.

    Computes loss metrics (QLIKE, MSE, MAE) and Diebold-Mariano tests for
    statistical comparison of forecast accuracy.

    Args:
        data: DataFrame with GARCH forecasts and realized measures.
        garch_forecast_col: Column name for GARCH variance forecasts.
        realized_variance_col: Column name for squared returns (proxy for RV).

    Returns:
        Dictionary with comparison results:
            {
                'garch_performance': {...},
                'realized_estimators_performance': {...},
                'diebold_mariano_tests': {...},
                'efficiency_ratios': {...},
            }

    Example:
        >>> aligned = align_forecasts_with_hloc(garch_fcst, hloc)
        >>> aligned = compute_realized_volatility_benchmarks(aligned)
        >>> results = compare_garch_vs_realized_estimators(aligned)
        >>> print(f"GARCH QLIKE: {results['garch_performance']['qlike']:.4f}")
    """
    # Realized estimator columns
    rv_estimators = ["Parkinson", "GarmanKlass", "RogersSatchell", "YangZhang"]

    # Filter valid observations
    valid_mask = (
        data[realized_variance_col].notna()
        & data[garch_forecast_col].notna()
        & (data[garch_forecast_col] > 0)
    )
    for est in rv_estimators:
        if est in data.columns:
            valid_mask &= data[est].notna() & (data[est] > 0)

    data_valid = data[valid_mask].copy()

    if len(data_valid) == 0:
        logger.warning("No valid observations for GARCH vs RV comparison")
        return {}

    # Actual realized variance (squared returns)
    rv_actual = cast(pd.Series, data_valid[realized_variance_col]).to_numpy(dtype=float)
    garch_forecast = cast(pd.Series, data_valid[garch_forecast_col]).to_numpy(dtype=float)
    residuals_series = cast(
        pd.Series,
        data_valid.get("resid", pd.Series(0.0, index=data_valid.index)),
    )
    residuals = residuals_series.to_numpy(dtype=float)

    # 1. GARCH performance
    garch_perf = {
        "n_obs": len(rv_actual),
        "qlike": qlike_loss(rv_actual, garch_forecast),
        "mse": mse_mae_variance(rv_actual, garch_forecast)["mse"],
        "mae": mse_mae_variance(rv_actual, garch_forecast)["mae"],
    }

    # 2. Realized estimators performance
    rv_estimators_perf = {}
    rv_estimators_forecasts = {}

    for est in rv_estimators:
        if est not in data_valid.columns:
            continue

        rv_est_series = cast(pd.Series, data_valid[est])
        rv_est_forecast = rv_est_series.to_numpy(dtype=float)
        rv_estimators_forecasts[est] = rv_est_forecast

        rv_estimators_perf[est] = {
            "n_obs": len(rv_actual),
            "qlike": qlike_loss(rv_actual, rv_est_forecast),
            "mse": mse_mae_variance(rv_actual, rv_est_forecast)["mse"],
            "mae": mse_mae_variance(rv_actual, rv_est_forecast)["mae"],
        }

    # 3. Diebold-Mariano tests: GARCH vs each RV estimator
    dm_tests = {}

    for est, rv_est_forecast in rv_estimators_forecasts.items():
        try:
            dm_result = diebold_mariano_test(
                e=residuals,
                sigma2_model1=garch_forecast,
                sigma2_model2=rv_est_forecast,
                loss_function="qlike",
                h=1,
            )
            dm_tests[f"GARCH_vs_{est}"] = dm_result
        except Exception as e:
            logger.warning(f"DM test failed for GARCH vs {est}: {e}")

    # 4. Efficiency ratios (variance of forecast errors)
    efficiency = {}
    garch_errors = rv_actual - garch_forecast

    for est, rv_est_forecast in rv_estimators_forecasts.items():
        est_errors = rv_actual - rv_est_forecast
        # Efficiency ratio: var(GARCH errors) / var(RV estimator errors)
        # < 1 means GARCH is more efficient
        var_garch = float(np.var(garch_errors, ddof=1))
        var_est = float(np.var(est_errors, ddof=1))

        if var_est > 0:
            efficiency[f"GARCH_vs_{est}"] = var_garch / var_est
        else:
            efficiency[f"GARCH_vs_{est}"] = float("nan")

    results = {
        "garch_performance": garch_perf,
        "realized_estimators_performance": rv_estimators_perf,
        "diebold_mariano_tests": dm_tests,
        "efficiency_ratios": efficiency,
        "n_observations": len(data_valid),
    }

    return results


def generate_benchmark_report(
    comparison_results: dict[str, Any],
    output_path: str | Path | None = None,
) -> None:
    """Generate human-readable benchmark report.

    Args:
        comparison_results: Results from compare_garch_vs_realized_estimators.
        output_path: Path to save report. If None, prints to logger.
    """
    if not comparison_results:
        logger.warning("No comparison results to report")
        return

    lines = []
    lines.append("=" * 80)
    lines.append("GARCH vs REALIZED VOLATILITY ESTIMATORS - BENCHMARK REPORT")
    lines.append("=" * 80)
    lines.append("")

    n_obs = comparison_results.get("n_observations", 0)
    lines.append(f"Number of observations: {n_obs}")
    lines.append("")

    # GARCH performance
    garch_perf = comparison_results.get("garch_performance", {})
    lines.append("GARCH MODEL PERFORMANCE:")
    lines.append(f"  QLIKE: {garch_perf.get('qlike', float('nan')):.6f}")
    lines.append(f"  MSE:   {garch_perf.get('mse', float('nan')):.6f}")
    lines.append(f"  MAE:   {garch_perf.get('mae', float('nan')):.6f}")
    lines.append("")

    # Realized estimators performance
    rv_perf = comparison_results.get("realized_estimators_performance", {})
    if rv_perf:
        lines.append("REALIZED VOLATILITY ESTIMATORS PERFORMANCE:")
        for est_name, metrics in rv_perf.items():
            lines.append(f"  {est_name}:")
            lines.append(f"    QLIKE: {metrics.get('qlike', float('nan')):.6f}")
            lines.append(f"    MSE:   {metrics.get('mse', float('nan')):.6f}")
            lines.append(f"    MAE:   {metrics.get('mae', float('nan')):.6f}")
        lines.append("")

    # Diebold-Mariano tests
    dm_tests = comparison_results.get("diebold_mariano_tests", {})
    if dm_tests:
        lines.append("DIEBOLD-MARIANO TESTS (GARCH vs RV Estimators):")
        lines.append("  (Negative statistic favors GARCH, positive favors RV estimator)")
        for test_name, result in dm_tests.items():
            stat = result.get("dm_statistic", float("nan"))
            pval = result.get("p_value", float("nan"))
            lines.append(f"  {test_name}:")
            lines.append(f"    DM statistic: {stat:+.4f}")
            lines.append(f"    p-value:      {pval:.4f}")
        lines.append("")

    # Efficiency ratios
    efficiency = comparison_results.get("efficiency_ratios", {})
    if efficiency:
        lines.append("EFFICIENCY RATIOS (GARCH vs RV Estimators):")
        lines.append("  (Ratio < 1 means GARCH is more efficient)")
        for name, ratio in efficiency.items():
            lines.append(f"  {name}: {ratio:.4f}")
        lines.append("")

    lines.append("=" * 80)

    report_text = "\n".join(lines)

    if output_path is None:
        # Print to logger
        for line in lines:
            logger.info(line)
    else:
        # Save to file
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_text)
        logger.info(f"Benchmark report saved to: {path}")


def run_realized_volatility_benchmark(
    garch_forecasts: pd.DataFrame | None = None,
    hloc_data: pd.DataFrame | None = None,
    *,
    save_report: bool = True,
) -> dict[str, Any]:
    """Run complete realized volatility benchmark analysis.

    This is the main entry point for comparing GARCH forecasts with realized
    volatility estimators.

    Args:
        garch_forecasts: GARCH forecasts DataFrame. If None, loads from default.
        hloc_data: HLOC price data. If None, loads from default.
        save_report: Whether to save benchmark report to file.

    Returns:
        Dictionary with benchmark results.

    Example:
        >>> results = run_realized_volatility_benchmark()
        >>> outperformed = sum(
        ...     1 for r in results['diebold_mariano_tests'].values()
        ...     if r['dm_statistic'] < 0
        ... )
        >>> print(f"GARCH outperformed {outperformed} estimators")
    """
    logger.info("=" * 80)
    logger.info("RUNNING REALIZED VOLATILITY BENCHMARK")
    logger.info("=" * 80)

    # Load GARCH forecasts if not provided
    if garch_forecasts is None:
        try:
            from src.garch.garch_eval.eval import forecast_on_test_from_trained_model

            logger.info("Loading GARCH forecasts from trained model...")
            garch_forecasts = forecast_on_test_from_trained_model()
        except Exception as e:
            logger.error(f"Failed to load GARCH forecasts: {e}")
            return {}

    # Load HLOC data if not provided
    if hloc_data is None:
        try:
            logger.info("Loading HLOC price data...")
            hloc_data = load_hloc_data()
        except FileNotFoundError as e:
            logger.warning(f"{e}. Skipping realized volatility benchmark.")
            return {}
        except Exception as e:
            logger.error(f"Failed to load HLOC data: {e}")
            return {}

    # Align forecasts with HLOC data
    logger.info("Aligning GARCH forecasts with HLOC data...")
    aligned = align_forecasts_with_hloc(garch_forecasts, hloc_data)

    if aligned.empty:
        logger.error("No overlapping data between GARCH forecasts and HLOC prices")
        return {}

    # Compute realized volatility measures
    logger.info("Computing realized volatility estimators...")
    aligned = compute_realized_volatility_benchmarks(aligned)

    # Compare GARCH vs realized estimators
    logger.info("Comparing GARCH vs realized volatility estimators...")
    results = compare_garch_vs_realized_estimators(aligned)

    # Generate report
    if save_report and results:
        report_path = Path(GARCH_RESULTS_DIR) / "realized_vol_benchmark_report.txt"
        generate_benchmark_report(results, output_path=report_path)

    logger.info("=" * 80)
    logger.info("REALIZED VOLATILITY BENCHMARK COMPLETE")
    logger.info("=" * 80)

    return results


__all__ = [
    "load_hloc_data",
    "align_forecasts_with_hloc",
    "compute_realized_volatility_benchmarks",
    "compare_garch_vs_realized_estimators",
    "generate_benchmark_report",
    "run_realized_volatility_benchmark",
]
