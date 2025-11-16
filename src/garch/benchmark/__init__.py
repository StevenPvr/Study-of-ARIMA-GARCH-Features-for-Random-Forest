"""Benchmark and model comparison modules for volatility forecasting.

This package provides tools for comparing volatility forecasting models:
- Statistical tests (Diebold-Mariano)
- Model Confidence Set (MCS) analysis
- Realized volatility benchmarks
- Baseline model implementations
- Volatility backtest orchestration
"""

from __future__ import annotations

# Volatility backtest
from .bench_volatility import run_benchmark_section4, run_vol_backtest, save_vol_backtest_outputs

# Baseline models
from .benchmark_models import HARRV, HistoricalVolatility, RiskMetricsEWMA, compare_benchmark_models

# Model comparison and evaluation
from .model_comparison import ModelComparisonEvaluator, run_advanced_evaluation

# Realized volatility benchmarking
from .realized_vol_comparison import (
    align_forecasts_with_hloc,
    compare_garch_vs_realized_estimators,
    compute_realized_volatility_benchmarks,
    generate_benchmark_report,
    load_hloc_data,
    run_realized_volatility_benchmark,
)

# Realized volatility estimators
from .realized_volatility import (
    compare_realized_estimators,
    compute_realized_measures,
    efficiency_ratio,
    garman_klass_estimator,
    parkinson_estimator,
    realized_variance_returns,
    rogers_satchell_estimator,
    yang_zhang_estimator,
)

# Statistical tests
from .statistical_tests import diebold_mariano_test

__all__ = [
    # Statistical tests
    "diebold_mariano_test",
    # Model comparison
    "ModelComparisonEvaluator",
    "run_advanced_evaluation",
    # Realized volatility estimators
    "compare_realized_estimators",
    "compute_realized_measures",
    "efficiency_ratio",
    "garman_klass_estimator",
    "parkinson_estimator",
    "realized_variance_returns",
    "rogers_satchell_estimator",
    "yang_zhang_estimator",
    # Realized volatility benchmarking
    "align_forecasts_with_hloc",
    "compare_garch_vs_realized_estimators",
    "compute_realized_volatility_benchmarks",
    "generate_benchmark_report",
    "load_hloc_data",
    "run_realized_volatility_benchmark",
    # Volatility backtest
    "run_benchmark_section4",
    "run_vol_backtest",
    "save_vol_backtest_outputs",
    # Baseline models
    "HARRV",
    "HistoricalVolatility",
    "RiskMetricsEWMA",
    "compare_benchmark_models",
]
