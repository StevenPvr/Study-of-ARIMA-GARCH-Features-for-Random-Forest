"""Constants for the S&P 500 Forecasting project."""

from __future__ import annotations

import math
import os
from datetime import datetime

# Re-export all paths from path.py for backward compatibility
from src.path import (  # noqa: F401
    ARIMA_DATA_VISU_PLOTS_DIR,
    ARIMA_EVALUATION_DIR,
    ARIMA_EVALUATION_PLOTS_DIR,
    ARIMA_OPTIMIZATION_DIR,
    ARIMA_OUTPUTS_DIR,
    ARIMA_PLOTS_DIR,
    ARIMA_RESULTS_DIR,
    ARIMA_TRAINING_DIR,
    ARTIFACTS_DIR,
    BENCHMARK_PLOTS_DIR,
    BENCHMARK_RESULTS_DIR,
    DATA_DIR,
    DATA_PLOTS_DIR,
    DATA_RESULTS_DIR,
    DATA_TICKERS_FULL_FILE,
    DATA_TICKERS_FULL_INDICATORS_FILE,
    DATA_TICKERS_FULL_INSIGHTS_FILE,
    DATA_TICKERS_FULL_INSIGHTS_INDICATORS_FILE,
    DATASET_FILE,
    DATASET_FILTERED_FILE,
    DATASET_FILTERED_PARQUET_FILE,
    FETCH_REPORT_FILE,
    GARCH_ACF_SQUARED_PLOT,
    GARCH_DATA_VISU_PLOTS_DIR,
    GARCH_DATASET_FILE,
    GARCH_DIAGNOSTIC_DIR,
    GARCH_DIAGNOSTICS_FILE,
    GARCH_DIAGNOSTICS_PLOTS_DIR,
    GARCH_DISTRIBUTION_DIAGNOSTICS_FILE,
    GARCH_ESTIMATION_DIR,
    GARCH_ESTIMATION_FILE,
    GARCH_EVAL_METRICS_FILE,
    GARCH_EVAL_VAR_RESIDUALS_PLOT,
    GARCH_EVAL_VAR_SCATTER_PLOT,
    GARCH_EVAL_VAR_TIMESERIES_PLOT,
    GARCH_EVAL_VAR_VIOLATIONS_TEMPLATE,
    GARCH_EVALUATION_DIR,
    GARCH_EVALUATION_PLOTS_DIR,
    GARCH_FORECASTS_FILE,
    GARCH_LJUNGBOX_FILE,
    GARCH_ML_DATASET_FILE,
    GARCH_MODEL_FILE,
    GARCH_MODEL_METADATA_FILE,
    GARCH_NUMERICAL_TESTS_FILE,
    GARCH_OPTIMIZATION_DIR,
    GARCH_OPTIMIZATION_RESULTS_FILE,
    GARCH_PLOTS_DIR,
    GARCH_RESIDUALS_OUTPUTS_FILE,
    GARCH_RESULTS_DIR,
    GARCH_RETURNS_CLUSTERING_PLOT,
    GARCH_ROLLING_DIR,
    GARCH_ROLLING_EVAL_FILE,
    GARCH_ROLLING_FORECASTS_FILE,
    GARCH_ROLLING_VARIANCE_FILE,
    GARCH_SQUARED_RESIDUALS_ACF_LB_PLOT,
    GARCH_STD_ACF_PACF_PLOT,
    GARCH_STD_QQ_PLOT,
    GARCH_STD_SQUARED_ACF_PACF_PLOT,
    GARCH_STRUCTURE_DIR,
    GARCH_STRUCTURE_PLOT,
    GARCH_STRUCTURE_PLOTS_DIR,
    GARCH_TRAINING_DIR,
    GARCH_VARIANCE_OUTPUTS_FILE,
    LIGHTGBM_ABLATION_DIR,
    LIGHTGBM_ABLATION_SIGMA2_RESULTS_FILE,
    LIGHTGBM_CORRELATION_PLOTS_DIR,
    LIGHTGBM_DATASET_COMPLETE_FILE,
    LIGHTGBM_DATASET_INSIGHTS_ONLY_FILE,
    LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE,
    LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE,
    LIGHTGBM_DATASET_TECHNICAL_INDICATORS_FILE,
    LIGHTGBM_DATASET_TECHNICAL_ONLY_NO_TARGET_LAGS_FILE,
    LIGHTGBM_DATASET_TECHNICAL_PLUS_INSIGHTS_NO_TARGET_LAGS_FILE,
    LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE,
    LIGHTGBM_DATASET_WITHOUT_SIGMA2_FILE,
    LIGHTGBM_EVAL_RESULTS_FILE,
    LIGHTGBM_EVALUATION_DIR,
    LIGHTGBM_LEAKAGE_TEST_RESULTS_FILE,
    LIGHTGBM_MODELS_DIR,
    LIGHTGBM_OPTIMIZATION_DIR,
    LIGHTGBM_OPTIMIZATION_RESULTS_FILE,
    LIGHTGBM_PERMUTATION_PLOTS_DIR,
    LIGHTGBM_PERMUTATION_RESULTS_FILE,
    LIGHTGBM_PLOTS_DIR,
    LIGHTGBM_RESULTS_DIR,
    LIGHTGBM_SHAP_PLOTS_DIR,
    LIGHTGBM_TRAINING_DIR,
    LIGHTGBM_TRAINING_RESULTS_FILE,
    LJUNGBOX_RESIDUALS_SARIMA_FILE,
    PLOTS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    ROLLING_PREDICTIONS_SARIMA_FILE,
    ROLLING_VALIDATION_METRICS_SARIMA_FILE,
    SARIMA_ARTIFACTS_DIR,
    SARIMA_BEST_MODELS_FILE,
    SARIMA_DATA_VISU_PLOTS_DIR,
    SARIMA_EVALUATION_PLOTS_DIR,
    SARIMA_OPTIMIZATION_RESULTS_FILE,
    SARIMA_RESIDUALS_LJUNGBOX_PLOT,
    SARIMA_TRAINED_MODEL_FILE,
    SARIMA_TRAINED_MODEL_METADATA_FILE,
    SP500_TICKERS_FILE,
    STATIONARITY_REPORT_FILE,
    VOL_BACKTEST_FORECASTS_FILE,
    VOL_BACKTEST_METRICS_FILE,
    VOL_BACKTEST_PLOTS_DIR,
    VOL_BACKTEST_VOLATILITY_PLOT,
    WEIGHTED_LOG_RETURNS_FILE,
    WEIGHTED_LOG_RETURNS_SPLIT_FILE,
)

# ============================================================================
# GARCH DATA VISUALISATION CONSTANTS
# ============================================================================

# GARCH data visualization figure sizes
GARCH_DATA_VISU_FIGURE_SIZE_RETURNS: tuple[int, int] = (12, 8)
GARCH_DATA_VISU_FIGURE_SIZE_AUTOCORR: tuple[int, int] = (12, 5)

# GARCH data visualization plot colors
GARCH_PLOT_COLOR_RETURNS: str = "blue"
GARCH_PLOT_COLOR_ABSOLUTE_RETURNS: str = "orange"
GARCH_PLOT_COLOR_SQUARED_RETURNS: str = "red"
GARCH_PLOT_COLOR_ZERO_LINE: str = "black"

# GARCH data visualization plot styling
GARCH_PLOT_LINEWIDTH: float = 0.5
GARCH_PLOT_ALPHA_MAIN: float = 0.7
GARCH_PLOT_ALPHA_GRID: float = 0.3

# GARCH structure plot constants
GARCH_STRUCTURE_FIGURE_SIZE: tuple[int, int] = (10, 6)
GARCH_STRUCTURE_PLOT_COLOR_RESIDUALS: str = "#1f77b4"
GARCH_STRUCTURE_PLOT_COLOR_ACF: str = "#ff7f0e"
GARCH_STRUCTURE_PLOT_COLOR_ZERO_LINE: str = "black"
GARCH_STRUCTURE_PLOT_COLOR_CONFIDENCE: str = "red"
GARCH_STRUCTURE_PLOT_BAR_WIDTH: float = 0.8
GARCH_STRUCTURE_PLOT_LINEWIDTH: float = 0.8
GARCH_STRUCTURE_PLOT_FONTSIZE_TITLE: int = 12

# GARCH diagnostics plot constants
GARCH_DIAGNOSTICS_FIGURE_SIZE: tuple[int, int] = (10, 6)
GARCH_DIAGNOSTICS_QQ_FIGURE_SIZE: tuple[int, int] = (6, 6)
GARCH_DIAGNOSTICS_PLOT_COLOR_ACF: str = "#1f77b4"
GARCH_DIAGNOSTICS_PLOT_COLOR_PACF: str = "#ff7f0e"
GARCH_DIAGNOSTICS_PLOT_COLOR_CONFIDENCE: str = "red"
GARCH_DIAGNOSTICS_PLOT_COLOR_ZERO_LINE: str = "black"
GARCH_DIAGNOSTICS_PLOT_COLOR_GRAY: str = "gray"
GARCH_DIAGNOSTICS_PLOT_COLOR_TRAIN: str = "#1f77b4"
GARCH_DIAGNOSTICS_PLOT_COLOR_TEST: str = "#ff7f0e"
GARCH_DIAGNOSTICS_PLOT_COLOR_TRAIN_STD: str = "#2ca02c"
GARCH_DIAGNOSTICS_PLOT_COLOR_TEST_STD: str = "#d62728"
GARCH_DIAGNOSTICS_PLOT_BAR_WIDTH: float = 0.8
GARCH_DIAGNOSTICS_PLOT_LINEWIDTH: float = 1.0
GARCH_DIAGNOSTICS_PLOT_ALPHA: float = 0.8
GARCH_DIAGNOSTICS_PLOT_SCATTER_SIZE: int = 8
GARCH_DIAGNOSTICS_PLOT_LINESTYLE_DASHED: str = "--"
GARCH_DIAGNOSTICS_PLOT_LINESTYLE_DOTTED: str = ":"
GARCH_DIAGNOSTICS_PLOT_LEGEND_LOC: str = "upper right"
GARCH_DIAGNOSTICS_PLOT_STD_ERROR_DENOMINATOR: float = 1.0
GARCH_DIAGNOSTICS_PLOT_QQ_PROB_OFFSET: float = 0.5

# ============================================================================
# MODEL PARAMETERS & DEFAULTS
# ============================================================================

# ARIMA/SARIMA defaults
SARIMA_DEFAULT_SEASONAL_ORDER: tuple[int, int, int, int] = (0, 0, 0, 5)
SARIMA_DEFAULT_SEASONAL_PERIOD: int = 5
SARIMA_REFIT_EVERY_DEFAULT: int = 20


# LightGBM train/test split ratio
LIGHTGBM_TRAIN_TEST_SPLIT_RATIO: float = 0.8
LJUNGBOX_SIGNIFICANCE_LEVEL: float = 0.05

# ARIMA Optuna optimization defaults
# NOTE: Development/test values below. For production, use larger values:
# - SARIMA_OPTIMIZATION_N_TRIALS: 150+ (current: 2 for testing)
# - SARIMA_OPTIMIZATION_N_SPLITS: 5 (current: 2 for testing)
SARIMA_OPTIMIZATION_N_TRIALS: int = 150  # TODO: Set to 150+ for production
SARIMA_OPTIMIZATION_N_SPLITS: int = 5  # Walk-forward CV splits during optimization (use 5 for prod)
SARIMA_OPTIMIZATION_TEST_SIZE: float = 0.2
# Ratio of training data to use for validation in walk-forward CV (20%)
SARIMA_OPTIMIZATION_TOP_K_RESULTS: int = 20  # Number of top optimization results to extract

# ARIMA parameter ranges for Optuna (evidence-based for financial log-returns)
# Based on: Box & Jenkins (1976), Tsay (2005), and empirical studies on stock returns
# Financial log-returns typically exhibit short-memory processes (p,q ≤ 3)
SARIMA_P_MAX: int = 3  # Conservative bound based on financial time series literature
SARIMA_D_MAX: int = 0  # Stationary series - no differencing needed
SARIMA_Q_MAX: int = 3  # Conservative bound for MA component in financial data
SARIMA_P_SEASONAL_MAX: int = 1  # Weak seasonal patterns in financial markets (Hamilton, 1994)
SARIMA_D_SEASONAL_MAX: int = 1  # Minimal seasonal differencing for financial series
SARIMA_Q_SEASONAL_MAX: int = 1  # Limited seasonal MA component in equity markets
SARIMA_P_MIN: int = 0
SARIMA_D_MIN: int = 0
SARIMA_Q_MIN: int = 0
SARIMA_P_SEASONAL_MIN: int = 0
SARIMA_D_SEASONAL_MIN: int = 0
SARIMA_Q_SEASONAL_MIN: int = 0

# ARIMA seasonal period (S) search space for Optuna (daily frequency)
# Based on financial markets literature: weak seasonal effects in equity returns
# (French, 1980; Lakonishok & Smidt, 1988)
# s=0: No seasonality (most appropriate for log-returns - efficient market hypothesis)
# s=5: Weekly pattern (trading week = 5 days) - day-of-week effects documented
# s=21: Monthly pattern (trading month ≈ 21 days) - turn-of-month effects
SARIMA_S_ALLOWED_VALUES: tuple[int, ...] = (0, 5, 21)

# ARIMA trend parameter search space for Optuna
# "n": no trend (RECOMMENDED for log returns - mean ≈ 0)
# "c": constant drift (may be considered for some financial series)
# "t": linear trend (NOT recommended for returns - implies exponential growth)
# "ct": constant + linear trend (NOT recommended for returns)
#
# IMPORTANT: For log returns (which have mean ≈ 0), only "n" (no trend) is
# econometrically appropriate. A constant or linear trend in log returns would
# imply unrealistic exponential or super-exponential growth in price levels.
#
# Current setting: Only "n" for log returns (optimal choice)
# To allow constant drift, use: ("n", "c")
SARIMA_TREND_ALLOWED_VALUES: tuple[str, ...] = ("n",)
# ARIMA refit_every parameter ranges for Optuna

# ARIMA data column defaults
SARIMA_DEFAULT_VALUE_COLUMN: str = "weighted_log_return"
# ARIMA optimization output defaults

SARIMA_HISTORY_COLUMNS: tuple[str, ...] = ("split", "date", "y_true", "y_pred", "sarima_resid")
# ARIMA optimization validation settings
SARIMA_VALIDATION_WEIGHT: float = 0.3  # Weight for validation metrics in composite score (0-1)
SARIMA_LJUNGBOX_P_VALUE_THRESHOLD: float = (
    0.10  # Relaxed 10% significance level for SARIMA residual independence
)
# Minimum p-value to consider residuals uncorrelated
SARIMA_LJUNGBOX_PENALTY_WEIGHT: float = 10.0  # Soft penalty weight for Ljung–Box deficit
# Optional override via environment variable (validated). If set but invalid → error.
_env_lb_penalty = os.getenv("SARIMA_LJUNGBOX_PENALTY_WEIGHT")
if _env_lb_penalty is not None:
    try:
        _penalty_val = float(_env_lb_penalty)
    except ValueError as exc:  # pragma: no cover - defensive path
        raise ValueError(
            "Environment variable SARIMA_LJUNGBOX_PENALTY_WEIGHT must be a float."
        ) from exc
    if not math.isfinite(_penalty_val):  # pragma: no cover - defensive path
        raise ValueError("Environment variable SARIMA_LJUNGBOX_PENALTY_WEIGHT must be finite.")
    SARIMA_LJUNGBOX_PENALTY_WEIGHT = float(_penalty_val)
SARIMA_MIN_TRAIN_SIZE_FOR_RESIDUALS: int = 60  # Minimum training window for reliable residuals
SARIMA_NORMALIZATION_EPSILON: float = 1e-8  # Epsilon for z-score normalization stability
SARIMA_MIN_STATS_SAMPLES: int = 2  # Minimum samples needed for meaningful statistics
SARIMA_REFIT_EVERY_OPTIONS: tuple[int, ...] = (5, 21)  # Options for refit frequency optimization
SARIMA_LJUNGBOX_LAGS_DEFAULT: int = 20  # Default lags for Ljung-Box test
SARIMA_ADF_ALPHA_DEFAULT: float = 0.05  # Default significance level for ADF test
SARIMA_MIN_SERIES_LENGTH_STATIONARITY: int = 10  # Minimum series length for stationarity tests
SARIMA_MIN_SERIES_LENGTH_DIFFERENCED: int = 10  # Minimum differenced series length for validation
SARIMA_BACKTEST_MIN_TRAIN_MARGIN: int = (
    16  # Minimum training margin for backtest (for differencing)
)

# GARCH defaults and constraints
GARCH_MIN_INIT_VAR: float = 1e-10
GARCH_DEFAULT_ALPHA: float = 0.05

GARCH_STUDENT_NU_MIN: float = 2.5
GARCH_STUDENT_NU_MAX: float = 200.0
GARCH_STUDENT_NU_INIT: float = 8.0
GARCH_SKEWT_LAMBDA_MIN: float = -0.99
GARCH_SKEWT_LAMBDA_MAX: float = 0.99
GARCH_SKEWT_LAMBDA_INIT: float = -0.1
# GARCH parameter estimation defaults
GARCH_ESTIMATION_MIN_OBSERVATIONS: int = 10
GARCH_ESTIMATION_PARALLEL_WORKERS: int = 3
GARCH_ESTIMATION_PENALTY_VALUE: float = 1e50
GARCH_ESTIMATION_BETA_MIN: float = -0.999
GARCH_ESTIMATION_BETA_MAX: float = 0.999
GARCH_ESTIMATION_OMEGA_BOUND_MIN: float = -50.0
GARCH_ESTIMATION_OMEGA_BOUND_MAX: float = 50.0
GARCH_ESTIMATION_ALPHA_BOUND_MIN: float = -5.0
GARCH_ESTIMATION_ALPHA_BOUND_MAX: float = 5.0
GARCH_ESTIMATION_GAMMA_BOUND_MIN: float = -5.0
GARCH_ESTIMATION_GAMMA_BOUND_MAX: float = 5.0
GARCH_ESTIMATION_INIT_BETA: float = 0.95
GARCH_ESTIMATION_INIT_BETA2: float = 0.01  # Small initial value for beta2 in EGARCH(o,2)
GARCH_ESTIMATION_INIT_ALPHA: float = 0.1
GARCH_ESTIMATION_INIT_GAMMA: float = 0.0
GARCH_ESTIMATION_NU_MIN_THRESHOLD: float = 2.0
GARCH_ESTIMATION_KAPPA_ADJUSTMENT_COEFF: float = 0.1
GARCH_ESTIMATION_KAPPA_EPSILON: float = 1e-12
# GARCH optimization convergence settings
GARCH_ESTIMATION_MAXITER: int = 1000  # Maximum iterations for SLSQP
GARCH_ESTIMATION_FTOL: float = 1e-7  # Function tolerance for convergence
GARCH_ESTIMATION_EPS: float = 1e-8  # Step size for numerical derivatives
# GARCH Skew-t distribution constants
GARCH_SKEWT_COEFF_A_MULTIPLIER: float = 4.0
GARCH_SKEWT_COEFF_B_SQ_TERM1: float = 1.0
GARCH_SKEWT_COEFF_B_SQ_TERM2: float = 3.0

# GARCH diagnostics defaults
GARCH_LM_LAGS_DEFAULT: int = 12
GARCH_ACF_LAGS_DEFAULT: int = 20
GARCH_PLOT_Z_CONF: float = 1.96
GARCH_LJUNG_BOX_LAGS_DEFAULT: int = 20
GARCH_LJUNG_BOX_SPECIFIC_LAGS: list[int] = [10, 20]  # Specific lags for comprehensive diagnostics
GARCH_STD_EPSILON: float = 1e-12  # Small epsilon for numerical stability in standardization

# GARCH numerical tests - minimum observation requirements
GARCH_NUMERICAL_MIN_OBS_BREUSCH_PAGAN: int = 3
GARCH_NUMERICAL_MIN_OBS_WHITE: int = 5

# GARCH numerical tests - polynomial orders for heteroskedasticity tests
GARCH_NUMERICAL_BREUSCH_PAGAN_POLYNOMIAL_ORDER: int = 2
GARCH_NUMERICAL_WHITE_POLYNOMIAL_ORDER: int = 4

# GARCH numerical tests - normalization epsilon for numerical stability
GARCH_NUMERICAL_WHITE_NORMALIZATION_EPSILON: float = 1e-10

# GARCH numerical tests - test names (internationalization ready)
GARCH_NUMERICAL_TEST_NAME_LJUNG_BOX_RESIDUALS: str = "Ljung-Box Test sur les résidus"
GARCH_NUMERICAL_TEST_NAME_LJUNG_BOX_SQUARED: str = "Ljung-Box Test sur les résidus au carré"
GARCH_NUMERICAL_TEST_NAME_ENGLE_ARCH_LM: str = "Engle ARCH LM Test"
GARCH_NUMERICAL_TEST_NAME_MCLEOD_LI: str = "McLeod-Li Test"

# Rolling GARCH defaults
GARCH_REFIT_EVERY_DEFAULT: int = 20
GARCH_REFIT_WINDOW_DEFAULT: str = "expanding"  # or "rolling"
GARCH_REFIT_WINDOW_SIZE_DEFAULT: int = 1000
GARCH_MIN_WINDOW_SIZE: int = 250  # Minimum window size to start forecasting (≈1 year of trading)
GARCH_INITIAL_WINDOW_SIZE_DEFAULT: int = 1000  # Initial window size for expanding window training
GARCH_LOG_VAR_MAX: float = 700.0  # Maximum log-variance for numerical stability
GARCH_LOG_VAR_MIN: float = -700.0  # Minimum log-variance for numerical stability
# Minimum residuals required to fit an initial EGARCH model.
# Lowered to 10 to enable small-sample offline tests while keeping
# explicit validation in EGARCHForecaster.
GARCH_FIT_MIN_SIZE: int = 10


# GARCH initial variance estimation defaults
GARCH_INIT_VAR_ESTIMATION_MIN_SIZE: int = 30
# Minimum observations for robust initial variance (increased from 10)


# GARCH calibration defaults
GARCH_CALIBRATION_EPS: float = 1e-12

# GARCH model specification defaults
EGARCH_DEFAULT_ARCH_ORDER: int = 1
# Default ARCH order (o) for EGARCH(o,p)
EGARCH_DEFAULT_GARCH_ORDER: int = 1
# Default GARCH order (p) for EGARCH(o,p)
EGARCH_DEFAULT_DISTRIBUTION: str = "normal"
# Default innovation distribution


# GARCH hyperparameter optimization defaults
GARCH_OPTIMIZATION_BURN_IN_RATIO: float = 0.3  # 30% of TRAIN for burn-in
GARCH_OPTIMIZATION_MIN_VALIDATION_SIZE: int = 20  # Minimum validation window size
GARCH_OPTIMIZATION_N_TRIALS: int = 100
# Number of Optuna trials (increased for larger search space)
GARCH_OPTIMIZATION_N_SPLITS: int = 5
# Number of walk-forward CV splits during optimization
GARCH_OPTIMIZATION_DISTRIBUTIONS: tuple[str, ...] = ("normal", "student", "skewt")

GARCH_OPTIMIZATION_WINDOW_TYPES: tuple[str, ...] = ("expanding", "rolling")
GARCH_OPTIMIZATION_ROLLING_WINDOW_SIZES: tuple[int, ...] = (500, 1000)

# GARCH hyperparameter optimization - composite objective weights
GARCH_OPTIMIZATION_QLIKE_WEIGHT: float = 0.7
# Weight for QLIKE loss in composite objective
GARCH_OPTIMIZATION_AIC_WEIGHT: float = 0.2
# Weight for AIC penalty in composite objective
GARCH_OPTIMIZATION_DIAGNOSTIC_WEIGHT: float = 0.1
# Weight for diagnostic penalty in composite objective

# GARCH hyperparameter optimization - diagnostic thresholds
GARCH_OPTIMIZATION_DIAGNOSTIC_PVALUE_THRESHOLD: float = 0.05
# P-value threshold for diagnostic tests
GARCH_OPTIMIZATION_ARCH_LM_LAGS: int = 5
# Number of lags for ARCH-LM test in optimization

# GARCH hyperparameter optimization - 3-phase validation split
GARCH_VALIDATION_TRAIN_RATIO: float = 0.6
# 60% for training
GARCH_VALIDATION_VAL_RATIO: float = 0.2
# 20% for validation (hyperparameter tuning)
GARCH_VALIDATION_TEST_RATIO: float = 0.2
# 20% for final test (evaluation only)

# GARCH evaluation defaults
GARCH_EVAL_DEFAULT_LEVEL: float = 0.95
GARCH_EVAL_DEFAULT_HORIZON: int = 5
GARCH_EVAL_DEFAULT_ALPHAS: tuple[float, float] = (0.01, 0.05)
GARCH_EVAL_EPSILON: float = 1e-12
GARCH_EVAL_MIN_ALPHA: float = 1e-6
GARCH_EVAL_AIC_MULTIPLIER: float = 2.0
GARCH_EVAL_DEFAULT_SLOPE: float = 1.0
GARCH_EVAL_PLOT_LIMIT_MULTIPLIER: float = 1.05
GARCH_EVAL_FIGURE_SIZE_DEFAULT: tuple[int, int] = (10, 4)
GARCH_EVAL_FIGURE_SIZE_SCATTER: tuple[int, int] = (5, 5)
GARCH_EVAL_FIGURE_SIZE_RESIDUALS: tuple[int, int] = (10, 3)
GARCH_EVAL_HALF: float = 0.5
GARCH_EVAL_MIN_OBS: int = 2  # Minimum observations for Mincer-Zarnowitz regression

# Evaluation-only warm-up override: earliest forecast start position.
# For evaluation/feature generation we want σ² forecasts to begin early
# to avoid losing too many rows in downstream ML datasets. This leaves
# training defaults unchanged and is opt-in from the evaluation pipeline.
GARCH_EVAL_FORCED_MIN_START_SIZE: int = 200


GARCH_MODEL_NAMES: tuple[str, ...] = ("egarch_normal", "egarch_student", "egarch_skewt")
GARCH_MODEL_PARAMS_COUNT: dict[str, int] = {
    "egarch_normal": 4,
    "egarch_student": 5,
    "egarch_skewt": 6,
}

# LightGBM defaults
# Lags restricted to 1–3 to avoid lookahead bias and over-engineering
LIGHTGBM_LAG_WINDOWS: tuple[int, ...] = (1, 2, 3)

# Base features shared across all datasets now include log_volatility (with lags)
LIGHTGBM_BASE_FEATURE_COLUMNS: tuple[str, ...] = ("log_volatility",)

# Legacy lag feature hints kept minimal; explicit selection happens in data prep


# ARIMA-GARCH insights columns used in LightGBM datasets
# ARIMA insights: sarima_pred and sarima_resid (from SARIMA model)
# GARCH insights: sigma2_garch, sigma_garch, std_resid_garch
# Additional ARIMA insights: arima_residual_return
LIGHTGBM_ARIMA_GARCH_INSIGHT_COLUMNS: tuple[str, ...] = (
    "sarima_pred",
    "sarima_resid",
    "sigma2_garch",
    "sigma_garch",
    "std_resid_garch",
    "arima_residual_return",
)
LIGHTGBM_SHAP_MAX_DISPLAY_DEFAULT: int = 20
LIGHTGBM_OPTIMIZATION_N_SPLITS: int = 5
LIGHTGBM_OPTIMIZATION_N_TRIALS: int = 100
LIGHTGBM_OPTIMIZATION_SAMPLE_FRACTION: float = 0.5  # Use 50% of train data for optimization
LIGHTGBM_OPTIMIZATION_MAX_WORKERS: int = 2  # Optimize 2 models in parallel


# Custom ML feature windows
LIGHTGBM_VOL_MA_SHORT_WINDOW: int = 5
LIGHTGBM_VOL_MA_LONG_WINDOW: int = 20
LIGHTGBM_TURNOVER_MA_WINDOW: int = 5
LIGHTGBM_REALIZED_VOL_WINDOW: int = 5

# New technical feature set used for lagging (calendar features are excluded from lags)
# Note: realized_vol_5d is not included as we use log_volatility (the target) for lags
LIGHTGBM_TECHNICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "log_return",
    "abs_ret",
    "ret_sq",
    "log_volatility",
    "log_volume",
    "log_volume_rel_ma_5",
    "log_volume_zscore_20",
    "log_turnover",
    "turnover_rel_ma_5",
    "obv",
    "atr",
)

# Calendar features (non-lagged) to include in technical indicator datasets
# Kept separate to avoid creating lags for these discrete/time-indexed signals
LIGHTGBM_CALENDAR_FEATURE_COLUMNS: tuple[str, ...] = (
    "day_of_week",
    "month",
    "is_month_end",
    "day_in_month_norm",
)


# Volatility baseline model parameters
VOL_RISKMETRICS_LAMBDA: float = 0.94  # RiskMetrics EWMA decay parameter (J.P. Morgan 1996)
VOL_EWMA_LAMBDA_DEFAULT = VOL_RISKMETRICS_LAMBDA  # Alias for backward compatibility
VOL_ROLLING_WINDOW_DEFAULT: int = 20
VOL_HAR_DAILY_LAG: int = 1  # HAR daily component lag
VOL_HAR_WEEK_WINDOW: int = 5  # HAR weekly window (5 trading days)
VOL_HAR_MONTH_WINDOW: int = 22  # HAR monthly window (22 trading days)

VOL_MIN_VARIANCE: float = 1e-12
VOL_MIN_OMEGA: float = 1e-6

# ============================================================================
# DATA PIPELINE CONSTANTS
# ============================================================================

# Data fetching constants
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

DATA_FETCH_END_DATE = datetime(2024, 12, 31)
DATA_FETCH_START_DATE = datetime(2013, 1, 1)
WIKIPEDIA_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) " "AppleWebKit/537.36 (KHTML, like Gecko)"
)
WIKIPEDIA_REQUEST_TIMEOUT = 15
REQUIRED_TICKER_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
DATA_FETCH_LOG_PROGRESS_INTERVAL = 50

# Data cleaning constants
TOP_N_TICKERS_REPORT: int = 10

# Data conversion constants
LIQUIDITY_WEIGHTS_WINDOW_DEFAULT: int = 20
MAX_ERROR_DATES_DISPLAY: int = 5

# Data preparation constants
TRAIN_RATIO_DEFAULT: float = 0.8
TIMESERIES_SPLIT_N_SPLITS: int = 2
STATIONARITY_RESAMPLE_FREQ: str = "W"

# Required column sets for data validation
REQUIRED_OHLCV_COLUMNS: list[str] = ["date", "tickers", "close", "volume"]
REQUIRED_COLS_TICKER_DATA: set[str] = {"date", "ticker", "high", "low", "close", "volume"}
REQUIRED_COLS_WEIGHTED_RETURNS: set[str] = {"date", "weighted_log_return"}
REQUIRED_COLS_SPLIT_DATA: set[str] = {"date", "weighted_log_return", "split"}
REQUIRED_COLS_LOG_RETURN: set[str] = {"ticker", "date", "log_return"}
REQUIRED_COLS_CLOSE_PRICE: set[str] = {"ticker", "date", "close"}

# ============================================================================
# DATA VISUALISATION CONSTANTS
# ============================================================================

# Year validation range
YEAR_MIN: int = 1900
YEAR_MAX: int = 2100

# Plotting figure sizes
PLOT_FIGURE_SIZE_WEIGHTED_SERIES: tuple[int, int] = (18, 6)
PLOT_FIGURE_SIZE_ACF_PACF: tuple[int, int] = (14, 4)
PLOT_FIGURE_SIZE_STATIONARITY: tuple[int, int] = (16, 12)
PLOT_FIGURE_SIZE_SEASONAL_FULL: tuple[int, int] = (18, 6)
PLOT_FIGURE_SIZE_SEASONAL_YEAR: tuple[int, int] = (14, 4)
PLOT_FIGURE_SIZE_SEASONAL_DAILY: tuple[int, int] = (16, 6)


# Plotting styling constants
PLOT_LINEWIDTH_DEFAULT: float = 0.6
PLOT_LINEWIDTH_BOLD: float = 1.5
PLOT_LINEWIDTH_MEDIUM: float = 1.0
PLOT_LINEWIDTH_THIN: float = 0.8
PLOT_ALPHA_DEFAULT: float = 0.8
PLOT_ALPHA_MEDIUM: float = 0.7
PLOT_ALPHA_LIGHT: float = 0.3
PLOT_ALPHA_FILL: float = 0.2
PLOT_FONTSIZE_TITLE: int = 14
PLOT_FONTSIZE_SUBTITLE: int = 12
PLOT_FONTSIZE_LABEL: int = 12
PLOT_FONTSIZE_AXIS: int = 10
PLOT_FONTSIZE_TEXT: int = 9
PLOT_DPI: int = 300

# ACF/PACF defaults
ACF_PACF_DEFAULT_LAGS: int = 30
ACF_PACF_CONFIDENCE_ALPHA: float = 0.05
ACF_PACF_MIN_LAGS: int = 1

# Stationarity analysis defaults
STATIONARITY_ROLLING_WINDOW_DEFAULT: int = 252
STATIONARITY_ALPHA_DEFAULT: float = 0.05
STATIONARITY_TEXT_BOX_X: float = 0.02
STATIONARITY_TEXT_BOX_Y: float = 0.98
STATIONARITY_SUPTITLE_Y: float = 0.995
STATIONARITY_SEPARATOR_LENGTH: int = 60

# Seasonal decomposition defaults
SEASONAL_DEFAULT_PERIOD_DAILY: int = 5
SEASONAL_DEFAULT_PERIOD_WEEKLY: int = 52
SEASONAL_DEFAULT_PERIOD_MONTHLY: int = 12
SEASONAL_DEFAULT_MODEL: str = "additive"
SEASONAL_DEFAULT_YEARS: int = 1
SEASONAL_MIN_PERIODS: int = 2
SEASONAL_RESAMPLE_FREQ_WEEKLY: str = "W"
SEASONAL_RESAMPLE_FREQ_MONTHLY: str = "ME"
SEASONAL_RESAMPLE_FREQ_BUSINESS: str = "B"

# Date axis formatting
DATE_AXIS_YEAR_LOCATOR_MONTHS: tuple[int, int] = (1, 7)
DATE_AXIS_DAILY_MONTHS: tuple[int, int, int, int] = (1, 4, 7, 10)
DATE_AXIS_ROTATION: int = 45

# Residuals analysis

# ARIMA visualization color scheme (consistent across all plots)
PLOT_COLOR_TRAIN: str = "#2E86AB"  # Blue for training data
PLOT_COLOR_TEST: str = "#A23B72"  # Purple for test data
PLOT_COLOR_PREDICTION: str = "#F18F01"  # Orange for predictions
PLOT_COLOR_ACTUAL: str = "#2E86AB"  # Blue for actual values
PLOT_COLOR_FITTED: str = "#A23B72"  # Purple for fitted values
PLOT_COLOR_RESIDUAL: str = "#2E86AB"  # Blue for residuals
PLOT_COLOR_NORMAL_FIT: str = "#A23B72"  # Purple for normal distribution fit
PLOT_COLOR_REFERENCE: str = "red"  # Red for reference lines
PLOT_COLOR_SPLIT_LINE: str = "red"  # Red for train/test split line
PLOT_COLOR_SERIES_ORIGINAL: str = "blue"  # Blue for original series

# ARIMA visualization plot defaults
RESIDUALS_HISTOGRAM_BINS_DEFAULT: int = 50
DISTRIBUTION_HISTOGRAM_BINS_DEFAULT: int = 100
PLOT_MAX_POINTS_SUBSAMPLE: int = 500  # Max points for readable plots
QQ_PLOT_QUANTILE_MIN: float = 0.01
QQ_PLOT_QUANTILE_MAX: float = 0.99
RESIDUALS_PLOT_FIGURE_SIZE: tuple[int, int] = (14, 5)

# Text box style constants for consistent styling
PLOT_TEXTBOX_STYLE_DEFAULT: dict[str, str | float] = {
    "boxstyle": "round",
    "facecolor": "wheat",
    "alpha": 0.8,
}
PLOT_TEXTBOX_STYLE_INFO: dict[str, str | float] = {
    "boxstyle": "round",
    "facecolor": "lightblue",
    "alpha": 0.8,
}
PLOT_TEXTBOX_STYLE_SUCCESS: dict[str, str | float] = {
    "boxstyle": "round",
    "facecolor": "lightgreen",
    "alpha": 0.7,
}
PLOT_TEXTBOX_STYLE_ERROR: dict[str, str | float] = {
    "boxstyle": "round",
    "facecolor": "lightcoral",
    "alpha": 0.7,
}

# Statistics display defaults
STATISTICS_PRECISION_DEFAULT: int = 6
STATISTICS_PRECISION_SHORT: int = 4

# ============================================================================
# GENERAL CONSTANTS
# ============================================================================

DEFAULT_RANDOM_STATE = 42
DATE_FORMAT_DEFAULT: str = "%Y-%m-%d"
PLOT_FIGURE_SIZE_DEFAULT: tuple[int, int] = (10, 4)

# JSON formatting
JSON_INDENT_DEFAULT: int = 2

# Plot DPI (evaluation uses lower DPI for faster generation)
PLOT_DPI_EVALUATION: int = 150

# Default placeholder date for fabricated datetime indices
DEFAULT_PLACEHOLDER_DATE: str = "2000-01-01"


# Rolling forecast progress reporting interval
ROLLING_FORECAST_PROGRESS_INTERVAL: int = 10


# Plot text annotation positions
PLOT_TEXT_POSITION_X: float = 0.02
PLOT_TEXT_POSITION_Y: float = 0.95

# Data cleaning constants
MIN_OBSERVATIONS_PER_TICKER: int = 252  # ~1 year of trading days
