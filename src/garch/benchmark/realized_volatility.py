"""Realized volatility estimators using High-Low-Open-Close (HLOC) prices.

This module implements various realized volatility estimators that use intraday
price information (high, low, open, close) to estimate daily variance more
efficiently than simple squared returns.

Academic References:
    - Parkinson (1980): "The extreme value method for estimating the variance"
    - Garman-Klass (1980): "On the estimation of security price volatilities"
    - Rogers-Satchell (1991): "Estimating variance from high, low and closing prices"
    - Yang-Zhang (2000): "Drift-independent volatility estimation"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


def _validate_price_arrays(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_price: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and convert price arrays for realized volatility estimators.

    Args:
        high: High prices array.
        low: Low prices array.
        close: Close prices array.
        open_price: Open prices array.

    Returns:
        Tuple of validated (high_arr, low_arr, close_arr, open_arr).

    Raises:
        ValueError: If arrays have mismatched lengths or invalid prices.
    """
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    open_arr = np.asarray(open_price, dtype=float)

    if not (high_arr.size == low_arr.size == close_arr.size == open_arr.size):
        msg = "all price arrays must match in length"
        raise ValueError(msg)

    if np.any(high_arr < 0) or np.any(low_arr < 0) or np.any(close_arr < 0) or np.any(open_arr < 0):
        msg = "prices must be positive"
        raise ValueError(msg)

    if np.any(high_arr < low_arr):
        msg = "high prices must be >= low prices"
        raise ValueError(msg)

    return high_arr, low_arr, close_arr, open_arr


def parkinson_estimator(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Compute Parkinson (1980) range-based variance estimator.

    Uses high-low range to estimate variance. More efficient than squared returns
    when high and low prices are available.

    Args:
        high: Array of high prices (must be positive).
        high: Array of low prices (must be positive).

    Returns:
        Array of variance estimates (one per period).

    Raises:
        ValueError: If prices are invalid (negative, high < low, mismatched lengths).

    Example:
        >>> high = np.array([105.0, 110.0])
        >>> low = np.array([95.0, 100.0])
        >>> var = parkinson_estimator(high, low)
        >>> assert np.all(var > 0)
    """
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)

    if high_arr.size != low_arr.size:
        msg = "high and low arrays must match in length"
        raise ValueError(msg)

    if np.any(high_arr < 0) or np.any(low_arr < 0):
        msg = "prices must be positive"
        raise ValueError(msg)

    if np.any(high_arr < low_arr):
        msg = "high prices must be >= low prices"
        raise ValueError(msg)

    # Parkinson estimator: var = (1/(4*ln(2))) * (ln(H/L))^2
    # Factor: 1/(4*ln(2)) ≈ 0.361
    log_range = np.log(high_arr / low_arr)
    variance = (1.0 / (4.0 * np.log(2.0))) * (log_range**2)

    return variance


def garman_klass_estimator(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_price: np.ndarray,
) -> np.ndarray:
    """Compute Garman-Klass (1980) HLOC variance estimator.

    Uses high, low, open, close prices to estimate variance. More efficient
    than Parkinson estimator as it incorporates close-open information.

    Args:
        high: Array of high prices.
        low: Array of low prices.
        close: Array of closing prices.
        open_price: Array of opening prices.

    Returns:
        Array of variance estimates (one per period).

    Raises:
        ValueError: If prices are invalid.

    Example:
        >>> high = np.array([105.0, 110.0])
        >>> low = np.array([95.0, 100.0])
        >>> close = np.array([100.0, 105.0])
        >>> open_price = np.array([98.0, 102.0])
        >>> var = garman_klass_estimator(high, low, close, open_price)
    """
    high_arr, low_arr, close_arr, open_arr = _validate_price_arrays(high, low, close, open_price)

    # Garman-Klass estimator
    # var = 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2
    log_hl = np.log(high_arr / low_arr)
    log_co = np.log(close_arr / open_arr)

    variance = 0.5 * (log_hl**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)

    # Ensure non-negative
    variance = np.maximum(variance, 0.0)

    return variance


def rogers_satchell_estimator(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_price: np.ndarray,
) -> np.ndarray:
    """Compute Rogers-Satchell (1991) drift-independent variance estimator.

    Uses high, low, open, close prices. Unlike Garman-Klass, this estimator
    is independent of drift (expected return).

    Args:
        high: Array of high prices.
        low: Array of low prices.
        close: Array of closing prices.
        open_price: Array of opening prices.

    Returns:
        Array of variance estimates (one per period).

    Raises:
        ValueError: If prices are invalid.

    Example:
        >>> high = np.array([105.0, 110.0])
        >>> low = np.array([95.0, 100.0])
        >>> close = np.array([100.0, 105.0])
        >>> open_price = np.array([98.0, 102.0])
        >>> var = rogers_satchell_estimator(high, low, close, open_price)
    """
    high_arr, low_arr, close_arr, open_arr = _validate_price_arrays(high, low, close, open_price)

    # Rogers-Satchell estimator
    # var = ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)
    log_hc = np.log(high_arr / close_arr)
    log_ho = np.log(high_arr / open_arr)
    log_lc = np.log(low_arr / close_arr)
    log_lo = np.log(low_arr / open_arr)

    variance = log_hc * log_ho + log_lc * log_lo

    # Ensure non-negative
    variance = np.maximum(variance, 0.0)

    return variance


def yang_zhang_estimator(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_price: np.ndarray,
    k: float = 0.34,
) -> np.ndarray:
    """Compute Yang-Zhang (2000) combined variance estimator.

    Combines overnight and intraday variance estimates. The first period
    will be NaN as it requires the previous period's close.

    Args:
        high: Array of high prices.
        low: Array of low prices.
        close: Array of closing prices.
        open_price: Array of opening prices.
        k: Weight parameter (default: 0.34, must be in [0, 1]).

    Returns:
        Array of variance estimates (first element is NaN).

    Raises:
        ValueError: If prices are invalid, k is out of range, or < 2 periods.

    Example:
        >>> high = np.array([105.0, 110.0, 108.0])
        >>> low = np.array([95.0, 100.0, 98.0])
        >>> close = np.array([100.0, 105.0, 102.0])
        >>> open_price = np.array([98.0, 102.0, 100.0])
        >>> var = yang_zhang_estimator(high, low, close, open_price)
    """
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    open_arr = np.asarray(open_price, dtype=float)

    if not (high_arr.size == low_arr.size == close_arr.size == open_arr.size):
        msg = "all price arrays must match in length"
        raise ValueError(msg)

    if high_arr.size < 2:
        msg = "Yang-Zhang estimator requires at least 2 periods"
        raise ValueError(msg)

    if not (0.0 <= k <= 1.0):
        msg = "k must be in [0, 1]"
        raise ValueError(msg)

    if np.any(high_arr < 0) or np.any(low_arr < 0) or np.any(close_arr < 0) or np.any(open_arr < 0):
        msg = "prices must be positive"
        raise ValueError(msg)

    if np.any(high_arr < low_arr):
        msg = "high prices must be >= low prices"
        raise ValueError(msg)

    n = high_arr.size
    variance = np.full(n, np.nan, dtype=float)

    # Overnight returns (close to open)
    overnight_returns = np.log(open_arr[1:] / close_arr[:-1])

    # Intraday returns (open to close)
    intraday_returns = np.log(close_arr[1:] / open_arr[1:])

    # Rogers-Satchell for intraday variance
    rs_intraday = rogers_satchell_estimator(
        high_arr[1:],
        low_arr[1:],
        close_arr[1:],
        open_arr[1:],
    )

    # Classical realized variance from overnight returns
    rv_overnight = overnight_returns**2

    # Classical realized variance from intraday returns
    rv_intraday = intraday_returns**2

    # Yang-Zhang estimator: k * rv_overnight + (1 - k) * rv_intraday + (1 - k) * RS
    variance[1:] = k * rv_overnight + (1.0 - k) * rv_intraday + (1.0 - k) * rs_intraday

    return variance


def realized_variance_returns(returns: np.ndarray) -> float:
    """Compute classical realized variance from returns.

    Simple sum of squared returns. This is the baseline estimator.

    Args:
        returns: Array of returns.

    Returns:
        Realized variance (sum of squared returns).

    Example:
        >>> returns = np.array([0.01, -0.02, 0.015])
        >>> rv = realized_variance_returns(returns)
        >>> assert rv > 0
    """
    returns_arr = np.asarray(returns, dtype=float)

    if returns_arr.size == 0:
        return 0.0

    return float(np.sum(returns_arr**2))


def compute_realized_measures(
    data: pd.DataFrame,
    *,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    open_col: str = "Open",
) -> pd.DataFrame:
    """Compute all realized volatility measures from HLOC DataFrame.

    Computes Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang, and
    classical RV from returns.

    Args:
        data: DataFrame with High, Low, Open, Close columns.
        high_col: Name of high price column.
        low_col: Name of low price column.
        close_col: Name of close price column.
        open_col: Name of open price column.

    Returns:
        DataFrame with columns: RV, Parkinson, GarmanKlass, RogersSatchell, YangZhang.

    Raises:
        ValueError: If required columns are missing or prices are invalid.

    Example:
        >>> data = pd.DataFrame({
        ...     "High": [105, 110],
        ...     "Low": [95, 100],
        ...     "Close": [100, 105],
        ...     "Open": [98, 102],
        ... })
        >>> measures = compute_realized_measures(data)
        >>> assert "Parkinson" in measures.columns
    """
    required_cols = [high_col, low_col, close_col, open_col]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        msg = f"missing required columns: {missing}"
        raise ValueError(msg)

    # Convert to numeric arrays explicitly to avoid datetime comparison issues
    high = np.asarray(data[high_col], dtype=float)
    low = np.asarray(data[low_col], dtype=float)
    close = np.asarray(data[close_col], dtype=float)
    open_price = np.asarray(data[open_col], dtype=float)

    # Validate prices
    if np.any(high < 0) or np.any(low < 0) or np.any(close < 0) or np.any(open_price < 0):
        msg = "prices must be positive"
        raise ValueError(msg)

    # Compute all measures
    result = pd.DataFrame(index=data.index)

    # Classical RV from returns
    returns = np.log(close / open_price)
    result["RV"] = returns**2

    # Range-based estimators
    result["Parkinson"] = parkinson_estimator(high, low)
    result["GarmanKlass"] = garman_klass_estimator(high, low, close, open_price)
    result["RogersSatchell"] = rogers_satchell_estimator(high, low, close, open_price)
    result["YangZhang"] = yang_zhang_estimator(high, low, close, open_price)

    return result


def efficiency_ratio(estimator1: np.ndarray, estimator2: np.ndarray) -> float:
    """Compute efficiency ratio between two variance estimators.

    Efficiency ratio = var(estimator1) / var(estimator2).
    A ratio < 1 means estimator1 is more efficient (lower variance).

    Args:
        estimator1: First variance estimator array.
        estimator2: Second variance estimator array.

    Returns:
        Efficiency ratio (variance ratio).

    Example:
        >>> est1 = np.array([0.01, 0.02, 0.015])
        >>> est2 = np.array([0.012, 0.018, 0.016])
        >>> ratio = efficiency_ratio(est1, est2)
    """
    est1_arr = np.asarray(estimator1, dtype=float)
    est2_arr = np.asarray(estimator2, dtype=float)

    if est1_arr.size != est2_arr.size:
        msg = "estimator arrays must match in length"
        raise ValueError(msg)

    # Remove NaN values
    valid_mask = np.isfinite(est1_arr) & np.isfinite(est2_arr)
    if not np.any(valid_mask):
        return float("nan")

    est1_valid = est1_arr[valid_mask]
    est2_valid = est2_arr[valid_mask]

    var1 = float(np.var(est1_valid, ddof=1))
    var2 = float(np.var(est2_valid, ddof=1))

    if var2 == 0.0:
        return float("nan")

    return var1 / var2


def compare_realized_estimators(data: pd.DataFrame) -> pd.DataFrame:
    """Compare all realized volatility estimators.

    Computes summary statistics (mean, std) and efficiency ratios
    relative to classical RV.

    Args:
        data: DataFrame with High, Low, Open, Close columns.

    Returns:
        DataFrame with index=estimator names, columns=mean, std, efficiency_vs_RV.

    Example:
        >>> data = pd.DataFrame({
        ...     "High": [105, 110, 108],
        ...     "Low": [95, 100, 98],
        ...     "Close": [100, 105, 102],
        ...     "Open": [98, 102, 100],
        ... })
        >>> comparison = compare_realized_estimators(data)
        >>> assert "RV" in comparison.index
    """
    measures = compute_realized_measures(data)

    estimators = ["RV", "Parkinson", "GarmanKlass", "RogersSatchell", "YangZhang"]
    results = []
    processed_names = []

    rv_values = np.asarray(measures["RV"], dtype=float)

    for est_name in estimators:
        if est_name not in measures.columns:
            continue

        est_values = np.asarray(measures[est_name], dtype=float)
        valid_mask = np.isfinite(est_values) & np.isfinite(rv_values)

        if not np.any(valid_mask):
            continue

        est_valid = est_values[valid_mask]
        rv_valid = rv_values[valid_mask]

        mean_val = float(np.mean(est_valid))
        std_val = float(np.std(est_valid, ddof=1))

        # Efficiency ratio vs RV
        if est_name == "RV":
            efficiency = 1.0
        else:
            efficiency = efficiency_ratio(est_valid, rv_valid)

        results.append(
            {
                "mean": mean_val,
                "std": std_val,
                "efficiency_vs_RV": efficiency,
            }
        )
        processed_names.append(est_name)

    result_df = pd.DataFrame(results, index=pd.Index(processed_names))
    return result_df


__all__ = [
    "parkinson_estimator",
    "garman_klass_estimator",
    "rogers_satchell_estimator",
    "yang_zhang_estimator",
    "realized_variance_returns",
    "compute_realized_measures",
    "efficiency_ratio",
    "compare_realized_estimators",
]
