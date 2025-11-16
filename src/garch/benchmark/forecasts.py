"""Volatility forecast functions for baseline models.

This module provides wrapper functions that adapt the class-based models
from benchmark_models.py to work with the rolling forecast API used in
volatility backtesting (using e_valid arrays and pos_test indices).
"""

from __future__ import annotations

import numpy as np

import src.constants as C

from .benchmark_models import HARRV, HistoricalVolatility, RiskMetricsEWMA


def ewma_forecast(e_valid: np.ndarray, pos_test: np.ndarray, lam: float) -> np.ndarray:
    """One-step EWMA variance: s2_{t+1} = lam*s2_t + (1-lam)*e_t^2.

    Wrapper around RiskMetricsEWMA class adapted for rolling forecast API.

    Args:
        e_valid: Valid residuals array.
        pos_test: Test positions in valid array.
        lam: EWMA decay factor (lambda).

    Returns:
        EWMA variance forecasts for test positions.
    """
    if pos_test.size == 0:
        return np.array([], dtype=float)

    # Use the unified RiskMetricsEWMA class
    model = RiskMetricsEWMA(lam=lam)

    # Compute forecasts for all data
    # RiskMetricsEWMA.forecast() returns sigma2 where sigma2[t] is forecast for t+1
    # given data up to t. But the recursion uses returns[t-1] to compute sigma2[t],
    # so sigma2[t] is based on data up to t-1
    all_forecasts = model.forecast(e_valid)

    # Extract forecasts at test positions
    out = np.empty(pos_test.size, dtype=float)
    for i, pos in enumerate(pos_test):
        # sigma2[pos] uses e_valid[pos-1], which is what we want
        out[i] = all_forecasts[int(pos)]

    return out


def rolling_var_forecast(e_valid: np.ndarray, pos_test: np.ndarray, window: int) -> np.ndarray:
    """One-step rolling variance forecast using last `window` observations.

    Wrapper around HistoricalVolatility class adapted for rolling forecast API.

    Args:
        e_valid: Valid residuals array.
        pos_test: Test positions in valid array.
        window: Rolling window size.

    Returns:
        Rolling variance forecasts for test positions.
    """
    if pos_test.size == 0:
        return np.array([], dtype=float)

    # Use the unified HistoricalVolatility class
    model = HistoricalVolatility(window=window)

    # Compute forecasts for all data
    # HistoricalVolatility.forecast() computes sigma2[t] using data up to t (inclusive)
    # For one-step-ahead forecast at position pos (using data up to pos-1),
    # we need sigma2[pos-1]
    all_forecasts = model.forecast(e_valid)

    # Extract forecasts at test positions (with one-step-ahead adjustment)
    out = np.empty(pos_test.size, dtype=float)
    for i, pos in enumerate(pos_test):
        if pos > 0:
            out[i] = max(all_forecasts[int(pos) - 1], C.VOL_MIN_VARIANCE)
        else:
            # Edge case: if pos==0, use first available variance
            out[i] = max(all_forecasts[0], C.VOL_MIN_VARIANCE)

    return out


def naive_forecast(e_valid: np.ndarray, pos_test: np.ndarray) -> np.ndarray:
    """Naive model: constant variance (historical mean).

    Args:
        e_valid: Valid residuals array.
        pos_test: Test positions in valid array.

    Returns:
        Naive variance forecasts (constant).
    """
    if pos_test.size == 0:
        return np.array([], dtype=float)
    p0 = int(pos_test[0])
    train_var = float(np.var(e_valid[:p0])) if p0 > 1 else float(np.var(e_valid[: max(1, p0)]))
    train_var = max(train_var, C.VOL_MIN_VARIANCE)
    return np.full(pos_test.size, train_var, dtype=float)


def garch11_forecast(e_valid: np.ndarray, pos_test: np.ndarray) -> np.ndarray:
    """GARCH(1,1): sigma2_{t+1} = omega + alpha * e_t^2 + beta * sigma2_t.

    Estimated on training data, then forecasted recursively.

    Args:
        e_valid: Valid residuals array.
        pos_test: Test positions in valid array.

    Returns:
        GARCH(1,1) variance forecasts.
    """
    out = np.empty(pos_test.size, dtype=float)
    if pos_test.size == 0:
        return out

    p0 = int(pos_test[0])
    train = e_valid[:p0]

    if train.size <= 2:
        omega, alpha, beta = C.VOL_MIN_OMEGA, 0.0, 0.0
    else:
        # Estimate GARCH(1,1) using OLS approximation on squared residuals
        # y_t = e_t^2, regress on constant, e_{t-1}^2, and y_{t-1}
        # y_t = e_t^2 for t >= 2 (need lag)
        y = (train[2:] ** 2).astype(float)  # y_t for t=2,3,...
        x_lag1 = (train[1:-1] ** 2).astype(float)  # e_{t-1}^2 for t=2,3,...
        y_lag1 = (train[1:-1] ** 2).astype(float)  # y_{t-1} = e_{t-1}^2 for t=2,3,...

        # All arrays should have same size now
        X = np.column_stack([np.ones_like(y), x_lag1, y_lag1])
        y_reg = y
        X_reg = X

        try:
            beta_coefs, *_ = np.linalg.lstsq(X_reg, y_reg, rcond=None)
            omega = float(max(beta_coefs[0], C.VOL_MIN_VARIANCE))
            alpha = float(max(beta_coefs[1], 0.0))
            beta = float(max(min(beta_coefs[2], 0.99), 0.0))
            # Ensure stationarity: alpha + beta < 1
            if alpha + beta >= 1.0:
                total = alpha + beta
                alpha = alpha / total * 0.99
                beta = beta / total * 0.99
        except Exception:
            omega, alpha, beta = C.VOL_MIN_OMEGA, 0.0, 0.0

    # Initialize variance for first forecast
    s2_prev = float(np.var(train)) if train.size > 1 else C.VOL_MIN_VARIANCE
    s2_prev = max(s2_prev, C.VOL_MIN_VARIANCE)

    # Forecast recursively
    for i, pos in enumerate(pos_test):
        e_prev = float(e_valid[int(pos) - 1])
        s2_next = omega + alpha * (e_prev**2) + beta * s2_prev
        s2_next = max(s2_next, C.VOL_MIN_VARIANCE)
        out[i] = s2_next
        s2_prev = s2_next

    return out


def arch1_forecast(e_valid: np.ndarray, pos_test: np.ndarray) -> np.ndarray:
    """ARCH(1): sigma2_{t+1} = omega + alpha * e_t^2 (estimated on train).

    Args:
        e_valid: Valid residuals array.
        pos_test: Test positions in valid array.

    Returns:
        ARCH(1) variance forecasts.
    """
    out = np.empty(pos_test.size, dtype=float)
    if pos_test.size == 0:
        return out
    p0 = int(pos_test[0])
    train = e_valid[:p0]
    if train.size <= 2:
        omega, alpha = C.VOL_MIN_OMEGA, 0.0
    else:
        y = (train[1:] ** 2).astype(float)
        x = (train[:-1] ** 2).astype(float)
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        omega = float(max(beta[0], C.VOL_MIN_VARIANCE))
        alpha = float(max(beta[1], 0.0))
    for i, pos in enumerate(pos_test):
        out[i] = float(omega + alpha * float(e_valid[int(pos) - 1] ** 2))
        out[i] = max(out[i], C.VOL_MIN_VARIANCE)
    return out


def har3_forecast(e_valid: np.ndarray, pos_test: np.ndarray) -> np.ndarray:
    """HAR(3) on squared returns: d (lag1), w (5-day avg), m (22-day avg).

    Wrapper around HARRV class adapted for rolling forecast API.

    Args:
        e_valid: Valid residuals array.
        pos_test: Test positions in valid array.

    Returns:
        HAR(3) variance forecasts for test positions.
    """
    if pos_test.size == 0:
        return np.array([], dtype=float)

    # Use the unified HARRV class
    model = HARRV()

    # Compute realized variance (squared residuals)
    rv = (e_valid**2).astype(float)

    # Fit model on training data
    p0 = int(pos_test[0])
    model.fit(rv[:p0])

    # Generate forecasts for test positions
    out = np.empty(pos_test.size, dtype=float)
    for i, pos in enumerate(pos_test):
        t = int(pos)
        # Use private method _build_features (ideally should be public)
        features = model._build_features(rv, t)
        if features is not None and model.beta is not None:
            s2 = float(np.dot(model.beta, features))
        else:
            # Fallback: use historical variance
            s2 = float(np.var(rv[:t])) if t > 1 else float(np.var(rv[: max(1, t)]))
        out[i] = max(s2, C.VOL_MIN_VARIANCE)

    return out


def compute_baseline_forecasts(
    e_valid: np.ndarray,
    pos_test: np.ndarray,
    ewma_lambda: float,
    rolling_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute all baseline volatility forecasts.

    Args:
        e_valid: Valid residuals array.
        pos_test: Test positions in valid array.
        ewma_lambda: EWMA decay factor.
        rolling_window: Rolling window size.

    Returns:
        Tuple of (s2_ewma, s2_garch11, s2_roll_var, s2_naive, s2_arch1, s2_har3).
    """
    s2_ewma = ewma_forecast(e_valid, pos_test, lam=ewma_lambda)
    s2_garch11 = garch11_forecast(e_valid, pos_test)
    s2_roll_var = rolling_var_forecast(e_valid, pos_test, window=rolling_window)
    s2_naive = naive_forecast(e_valid, pos_test)
    s2_arch1 = arch1_forecast(e_valid, pos_test)
    s2_har3 = har3_forecast(e_valid, pos_test)
    return s2_ewma, s2_garch11, s2_roll_var, s2_naive, s2_arch1, s2_har3
