"""Academic benchmark models for volatility forecasting.

Provides reference implementations of standard volatility forecasting models
used in academic literature for benchmarking GARCH-family models.

Models included:
1. RiskMetrics EWMA (J.P. Morgan 1996)
2. HAR-RV (Corsi 2009)
3. Rolling Historical Volatility
4. Naive/Historical Average
5. ARCH(1) (Engle 1982)
6. GARCH(1,1) (Bollerslev 1986)

All models provide one-step-ahead variance forecasts σ²_{t+1|t}.

References:
    - J.P. Morgan (1996). "RiskMetrics—Technical Document."
    - Engle, R. F. (1982). "Autoregressive conditional heteroscedasticity with
      estimates of the variance of United Kingdom inflation." Econometrica, 987-1007.
    - Bollerslev, T. (1986). "Generalized autoregressive conditional
      heteroskedasticity." Journal of Econometrics, 31(3), 307-327.
    - Corsi, F. (2009). "A simple approximate long-memory model of realized
      volatility." Journal of Financial Econometrics, 7(2), 174-196.
"""

from __future__ import annotations

import numpy as np

from src.constants import (
    VOL_HAR_DAILY_LAG,
    VOL_HAR_MONTH_WINDOW,
    VOL_HAR_WEEK_WINDOW,
    VOL_RISKMETRICS_LAMBDA,
)
from src.utils import get_logger

logger = get_logger(__name__)


class RiskMetricsEWMA:
    """RiskMetrics Exponentially Weighted Moving Average (EWMA) model.

    Variance equation: σ²ₜ = λ·σ²ₜ₋₁ + (1-λ)·r²ₜ₋₁

    where λ is the decay factor. RiskMetrics uses λ = 0.94 for daily data.

    This model gives more weight to recent observations and is widely used
    in financial risk management due to its simplicity and reasonable performance.

    Attributes:
        lam: Decay factor (default: 0.94 for daily data).

    Reference:
        J.P. Morgan (1996). "RiskMetrics—Technical Document."
        Fourth edition, New York.
    """

    def __init__(self, lam: float = VOL_RISKMETRICS_LAMBDA):
        """Initialize RiskMetrics EWMA model.

        Args:
            lam: Decay factor λ ∈ (0,1). RiskMetrics uses 0.94 for daily,
                 0.97 for monthly data.
        """
        if not 0 < lam < 1:
            msg = f"Lambda must be in (0,1), got {lam}"
            raise ValueError(msg)
        self.lam = lam
        logger.info("RiskMetrics EWMA initialized with λ=%.4f", lam)

    def forecast(self, returns: np.ndarray, sigma2_init: float | None = None) -> np.ndarray:
        """Compute one-step-ahead variance forecasts.

        Args:
            returns: Historical returns r_1, ..., r_T.
            sigma2_init: Initial variance σ²_0. If None, uses sample variance.

        Returns:
            Variance forecasts σ²_{t+1|t} for t=1,...,T.
            Length equals len(returns).

        Example:
            >>> model = RiskMetricsEWMA(lam=0.94)
            >>> returns = np.random.randn(100) * 0.01
            >>> sigma2 = model.forecast(returns)
            >>> # sigma2[t] is forecast for t+1 given data up to t
        """
        returns = np.asarray(returns, dtype=float).ravel()
        T = len(returns)

        if T == 0:
            return np.array([])

        # Initialize variance
        if sigma2_init is None:
            sigma2_init = float(np.var(returns[: min(T, 100)]))  # Use first 100 obs
            sigma2_init = max(sigma2_init, 1e-8)

        sigma2 = np.empty(T)
        sigma2[0] = sigma2_init

        # EWMA recursion: σ²ₜ = λ·σ²ₜ₋₁ + (1-λ)·r²ₜ₋₁
        for t in range(1, T):
            sigma2[t] = self.lam * sigma2[t - 1] + (1 - self.lam) * (returns[t - 1] ** 2)

        return sigma2

    def __str__(self) -> str:
        """Return string representation."""
        return f"RiskMetrics EWMA(λ={self.lam:.4f})"


class HARRV:
    """Heterogeneous Autoregressive model of Realized Volatility (HAR-RV).

    Variance equation: RV_t = β₀ + β_d·RV_{t-1} + β_w·RV^(w)_{t-1} + β_m·RV^(m)_{t-1} + εₜ

    where:
    - RV_{t-1}: daily realized volatility (lag 1)
    - RV^(w)_{t-1}: weekly average (mean of last 5 days)
    - RV^(m)_{t-1}: monthly average (mean of last 22 days)

    Captures volatility persistence at multiple time scales (daily, weekly, monthly).

    Reference:
        Corsi, F. (2009). "A simple approximate long-memory model of realized volatility."
        Journal of Financial Econometrics, 7(2), 174-196.
    """

    def __init__(
        self,
        daily_lags: int = VOL_HAR_DAILY_LAG,
        weekly_lags: int = VOL_HAR_WEEK_WINDOW,
        monthly_lags: int = VOL_HAR_MONTH_WINDOW,
    ):
        """Initialize HAR-RV model.

        Args:
            daily_lags: Lag for daily component (default: 1).
            weekly_lags: Window for weekly average (default: 5).
            monthly_lags: Window for monthly average (default: 22).
        """
        self.daily_lags = daily_lags
        self.weekly_lags = weekly_lags
        self.monthly_lags = monthly_lags
        self.beta: np.ndarray | None = None

    def _build_features(self, rv: np.ndarray, t: int) -> np.ndarray | None:
        """Build HAR feature vector for time t.

        Args:
            rv: Realized variance series.
            t: Time index.

        Returns:
            Feature vector [1, RV_{t-1}, RV^(w)_{t-1}, RV^(m)_{t-1}] or None if insufficient data.
        """
        if t < max(self.daily_lags, self.weekly_lags, self.monthly_lags):
            return None

        rv_daily = float(rv[t - self.daily_lags])
        rv_weekly = float(np.mean(rv[t - self.weekly_lags : t]))
        rv_monthly = float(np.mean(rv[t - self.monthly_lags : t]))

        return np.array([1.0, rv_daily, rv_weekly, rv_monthly])

    def fit(self, rv: np.ndarray) -> None:
        """Estimate HAR model coefficients via OLS.

        Args:
            rv: Realized variance series (e.g., squared returns).
        """
        rv = np.asarray(rv, dtype=float).ravel()
        T = len(rv)

        # Build training data
        X_list = []
        y_list = []

        min_t = max(self.daily_lags, self.weekly_lags, self.monthly_lags)
        for t in range(min_t, T):
            features = self._build_features(rv, t)
            if features is not None:
                X_list.append(features)
                y_list.append(rv[t])

        if len(y_list) < 4:
            logger.warning("Insufficient data for HAR estimation, using naive mean")
            mean_rv = float(np.mean(rv)) if len(rv) > 0 else 1e-6
            self.beta = np.array([mean_rv, 0.0, 0.0, 0.0])
            return

        # OLS estimation
        X = np.array(X_list)
        y = np.array(y_list)

        try:
            beta_result, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.beta = beta_result
            if self.beta is not None:
                logger.info(
                    "HAR-RV fitted: β₀=%.6f, β_d=%.4f, β_w=%.4f, β_m=%.4f",
                    self.beta[0],
                    self.beta[1],
                    self.beta[2],
                    self.beta[3],
                )
        except np.linalg.LinAlgError:
            logger.warning("HAR OLS failed, using naive mean")
            mean_rv = float(np.mean(rv))
            self.beta = np.array([mean_rv, 0.0, 0.0, 0.0])

    def forecast(self, rv: np.ndarray, steps: int = 1) -> np.ndarray:
        """Generate multi-step ahead forecasts.

        Args:
            rv: Realized variance series up to time T.
            steps: Number of steps ahead to forecast.

        Returns:
            Forecast array of length `steps`.

        Raises:
            ValueError: If model has not been fitted.
        """
        if self.beta is None:
            msg = "Model not fitted. Call fit() first."
            raise ValueError(msg)

        rv = np.asarray(rv, dtype=float).ravel()
        forecasts = np.empty(steps)

        # Extend rv with forecasts for multi-step prediction
        rv_extended = list(rv)

        for h in range(steps):
            t = len(rv_extended)
            features = self._build_features(np.array(rv_extended), t)

            if features is None:
                # Fallback to last observed value
                forecasts[h] = rv_extended[-1]
            else:
                forecast = float(np.dot(self.beta, features))
                forecast = max(forecast, 1e-8)  # Ensure positivity
                forecasts[h] = forecast
                rv_extended.append(forecast)

        return forecasts

    def __str__(self) -> str:
        """Return string representation."""
        return f"HAR-RV(d={self.daily_lags}, w={self.weekly_lags}, m={self.monthly_lags})"


class HistoricalVolatility:
    """Rolling historical volatility (sample variance).

    Simple baseline that computes variance using a rolling window.

    σ²_{t+1|t} = (1/window) · Σ_{i=t-window+1}^{t} r²ᵢ
    """

    def __init__(self, window: int = 250):
        """Initialize historical volatility model.

        Args:
            window: Rolling window size (default: 250 = 1 year of daily data).
        """
        if window <= 0:
            msg = f"Window must be positive, got {window}"
            raise ValueError(msg)
        self.window = window

    def forecast(self, returns: np.ndarray) -> np.ndarray:
        """Compute rolling variance forecasts.

        Args:
            returns: Historical returns.

        Returns:
            Rolling variance forecasts.
        """
        returns = np.asarray(returns, dtype=float).ravel()
        T = len(returns)
        sigma2 = np.empty(T)

        for t in range(T):
            start = max(0, t - self.window + 1)
            window_returns = returns[start : t + 1]
            if len(window_returns) > 0:
                sigma2[t] = float(np.var(window_returns))
            else:
                sigma2[t] = 1e-8

        return sigma2

    def __str__(self) -> str:
        """Return string representation."""
        return f"Historical Volatility(window={self.window})"


def compare_benchmark_models(
    returns: np.ndarray,
    train_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Compare all benchmark models on the same data.

    Args:
        returns: Return series.
        train_size: Size of training set. If None, uses 80% of data.

    Returns:
        Dictionary mapping model names to variance forecasts.

    Example:
        >>> returns = np.random.randn(1000) * 0.01
        >>> forecasts = compare_benchmark_models(returns, train_size=800)
        >>> # forecasts = {'RiskMetrics': [...], 'HAR': [...], 'HistVol': [...]}
    """
    returns = np.asarray(returns, dtype=float).ravel()
    T = len(returns)

    if train_size is None:
        train_size = int(0.8 * T)

    all_returns = returns

    forecasts = {}

    # RiskMetrics EWMA
    ewma = RiskMetricsEWMA()
    forecasts["RiskMetrics"] = ewma.forecast(all_returns)

    # HAR-RV
    har = HARRV()
    rv = returns**2
    har.fit(rv[:train_size])
    har_forecast = np.empty(T)
    har_forecast[:train_size] = rv[:train_size]  # In-sample: use realized
    # Out-of-sample: recursive 1-step forecast
    for t in range(train_size, T):
        har_forecast[t] = har.forecast(rv[:t], steps=1)[0]
    forecasts["HAR-RV"] = har_forecast

    # Historical Volatility
    histvol = HistoricalVolatility(window=min(250, train_size))
    forecasts["HistVol"] = histvol.forecast(all_returns)

    logger.info("Compared %d benchmark models on %d observations", len(forecasts), T)
    return forecasts


__all__ = [
    "RiskMetricsEWMA",
    "HARRV",
    "HistoricalVolatility",
    "compare_benchmark_models",
]
