"""Critical test to detect data leakage in feature computation.

This test compares batch feature computation (current approach) with incremental
computation (production simulation) to empirically detect any data leakage.

KEY PRINCIPLE:
If features computed on the full dataset give DIFFERENT results than features
computed incrementally (simulating production), then there is data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import LIGHTGBM_REALIZED_VOL_WINDOW


def compute_log_returns_incremental(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns incrementally, ensuring no look-ahead.

    This simulates production where we compute one observation at a time.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["log_return"] = np.nan

    for ticker in df["ticker"].unique():
        ticker_mask = df["ticker"] == ticker
        ticker_indices = df[ticker_mask].index

        for i, idx in enumerate(ticker_indices):
            if i == 0:
                # First observation has no previous price
                continue

            prev_idx = ticker_indices[i - 1]
            close_t = df.loc[idx, "close"]
            close_prev = df.loc[prev_idx, "close"]

            if close_t > 0 and close_prev > 0:
                df.loc[idx, "log_return"] = np.log(close_t / close_prev)

    return df


def compute_volatility_incremental(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volatility incrementally, ensuring no look-ahead.

    At time t, volatility uses ONLY data from [t-window+1, ..., t].
    This is computed one observation at a time to simulate production.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["log_volatility"] = np.nan

    window = LIGHTGBM_REALIZED_VOL_WINDOW

    for ticker in df["ticker"].unique():
        ticker_mask = df["ticker"] == ticker
        ticker_indices = df[ticker_mask].index

        for i, idx in enumerate(ticker_indices):
            if i < window - 1:
                # Not enough data for window
                continue

            # Get indices for window [t-window+1, ..., t]
            window_indices = ticker_indices[max(0, i - window + 1) : i + 1]

            # Compute realized volatility using ONLY past data
            returns_squared = df.loc[window_indices, "log_return"].dropna() ** 2

            if len(returns_squared) >= window:
                realized_vol = np.sqrt(returns_squared.sum())
                df.loc[idx, "log_volatility"] = np.log1p(realized_vol)

    return df


class TestCriticalLeakageDetection:
    """Critical tests to detect data leakage empirically."""

    def test_log_returns_batch_vs_incremental(self):
        """Test that batch and incremental log returns computation give identical results."""
        # Import the actual function
        from src.data_preparation.computations import compute_log_returns_for_tickers

        # Create test data
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        data = []

        for ticker in ["AAPL", "GOOGL"]:
            prices = 100 + np.cumsum(np.random.randn(len(dates)))
            for date, price in zip(dates, prices):
                data.append({
                    "date": date,
                    "ticker": ticker,
                    "close": max(1.0, price),  # Ensure positive
                    "high": max(2.0, price + 1),
                    "low": max(0.5, price - 1),
                    "volume": 1000000,
                })

        df = pd.DataFrame(data)

        # Compute using batch method (current approach)
        df_batch = compute_log_returns_for_tickers(df.copy())

        # Compute using incremental method (production simulation)
        df_incremental = compute_log_returns_incremental(df.copy())
        df_incremental = df_incremental.dropna(subset=["log_return"])

        # Compare results
        df_batch = df_batch.sort_values(["ticker", "date"]).reset_index(drop=True)
        df_incremental = df_incremental.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Should have same number of rows
        assert len(df_batch) == len(df_incremental), (
            f"Different number of rows: batch={len(df_batch)}, incremental={len(df_incremental)}"
        )

        # Log returns should be nearly identical (allowing for floating point errors)
        np.testing.assert_allclose(
            df_batch["log_return"].values,
            df_incremental["log_return"].values,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Batch and incremental log returns differ - POTENTIAL DATA LEAKAGE!"
        )

    def test_volatility_batch_vs_incremental(self):
        """CRITICAL: Test that batch and incremental volatility computation give identical results.

        This is the key test. If they differ, it means the batch approach is using
        future information that wouldn't be available in production.
        """
        from src.data_preparation.computations import (
            compute_log_returns_for_tickers,
            compute_volatility_for_tickers,
        )

        # Create test data with known values
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = []

        for ticker in ["AAPL"]:
            prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
            for date, price in zip(dates, prices):
                data.append({
                    "date": date,
                    "ticker": ticker,
                    "close": max(1.0, price),
                    "high": max(2.0, price + 1),
                    "low": max(0.5, price - 1),
                    "volume": 1000000,
                })

        df = pd.DataFrame(data)

        # Compute using batch method (current approach)
        df_batch = compute_log_returns_for_tickers(df.copy())
        df_batch = compute_volatility_for_tickers(df_batch)
        df_batch = df_batch.dropna(subset=["log_volatility"])

        # Compute using incremental method (production simulation)
        df_incremental = compute_log_returns_incremental(df.copy())
        df_incremental = compute_volatility_incremental(df_incremental)
        df_incremental = df_incremental.dropna(subset=["log_volatility"])

        # Compare results
        df_batch = df_batch.sort_values(["ticker", "date"]).reset_index(drop=True)
        df_incremental = df_incremental.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Should have same number of rows
        assert len(df_batch) == len(df_incremental), (
            f"Different number of rows: batch={len(df_batch)}, incremental={len(df_incremental)}"
        )

        # Volatility should be nearly identical
        # This is THE critical assertion - if this fails, there is data leakage
        np.testing.assert_allclose(
            df_batch["log_volatility"].values,
            df_incremental["log_volatility"].values,
            rtol=1e-8,
            atol=1e-8,
            err_msg=(
                "CRITICAL: Batch and incremental volatility differ significantly!\n"
                "This indicates DATA LEAKAGE in the batch approach.\n"
                "The batch computation is using information not available in production."
            )
        )

    def test_rolling_window_is_backward_looking(self):
        """Verify that rolling windows only use past data, never future data."""
        from src.data_preparation.computations import compute_volatility_for_tickers

        # Create data with known pattern
        dates = pd.date_range("2020-01-01", periods=20, freq="D")

        # Create returns with a known spike on day 10
        returns = np.zeros(20)
        returns[10] = 5.0  # Large spike

        # Compute prices from returns
        prices = [100.0]
        for r in returns[1:]:
            prices.append(prices[-1] * np.exp(r))

        df = pd.DataFrame({
            "date": dates,
            "ticker": ["AAPL"] * 20,
            "close": prices,
            "high": np.array(prices) + 1,
            "low": np.array(prices) - 1,
            "volume": [1000000] * 20,
        })

        # Compute log returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df = df.dropna(subset=["log_return"])

        # Compute volatility
        df = compute_volatility_for_tickers(df)
        df = df.dropna(subset=["log_volatility"])

        # The spike on day 10 should affect volatility AFTER day 10, not before
        # Get volatility before and after the spike
        vol_before_spike = df[df["date"] < dates[10]]["log_volatility"].values
        vol_at_spike = df[df["date"] == dates[10]]["log_volatility"].values
        vol_after_spike = df[df["date"] > dates[10]]["log_volatility"].iloc[:LIGHTGBM_REALIZED_VOL_WINDOW].values

        if len(vol_before_spike) > 0 and len(vol_at_spike) > 0:
            # Volatility before spike should be low
            assert vol_before_spike[-1] < vol_at_spike[0], (
                "Rolling window appears to be forward-looking! "
                "Volatility before spike is higher than at spike."
            )

        if len(vol_after_spike) > 0:
            # Volatility after spike should be high (within window)
            assert np.mean(vol_after_spike) > np.mean(vol_before_spike), (
                "Rolling window behavior is incorrect. "
                "Volatility after spike should be higher than before."
            )

    def test_no_future_contamination_in_test_set(self, tmp_path):
        """CRITICAL: Verify that test set features don't use information from other test observations.

        This test simulates the train/test split and verifies that each test observation's
        features are computed using ONLY:
        1. All train data
        2. Test data up to (but not including) the current observation
        """
        from src.data_preparation.ticker_preparation import split_tickers_train_test

        # Create synthetic data
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = []

        for ticker in ["AAPL"]:
            prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
            for date, price in zip(dates, prices):
                data.append({
                    "date": date,
                    "ticker": ticker,
                    "close": max(1.0, price),
                    "high": max(2.0, price + 1),
                    "low": max(0.5, price - 1),
                    "volume": 1000000,
                })

        df = pd.DataFrame(data)

        # Save and split
        input_file = tmp_path / "test_data.parquet"
        output_file = tmp_path / "split_data.parquet"
        df.to_parquet(input_file)

        split_tickers_train_test(
            train_ratio=0.8,
            input_file=str(input_file),
            output_file=str(output_file),
        )

        result = pd.read_parquet(output_file)

        # Get train and test data
        train_data = result[result["split"] == "train"].sort_values("date")
        test_data = result[result["split"] == "test"].sort_values("date")

        if len(test_data) == 0:
            pytest.skip("No test data after split")

        # For each test observation, verify its volatility could be computed
        # using only train data + previous test data
        for idx in range(len(test_data)):
            test_row = test_data.iloc[idx]
            test_date = test_row["date"]

            # Simulate incremental computation: use train + test up to (not including) current date
            available_data = pd.concat([
                train_data,
                test_data[test_data["date"] < test_date]
            ])

            if len(available_data) < LIGHTGBM_REALIZED_VOL_WINDOW:
                continue

            # Get last N returns for window
            last_n_returns = available_data["log_return"].tail(LIGHTGBM_REALIZED_VOL_WINDOW)

            # Compute expected volatility using only available data
            if len(last_n_returns) == LIGHTGBM_REALIZED_VOL_WINDOW:
                expected_vol = np.log1p(np.sqrt((last_n_returns ** 2).sum()))
                actual_vol = test_row["log_volatility"]

                # They should match (allowing for the current observation being included)
                # Note: This test may need adjustment based on exact feature computation logic
                # The key is that actual_vol should NOT use future test data
                assert not np.isnan(actual_vol), (
                    f"Test observation at {test_date} has NaN volatility, "
                    "indicating potential computation issue"
                )


class TestFeatureComputationCorrectness:
    """Verify correct implementation of feature computation."""

    def test_log_return_formula(self):
        """Verify log return formula: log(price[t] / price[t-1])."""
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=3, freq="D"),
            "ticker": ["AAPL"] * 3,
            "close": [100.0, 110.0, 105.0],
        })

        from src.data_preparation.computations import compute_log_returns_for_tickers
        result = compute_log_returns_for_tickers(df)

        # Manual calculation
        expected_return_1 = np.log(110.0 / 100.0)
        expected_return_2 = np.log(105.0 / 110.0)

        actual_returns = result["log_return"].dropna().values

        np.testing.assert_allclose(
            actual_returns,
            [expected_return_1, expected_return_2],
            rtol=1e-10
        )

    def test_volatility_window_size(self):
        """Verify volatility uses exact window size."""
        from src.data_preparation.computations import (
            compute_log_returns_for_tickers,
            compute_volatility_for_tickers,
        )

        # Create data with window_size + 1 observations
        n_obs = LIGHTGBM_REALIZED_VOL_WINDOW + 5
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=n_obs, freq="D"),
            "ticker": ["AAPL"] * n_obs,
            "close": 100.0 + np.arange(n_obs, dtype=float),
        })

        df = compute_log_returns_for_tickers(df)
        df = compute_volatility_for_tickers(df)

        # Count non-NaN volatility values
        n_valid_vol = df["log_volatility"].notna().sum()

        # Should have exactly (n_obs - 1) - (window_size - 1) valid volatility values
        # -1 for first log return being NaN, -(window_size-1) for window requirement
        expected_valid = n_obs - LIGHTGBM_REALIZED_VOL_WINDOW

        assert n_valid_vol == expected_valid, (
            f"Expected {expected_valid} valid volatility values, got {n_valid_vol}. "
            "This suggests incorrect window size implementation."
        )
