"""Tests for data leakage validation in data preparation pipeline.

This module contains critical tests to ensure no future information
leaks into past predictions through rolling window features or temporal splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import LIGHTGBM_REALIZED_VOL_WINDOW
from src.data_preparation.ticker_preparation import (
    _remove_contaminated_test_observations,
    split_tickers_train_test,
)


class TestRollingWindowLeakagePrevention:
    """Test that rolling window features don't leak across train/test boundary."""

    def test_contaminated_observations_removed(self, tmp_path):
        """Test that first N test observations are removed (N = window_size - 1)."""
        # Create synthetic data with clear train/test split
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = []

        for ticker in ["AAPL", "GOOGL"]:
            for date in dates:
                data.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "close": 100.0 + np.random.randn(),
                        "high": 101.0,
                        "low": 99.0,
                        "volume": 1000000,
                    }
                )

        df = pd.DataFrame(data)

        # Save to temp file
        input_file = tmp_path / "test_data.parquet"
        output_file = tmp_path / "split_data.parquet"
        df.to_parquet(input_file)

        # Run split with train_ratio=0.8
        split_tickers_train_test(
            train_ratio=0.8,
            input_file=str(input_file),
            output_file=str(output_file),
        )

        # Load result
        result = pd.read_parquet(output_file)

        # For each ticker, verify first N test observations are removed
        n_contaminated = LIGHTGBM_REALIZED_VOL_WINDOW - 1

        for ticker in ["AAPL", "GOOGL"]:
            ticker_data = result[result["tickers"] == ticker].sort_values("date")

            # Get split date (last train date)
            train_data = ticker_data[ticker_data["split"] == "train"]
            test_data = ticker_data[ticker_data["split"] == "test"]

            if len(train_data) > 0 and len(test_data) > 0:
                split_date = train_data["date"].max()
                first_test_date = test_data["date"].min()

                # Calculate expected first test date (after removing contaminated obs)
                # Expected: split_date + n_contaminated + 1 days
                expected_gap_days = n_contaminated + 1

                actual_gap = (first_test_date - split_date).days

                # Allow some flexibility for weekends/gaps
                assert (
                    actual_gap >= expected_gap_days
                ), f"Ticker {ticker}: First test observation is too close to split date. "
                f"Gap: {actual_gap} days, expected >= {expected_gap_days} days"

    def test_remove_contaminated_observations_function(self):
        """Test the _remove_contaminated_test_observations function directly."""
        # Create test data with known split
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "ticker": ["AAPL"] * 20,
                "close": np.arange(20, dtype=float),
                "split": ["train"] * 16 + ["test"] * 4,  # 16 train, 4 test
            }
        )

        # Remove contaminated observations
        result = _remove_contaminated_test_observations(df)

        # With window_size=5, should remove first 4 test observations
        # So all 4 test observations should be removed
        n_test = (result["split"] == "test").sum()
        expected_remaining = max(0, 4 - (LIGHTGBM_REALIZED_VOL_WINDOW - 1))

        assert n_test == expected_remaining, (
            f"Expected {expected_remaining} test observations remaining, "
            f"but got {n_test}"
        )

    def test_sufficient_test_data_after_cleaning(self):
        """Test that sufficient test data remains after removing contaminated obs."""
        # Create data with enough test observations
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "ticker": ["AAPL"] * 30,
                "close": np.arange(30, dtype=float),
                "split": ["train"] * 20 + ["test"] * 10,  # 20 train, 10 test
            }
        )

        result = _remove_contaminated_test_observations(df)

        n_test = (result["split"] == "test").sum()
        expected = 10 - (LIGHTGBM_REALIZED_VOL_WINDOW - 1)

        assert n_test == expected, f"Expected {expected} test observations, got {n_test}"


class TestTemporalSplitIntegrity:
    """Test that train/test splits maintain temporal ordering."""

    def test_no_temporal_overlap(self, tmp_path):
        """Test that no test observation date is before any train observation date."""
        # Create synthetic data
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = []

        for ticker in ["AAPL", "GOOGL", "MSFT"]:
            for date in dates:
                data.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "close": 100.0 + np.random.randn(),
                        "high": 101.0,
                        "low": 99.0,
                        "volume": 1000000,
                    }
                )

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

        # For each ticker, verify temporal ordering
        for ticker in result["tickers"].unique():
            ticker_data = result[result["tickers"] == ticker]
            train_dates = ticker_data[ticker_data["split"] == "train"]["date"]
            test_dates = ticker_data[ticker_data["split"] == "test"]["date"]

            if len(train_dates) > 0 and len(test_dates) > 0:
                max_train_date = train_dates.max()
                min_test_date = test_dates.min()

                assert min_test_date > max_train_date, (
                    f"Ticker {ticker}: Test dates overlap with train dates. "
                    f"Max train: {max_train_date}, Min test: {min_test_date}"
                )

    def test_features_exist_before_split(self, tmp_path):
        """Test that features are computed before temporal split."""
        # Create synthetic data
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        data = []

        for ticker in ["AAPL"]:
            for date in dates:
                data.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "close": 100.0 + np.random.randn(),
                        "high": 101.0,
                        "low": 99.0,
                        "volume": 1000000,
                    }
                )

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

        # Verify that all rows have log_return and log_volatility
        # (indicating features were computed before split)
        assert "log_return" in result.columns
        assert "log_volatility" in result.columns

        # No NaN values should exist (after cleaning)
        assert result["log_return"].notna().all()
        assert result["log_volatility"].notna().all()


class TestDataLossTransparency:
    """Test that data loss is properly logged and transparent."""

    def test_no_silent_data_loss(self, tmp_path, caplog):
        """Test that any data loss is logged with statistics."""
        # Create data with some problematic values
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        data = []

        for i, date in enumerate(dates):
            close_price = 100.0 if i > 0 else 0.0  # First price is zero
            data.append(
                {
                    "date": date,
                    "ticker": "AAPL",
                    "close": close_price,
                    "high": 101.0,
                    "low": 99.0,
                    "volume": 1000000,
                }
            )

        df = pd.DataFrame(data)

        input_file = tmp_path / "test_data.parquet"
        output_file = tmp_path / "split_data.parquet"
        df.to_parquet(input_file)

        # Run split
        split_tickers_train_test(
            train_ratio=0.8,
            input_file=str(input_file),
            output_file=str(output_file),
        )

        # Verify logging occurred
        log_messages = [record.message for record in caplog.records]

        # Should have messages about:
        # 1. Non-positive prices detected
        # 2. Ticker contamination warnings
        # 3. NaN handling

        assert any("non-positive close prices" in msg for msg in log_messages)
        assert any("contaminated" in msg for msg in log_messages)
        assert any("NaN" in msg for msg in log_messages)
