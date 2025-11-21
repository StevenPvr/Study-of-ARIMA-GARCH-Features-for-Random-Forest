"""Tests for random baseline model."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.lightgbm.baseline.baseline import (
    _generate_random_predictions,
    compute_random_baseline_metrics,
)


class TestGenerateRandomPredictions:
    """Tests for _generate_random_predictions function."""

    def test_generate_random_predictions_shape(self) -> None:
        """Test that random predictions have correct shape."""
        y_test = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        result = _generate_random_predictions(y_test, random_state=42)

        assert len(result) == len(y_test)
        assert isinstance(result, np.ndarray)

    def test_generate_random_predictions_distribution(self) -> None:
        """Test that random predictions follow expected distribution."""
        y_test = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 100)  # Large sample

        result = _generate_random_predictions(y_test, random_state=42)

        # Check that predictions are roughly normally distributed
        # with same mean and std as y_test
        assert abs(np.mean(result) - np.mean(y_test)) < 0.1
        assert abs(np.std(result) - np.std(y_test)) < 0.1

    def test_generate_random_predictions_reproducibility(self) -> None:
        """Test that random predictions are reproducible with same seed."""
        y_test = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        result1 = _generate_random_predictions(y_test, random_state=42)
        result2 = _generate_random_predictions(y_test, random_state=42)

        np.testing.assert_array_equal(result1, result2)


class TestComputeRandomBaselineMetrics:
    """Tests for compute_random_baseline_metrics function."""

    def test_compute_random_baseline_metrics_return_structure(self) -> None:
        """Test that function returns expected dictionary structure."""
        # Create mock test data
        test_data = pd.DataFrame(
            {
                "split": ["test"] * 10,
                "log_volatility": np.random.randn(10),
                "feature1": np.random.randn(10),
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = Path(f.name)
            test_data.to_parquet(temp_path)

        try:
            with patch("src.lightgbm.baseline.baseline.load_test_data") as mock_load:
                mock_load.return_value = (
                    pd.DataFrame({"feature1": np.random.randn(10)}),
                    pd.Series(np.random.randn(10)),
                )

                result = compute_random_baseline_metrics(temp_path, random_state=42)

                expected_keys = {
                    "model_name",
                    "test_metrics",
                    "test_size",
                    "n_features",
                    "feature_importances",
                }
                assert set(result.keys()) == expected_keys
                assert result["model_name"] == "random_baseline"
                assert result["n_features"] == 0
                assert result["feature_importances"] == {}
                assert isinstance(result["test_metrics"], dict)
                assert "mae" in result["test_metrics"]

        finally:
            temp_path.unlink()

    def test_compute_random_baseline_metrics_with_valid_data(self) -> None:
        """Test computation with valid test data."""
        X_test = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
        y_test = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = Path(f.name)
            # Create dummy data for the file
            pd.DataFrame({"dummy": [1]}).to_parquet(temp_path)

        try:
            with patch("src.lightgbm.baseline.baseline.load_test_data") as mock_load:
                mock_load.return_value = (X_test, y_test)

                result = compute_random_baseline_metrics(temp_path, random_state=42)

                assert result["test_size"] == len(X_test)
                assert all(key in result["test_metrics"] for key in ["mae", "mse", "rmse", "r2"])  # type: ignore[operator]

        finally:
            temp_path.unlink()

    def test_compute_random_baseline_metrics_invalid_file(self) -> None:
        """Test that function raises error for non-existent file."""
        non_existent_path = Path("/non/existent/file.parquet")

        with pytest.raises(FileNotFoundError):
            compute_random_baseline_metrics(non_existent_path)
