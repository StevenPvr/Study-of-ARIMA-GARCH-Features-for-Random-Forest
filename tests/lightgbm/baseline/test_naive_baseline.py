"""Tests for naive baseline model."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.lightgbm.baseline.naive_baseline import (
    _generate_naive_predictions,
    compute_naive_baseline_metrics,
)


class TestGenerateNaivePredictions:
    """Tests for _generate_naive_predictions function."""

    def test_generate_naive_predictions_with_valid_feature(self) -> None:
        """Test naive predictions using log_volatility_lag_1."""
        X_test = pd.DataFrame(
            {
                "log_volatility_lag_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "other_feature": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        y_test = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5])

        result = _generate_naive_predictions(X_test, y_test)

        # Should return the lag_1 values as predictions
        expected = X_test["log_volatility_lag_1"].to_numpy()
        np.testing.assert_array_equal(result, expected)

    def test_generate_naive_predictions_missing_feature(self) -> None:
        """Test that function raises error when log_volatility_lag_1 is missing."""
        X_test = pd.DataFrame({"other_feature": [0.1, 0.2, 0.3, 0.4, 0.5]})
        y_test = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5])

        with pytest.raises(ValueError, match="Feature 'log_volatility_lag_1' not found"):
            _generate_naive_predictions(X_test, y_test)

    def test_generate_naive_predictions_shape(self) -> None:
        """Test that predictions have correct shape."""
        X_test = pd.DataFrame({"log_volatility_lag_1": [1.0, 2.0, 3.0]})
        y_test = pd.Series([1.5, 2.5, 3.5])

        result = _generate_naive_predictions(X_test, y_test)

        assert len(result) == len(y_test)
        assert isinstance(result, np.ndarray)


class TestComputeNaiveBaselineMetrics:
    """Tests for compute_naive_baseline_metrics function."""

    def test_compute_naive_baseline_metrics_return_structure(self) -> None:
        """Test that function returns expected dictionary structure."""
        # Create mock test data
        X_test = pd.DataFrame(
            {"log_volatility_lag_1": [1.0, 2.0, 3.0], "other_feature": [0.1, 0.2, 0.3]}
        )
        y_test = pd.Series([1.5, 2.5, 3.5])

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = Path(f.name)
            pd.DataFrame({"dummy": [1]}).to_parquet(temp_path)

        try:
            with patch("src.lightgbm.baseline.naive_baseline.load_test_data") as mock_load:
                mock_load.return_value = (X_test, y_test)

                result = compute_naive_baseline_metrics(temp_path)

                expected_keys = {
                    "model_name",
                    "test_metrics",
                    "test_size",
                    "n_features",
                    "feature_importances",
                }
                assert set(result.keys()) == expected_keys
                assert result["model_name"] == "naive_persistence_baseline"
                assert result["n_features"] == 1
                assert result["feature_importances"] == {"log_volatility_lag_1": 1.0}
                assert isinstance(result["test_metrics"], dict)
                assert "mae" in result["test_metrics"]

        finally:
            temp_path.unlink()

    def test_compute_naive_baseline_metrics_with_valid_data(self) -> None:
        """Test computation with valid test data."""
        X_test = pd.DataFrame({"log_volatility_lag_1": [1.0, 2.0, 3.0, 4.0, 5.0]})
        y_test = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5])

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = Path(f.name)
            pd.DataFrame({"dummy": [1]}).to_parquet(temp_path)

        try:
            with patch("src.lightgbm.baseline.naive_baseline.load_test_data") as mock_load:
                mock_load.return_value = (X_test, y_test)

                result = compute_naive_baseline_metrics(temp_path)

                assert result["test_size"] == len(X_test)
                test_metrics = cast(dict[str, float], result["test_metrics"])
                assert all(key in test_metrics for key in ["mae", "mse", "rmse", "r2"])

        finally:
            temp_path.unlink()

    def test_compute_naive_baseline_metrics_invalid_file(self) -> None:
        """Test that function raises error for non-existent file."""
        non_existent_path = Path("/non/existent/file.parquet")

        with pytest.raises(FileNotFoundError):
            compute_naive_baseline_metrics(non_existent_path)
