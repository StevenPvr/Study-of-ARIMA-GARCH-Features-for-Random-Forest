"""Tests for leakage_test module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.constants import (
    LEAKAGE_R2_THRESHOLD,
    LIGHTGBM_COMPLETE_MODEL_NAME,
    LIGHTGBM_DATASET_COMPLETE_MODEL_NAME,
    TEST_SPLIT_LABEL,
    TRAIN_SPLIT_LABEL,
)
from src.lightgbm.data_leakage_checkup.leakage_test import (
    _compute_delta_metrics,
    _compute_metrics,
    _detect_leakage,
    _load_json_results,
    _log_leakage_result,
    _log_metrics_comparison,
    _log_test_header,
    _shuffle_target,
)


class TestLoadJsonResults:
    """Tests for _load_json_results function."""

    def test_load_json_results_success(self):
        """Test successful loading of JSON results."""
        test_data = {"key1": {"nested": "value"}, "key2": "value2"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            result = _load_json_results(temp_path, "key1", "test results")
            assert result == {"nested": "value"}
        finally:
            temp_path.unlink()

    def test_load_json_results_file_not_found(self):
        """Test FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="test results not found"):
            _load_json_results(Path("/nonexistent/file.json"), "key", "test results")

    def test_load_json_results_key_not_found(self):
        """Test KeyError when key doesn't exist."""
        test_data = {"existing_key": "value"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(KeyError, match="nonexistent_key not found in test results"):
                _load_json_results(temp_path, "nonexistent_key", "test results")
        finally:
            temp_path.unlink()


class TestComputeMetrics:
    """Tests for _compute_metrics function."""

    def test_compute_metrics_perfect_predictions(self):
        """Test metrics computation with perfect predictions."""
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        metrics = _compute_metrics(y_true, y_pred)

        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 1.0

    def test_compute_metrics_with_errors(self):
        """Test metrics computation with prediction errors."""
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])

        metrics = _compute_metrics(y_true, y_pred)

        assert metrics["mae"] > 0.0
        assert metrics["mse"] > 0.0
        assert metrics["rmse"] > 0.0
        assert metrics["r2"] < 1.0


class TestShuffleTarget:
    """Tests for _shuffle_target function."""

    def test_shuffle_target_preserves_index(self):
        """Test that shuffling preserves original index."""
        original_index = pd.Index([10, 20, 30])
        y_train = pd.Series([1.0, 2.0, 3.0], index=original_index)

        result = _shuffle_target(y_train, random_state=42)

        # Index should be preserved
        pd.testing.assert_index_equal(result.index, original_index)

        # Values should be shuffled (but we can't test exact shuffle due to randomness)
        assert len(result) == len(y_train)
        assert set(result.values) == set(y_train.values)

    def test_shuffle_target_different_random_state(self):
        """Test that different random states produce different shuffles."""
        y_train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        result1 = _shuffle_target(y_train, random_state=1)
        result2 = _shuffle_target(y_train, random_state=2)

        # Results should be different (with high probability for 5 elements)
        assert not result1.equals(result2)


class TestComputeDeltaMetrics:
    """Tests for _compute_delta_metrics function."""

    def test_compute_delta_metrics(self):
        """Test delta metrics computation."""
        shuffled = {"mae": 2.0, "mse": 4.0, "rmse": 2.0, "r2": 0.5}
        original = {"mae": 1.0, "mse": 1.0, "rmse": 1.0, "r2": 0.8}

        delta = _compute_delta_metrics(shuffled, original)

        assert delta["delta_mae"] == 1.0
        assert delta["delta_mse"] == 3.0
        assert delta["delta_rmse"] == 1.0
        assert abs(delta["delta_r2"] - (-0.3)) < 1e-10  # Handle floating point precision


class TestDetectLeakage:
    """Tests for _detect_leakage function."""

    def test_detect_leakage_above_threshold(self):
        """Test leakage detection when R² is above threshold."""
        assert _detect_leakage(LEAKAGE_R2_THRESHOLD + 0.1) is True

    def test_detect_leakage_below_threshold(self):
        """Test leakage detection when R² is below threshold."""
        assert _detect_leakage(LEAKAGE_R2_THRESHOLD - 0.1) is False

    def test_detect_leakage_at_threshold(self):
        """Test leakage detection when R² is exactly at threshold."""
        assert _detect_leakage(LEAKAGE_R2_THRESHOLD) is False


class TestLogFunctions:
    """Tests for logging functions."""

    @patch("src.lightgbm.data_leakage_checkup.leakage_test.logger")
    def test_log_test_header(self, mock_logger):
        """Test test header logging."""
        _log_test_header()

        mock_logger.info.assert_called()
        assert mock_logger.info.call_count == 3

    @patch("src.lightgbm.data_leakage_checkup.leakage_test.logger")
    def test_log_metrics_comparison(self, mock_logger):
        """Test metrics comparison logging."""
        shuffled = {"mae": 1.0, "mse": 2.0, "rmse": 1.5, "r2": 0.8}
        original = {"mae": 0.5, "mse": 1.0, "rmse": 1.0, "r2": 0.9}

        _log_metrics_comparison(shuffled, original)

        assert (
            mock_logger.info.call_count == 10
        )  # 5 shuffled (header + 4 metrics) + 5 original (header + 4 metrics)

    @patch("src.lightgbm.data_leakage_checkup.leakage_test.logger")
    def test_log_leakage_result_no_leakage(self, mock_logger):
        """Test logging when no leakage detected."""
        result = _log_leakage_result(False, 0.05)

        assert "NO LEAKAGE" in result
        mock_logger.info.assert_called()
        mock_logger.warning.assert_not_called()

    @patch("src.lightgbm.data_leakage_checkup.leakage_test.logger")
    def test_log_leakage_result_with_leakage(self, mock_logger):
        """Test logging when leakage is detected."""
        result = _log_leakage_result(True, 0.15)

        assert "LEAKAGE DETECTED" in result
        mock_logger.warning.assert_called()
        mock_logger.info.assert_called()


class TestConstants:
    """Tests for constant usage."""

    def test_split_labels_are_strings(self):
        """Test that split labels are proper strings."""
        assert isinstance(TRAIN_SPLIT_LABEL, str)
        assert isinstance(TEST_SPLIT_LABEL, str)
        assert TRAIN_SPLIT_LABEL == "train"
        assert TEST_SPLIT_LABEL == "test"

    def test_model_names_are_strings(self):
        """Test that model names are proper strings."""
        assert isinstance(LIGHTGBM_COMPLETE_MODEL_NAME, str)
        assert isinstance(LIGHTGBM_DATASET_COMPLETE_MODEL_NAME, str)

    def test_leakage_threshold_is_float(self):
        """Test that leakage threshold is a proper float."""
        assert isinstance(LEAKAGE_R2_THRESHOLD, float)
        assert LEAKAGE_R2_THRESHOLD == 0.1
