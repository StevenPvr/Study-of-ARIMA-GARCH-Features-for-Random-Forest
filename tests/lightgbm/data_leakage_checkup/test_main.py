"""Tests for main module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.constants import (
    DEFAULT_RANDOM_STATE,
    LIGHTGBM_DATASET_COMPLETE_FILE,
    LIGHTGBM_LEAKAGE_TEST_RESULTS_FILE,
)
from src.lightgbm.data_leakage_checkup.main import _save_test_results, main


class TestSaveTestResults:
    """Tests for _save_test_results function."""

    @patch("src.lightgbm.data_leakage_checkup.main.ensure_output_dir")
    @patch("src.lightgbm.data_leakage_checkup.main.save_json_pretty")
    @patch("src.lightgbm.data_leakage_checkup.main.logger")
    def test_save_test_results(self, mock_logger, mock_save_json, mock_ensure_dir):
        """Test saving test results."""
        test_results = {"test_name": "target_shuffle", "leakage_detected": False}

        _save_test_results(test_results)

        mock_ensure_dir.assert_called_once_with(LIGHTGBM_LEAKAGE_TEST_RESULTS_FILE)
        mock_save_json.assert_called_once_with(test_results, LIGHTGBM_LEAKAGE_TEST_RESULTS_FILE)
        mock_logger.info.assert_called_once()


class TestMain:
    """Tests for main function."""

    @patch("src.lightgbm.data_leakage_checkup.main._save_test_results")
    @patch("src.lightgbm.data_leakage_checkup.main.run_target_shuffle_test")
    @patch("src.lightgbm.data_leakage_checkup.main.resolve_dataset_path")
    def test_main_success_no_leakage(self, mock_resolve_path, mock_run_test, mock_save_results):
        """Test main function when no leakage is detected."""
        # Setup mocks
        mock_resolve_path.return_value = Path("/fake/dataset.parquet")
        mock_run_test.return_value = {"leakage_detected": False, "interpretation": "NO LEAKAGE"}

        # Run main
        result = main(dataset_path=Path("/input/dataset.parquet"), random_state=123)

        # Verify calls
        mock_resolve_path.assert_called_once_with(
            Path("/input/dataset.parquet"), LIGHTGBM_DATASET_COMPLETE_FILE
        )
        mock_run_test.assert_called_once_with(
            dataset_path=Path("/fake/dataset.parquet"), random_state=123
        )
        mock_save_results.assert_called_once_with(result)

        # Verify return value
        assert result["leakage_detected"] is False

    @patch("src.lightgbm.data_leakage_checkup.main._save_test_results")
    @patch("src.lightgbm.data_leakage_checkup.main.run_target_shuffle_test")
    @patch("src.lightgbm.data_leakage_checkup.main.resolve_dataset_path")
    @patch("src.lightgbm.data_leakage_checkup.main.logger")
    def test_main_with_leakage(
        self, mock_logger, mock_resolve_path, mock_run_test, mock_save_results
    ):
        """Test main function when leakage is detected."""
        # Setup mocks
        mock_resolve_path.return_value = Path("/fake/dataset.parquet")
        mock_run_test.return_value = {
            "leakage_detected": True,
            "interpretation": "LEAKAGE DETECTED",
        }

        # Run main - should exit with code 1
        with pytest.raises(SystemExit) as exc_info:
            main(dataset_path=Path("/input/dataset.parquet"), random_state=123)

        assert exc_info.value.code == 1

        # Verify error logging
        mock_logger.error.assert_called_once_with(
            "Leakage test FAILED: Potential data leakage detected!"
        )

    @patch("src.lightgbm.data_leakage_checkup.main._save_test_results")
    @patch("src.lightgbm.data_leakage_checkup.main.run_target_shuffle_test")
    @patch("src.lightgbm.data_leakage_checkup.main.resolve_dataset_path")
    def test_main_with_default_parameters(
        self, mock_resolve_path, mock_run_test, mock_save_results
    ):
        """Test main function with default parameters."""
        # Setup mocks
        mock_resolve_path.return_value = Path("/default/dataset.parquet")
        mock_run_test.return_value = {"leakage_detected": False, "interpretation": "NO LEAKAGE"}

        # Run main with defaults
        result = main()

        # Verify calls with defaults
        mock_resolve_path.assert_called_once()
        mock_run_test.assert_called_once_with(
            dataset_path=Path("/default/dataset.parquet"), random_state=DEFAULT_RANDOM_STATE
        )
        mock_save_results.assert_called_once_with(result)
