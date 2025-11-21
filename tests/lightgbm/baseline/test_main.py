"""Tests for baseline main module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import json
from unittest.mock import ANY


from src.constants import DEFAULT_RANDOM_STATE
from src.lightgbm.baseline.main import (
    _compute_all_baseline_metrics,
    _load_existing_results,
    _save_results_to_file,
    main,
)


class TestLoadExistingResults:
    """Tests for _load_existing_results function."""

    def test_load_existing_results_when_file_exists(self) -> None:
        """Test loading results from existing file."""
        expected_data = {"model1": {"metric": 0.5}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)
            json.dump(expected_data, f)

        try:
            # Mock the LIGHTGBM_EVAL_RESULTS_FILE to use our temp file
            with patch("src.lightgbm.baseline.main.LIGHTGBM_EVAL_RESULTS_FILE", temp_path):
                result = _load_existing_results()

                assert result == expected_data

        finally:
            temp_path.unlink()

    def test_load_existing_results_when_file_not_exists(self) -> None:
        """Test returning empty dict when file doesn't exist."""
        nonexistent_file = Path("/non/existent/file.json")
        with patch("src.lightgbm.baseline.main.LIGHTGBM_EVAL_RESULTS_FILE", nonexistent_file):
            result = _load_existing_results()

            assert result == {}


class TestComputeAllBaselineMetrics:
    """Tests for _compute_all_baseline_metrics function."""

    def test_compute_all_baseline_metrics_return_structure(self) -> None:
        """Test that function returns results for both baseline models."""
        with (
            patch("src.lightgbm.baseline.main.compute_random_baseline_metrics") as mock_random,
            patch("src.lightgbm.baseline.main.compute_naive_baseline_metrics") as mock_naive,
        ):
            mock_random.return_value = {"model_name": "random", "test_metrics": {"mae": 0.1}}
            mock_naive.return_value = {"model_name": "naive", "test_metrics": {"mae": 0.2}}

            result = _compute_all_baseline_metrics(Path("dummy.parquet"), random_state=42)

            assert "random_baseline" in result
            assert "naive_persistence_baseline" in result
            assert result["random_baseline"]["model_name"] == "random"
            assert result["naive_persistence_baseline"]["model_name"] == "naive"

    def test_compute_all_baseline_metrics_calls_both_functions(self) -> None:
        """Test that both baseline functions are called."""
        dataset_path = Path("dummy.parquet")

        with (
            patch("src.lightgbm.baseline.main.compute_random_baseline_metrics") as mock_random,
            patch("src.lightgbm.baseline.main.compute_naive_baseline_metrics") as mock_naive,
        ):
            mock_random.return_value = {"model_name": "random"}
            mock_naive.return_value = {"model_name": "naive"}

            _compute_all_baseline_metrics(dataset_path, random_state=42)

            mock_random.assert_called_once_with(dataset_path=dataset_path, random_state=42)
            mock_naive.assert_called_once_with(dataset_path=dataset_path)


class TestSaveResultsToFile:
    """Tests for _save_results_to_file function."""

    def test_save_results_to_file_creates_output_dir(self) -> None:
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "subdir" / "results.json"
            assert not output_path.parent.exists()

            results = {"model1": {"metric": 0.5}}

            # Mock _load_existing_results to return empty dict
            with patch("src.lightgbm.baseline.main._load_existing_results") as mock_load:
                mock_load.return_value = {}

                _save_results_to_file(results, output_path)

                assert output_path.exists()
                assert output_path.parent.exists()

                with open(output_path) as f:
                    saved_data = json.load(f)
                    assert saved_data == results

    def test_save_results_to_file_merges_with_existing(self) -> None:
        """Test that new results are merged with existing results."""
        existing_data = {"existing_model": {"metric": 0.3}}
        new_results = {"new_model": {"metric": 0.7}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)
            json.dump(existing_data, f)

        try:
            # Mock _load_existing_results to return our existing data
            with patch("src.lightgbm.baseline.main._load_existing_results") as mock_load:
                mock_load.return_value = existing_data

                _save_results_to_file(new_results, temp_path)

                with open(temp_path) as f:
                    saved_data = json.load(f)

                    assert "existing_model" in saved_data
                    assert "new_model" in saved_data
                    assert saved_data["existing_model"]["metric"] == 0.3
                    assert saved_data["new_model"]["metric"] == 0.7

        finally:
            temp_path.unlink()


class TestMain:
    """Tests for main function."""

    def test_main_with_default_dataset(self) -> None:
        """Test main function with default dataset resolution."""
        with (
            patch("src.lightgbm.baseline.main.resolve_dataset_path") as mock_resolve,
            patch("src.lightgbm.baseline.main._compute_all_baseline_metrics") as mock_compute,
            patch("src.lightgbm.baseline.main._save_results_to_file") as mock_save,
        ):
            mock_resolve.return_value = Path("resolved.parquet")
            mock_compute.return_value = {"results": "dummy"}

            result = main()

            mock_resolve.assert_called_once()
            mock_compute.assert_called_once_with(Path("resolved.parquet"), DEFAULT_RANDOM_STATE)
            mock_save.assert_called_once_with({"results": "dummy"})
            assert result == {"results": "dummy"}

    def test_main_with_custom_dataset(self) -> None:
        """Test main function with custom dataset path."""
        custom_path = Path("custom.parquet")

        with (
            patch("src.lightgbm.baseline.main.resolve_dataset_path") as mock_resolve,
            patch("src.lightgbm.baseline.main._compute_all_baseline_metrics") as mock_compute,
            patch("src.lightgbm.baseline.main._save_results_to_file") as mock_save,
        ):
            mock_resolve.return_value = custom_path
            mock_compute.return_value = {"results": "dummy"}

            result = main(dataset_path=custom_path, random_state=123)

            mock_resolve.assert_called_once_with(custom_path, ANY)
            mock_compute.assert_called_once_with(custom_path, 123)
            mock_save.assert_called_once_with({"results": "dummy"})
            assert result == {"results": "dummy"}
