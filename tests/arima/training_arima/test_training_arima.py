"""Unit tests for training_arima module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.arima.training_arima.training_arima import (
    load_trained_model,
    save_trained_model,
    train_arima_model,
    train_best_model,
)
from src.constants import ARIMA_DEFAULT_ORDER, ARIMA_DEFAULT_REFIT_EVERY
from src.arima.training_arima.utils import validate_arima_parameters


class TestTrainArimaModel:
    """Tests for train_arima_model function."""

    @patch("src.arima.training_arima.training_arima.fit_arima_model")
    @patch("src.arima.training_arima.training_arima.logger")
    def test_train_arima_model_success(
        self,
        mock_logger: MagicMock,
        mock_fit_arima: MagicMock,
    ) -> None:
        """Test successful ARIMA model training."""
        train_series = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 100)
        order = (1, 0, 1)

        mock_fitted = MagicMock()
        mock_fitted.aic = -100.0
        mock_fit_arima.return_value = mock_fitted

        fitted_model = train_arima_model(train_series, order)

        assert fitted_model == mock_fitted
        mock_fit_arima.assert_called_once_with(train_series, order=order, verbose=False)

    @patch("src.arima.training_arima.training_arima.fit_arima_model")
    def test_train_arima_model_failure(self, mock_fit_arima: MagicMock) -> None:
        """Test ARIMA model training failure."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 10)
        order = (1, 0, 1)

        mock_fit_arima.side_effect = Exception("Training failed")

        with pytest.raises(RuntimeError, match="Failed to train"):
            train_arima_model(train_series, order)

    @patch("src.arima.training_arima.training_arima.fit_arima_model")
    def test_train_arima_model_runtime_error(self, mock_fit_arima: MagicMock) -> None:
        """Test ARIMA model training with RuntimeError."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 10)
        order = (1, 0, 1)

        mock_fit_arima.side_effect = RuntimeError("Model convergence failed")

        with pytest.raises(RuntimeError, match="Model convergence failed"):
            train_arima_model(train_series, order)


class TestTrainBestModel:
    """Tests for train_best_model function (fixed ARIMA)."""

    @patch("src.arima.training_arima.training_arima.train_arima_model")
    @patch("src.arima.training_arima.training_arima.logger")
    def test_train_best_model_trains_fixed_order(
        self,
        _mock_logger: MagicMock,
        mock_train: MagicMock,
    ) -> None:
        """Ensure train_best_model trains with ARIMA_DEFAULT_ORDER and exports metadata."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 100)

        mock_fitted = MagicMock()
        mock_fitted.aic = -100.0
        mock_train.return_value = mock_fitted

        fitted_model, model_info = train_best_model(train_series)

        assert fitted_model == mock_fitted
        mock_train.assert_called_once_with(train_series, ARIMA_DEFAULT_ORDER)
        assert model_info["order"] == ARIMA_DEFAULT_ORDER
        assert "params" in model_info
        assert model_info["params"]["refit_every"] == ARIMA_DEFAULT_REFIT_EVERY

    def test_train_best_model_empty_series(self) -> None:
        """train_best_model must raise on empty series (no silent fallback)."""
        train_series = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="Training series cannot be empty"):
            train_best_model(train_series)


class TestSaveTrainedModel:
    """Tests for save_trained_model function."""

    @patch("src.arima.training_arima.training_arima.ARIMA_TRAINED_MODEL_FILE")
    @patch("src.arima.training_arima.training_arima.ARIMA_TRAINED_MODEL_METADATA_FILE")
    @patch("src.arima.training_arima.training_arima.joblib.dump")
    @patch("src.arima.training_arima.training_arima.logger")
    def test_save_trained_model_success(
        self,
        mock_logger: MagicMock,
        mock_dump: MagicMock,
        _mock_metadata_file: MagicMock,
        mock_model_file: MagicMock,
    ) -> None:
        """Test successful saving of trained model."""
        mock_fitted = MagicMock()
        model_info = {"params": "ARIMA(1,0,1)(0,0,0)[12]", "p": 1, "d": 0, "q": 1}

        mock_model_file.parent.mkdir = MagicMock()

        with patch("builtins.open", mock_open()):
            save_trained_model(mock_fitted, model_info)

        mock_dump.assert_called_once()
        mock_logger.info.assert_called()

    def test_save_trained_model_fitted_model_none(self) -> None:
        """Test save_trained_model when fitted_model is None."""
        model_info = {"params": "ARIMA(1,0,1)(0,0,0)[12]", "p": 1, "d": 0, "q": 1}

        with pytest.raises(ValueError, match="fitted_model cannot be None"):
            save_trained_model(None, model_info)

    def test_save_trained_model_model_info_none(self) -> None:
        """Test save_trained_model when model_info is None."""
        mock_fitted = MagicMock()

        with pytest.raises(ValueError, match="model_info cannot be None"):
            save_trained_model(mock_fitted, None)

    @patch("src.arima.training_arima.training_arima.ARIMA_TRAINED_MODEL_FILE")
    @patch("src.arima.training_arima.training_arima.joblib.dump")
    def test_save_trained_model_save_error(
        self, mock_dump: MagicMock, mock_model_file: MagicMock
    ) -> None:
        """Test save_trained_model when saving fails."""
        mock_fitted = MagicMock()
        model_info = {"params": "ARIMA(1,0,1)(0,0,0)[12]", "p": 1, "d": 0, "q": 1}

        mock_model_file.parent.mkdir = MagicMock()
        mock_dump.side_effect = Exception("Save failed")

        with pytest.raises(RuntimeError, match="Failed to save trained model"):
            save_trained_model(mock_fitted, model_info)


class TestValidateArimaParameters:
    """Tests for validate_arima_parameters function."""

    def test_validate_arima_parameters_success(self) -> None:
        """Test successful validation of ARIMA parameters."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 100)
        order = (1, 0, 1)

        # Should not raise
        validate_arima_parameters(train_series, order)

    def test_validate_arima_parameters_empty_series(self) -> None:
        """Test validation with empty series."""
        train_series = pd.Series([], dtype=float)
        order = (1, 0, 1)
        with pytest.raises(ValueError, match="Training series cannot be empty"):
            validate_arima_parameters(train_series, order)

    def test_validate_arima_parameters_invalid_order_length(self) -> None:
        """Test validation with invalid order length."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 100)
        order = (1, 0)  # Invalid: should be 3 values

        with pytest.raises(ValueError, match="Invalid order"):
            validate_arima_parameters(train_series, order)  # type: ignore[arg-type]

    def test_validate_arima_parameters_invalid_order_negative(self) -> None:
        """Test validation with negative order values."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 100)
        order = (1, -1, 1)  # Invalid: negative value

        with pytest.raises(ValueError, match="Invalid order"):
            validate_arima_parameters(train_series, order)


class TestLoadTrainedModel:
    """Tests for load_trained_model function."""

    @patch("src.arima.training_arima.training_arima.load_json_data")
    @patch("src.arima.training_arima.training_arima.joblib.load")
    @patch("src.arima.training_arima.training_arima.validate_file_exists")
    def test_load_trained_model_success(
        self,
        mock_validate: MagicMock,
        mock_joblib_load: MagicMock,
        mock_load_json: MagicMock,
    ) -> None:
        """Test successful loading of trained model."""
        mock_fitted = MagicMock()
        mock_fitted.aic = -100.0
        model_info = {"params": "ARIMA(1,0,1)(0,0,0)[12]", "p": 1, "d": 0, "q": 1}

        mock_joblib_load.return_value = mock_fitted
        mock_load_json.return_value = model_info

        fitted_model, loaded_info = load_trained_model()

        assert fitted_model == mock_fitted
        assert loaded_info == model_info
        assert mock_validate.call_count == 2  # Called for both model file and metadata file

    @patch("src.arima.training_arima.training_arima.validate_file_exists")
    def test_load_trained_model_file_not_found(self, mock_validate: MagicMock) -> None:
        """Test load_trained_model when file doesn't exist."""
        mock_validate.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            load_trained_model()

    @patch("src.arima.training_arima.training_arima.load_json_data")
    @patch("src.arima.training_arima.training_arima.joblib.load")
    @patch("src.arima.training_arima.training_arima.validate_file_exists")
    def test_load_trained_model_loading_error(
        self,
        mock_validate: MagicMock,
        mock_joblib_load: MagicMock,
        mock_load_json: MagicMock,
    ) -> None:
        """Test load_trained_model when loading fails."""
        mock_joblib_load.side_effect = Exception("Loading failed")

        with pytest.raises(RuntimeError, match="Failed to load trained model"):
            load_trained_model()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
