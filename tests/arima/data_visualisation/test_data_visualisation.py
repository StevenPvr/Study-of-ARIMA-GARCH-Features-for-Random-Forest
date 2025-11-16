"""Unit tests for data_visualisation module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Dependencies are mocked in conftest.py before imports
from src.arima.data_visualisation.pre_modeling import (
    plot_acf_pacf,
    plot_seasonality_daily,
    plot_seasonality_for_year,
    plot_seasonality_full_period,
    plot_seasonality_monthly,
    plot_stationarity,
    plot_weighted_series,
)
from src.arima.stationnarity_check.stationnarity_check import StationarityReport
from src.constants import (
    SEASONAL_DEFAULT_MODEL,
    SEASONAL_DEFAULT_PERIOD_DAILY,
    SEASONAL_DEFAULT_PERIOD_MONTHLY,
    SEASONAL_DEFAULT_PERIOD_WEEKLY,
    SEASONAL_DEFAULT_YEARS,
    SEASONAL_MIN_PERIODS,
)


class MockAxesArray:
    """Mock class that simulates numpy array indexing for matplotlib axes.

    This allows us to use axes[i, j] syntax while returning MagicMock objects.
    """

    def __init__(self, axes_grid: list[list[MagicMock]]) -> None:
        """Initialize with a 2D grid of MagicMock axes.

        Args:
            axes_grid: 2D list of MagicMock axes objects.
        """
        self._axes = axes_grid

    def __getitem__(self, key: tuple[int, int] | int) -> MagicMock | list[MagicMock]:
        """Enable indexing with tuple (i, j) or single index.

        Args:
            key: Tuple (i, j) or single index.

        Returns:
            MagicMock axes object for tuple key, or list of MagicMock for single index.
        """
        if isinstance(key, tuple):
            i, j = key
            return self._axes[i][j]
        # For single index, return the row (list of MagicMock)
        return self._axes[key]


class TestPlotWeightedSeries:
    """Tests for plot_weighted_series function."""

    @patch("src.arima.data_visualisation.pre_modeling._save_plot")
    @patch("src.arima.data_visualisation.pre_modeling.create_standard_figure")
    @patch("src.arima.data_visualisation.data_loading.load_dataframe")
    @patch("src.arima.data_visualisation.pre_modeling.logger")
    def test_plot_weighted_series_success(
        self,
        mock_logger: MagicMock,
        mock_load_df: MagicMock,
        mock_create_figure: MagicMock,
        mock_save_plot: MagicMock,
    ) -> None:
        """Test successful plotting of weighted series."""
        # Mock load_dataframe to return test data
        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100, freq="D"),
                "weighted_log_return": np.random.randn(100),
            }
        )
        mock_load_df.return_value = mock_df

        # Mock create_standard_figure to return (figure, axes)
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_create_figure.return_value = (mock_fig, mock_ax)

        # Execute
        plot_weighted_series(
            data_file="test_data.csv",
            output_file="test_output.png",
        )

        # Verify
        mock_load_df.assert_called_once()
        mock_create_figure.assert_called_once()
        mock_save_plot.assert_called_once()
        mock_logger.info.assert_called()

    @patch("src.arima.data_visualisation.pre_modeling._save_plot")
    @patch("src.arima.data_visualisation.pre_modeling.create_standard_figure")
    @patch("src.arima.data_visualisation.data_loading.load_dataframe")
    @patch("src.arima.data_visualisation.pre_modeling.logger")
    def test_plot_weighted_series_custom_paths(
        self,
        mock_logger: MagicMock,
        mock_load_df: MagicMock,
        mock_create_figure: MagicMock,
        mock_save_plot: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test plotting with custom file paths."""

        # Mock load_dataframe to return test data
        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="D"),
                "weighted_log_return": np.random.randn(50),
            }
        )
        mock_load_df.return_value = mock_df

        # Mock create_standard_figure to return (figure, axes)
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_create_figure.return_value = (mock_fig, mock_ax)

        # Use tmp_path instead of relative path to avoid creating permanent files
        custom_output = tmp_path / "custom_output.png"
        plot_weighted_series(data_file="custom_data.csv", output_file=str(custom_output))

        # Verify load_dataframe was called (it internally uses read_csv/read_parquet)
        mock_load_df.assert_called_once()
        # Verify _save_plot was called with the figure and output path
        mock_save_plot.assert_called_once()
        # Verify the output file path was passed
        call_args = mock_save_plot.call_args
        assert str(custom_output) in str(call_args)


class TestPlotAcfPacf:
    """Tests for plot_acf_pacf function."""

    @patch("src.arima.data_visualisation.pre_modeling._save_plot")
    @patch("src.arima.data_visualisation.pre_modeling.create_standard_figure")
    @patch("src.arima.data_visualisation.data_loading.load_dataframe")
    @patch("src.arima.data_visualisation.pre_modeling.plot_acf")
    @patch("src.arima.data_visualisation.pre_modeling.plot_pacf")
    @patch("src.arima.data_visualisation.pre_modeling.logger")
    def test_plot_acf_pacf_success(
        self,
        mock_logger: MagicMock,
        mock_plot_pacf: MagicMock,
        mock_plot_acf: MagicMock,
        mock_load_df: MagicMock,
        mock_create_figure: MagicMock,
        mock_save_plot: MagicMock,
    ) -> None:
        """Test successful plotting of ACF/PACF."""
        # Setup mocks

        # Mock load_dataframe to return test data
        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100, freq="D"),
                "weighted_log_return": np.random.randn(100),
            }
        )
        mock_load_df.return_value = mock_df

        mock_axes = [MagicMock(), MagicMock()]
        mock_create_figure.return_value = (MagicMock(), mock_axes)

        # Execute
        plot_acf_pacf(
            data_file="test_data.csv",
            output_file="test_output.png",
        )

        # Verify
        mock_load_df.assert_called_once()
        mock_create_figure.assert_called_once()
        mock_plot_acf.assert_called_once()
        mock_plot_pacf.assert_called_once()
        mock_save_plot.assert_called_once()

    @patch("src.arima.data_visualisation.pre_modeling.plt")
    @patch("src.arima.data_visualisation.data_loading.load_dataframe")
    @patch("src.arima.data_visualisation.pre_modeling.plot_acf")
    @patch("src.arima.data_visualisation.pre_modeling.plot_pacf")
    @patch("src.arima.data_visualisation.pre_modeling.logger")
    def test_plot_acf_pacf_custom_lags(
        self,
        mock_logger: MagicMock,
        mock_plot_pacf: MagicMock,
        mock_plot_acf: MagicMock,
        mock_load_df: MagicMock,
        mock_plt: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test plotting with custom number of lags.

        When requesting more lags than available (50 lags with 50 data points),
        the function should automatically adjust to the maximum possible (49 lags).
        """

        # Create 50 data points - can only compute up to 49 lags (need lags+1 points)
        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="D"),
                "weighted_log_return": np.random.randn(50),
            }
        )
        mock_load_df.return_value = mock_df

        mock_axes = [MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (MagicMock(), mock_axes)

        # Use tmp_path instead of relative path to avoid creating permanent files
        test_output = tmp_path / "test_output.png"
        plot_acf_pacf(
            data_file="test_data.csv",
            output_file=str(test_output),
            lags=50,  # Request 50 lags, but will be adjusted to 49
        )

        mock_plot_acf.assert_called_once()
        args, kwargs = mock_plot_acf.call_args
        # With 50 points, max lags is 49 (need lags+1 points to compute lag 'lags')
        assert kwargs["lags"] == 49
        # Verify warning was logged about adjustment
        mock_logger.warning.assert_called_once()


class TestPlotStationarity:
    """Tests for plot_stationarity function."""

    @patch("src.arima.data_visualisation.pre_modeling.logger")
    @patch("src.arima.data_visualisation.pre_modeling._save_plot")
    @patch("src.arima.data_visualisation.pre_modeling.add_stationarity_text_box")
    @patch(
        "src.arima.data_visualisation.pre_modeling.format_stationarity_test_text",
        return_value="formatted",
    )
    @patch("src.arima.data_visualisation.pre_modeling.plot_stationarity_rolling_std")
    @patch("src.arima.data_visualisation.pre_modeling.plot_stationarity_rolling_mean")
    @patch("src.arima.data_visualisation.pre_modeling.plot_stationarity_series_with_bands")
    @patch("src.arima.data_visualisation.pre_modeling.evaluate_stationarity")
    @patch("src.arima.data_visualisation.pre_modeling.load_and_validate_data")
    @patch("src.arima.data_visualisation.pre_modeling.plt")
    def test_plot_stationarity_success(
        self,
        mock_plt: MagicMock,
        mock_load: MagicMock,
        mock_evaluate: MagicMock,
        mock_series_bands: MagicMock,
        _mock_roll_mean: MagicMock,
        _mock_roll_std: MagicMock,
        mock_format_text: MagicMock,
        mock_add_box: MagicMock,
        mock_save: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        dates = pd.date_range("2020-01-01", periods=260, freq="B")
        values = np.linspace(-0.1, 0.1, len(dates))
        dataframe = pd.DataFrame({"weighted_log_return": values}, index=dates)
        dataframe.index.name = "date"
        mock_load.return_value = dataframe
        axes = np.array([MagicMock(), MagicMock(), MagicMock()], dtype=object)
        mock_plt.subplots.return_value = (MagicMock(), axes)
        report = StationarityReport(
            stationary=True,
            alpha=0.05,
            adf={
                "statistic": -2.5,
                "p_value": 0.04,
                "lags": 2,
                "nobs": len(dates),
                "critical_values": {},
            },
            kpss={
                "statistic": 0.12,
                "p_value": 0.1,
                "lags": 3,
                "nobs": len(dates),
                "critical_values": {},
            },
            zivot_andrews=None,
        )
        mock_evaluate.return_value = report

        plot_stationarity("input.csv", "stationarity.png", rolling_window=10, alpha=0.05)

        pd.testing.assert_series_equal(
            mock_evaluate.call_args.args[0], dataframe["weighted_log_return"]
        )
        assert mock_series_bands.call_args.args[4] == 10
        mock_format_text.assert_called_once()
        assert mock_add_box.call_count == 1
        add_args, add_kwargs = mock_add_box.call_args
        assert add_args[1] == "formatted"
        assert len(add_args) == 2
        mock_save.assert_called_once_with("stationarity.png")
        assert mock_logger.info.call_count >= 2


class TestPlotSeasonalityFullPeriod:
    """Tests for plot_seasonality_full_period function."""

    @patch("src.arima.data_visualisation.pre_modeling.logger")
    @patch("src.arima.data_visualisation.pre_modeling._save_plot")
    @patch("src.arima.data_visualisation.pre_modeling._plot_seasonal_component_only")
    @patch("src.arima.data_visualisation.pre_modeling._decompose_seasonal_component")
    @patch("src.arima.data_visualisation.pre_modeling.load_and_validate_data")
    @patch("src.arima.data_visualisation.pre_modeling.validate_seasonal_params")
    def test_plot_seasonality_full_period_defaults(
        self,
        mock_validate: MagicMock,
        mock_load: MagicMock,
        mock_decompose: MagicMock,
        mock_plot: MagicMock,
        mock_save: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        dates = pd.date_range("2015-01-05", periods=520, freq="B")
        dataframe = pd.DataFrame(
            {"weighted_log_return": np.linspace(0.0, 1.0, len(dates))}, index=dates
        )
        dataframe.index.name = "date"
        mock_load.return_value = dataframe
        seasonal_component = pd.Series(
            np.linspace(-0.2, 0.2, 100),
            index=pd.date_range("2015-01-11", periods=100, freq="W"),
            name="seasonal",
        )
        mock_decompose.return_value = seasonal_component

        plot_seasonality_full_period(data_file="input.csv", output_file="full.png")

        mock_validate.assert_called_once_with(SEASONAL_DEFAULT_MODEL, None)
        expected_series = dataframe["weighted_log_return"].resample("W").mean().dropna()
        pd.testing.assert_series_equal(mock_decompose.call_args.args[0], expected_series)
        assert mock_decompose.call_args.kwargs["period"] == SEASONAL_DEFAULT_PERIOD_WEEKLY
        mock_plot.assert_called_once_with(seasonal_component, title=ANY, full_period=True)
        mock_save.assert_called_once_with("full.png")
        mock_logger.info.assert_called()


class TestPlotSeasonalityDaily:
    """Tests for plot_seasonality_daily function."""

    @patch("src.arima.data_visualisation.pre_modeling.logger")
    @patch("src.arima.data_visualisation.pre_modeling._save_plot")
    @patch("src.arima.data_visualisation.pre_modeling.plot_seasonal_daily_long_period")
    @patch("src.arima.data_visualisation.pre_modeling._decompose_seasonal_component")
    @patch("src.arima.data_visualisation.pre_modeling.validate_minimum_periods")
    @patch("src.arima.data_visualisation.pre_modeling._filter_series_to_recent_years")
    @patch("src.arima.data_visualisation.pre_modeling.load_and_validate_data")
    @patch("src.arima.data_visualisation.pre_modeling.validate_seasonal_params")
    def test_plot_seasonality_daily_defaults(
        self,
        mock_validate: MagicMock,
        mock_load: MagicMock,
        mock_filter: MagicMock,
        mock_validate_minimum: MagicMock,
        mock_decompose: MagicMock,
        mock_plot_daily: MagicMock,
        mock_save: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        dates = pd.date_range("2023-01-02", periods=260, freq="B")
        dataframe = pd.DataFrame(
            {"weighted_log_return": np.linspace(0.0, 0.05, len(dates))}, index=dates
        )
        dataframe.index.name = "date"
        mock_load.return_value = dataframe
        filtered = pd.Series(
            np.linspace(-0.01, 0.02, 80),
            index=pd.date_range("2023-04-03", periods=80, freq="B"),
            name="weighted_log_return",
        )
        mock_filter.return_value = filtered
        seasonal = pd.Series(
            np.linspace(-0.005, 0.005, 20),
            index=pd.date_range("2023-04-03", periods=20, freq="B"),
            name="seasonal",
        )
        mock_decompose.return_value = seasonal

        plot_seasonality_daily(data_file="input.csv", output_file="daily.png")

        mock_validate.assert_called_once_with(SEASONAL_DEFAULT_MODEL, SEASONAL_DEFAULT_PERIOD_DAILY)
        pd.testing.assert_series_equal(
            mock_filter.call_args.args[0], dataframe["weighted_log_return"]
        )
        assert mock_filter.call_args.args[1] == SEASONAL_DEFAULT_YEARS
        mock_validate_minimum.assert_called_once_with(
            filtered, SEASONAL_DEFAULT_PERIOD_DAILY, min_periods=SEASONAL_MIN_PERIODS
        )
        mock_decompose.assert_called_once_with(
            filtered, model=SEASONAL_DEFAULT_MODEL, period=SEASONAL_DEFAULT_PERIOD_DAILY
        )
        mock_plot_daily.assert_called_once_with(seasonal, title=ANY)
        mock_save.assert_called_once_with("daily.png")
        mock_logger.info.assert_called()


class TestPlotSeasonalityMonthly:
    """Tests for plot_seasonality_monthly function."""

    @patch("src.arima.data_visualisation.pre_modeling.logger")
    @patch("src.arima.data_visualisation.pre_modeling._save_plot")
    @patch("src.arima.data_visualisation.pre_modeling._plot_seasonal_component_only")
    @patch("src.arima.data_visualisation.pre_modeling._decompose_seasonal_component")
    @patch("src.arima.data_visualisation.pre_modeling.validate_minimum_periods")
    @patch("src.arima.data_visualisation.pre_modeling.load_and_validate_data")
    @patch("src.arima.data_visualisation.pre_modeling.validate_seasonal_params")
    def test_plot_seasonality_monthly_defaults(
        self,
        mock_validate: MagicMock,
        mock_load: MagicMock,
        mock_validate_minimum: MagicMock,
        mock_decompose: MagicMock,
        mock_plot: MagicMock,
        mock_save: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        dates = pd.date_range("2010-01-04", periods=600, freq="B")
        dataframe = pd.DataFrame(
            {"weighted_log_return": np.linspace(-0.03, 0.03, len(dates))}, index=dates
        )
        dataframe.index.name = "date"
        mock_load.return_value = dataframe
        seasonal = pd.Series(
            np.linspace(-0.02, 0.02, 48),
            index=pd.date_range("2010-01-31", periods=48, freq="ME"),
            name="seasonal",
        )
        mock_decompose.return_value = seasonal

        plot_seasonality_monthly(data_file="input.csv", output_file="monthly.png")

        mock_validate.assert_called_once_with(
            SEASONAL_DEFAULT_MODEL, SEASONAL_DEFAULT_PERIOD_MONTHLY
        )
        expected_series = dataframe["weighted_log_return"].resample("ME").mean().dropna()
        pd.testing.assert_series_equal(mock_decompose.call_args.args[0], expected_series)
        assert mock_validate_minimum.call_count == 1
        val_args, val_kwargs = mock_validate_minimum.call_args
        pd.testing.assert_series_equal(val_args[0], expected_series)
        assert val_args[1] == SEASONAL_DEFAULT_PERIOD_MONTHLY
        assert val_kwargs == {"min_periods": SEASONAL_MIN_PERIODS}
        mock_plot.assert_called_once_with(seasonal, title=ANY, full_period=True)
        mock_save.assert_called_once_with("monthly.png")
        mock_logger.info.assert_called()


class TestPlotSeasonalityForYear:
    """Tests for plot_seasonality_for_year function."""

    @patch("src.arima.data_visualisation.pre_modeling._save_plot")
    @patch("src.arima.data_visualisation.pre_modeling.create_standard_figure")
    @patch("src.arima.data_visualisation.data_loading.load_dataframe")
    @patch("src.arima.data_visualisation.seasonal.seasonal_decompose")
    @patch("src.arima.data_visualisation.pre_modeling.logger")
    def test_plot_seasonality_for_year(
        self,
        mock_logger: MagicMock,
        mock_decompose: MagicMock,
        mock_load_df: MagicMock,
        mock_create_figure: MagicMock,
        mock_save_plot: MagicMock,
    ) -> None:

        # Build a year of business-day data
        idx = pd.bdate_range("2023-01-02", "2023-12-29")
        mock_df = pd.DataFrame({"date": idx, "weighted_log_return": np.random.randn(len(idx))})
        mock_load_df.return_value = mock_df

        # Mock create_standard_figure
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_create_figure.return_value = (mock_fig, mock_ax)

        mock_res = MagicMock()
        mock_res.seasonal = pd.Series(np.random.randn(len(idx)), index=idx)
        mock_decompose.return_value = mock_res

        plot_seasonality_for_year(
            2023,
            data_file="test_data.csv",
            output_file="season_year.png",
        )

        mock_decompose.assert_called_once()
        mock_save_plot.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
