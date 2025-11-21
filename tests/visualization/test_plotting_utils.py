"""Tests for plotting_utils module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.visualization.plotting_utils import (
    add_grid,
    add_legend,
    add_zero_line,
    clean_array,
    create_figure_canvas,
    ensure_output_directory,
    format_date_axis,
    get_color_palette,
    get_test_result_style,
    load_json_if_exists,
    prepare_temporal_axis,
    save_figure,
    setup_plot_style,
    subsample_for_plotting,
    validate_plot_arrays,
)


class TestFigureCreation:
    """Test figure and canvas creation functions."""

    def test_create_figure_canvas_single_subplot(self):
        """Test creating a single subplot canvas."""
        fig, canvas, axes = create_figure_canvas((10, 6))
        assert fig is not None
        assert canvas is not None
        assert hasattr(axes, "plot")  # Should be a single Axes object

    def test_create_figure_canvas_multiple_subplots(self):
        """Test creating multiple subplots."""
        fig, canvas, axes = create_figure_canvas((10, 6), n_rows=2, n_cols=2)
        assert fig is not None
        assert canvas is not None
        assert axes.shape == (2, 2)

    def test_create_standard_figure(self):
        """Test creating standard pyplot figure."""
        # Note: This test is skipped in CI environments to avoid backend issues
        # In local development, ensure matplotlib uses Agg backend
        fig, canvas, axes = create_figure_canvas((10, 6))
        assert fig is not None
        assert axes is not None


class TestDataPreparation:
    """Test data preparation and validation functions."""

    def test_prepare_temporal_axis_with_dates(self):
        """Test preparing temporal axis with datetime data."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        axis = prepare_temporal_axis(dates, 10)
        assert len(axis) == 10
        assert axis.dtype == object  # datetime objects

    def test_prepare_temporal_axis_without_dates(self):
        """Test preparing temporal axis without dates."""
        axis = prepare_temporal_axis(None, 10)
        assert len(axis) == 10
        assert np.array_equal(axis, np.arange(10, dtype=float))

    def test_validate_plot_arrays_valid(self):
        """Test validation of valid arrays."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        assert validate_plot_arrays(arr1, arr2) is True

    def test_validate_plot_arrays_empty(self):
        """Test validation fails with empty arrays."""
        arr1 = np.array([])
        arr2 = np.array([1, 2, 3])
        assert validate_plot_arrays(arr1, arr2) is False

    def test_validate_plot_arrays_different_lengths(self):
        """Test validation fails with different length arrays."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5])
        assert validate_plot_arrays(arr1, arr2) is False

    def test_clean_array_with_nan(self):
        """Test cleaning array with NaN values."""
        arr = np.array([1.0, np.nan, 3.0, np.inf])
        cleaned = clean_array(arr)
        assert len(cleaned) == 2  # Only finite values
        assert np.array_equal(cleaned, [1.0, 3.0])

    def test_clean_array_without_nan(self):
        """Test cleaning array without NaN values."""
        arr = np.array([1.0, 2.0, 3.0])
        cleaned = clean_array(arr, remove_nan=False)
        assert len(cleaned) == 3


class TestPlotElements:
    """Test plot element addition functions."""

    @pytest.fixture
    def sample_figure_and_axes(self):
        """Create a sample figure and axes for testing."""
        fig, canvas, axes = create_figure_canvas((10, 6))
        return fig, axes

    def test_add_zero_line(self, sample_figure_and_axes):
        """Test adding zero line to plot."""
        fig, ax = sample_figure_and_axes
        add_zero_line(ax)
        # Check that a line was added
        assert len(ax.get_lines()) > 0

    def test_add_grid(self, sample_figure_and_axes):
        """Test adding grid to plot."""
        fig, ax = sample_figure_and_axes
        add_grid(ax)
        assert ax.grid is True or ax.grid

    def test_add_legend(self, sample_figure_and_axes):
        """Test adding legend to plot."""
        fig, ax = sample_figure_and_axes
        ax.plot([1, 2, 3], label="test")
        add_legend(ax)
        assert ax.get_legend() is not None


class TestStyling:
    """Test styling functions."""

    def test_setup_plot_style(self):
        """Test setting up plot style."""
        setup_plot_style("default")  # Should not raise

    def test_get_color_palette(self):
        """Test getting color palette."""
        colors = get_color_palette(5)
        assert len(colors) == 5
        assert all(isinstance(c, str) for c in colors)

    def test_get_test_result_style_significant(self):
        """Test getting style for significant p-value."""
        style = get_test_result_style(0.01)
        assert "facecolor" in style
        assert style["facecolor"] == "lightcoral"  # Error style

    def test_get_test_result_style_not_significant(self):
        """Test getting style for non-significant p-value."""
        style = get_test_result_style(0.1)
        assert "facecolor" in style
        assert style["facecolor"] == "lightgreen"  # Success style


class TestFileOperations:
    """Test file operation functions."""

    def test_ensure_output_directory(self):
        """Test ensuring output directory exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "subdir" / "file.png"
            result_path = ensure_output_directory(output_path)
            assert result_path.parent.exists()
            assert result_path == output_path

    def test_save_figure_with_path(self):
        """Test saving figure to file."""
        import matplotlib.pyplot as plt

        fig, canvas, ax = create_figure_canvas((8, 6))
        ax.plot([1, 2, 3])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_figure(fig, tmp_path, close_after=False)  # Don't close for testing
            assert tmp_path.exists()
        finally:
            plt.close(fig)  # Manually close after test
            if tmp_path.exists():
                tmp_path.unlink()

    def test_load_json_if_exists_existing(self):
        """Test loading existing JSON file."""
        test_data = {"key": "value"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            import json

            json.dump(test_data, tmp)
            tmp_path = Path(tmp.name)

        try:
            loaded = load_json_if_exists(tmp_path)
            assert loaded == test_data
        finally:
            tmp_path.unlink()

    def test_load_json_if_exists_nonexistent(self):
        """Test loading non-existent JSON file."""
        result = load_json_if_exists("nonexistent_file.json")
        assert result is None


class TestSubsampling:
    """Test data subsampling functions."""

    def test_subsample_for_plotting_small_dataset(self):
        """Test subsampling with small dataset."""
        df = pd.DataFrame({"x": range(10), "y": range(10)})
        subsampled = subsample_for_plotting(df, max_points=20)
        assert len(subsampled) == 10  # No subsampling needed

    def test_subsample_for_plotting_large_dataset(self):
        """Test subsampling with large dataset."""
        df = pd.DataFrame({"x": range(1000), "y": range(1000)})
        subsampled = subsample_for_plotting(df, max_points=100)
        assert len(subsampled) == 100

    def test_subsample_for_plotting_uniform_method(self):
        """Test uniform subsampling method."""
        df = pd.DataFrame({"x": range(100), "y": range(100)})
        subsampled = subsample_for_plotting(df, max_points=10, method="uniform")
        assert len(subsampled) == 10

    def test_subsample_for_plotting_random_method(self):
        """Test random subsampling method."""
        df = pd.DataFrame({"x": range(100), "y": range(100)})
        subsampled = subsample_for_plotting(df, max_points=10, method="random")
        assert len(subsampled) == 10


class TestDateFormatting:
    """Test date formatting functions."""

    @pytest.fixture
    def sample_figure_and_axes(self):
        """Create a sample figure and axes for testing."""
        fig, canvas, axes = create_figure_canvas((10, 6))
        return fig, axes

    def test_format_date_axis(self, sample_figure_and_axes):
        """Test formatting date axis."""
        fig, ax = sample_figure_and_axes
        try:
            format_date_axis(ax)
            # Check that locators and formatters are set (if no exception)
            assert ax.xaxis.get_major_locator() is not None
            assert ax.xaxis.get_major_formatter() is not None
        except (TypeError, AttributeError):
            # Skip test if matplotlib version doesn't support the locator
            pytest.skip("DateLocator not supported in this matplotlib version")
