"""Unit tests for GARCH numerical tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.garch.garch_numerical_test.garch_numerical import (
    engle_arch_lm_test,
    ljung_box_squared_test,
    ljung_box_test,
    mcleod_li_test,
    run_all_tests,
)
from src.garch.garch_numerical_test.utils import (
    validate_all_tests_result,
    validate_arch_lm_result,
    validate_ljung_box_result,
)


def test_ljung_box_test() -> None:
    """Test Ljung-Box test on residuals."""
    rng = np.random.default_rng(42)
    white_noise = rng.standard_normal(100)
    result = ljung_box_test(white_noise, lags=5, alpha=0.05)
    validate_ljung_box_result(result, 5)


def test_ljung_box_squared_test() -> None:
    """Test Ljung-Box test on squared residuals."""
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal(100)
    result = ljung_box_squared_test(residuals, lags=5, alpha=0.05)
    validate_ljung_box_result(result, 5)


def test_engle_arch_lm_test() -> None:
    """Test Engle ARCH-LM test."""
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal(100)
    result = engle_arch_lm_test(residuals, lags=5, alpha=0.05)
    validate_arch_lm_result(result, 5)


def test_mcleod_li_test() -> None:
    """Test McLeod-Li test."""
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal(100)
    result = mcleod_li_test(residuals, lags=5, alpha=0.05)
    validate_ljung_box_result(result, 5)


def test_run_all_tests() -> None:
    """Test running all tests together."""
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal(100)
    results = run_all_tests(residuals, ljung_box_lags=5, arch_lm_lags=5, alpha=0.05)
    validate_all_tests_result(results)


def test_empty_residuals() -> None:
    """Test handling of empty residuals."""
    empty = np.array([])
    result = ljung_box_test(empty, lags=5)
    assert result["n"] == 0
    assert not result["reject"]


def test_nan_handling() -> None:
    """Test handling of NaN values."""
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal(100)
    residuals[10:20] = np.nan
    result = ljung_box_test(residuals, lags=5)
    assert result["n"] == 90  # Should exclude NaNs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
