"""Tests for realized volatility estimators using HLOC prices."""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent.parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import pytest

from src.garch.benchmark.realized_volatility import (
    compare_realized_estimators,
    compute_realized_measures,
    efficiency_ratio,
    garman_klass_estimator,
    parkinson_estimator,
    realized_variance_returns,
    rogers_satchell_estimator,
    yang_zhang_estimator,
)


class TestParkinsonEstimator:
    """Tests for Parkinson (1980) range-based estimator."""

    def test_parkinson_basic(self) -> None:
        """Test basic Parkinson estimator computation."""
        high = np.array([105.0, 110.0, 108.0])
        low = np.array([95.0, 100.0, 98.0])

        var_p = parkinson_estimator(high, low)

        # Should return positive variances
        assert np.all(var_p > 0)
        assert len(var_p) == 3

    def test_parkinson_no_range(self) -> None:
        """Test Parkinson when high equals low."""
        high = np.array([100.0, 100.0])
        low = np.array([100.0, 100.0])

        var_p = parkinson_estimator(high, low)

        # Variance should be zero when no range
        assert np.all(var_p == 0.0)

    def test_parkinson_invalid_prices(self) -> None:
        """Test that invalid prices raise errors."""
        # High < Low
        with pytest.raises(ValueError, match="high prices must be >= low prices"):
            parkinson_estimator(np.array([90.0]), np.array([100.0]))

        # Negative prices
        with pytest.raises(ValueError, match="must be positive"):
            parkinson_estimator(np.array([-10.0]), np.array([-20.0]))

        # Mismatched lengths
        with pytest.raises(ValueError, match="must match"):
            parkinson_estimator(np.array([100.0, 105.0]), np.array([95.0]))


class TestGarmanKlassEstimator:
    """Tests for Garman-Klass (1980) HLOC estimator."""

    def test_garman_klass_basic(self) -> None:
        """Test basic Garman-Klass estimator computation."""
        high = np.array([105.0, 110.0])
        low = np.array([95.0, 100.0])
        close = np.array([100.0, 105.0])
        open_price = np.array([98.0, 102.0])

        var_gk = garman_klass_estimator(high, low, close, open_price)

        # Should return non-negative variances
        assert np.all(var_gk >= 0)
        assert len(var_gk) == 2

    def test_garman_klass_more_efficient_than_parkinson(self) -> None:
        """Test that GK uses more information than Parkinson."""
        # GK should use both range and close-open information
        high = np.array([110.0])
        low = np.array([90.0])
        close = np.array([100.0])
        open_price = np.array([95.0])

        var_gk = garman_klass_estimator(high, low, close, open_price)
        var_p = parkinson_estimator(high, low)

        # Both should be positive
        assert var_gk[0] > 0
        assert var_p[0] > 0


class TestRogersSatchellEstimator:
    """Tests for Rogers-Satchell (1991) drift-independent estimator."""

    def test_rogers_satchell_basic(self) -> None:
        """Test basic Rogers-Satchell estimator computation."""
        high = np.array([105.0, 110.0])
        low = np.array([95.0, 100.0])
        close = np.array([100.0, 105.0])
        open_price = np.array([98.0, 102.0])

        var_rs = rogers_satchell_estimator(high, low, close, open_price)

        # Should return non-negative variances
        assert np.all(var_rs >= 0)
        assert len(var_rs) == 2

    def test_rogers_satchell_no_volatility(self) -> None:
        """Test RS when all prices are equal (no volatility)."""
        price = 100.0
        high = np.array([price, price])
        low = np.array([price, price])
        close = np.array([price, price])
        open_price = np.array([price, price])

        var_rs = rogers_satchell_estimator(high, low, close, open_price)

        # Should be zero when no volatility
        assert np.all(var_rs == 0.0)


class TestYangZhangEstimator:
    """Tests for Yang-Zhang (2000) combined estimator."""

    def test_yang_zhang_basic(self) -> None:
        """Test basic Yang-Zhang estimator computation."""
        high = np.array([105.0, 110.0, 108.0])
        low = np.array([95.0, 100.0, 98.0])
        close = np.array([100.0, 105.0, 102.0])
        open_price = np.array([98.0, 102.0, 100.0])

        var_yz = yang_zhang_estimator(high, low, close, open_price)

        # Should return non-negative variances (except first which is NaN)
        assert np.isnan(var_yz[0])  # No overnight return for first period
        assert np.all(var_yz[1:] >= 0)
        assert len(var_yz) == 3

    def test_yang_zhang_requires_two_periods(self) -> None:
        """Test that YZ requires at least 2 periods."""
        high = np.array([100.0])
        low = np.array([95.0])
        close = np.array([98.0])
        open_price = np.array([97.0])

        with pytest.raises(ValueError, match="at least 2 periods"):
            yang_zhang_estimator(high, low, close, open_price)

    def test_yang_zhang_k_parameter(self) -> None:
        """Test YZ with different k values."""
        high = np.array([105.0, 110.0])
        low = np.array([95.0, 100.0])
        close = np.array([100.0, 105.0])
        open_price = np.array([98.0, 102.0])

        # Test different k values
        for k in [0.0, 0.34, 1.0]:
            var_yz = yang_zhang_estimator(high, low, close, open_price, k=k)
            assert np.all(var_yz[1:] >= 0)

    def test_yang_zhang_invalid_k(self) -> None:
        """Test that invalid k raises error."""
        high = np.array([105.0, 110.0])
        low = np.array([95.0, 100.0])
        close = np.array([100.0, 105.0])
        open_price = np.array([98.0, 102.0])

        with pytest.raises(ValueError, match="k must be in"):
            yang_zhang_estimator(high, low, close, open_price, k=1.5)


class TestRealizedVarianceReturns:
    """Tests for classical realized variance from returns."""

    def test_rv_returns_basic(self) -> None:
        """Test basic RV computation."""
        returns = np.array([0.01, -0.02, 0.015, -0.01])

        rv = realized_variance_returns(returns)

        # RV = sum of squared returns
        expected = np.sum(returns**2)
        assert rv == pytest.approx(expected)

    def test_rv_returns_empty(self) -> None:
        """Test RV with empty returns."""
        returns = np.array([])

        rv = realized_variance_returns(returns)

        assert rv == 0.0


class TestComputeRealizedMeasures:
    """Tests for computing all realized measures at once."""

    def test_compute_all_measures(self) -> None:
        """Test computing all realized measures."""
        data = pd.DataFrame(
            {
                "High": [105, 110, 108],
                "Low": [95, 100, 98],
                "Close": [100, 105, 102],
                "Open": [98, 102, 100],
            }
        )

        result = compute_realized_measures(data)

        # Check all columns are present
        expected_cols = ["RV", "Parkinson", "GarmanKlass", "RogersSatchell", "YangZhang"]
        for col in expected_cols:
            assert col in result.columns

        # Check shape
        assert len(result) == len(data)

        # Check non-negative (except NaN)
        assert np.all(result["Parkinson"] >= 0)
        assert np.all(result["GarmanKlass"] >= 0)
        assert np.all(result["RogersSatchell"] >= 0)

    def test_compute_measures_invalid_prices(self) -> None:
        """Test that negative prices raise error."""
        data = pd.DataFrame(
            {
                "High": [-105, 110],
                "Low": [-95, 100],
                "Close": [-100, 105],
                "Open": [-98, 102],
            }
        )

        with pytest.raises(ValueError, match="must be positive"):
            compute_realized_measures(data)


class TestEfficiencyRatio:
    """Tests for efficiency ratio computation."""

    def test_efficiency_ratio_basic(self) -> None:
        """Test basic efficiency ratio computation."""
        # Estimator 1 has higher variance (less efficient)
        est1 = np.random.randn(100) * 2.0
        est2 = np.random.randn(100) * 1.0

        ratio = efficiency_ratio(est1, est2)

        # ratio = var(est1) / var(est2)
        # Since est1 has ~4x variance, ratio should be > 1
        assert ratio > 1.0

    def test_efficiency_ratio_symmetric(self) -> None:
        """Test efficiency ratio symmetry."""
        np.random.seed(42)
        est1 = np.random.randn(100)
        est2 = np.random.randn(100) * 2.0

        ratio12 = efficiency_ratio(est1, est2)
        ratio21 = efficiency_ratio(est2, est1)

        # ratio12 * ratio21 should equal 1
        assert (ratio12 * ratio21) == pytest.approx(1.0, rel=0.01)

    def test_efficiency_ratio_with_nans(self) -> None:
        """Test efficiency ratio handles NaN values."""
        est1 = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        est2 = np.array([1.5, 2.5, 3.5, np.nan, 4.5])

        # Should compute on valid values only
        ratio = efficiency_ratio(est1, est2)

        assert np.isfinite(ratio)


class TestCompareRealizedEstimators:
    """Tests for comparing realized estimators."""

    def test_compare_estimators_basic(self) -> None:
        """Test basic estimator comparison."""
        # Generate realistic HLOC data
        np.random.seed(42)
        n = 100
        close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
        high = close * (1 + np.abs(np.random.randn(n)) * 0.01)
        low = close * (1 - np.abs(np.random.randn(n)) * 0.01)
        open_price = close * (1 + np.random.randn(n) * 0.005)

        data = pd.DataFrame(
            {
                "High": high,
                "Low": low,
                "Close": close,
                "Open": open_price,
            }
        )

        comparison = compare_realized_estimators(data)

        # Check structure
        assert "mean" in comparison.columns
        assert "std" in comparison.columns
        assert "efficiency_vs_RV" in comparison.columns

        # Check that RV has efficiency 1.0
        rv_efficiency = comparison.loc["RV", "efficiency_vs_RV"]
        assert rv_efficiency == pytest.approx(1.0)

        # Check that all estimators are present
        expected_estimators = ["RV", "Parkinson", "GarmanKlass", "RogersSatchell", "YangZhang"]
        for est in expected_estimators:
            assert est in comparison.index


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])  # pragma: no cover
