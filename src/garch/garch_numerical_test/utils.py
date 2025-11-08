"""Utility functions for GARCH numerical tests.

Contains helper functions used across the module for:
- Statistical computations (chi-square, Ljung-Box, ARCH-LM)
- Data preparation and validation
- Output formatting
- Test result validation
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# Statistical computation utilities
# ============================================================================


def prepare_lags_list(lags: int | list[int]) -> list[int]:
    """Convert lags to a list of integers.

    Args:
        lags: Single lag or list of lags.

    Returns:
        List of lag integers.
    """
    if isinstance(lags, int):
        return list(range(1, lags + 1))
    return sorted(set(int(lag) for lag in lags if lag > 0))


def chi2_sf(x: float, df: int) -> float:
    """Chi-square survival function P[X >= x].

    Args:
        x: Test statistic value.
        df: Degrees of freedom.

    Returns:
        P-value.
    """
    try:
        from scipy.stats import chi2  # type: ignore

        return float(chi2.sf(x, df))
    except Exception:
        return float("nan")


def compute_ljung_box_manual(
    res_centered: np.ndarray, lags_list: list[int], n: float
) -> tuple[list[float], list[float]]:
    """Compute Ljung-Box statistics manually.

    Args:
        res_centered: Centered residual series.
        lags_list: List of lag values to test.
        n: Sample size.

    Returns:
        Tuple of (lb_stats, lb_pvalues) lists.
    """
    lb_stats = []
    lb_pvalues = []
    s = 0.0
    for h in lags_list:
        if h >= n:
            break
        acf_h = np.sum(res_centered[h:] * res_centered[:-h]) / np.sum(res_centered**2)
        s += (acf_h * acf_h) / max(1.0, (n - h))
        q = n * (n + 2.0) * s
        lb_stats.append(float(q))
        p_val = chi2_sf(q, h)
        lb_pvalues.append(p_val)
    return lb_stats, lb_pvalues


def compute_ljung_box_stats(
    res: np.ndarray, lags_list: list[int]
) -> tuple[list[float], list[float]]:
    """Compute Ljung-Box statistics using statsmodels or fallback.

    Args:
        res: Residual series (already cleaned).
        lags_list: List of lag values to test.

    Returns:
        Tuple of (lb_stats, lb_pvalues) lists.
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox  # type: ignore

        df = acorr_ljungbox(res, lags=lags_list, return_df=True)
        return [float(x) for x in df["lb_stat"]], [float(x) for x in df["lb_pvalue"]]
    except Exception as e:
        logger.debug(f"Fallback in Ljung-Box computation: {e}")
        n = float(res.size)
        mean_res = np.mean(res)
        res_centered = res - mean_res
        return compute_ljung_box_manual(res_centered, lags_list, n)


def compute_arch_lm_statistic(e2: np.ndarray, lags: int) -> float:
    """Compute ARCH-LM test statistic.

    Args:
        e2: Squared residuals.
        lags: Number of lags in regression.

    Returns:
        LM test statistic value.
    """
    n = int(e2.size)
    Y = e2[lags:]
    X = np.ones((n - lags, lags + 1), dtype=float)
    for j in range(1, lags + 1):
        X[:, j] = e2[lags - j : n - j]

    try:
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        Y_hat = X @ beta
        ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
        ss_res = float(np.sum((Y - Y_hat) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return (n - lags) * max(0.0, r2)
    except Exception:
        return float("nan")


# ============================================================================
# Output formatting utilities
# ============================================================================


def prepare_output(results: dict[str, Any], n_residuals: int) -> dict[str, Any]:
    """Prepare output dictionary for test results.

    Args:
        results: Dictionary containing test results.
        n_residuals: Number of residuals tested.

    Returns:
        Formatted output dictionary.
    """
    from src.constants import GARCH_DATASET_FILE

    return {
        "source": str(GARCH_DATASET_FILE),
        "n_residuals": n_residuals,
        "tests": {
            "ljung_box_residuals": {
                "name": "Ljung-Box Test sur les résidus",
                "result": results["ljung_box_residuals"],
            },
            "ljung_box_squared": {
                "name": "Ljung-Box Test sur les résidus au carré",
                "result": results["ljung_box_squared"],
            },
            "engle_arch_lm": {
                "name": "Engle ARCH LM Test",
                "result": results["engle_arch_lm"],
            },
            "mcleod_li": {
                "name": "McLeod-Li Test",
                "result": results["mcleod_li"],
            },
        },
    }


def log_test_summary(results: dict[str, Any]) -> None:
    """Log summary of test results.

    Args:
        results: Dictionary containing test results.
    """
    logger.info("Test Results Summary:")
    logger.info("  Ljung-Box (residuals): reject=%s", results["ljung_box_residuals"]["reject"])
    logger.info("  Ljung-Box (squared): reject=%s", results["ljung_box_squared"]["reject"])
    logger.info("  Engle ARCH-LM: reject=%s", results["engle_arch_lm"]["reject"])
    logger.info("  McLeod-Li: reject=%s", results["mcleod_li"]["reject"])


# ============================================================================
# Test validation utilities
# ============================================================================


def assert_keys_present(result: dict[str, Any], required_keys: list[str]) -> None:
    """Assert that all required keys are present in result.

    Args:
        result: Test result dictionary.
        required_keys: List of required key names.
    """
    for key in required_keys:
        assert key in result


def validate_ljung_box_result(result: dict[str, Any], expected_lags: int) -> None:
    """Validate Ljung-Box test result structure.

    Args:
        result: Test result dictionary.
        expected_lags: Expected number of lags.
    """
    assert_keys_present(result, ["lb_stat", "lb_pvalue", "reject"])
    assert len(result["lb_stat"]) == expected_lags
    assert len(result["lb_pvalue"]) == expected_lags


def validate_arch_lm_result(result: dict[str, Any], expected_df: int) -> None:
    """Validate ARCH-LM test result structure.

    Args:
        result: Test result dictionary.
        expected_df: Expected degrees of freedom.
    """
    assert_keys_present(result, ["lm_stat", "p_value", "df", "reject"])
    assert result["df"] == expected_df


def validate_all_tests_result(results: dict[str, Any]) -> None:
    """Validate that all test results are present and valid.

    Args:
        results: Dictionary containing all test results.
    """
    required_keys = ["ljung_box_residuals", "ljung_box_squared", "engle_arch_lm", "mcleod_li"]
    for key in required_keys:
        assert key in results
        assert isinstance(results[key], dict)


__all__ = [
    # Statistical computation utilities
    "prepare_lags_list",
    "chi2_sf",
    "compute_ljung_box_manual",
    "compute_ljung_box_stats",
    "compute_arch_lm_statistic",
    # Output formatting utilities
    "prepare_output",
    "log_test_summary",
    # Test validation utilities
    "assert_keys_present",
    "validate_ljung_box_result",
    "validate_arch_lm_result",
    "validate_all_tests_result",
]
