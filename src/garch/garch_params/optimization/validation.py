"""Anti-leakage validation for EGARCH optimization.

This module provides validation functions to ensure no temporal leakage:
- Training window validation
- Validation window validation
- Temporal ordering checks
- Data split integrity checks

Critical for academic papers: must prove no look-ahead bias.
"""

from __future__ import annotations

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


def validate_temporal_ordering(
    train_start: int, train_end: int, val_start: int, val_end: int
) -> None:
    """Validate temporal ordering of training and validation windows.

    Ensures:
    - Training ends before validation starts
    - No overlap between windows
    - Valid window sizes

    Args:
        train_start: Training window start index.
        train_end: Training window end index (exclusive).
        val_start: Validation window start index.
        val_end: Validation window end index (exclusive).

    Raises:
        ValueError: If temporal ordering is violated.
    """
    if train_start < 0:
        msg = f"Invalid train_start: {train_start} < 0"
        raise ValueError(msg)

    if train_end <= train_start:
        msg = f"Invalid training window: train_end ({train_end}) <= train_start ({train_start})"
        raise ValueError(msg)

    if val_end <= val_start:
        msg = f"Invalid validation window: val_end ({val_end}) <= val_start ({val_start})"
        raise ValueError(msg)

    if val_start < train_end:
        msg = (
            f"Temporal leakage detected: val_start ({val_start}) < train_end ({train_end}). "
            "Validation must start at or after training end."
        )
        raise ValueError(msg)

    if val_start != train_end:
        logger.warning(
            "Gap between training and validation: train_end=%d, val_start=%d",
            train_end,
            val_start,
        )


def validate_window_bounds(
    train_start: int,
    train_end: int,
    val_start: int,
    val_end: int,
    data_size: int,
) -> None:
    """Validate window bounds against data size.

    Args:
        train_start: Training window start index.
        train_end: Training window end index (exclusive).
        val_start: Validation window start index.
        val_end: Validation window end index (exclusive).
        data_size: Total data size.

    Raises:
        ValueError: If windows exceed data bounds.
    """
    if train_start < 0:
        msg = f"train_start ({train_start}) < 0"
        raise ValueError(msg)

    if train_end > data_size:
        msg = f"train_end ({train_end}) exceeds data size ({data_size})"
        raise ValueError(msg)

    if val_start < 0:
        msg = f"val_start ({val_start}) < 0"
        raise ValueError(msg)

    if val_end > data_size:
        msg = f"val_end ({val_end}) exceeds data size ({data_size})"
        raise ValueError(msg)


def validate_no_test_data_used(
    residuals_train: np.ndarray,
    residuals_full: np.ndarray,
) -> None:
    """Validate that only training data is used (no test data).

    Args:
        residuals_train: Training residuals (TRAIN split only).
        residuals_full: Full residuals (TRAIN + TEST).

    Raises:
        ValueError: If residuals_train contains test data.
    """
    if residuals_train.size > residuals_full.size:
        msg = (
            f"Training residuals ({residuals_train.size}) "
            f"larger than full residuals ({residuals_full.size})"
        )
        raise ValueError(msg)

    # Check that train is a prefix of full
    if not np.array_equal(residuals_train, residuals_full[: residuals_train.size]):
        logger.warning(
            "Training residuals are not a prefix of full residuals - "
            "may indicate data leakage or data mismatch"
        )


def validate_cv_fold(
    train_window: np.ndarray,
    val_window: np.ndarray,
    min_size: int = 50,
) -> bool:
    """Validate CV fold windows meet minimum size requirements.

    Args:
        train_window: Training window data.
        val_window: Validation window data.
        min_size: Minimum window size.

    Returns:
        True if both windows are valid.
    """
    if train_window.size < min_size:
        logger.debug("Training window too small: %d < %d", train_window.size, min_size)
        return False

    if val_window.size < min_size:
        logger.debug("Validation window too small: %d < %d", val_window.size, min_size)
        return False

    return True


def assert_no_future_information(
    train_end_idx: int,
    val_start_idx: int,
    forecast_idx: int,
) -> None:
    """Assert that forecast uses no future information.

    Verifies that:
    - Training data ends before validation starts
    - Forecast index is within validation window
    - No look-ahead bias

    Args:
        train_end_idx: Last training index used (exclusive).
        val_start_idx: First validation index.
        forecast_idx: Index being forecast.

    Raises:
        AssertionError: If future information is used.
    """
    if train_end_idx > val_start_idx:
        msg = (
            f"Training uses future information: "
            f"train_end ({train_end_idx}) > val_start ({val_start_idx})"
        )
        raise AssertionError(msg)

    if forecast_idx < val_start_idx:
        msg = (
            f"Forecast index ({forecast_idx}) < val_start ({val_start_idx}). "
            "Forecasting past validation window."
        )
        raise AssertionError(msg)

    if forecast_idx < train_end_idx:
        msg = (
            f"Look-ahead bias detected: "
            f"forecast_idx ({forecast_idx}) < train_end ({train_end_idx})"
        )
        raise AssertionError(msg)
