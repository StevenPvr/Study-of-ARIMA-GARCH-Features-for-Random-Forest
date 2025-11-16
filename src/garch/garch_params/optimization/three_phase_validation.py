"""Three-phase validation split for EGARCH hyperparameter optimization.

This module implements train/validation/test splitting to prevent overfitting:
- TRAIN: Used for model fitting during each CV fold
- VALIDATION: Used for hyperparameter selection (replaces old TRAIN)
- TEST: Held out for final evaluation only (never used in optimization)

This ensures that hyperparameter selection does not overfit to test data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.constants import (
    GARCH_VALIDATION_TEST_RATIO,
    GARCH_VALIDATION_TRAIN_RATIO,
    GARCH_VALIDATION_VAL_RATIO,
)
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ThreePhaseDataset:
    """Container for three-phase validation split.

    Attributes:
        train: Training residuals for model fitting.
        validation: Validation residuals for hyperparameter tuning.
        test: Test residuals for final evaluation (held out).
        train_indices: Indices of training data in original array.
        val_indices: Indices of validation data in original array.
        test_indices: Indices of test data in original array.
    """

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_indices: tuple[int, int]
    val_indices: tuple[int, int]
    test_indices: tuple[int, int]

    def __post_init__(self) -> None:
        """Validate split ratios and sizes."""
        total_ratio = (
            GARCH_VALIDATION_TRAIN_RATIO + GARCH_VALIDATION_VAL_RATIO + GARCH_VALIDATION_TEST_RATIO
        )
        if not np.isclose(total_ratio, 1.0):
            msg = f"Validation ratios must sum to 1.0, got {total_ratio}"
            raise ValueError(msg)

        if self.train.size == 0:
            msg = "Training split is empty"
            raise ValueError(msg)
        if self.validation.size == 0:
            msg = "Validation split is empty"
            raise ValueError(msg)
        if self.test.size == 0:
            msg = "Test split is empty"
            raise ValueError(msg)


def split_three_phase(residuals: np.ndarray) -> ThreePhaseDataset:
    """Split residuals into train/validation/test sets.

    Uses ratios from constants to split sequentially (temporal order preserved).

    Args:
        residuals: Full residual array.

    Returns:
        ThreePhaseDataset with train/validation/test splits.

    Raises:
        ValueError: If residuals are insufficient for splitting.
    """
    n_total = residuals.size

    # Compute split sizes
    n_train = int(n_total * GARCH_VALIDATION_TRAIN_RATIO)
    n_val = int(n_total * GARCH_VALIDATION_VAL_RATIO)
    n_test = n_total - n_train - n_val  # Remaining goes to test

    if n_train == 0 or n_val == 0 or n_test == 0:
        msg = (
            f"Insufficient residuals for 3-phase split: n={n_total}, "
            f"train={n_train}, val={n_val}, test={n_test}"
        )
        raise ValueError(msg)

    # Split sequentially (temporal order)
    train_residuals = residuals[:n_train]
    val_residuals = residuals[n_train : n_train + n_val]
    test_residuals = residuals[n_train + n_val :]

    dataset = ThreePhaseDataset(
        train=train_residuals,
        validation=val_residuals,
        test=test_residuals,
        train_indices=(0, n_train),
        val_indices=(n_train, n_train + n_val),
        test_indices=(n_train + n_val, n_total),
    )

    logger.info(
        "3-phase split: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
        n_train,
        100 * GARCH_VALIDATION_TRAIN_RATIO,
        n_val,
        100 * GARCH_VALIDATION_VAL_RATIO,
        n_test,
        100 * GARCH_VALIDATION_TEST_RATIO,
    )

    return dataset


def combine_train_validation(dataset: ThreePhaseDataset) -> np.ndarray:
    """Combine train and validation sets for final model training.

    This is used after hyperparameter selection to retrain the best model
    on the combined train+validation data before final test evaluation.

    Args:
        dataset: ThreePhaseDataset.

    Returns:
        Combined train+validation residuals.
    """
    combined = np.concatenate([dataset.train, dataset.validation])
    logger.debug(
        "Combined train+val: %d observations (%d train + %d val)",
        combined.size,
        dataset.train.size,
        dataset.validation.size,
    )
    return combined


__all__ = [
    "ThreePhaseDataset",
    "split_three_phase",
    "combine_train_validation",
]
