"""Data loading utilities for GARCH diagnostics.

Contains functions for loading estimation parameters and preparing residuals.
"""

from __future__ import annotations

import json

import numpy as np

from src.constants import GARCH_DATASET_FILE, GARCH_ESTIMATION_FILE
from src.garch.garch_diagnostic.param_extraction import extract_nu_from_params, extract_params_dict
from src.garch.structure_garch.utils import load_garch_dataset, prepare_residuals
from src.utils import get_logger

logger = get_logger(__name__)


def check_converged_params(params: dict | None) -> bool:
    """Check if parameters dictionary indicates convergence."""
    return isinstance(params, dict) and params.get("converged", False)


def try_new_format_params(est_payload: dict) -> tuple[str | None, dict | None]:
    """Try to extract parameters from new format keys.

    Checks in preference order: egarch_skewt → egarch_student → egarch_normal.
    """
    egarch_skewt = est_payload.get("egarch_skewt")
    if check_converged_params(egarch_skewt):
        return "skewt", egarch_skewt
    egarch_student = est_payload.get("egarch_student")
    if check_converged_params(egarch_student):
        return "student", egarch_student
    egarch_normal = est_payload.get("egarch_normal")
    if check_converged_params(egarch_normal):
        return "normal", egarch_normal
    return None, None


def try_legacy_format_params(est_payload: dict) -> tuple[str | None, dict | None]:
    """Try to extract parameters from legacy format (student, normal)."""
    student = est_payload.get("student")
    if check_converged_params(student):
        return "student", student
    normal = est_payload.get("normal")
    if check_converged_params(normal):
        return "normal", normal
    return None, None


def choose_best_params(est_payload: dict) -> tuple[str, dict]:
    """Choose best converged EGARCH parameters from estimation payload.

    Args:
    ----
        est_payload: Estimation payload dictionary.

    Returns:
    -------
        Tuple of (distribution, parameters).

    Raises:
    ------
        ValueError: If no converged EGARCH model found.

    """
    dist, params = try_new_format_params(est_payload)
    if params is not None:
        return dist, params  # type: ignore[return-value]

    dist, params = try_legacy_format_params(est_payload)
    if params is not None:
        return dist, params  # type: ignore[return-value]

    msg = "No converged EGARCH model found in estimation payload"
    raise ValueError(msg)


def load_estimation_file() -> dict:
    """Load estimation JSON file.

    Raises:
        FileNotFoundError: If file is missing.
        ValueError: If JSON is invalid.
    """
    try:
        with open(GARCH_ESTIMATION_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Estimation file not found: %s", GARCH_ESTIMATION_FILE)
        raise
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in estimation file: %s", e)
        raise ValueError(f"Invalid JSON in {GARCH_ESTIMATION_FILE}") from e


def load_and_prepare_residuals() -> np.ndarray:
    """Load dataset and prepare test residuals.

    Raises:
        ValueError: If no valid residuals found.
    """
    data_frame = load_garch_dataset(str(GARCH_DATASET_FILE))
    resid_test = prepare_residuals(data_frame, use_test_only=True)
    resid_test = resid_test[np.isfinite(resid_test)]
    if resid_test.size == 0:
        logger.error("No valid residuals found in test set")
        raise ValueError("No valid residuals found in test set")
    return resid_test


def load_data_and_params() -> tuple[np.ndarray, str, dict, float | None]:
    """Load dataset, residuals, and best EGARCH parameters.

    Returns:
    -------
        Tuple of (residuals, distribution, params, nu).

    Raises:
    ------
        FileNotFoundError: If required files are missing.
        ValueError: If no converged model found or data loading fails.

    """
    est = load_estimation_file()
    dist, best = choose_best_params(est)

    params_dict = extract_params_dict(best)
    nu = extract_nu_from_params(best)
    resid_test = load_and_prepare_residuals()
    return resid_test, dist, params_dict, nu
