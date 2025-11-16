"""Batch EGARCH parameter estimation for all distributions.

This module estimates EGARCH parameters for all supported distributions
(normal, student, skewt) on TRAIN data and saves results to estimation.json.

This is a PREPARATION step that must be run BEFORE hyperparameter optimization.
The estimation.json file is required by garch_diagnostic module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import GARCH_ESTIMATION_FILE
from src.garch.garch_params.estimation.mle import estimate_egarch_mle
from src.utils import get_logger, save_json_pretty

logger = get_logger(__name__)


def estimate_all_distributions(
    residuals: np.ndarray,
    o: int,
    p: int,
) -> dict[str, dict]:
    """Estimate EGARCH parameters for all distributions.

    Args:
        residuals: TRAIN residuals from SARIMA model.
        o: ARCH order.
        p: GARCH order.

    Returns:
        Dictionary with keys: egarch_normal, egarch_student, egarch_skewt.
        Each value is a dict with: converged, params, loglik.
    """
    results = {}

    # Estimate for each distribution
    distributions = ["normal", "student", "skewt"]
    for dist in distributions:
        logger.info("Estimating EGARCH(%d,%d) with %s distribution...", o, p, dist)
        try:
            params_dict, convergence = estimate_egarch_mle(
                residuals,
                o=o,
                p=p,
                dist=dist,
            )

            # Build result for this distribution
            result = {
                "converged": convergence.converged,
                "params": params_dict,
                "loglik": convergence.final_loglik,
                "n_iterations": convergence.n_iterations,
                "message": convergence.message,
            }

            # Store with egarch_ prefix for new format
            key = f"egarch_{dist}"
            results[key] = result

            if convergence.converged:
                logger.info(
                    "✓ EGARCH(%d,%d) %s converged: loglik=%.2f",
                    o,
                    p,
                    dist,
                    convergence.final_loglik,
                )
            else:
                logger.warning(
                    "✗ EGARCH(%d,%d) %s failed to converge: %s", o, p, dist, convergence.message
                )

        except Exception as e:
            logger.error("Error estimating EGARCH(%d,%d) %s: %s", o, p, dist, e)
            # Store failed result
            results[f"egarch_{dist}"] = {
                "converged": False,
                "params": {},
                "loglik": None,
                "error": str(e),
            }

    return results


def run_batch_estimation_and_save(
    df: pd.DataFrame,
    o: int,
    p: int,
) -> None:
    """Run batch estimation for all distributions and save to estimation.json.

    This function:
    1. Filters TRAIN data only
    2. Estimates EGARCH for normal, student, and skewt distributions
    3. Saves results to results/garch/estimation/estimation.json

    This is a PREPARATION step that must be run BEFORE hyperparameter optimization.

    Args:
        df: DataFrame with date, split, and residual columns.
        o: ARCH order.
        p: GARCH order.

    Raises:
        ValueError: If no valid TRAIN residuals found.
    """
    logger.info("=" * 60)
    logger.info("BATCH ESTIMATION: EGARCH(%d,%d) for all distributions", o, p)
    logger.info("=" * 60)

    # Filter TRAIN data only
    df_train = df[df["split"] == "train"].copy()
    if df_train.empty:
        msg = "No TRAIN data found in dataset"
        logger.error(msg)
        raise ValueError(msg)

    # Extract residuals (assuming column is 'sarima_resid' or first numeric column)
    if "sarima_resid" in df_train.columns:
        residuals_series = df_train["sarima_resid"]
    else:
        # Try to find residual column
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ["date", "split"]]
        if not numeric_cols:
            msg = "No residual column found in dataset"
            logger.error(msg)
            raise ValueError(msg)
        residuals_series = df_train[numeric_cols[0]]

    residuals = np.asarray(residuals_series, dtype=float)

    # Filter out non-finite values
    residuals_clean = residuals[np.isfinite(residuals)]
    if len(residuals_clean) == 0:
        msg = "No valid (finite) residuals found in TRAIN data"
        logger.error(msg)
        raise ValueError(msg)

    logger.info("TRAIN residuals: n=%d (valid=%d)", len(residuals), len(residuals_clean))

    # Estimate for all distributions
    estimation_results = estimate_all_distributions(residuals_clean, o, p)

    # Save to file
    output_file = Path(GARCH_ESTIMATION_FILE)

    save_json_pretty(estimation_results, output_file)

    logger.info("=" * 60)
    logger.info("Saved estimation results to: %s", output_file)
    logger.info("=" * 60)

    # Log summary
    n_converged = sum(1 for v in estimation_results.values() if v.get("converged", False))
    logger.info(
        "Estimation summary: %d/%d distributions converged",
        n_converged,
        len(estimation_results),
    )
