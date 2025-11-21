"""Main entry point for random baseline model evaluation.

This script computes baseline metrics using random predictions and saves
them to the same results file as the LightGBM models for comparison.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import json

from src.constants import (
    DEFAULT_RANDOM_STATE,
    LIGHTGBM_DATASET_COMPLETE_FILE,
    LIGHTGBM_EVAL_RESULTS_FILE,
)
from src.lightgbm.baseline.baseline import compute_random_baseline_metrics
from src.lightgbm.baseline.naive_baseline import compute_naive_baseline_metrics
from src.lightgbm.shared_utils import resolve_dataset_path
from src.utils import ensure_output_dir, get_logger

logger = get_logger(__name__)


def _load_existing_results() -> dict:
    """Load existing evaluation results.

    Returns:
        Dictionary of existing results, or empty dict if file doesn't exist.
    """
    if LIGHTGBM_EVAL_RESULTS_FILE.exists():
        with open(LIGHTGBM_EVAL_RESULTS_FILE) as f:
            return json.load(f)
    return {}


def _compute_all_baseline_metrics(
    dataset_path: Path,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, dict]:
    """Compute metrics for all baseline models.

    Args:
        dataset_path: Path to dataset file.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with baseline model results.
    """
    results = {}

    logger.info("Computing random baseline metrics...")
    random_results = compute_random_baseline_metrics(
        dataset_path=dataset_path,
        random_state=random_state,
    )
    results["random_baseline"] = random_results

    logger.info("Computing naive baseline metrics...")
    naive_results = compute_naive_baseline_metrics(
        dataset_path=dataset_path,
    )
    results["naive_persistence_baseline"] = naive_results

    return results


def _save_results_to_file(results: dict, output_path: Path = LIGHTGBM_EVAL_RESULTS_FILE) -> None:
    """Save results merged with existing results to file.

    Args:
        results: New results to save.
        output_path: Path to save merged results.
    """
    # Load existing results and merge
    all_existing_results = _load_existing_results()
    all_existing_results.update(results)

    # Save merged results
    ensure_output_dir(output_path)
    with open(output_path, "w") as f:
        json.dump(all_existing_results, f, indent=2)

    logger.info(f"Saved baseline results to {output_path}")
    logger.info(f"Total models in results: {len(all_existing_results)}")


def main(
    dataset_path: Path | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """Run baseline model evaluation.

    Computes both random and naive baseline models automatically.

    Args:
        dataset_path: Path to dataset file. Defaults to complete dataset.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with both baseline results.
    """
    dataset_path = resolve_dataset_path(dataset_path, LIGHTGBM_DATASET_COMPLETE_FILE)
    logger.info(f"Using dataset: {dataset_path}")

    # Compute baseline metrics
    results = _compute_all_baseline_metrics(dataset_path, random_state)

    # Save results
    _save_results_to_file(results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute baseline metrics for LightGBM benchmarking"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to dataset file (default: complete dataset)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_STATE})",
    )

    args = parser.parse_args()

    try:
        main(dataset_path=args.dataset, random_state=args.random_state)
    except Exception as e:
        logger.error(f"Baseline evaluation failed: {e}")
        raise
