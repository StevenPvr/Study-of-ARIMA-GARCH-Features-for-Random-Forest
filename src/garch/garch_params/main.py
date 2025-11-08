"""CLI for estimating EGARCH(1,1) parameters via conditional MLE.

Estimates parameters for three distributions simultaneously:
- Normal
- Student-t
- Skew-t

Uses conditional maximum likelihood estimation with variance recursion.
Results saved to results/garch/eval/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.constants import (
    GARCH_DATASET_FILE,
    GARCH_ESTIMATION_DIR,
    GARCH_ESTIMATION_FILE,
)
from src.garch.garch_params.utils import (
    estimate_egarch_models,
    load_and_prepare_data,
)
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Estimate EGARCH(1,1) parameters via conditional MLE.

    Methodology:
    - Assumes parametric distribution for innovations zt (Normal, Student-t, Skew-t)
    - Maximizes conditional log-likelihood by recursing conditional variance σt²
    - Optimizes all three distributions simultaneously

    Results saved to results/garch/estimation/
    """
    logger.info("=" * 60)
    logger.info("GARCH ESTIMATION (Conditional MLE)")
    logger.info("Optimizing: Normal, Student-t, Skew-t")
    logger.info("=" * 60)

    resid_train, resid_test = load_and_prepare_data()
    egarch_normal, egarch_student, egarch_skewt = estimate_egarch_models(resid_train)

    payload = {
        "source": str(GARCH_DATASET_FILE),
        "methodology": "Conditional maximum likelihood estimation",
        "n_obs_train": int(resid_train.size),
        "n_obs_test": int(resid_test.size),
        "egarch_normal": egarch_normal,
        "egarch_student": egarch_student,
        "egarch_skewt": egarch_skewt,
    }

    GARCH_ESTIMATION_DIR.mkdir(parents=True, exist_ok=True)
    with GARCH_ESTIMATION_FILE.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved GARCH MLE results: %s", GARCH_ESTIMATION_FILE)


if __name__ == "__main__":
    main()
