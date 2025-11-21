"""LightGBM training module - backward compatibility layer."""

from __future__ import annotations

# Re-export functions for backward compatibility
from src.lightgbm.training.data_loading import load_dataset, load_optimization_results
from src.lightgbm.training.model_training import _run_single_training, save_model, train_lightgbm
from src.lightgbm.training.training_orchestration import run_training

__all__ = [
    "load_dataset",
    "load_optimization_results",
    "run_training",
    "save_model",
    "train_lightgbm",
    "_run_single_training",  # For tests
]
