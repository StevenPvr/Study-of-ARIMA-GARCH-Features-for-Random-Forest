"""Dataset utility functions (without_sigma2, ensure functions)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import LIGHTGBM_DATASET_COMPLETE_FILE, LIGHTGBM_DATASET_WITHOUT_SIGMA2_FILE
from src.lightgbm.data_preparation.data_loading import ensure_ticker_id_column
from src.lightgbm.data_preparation.dataset_variants import (
    create_dataset_without_sigma2 as _create_dataset_without_sigma2,
)
from src.utils import get_logger, save_parquet_and_csv

logger = get_logger(__name__)


def create_dataset_without_sigma2(
    df: pd.DataFrame | None = None, output_path: Path | None = None
) -> pd.DataFrame:
    """Create dataset without sigma2_garch columns for ablation study.

    Args:
        df: Complete dataset. If None, loads from LIGHTGBM_DATASET_COMPLETE_FILE.
        output_path: Path to save dataset. If None, uses LIGHTGBM_DATASET_WITHOUT_SIGMA2_FILE.

    Returns:
        Dataset without sigma2_garch columns.
    """
    if df is None:
        complete_file = LIGHTGBM_DATASET_COMPLETE_FILE.with_suffix(".parquet")
        logger.info(f"Loading complete dataset from {complete_file}")
        df = pd.read_parquet(complete_file)

    df_without_sigma2 = _create_dataset_without_sigma2(df)
    df_without_sigma2 = ensure_ticker_id_column(df_without_sigma2)

    if output_path is None:
        output_path = LIGHTGBM_DATASET_WITHOUT_SIGMA2_FILE.with_suffix(".parquet")

    save_parquet_and_csv(df_without_sigma2, output_path)
    logger.info(
        f"Saved dataset without sigma2_garch: {output_path} "
        f"({len(df_without_sigma2)} rows, {len(df_without_sigma2.columns)} columns)"
    )

    return df_without_sigma2
