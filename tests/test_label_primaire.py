from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd

from src.label_primaire import run_label_primaire_pipeline
from src.constants import TRAIN_SPLIT_LABEL, TEST_SPLIT_LABEL


def test_label_primaire_pipeline_outputs(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "target": [100, 102, 101, 103, 105, 106, 104, 107, 108, 110],
            "feature": range(10),
        }
    )

    data_path = tmp_path / "label_primaire.parquet"
    df.to_parquet(data_path, index=False)

    output_dir = tmp_path / "out"
    run_label_primaire_pipeline(
        data_path=str(data_path),
        model_name="linear_regression",
        target_column="target",
        price_column="target",
        horizon=1,
        upper_pct=0.01,
        lower_pct=0.01,
        cv_splits=2,
        output_dir=output_dir,
    )

    parquet_path = output_dir / "label_primaire_labels.parquet"
    csv_path = output_dir / "label_primaire_labels.csv"
    assert parquet_path.exists()
    assert csv_path.exists()

    annotated = pd.read_parquet(parquet_path)
    assert set([TRAIN_SPLIT_LABEL, TEST_SPLIT_LABEL]) == set(annotated["split"].unique())
    assert "triple_barrier_label" in annotated.columns

    train_labels = annotated.loc[annotated["split"] == TRAIN_SPLIT_LABEL, "triple_barrier_label"]
    assert train_labels.isna().all()

    test_labels = annotated.loc[annotated["split"] == TEST_SPLIT_LABEL, "triple_barrier_label"].reset_index(drop=True)
    # Avec un horizon 1 et une hausse de 1 %, la première observation test franchit la barrière haute
    assert test_labels.iloc[0] == 1
    # La dernière observation n'a pas de futur : label manquant
    assert pd.isna(test_labels.iloc[-1])
