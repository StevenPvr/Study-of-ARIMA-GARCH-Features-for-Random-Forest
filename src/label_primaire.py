"""Pipeline simplifiée pour labelliser le dataset ``label_primaire``.

Cette étape permet de choisir un modèle existant, d'optimiser ses
hyperparamètres, d'entraîner le modèle sur le split d'entraînement puis de
calculer les labels triple-barrière (López de Prado) sur le split test.
Le dataset complet avec les labels est ensuite exporté en Parquet et en CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys
from typing import Callable, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error

from src.config_logging import setup_logging
from src.constants import (
    DEFAULT_RANDOM_STATE,
    LABEL_PRIMAIRE_OUTPUT_CSV,
    LABEL_PRIMAIRE_OUTPUT_PARQUET,
    LABEL_PRIMAIRE_RESULTS_DIR,
    LABEL_PRIMAIRE_SOURCE_CSV,
    LABEL_PRIMAIRE_SOURCE_PARQUET,
    LIGHTGBM_TRAIN_TEST_SPLIT_RATIO,
    TEST_SPLIT_LABEL,
    TRAIN_SPLIT_LABEL,
)
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration minimale pour un modèle sélectionnable."""

    build_estimator: Callable[[], object]
    param_grid: dict[str, Iterable]


def _get_model_registry() -> dict[str, ModelConfig]:
    """Recense les modèles disponibles dans le projet."""

    return {
        "lightgbm": ModelConfig(
            build_estimator=lambda: lgb.LGBMRegressor(
                objective="regression",
                random_state=DEFAULT_RANDOM_STATE,
                verbosity=-1,
            ),
            param_grid={
                "num_leaves": [31, 63],
                "learning_rate": [0.05, 0.1],
                "max_depth": [-1, 5, 10],
                "n_estimators": [200, 400],
            },
        ),
        "random_forest": ModelConfig(
            build_estimator=lambda: RandomForestRegressor(
                random_state=DEFAULT_RANDOM_STATE,
                n_jobs=-1,
            ),
            param_grid={
                "n_estimators": [200, 400],
                "max_depth": [None, 10, 20],
                "max_features": ["sqrt", "log2"],
            },
        ),
        "gradient_boosting": ModelConfig(
            build_estimator=lambda: GradientBoostingRegressor(random_state=DEFAULT_RANDOM_STATE),
            param_grid={
                "learning_rate": [0.05, 0.1],
                "n_estimators": [200, 400],
                "max_depth": [2, 3, 4],
            },
        ),
        "linear_regression": ModelConfig(
            build_estimator=LinearRegression,
            param_grid={},
        ),
    }


def _load_dataset(custom_path: str | None) -> pd.DataFrame:
    """Charge le dataset label_primaire à partir d'un chemin explicite ou par défaut."""

    if custom_path:
        path = Path(custom_path)
    elif LABEL_PRIMAIRE_SOURCE_PARQUET.exists():
        path = LABEL_PRIMAIRE_SOURCE_PARQUET
    elif LABEL_PRIMAIRE_SOURCE_CSV.exists():
        path = LABEL_PRIMAIRE_SOURCE_CSV
    else:
        raise FileNotFoundError(
            "Aucun fichier label_primaire trouvé. Fournir --data-path ou placer "
            "label_primaire.parquet/csv dans le répertoire data/."
        )

    logger.info("Chargement du dataset label_primaire depuis %s", path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Format de fichier non supporté: {path.suffix}")


def _split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Réalise un split chronologique train/test basé sur le ratio global."""

    split_idx = int(len(df) * LIGHTGBM_TRAIN_TEST_SPLIT_RATIO)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    logger.info("Split effectué: %d lignes train, %d lignes test", len(train_df), len(test_df))
    return train_df, test_df


def _prepare_features(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Supprime la colonne cible et conserve uniquement les features numériques."""

    features = df.drop(columns=[target_column])
    numeric_features = features.select_dtypes(include=["number", "bool"]).copy()
    return numeric_features.ffill().bfill()


def _optimize_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int,
) -> object:
    """Optimise le modèle choisi à l'aide d'une recherche en grille temporelle."""

    registry = _get_model_registry()
    if model_name not in registry:
        raise ValueError(f"Modèle inconnu: {model_name}. Choisir parmi {list(registry)}")

    config = registry[model_name]
    estimator = config.build_estimator()
    if not config.param_grid:
        logger.info("Pas d'hyperparamètres à optimiser pour %s; entraînement direct", model_name)
        estimator.fit(X_train, y_train)
        return estimator

    logger.info("Optimisation des hyperparamètres pour %s", model_name)
    cv = TimeSeriesSplit(n_splits=cv_splits)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=config.param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    logger.info("Meilleure configuration: %s", search.best_params_)
    return best_model


def _apply_triple_barrier(
    prices: pd.Series,
    horizon: int,
    upper_pct: float,
    lower_pct: float,
) -> pd.Series:
    """Calcule les labels triple-barrière sur une série de prix."""

    labels = pd.Series(index=prices.index, dtype="Int64")
    price_values = prices.to_numpy()
    n = len(price_values)

    for i in range(n):
        if i + 1 >= n:
            labels.iloc[i] = pd.NA
            continue

        upper_barrier = price_values[i] * (1 + upper_pct)
        lower_barrier = price_values[i] * (1 - lower_pct)
        window_end = min(n, i + horizon + 1)
        future_prices = price_values[i + 1 : window_end]

        label_value: int | type(pd.NA)
        if np.any(future_prices >= upper_barrier):
            label_value = 1
        elif np.any(future_prices <= lower_barrier):
            label_value = -1
        elif window_end < n:
            label_value = 0
        else:
            label_value = pd.NA

        labels.iloc[i] = label_value

    return labels


def _attach_labels_and_save(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_labels: pd.Series,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Assemble le dataset annoté et l'enregistre en Parquet/CSV."""

    annotated = pd.concat([train_df, test_df], axis=0)
    annotated["split"] = [TRAIN_SPLIT_LABEL] * len(train_df) + [TEST_SPLIT_LABEL] * len(test_df)
    annotated["triple_barrier_label"] = pd.concat(
        [pd.Series([pd.NA] * len(train_df), index=train_df.index, dtype="Int64"), test_labels]
    ).values

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / LABEL_PRIMAIRE_OUTPUT_PARQUET.name
    csv_path = output_dir / LABEL_PRIMAIRE_OUTPUT_CSV.name
    annotated.to_parquet(parquet_path, index=False)
    annotated.to_csv(csv_path, index=False)
    logger.info("Dataset annoté sauvegardé: %s et %s", parquet_path, csv_path)
    return parquet_path, csv_path


def run_label_primaire_pipeline(
    data_path: str | None = None,
    model_name: str = "lightgbm",
    target_column: str = "target",
    price_column: str = "target",
    horizon: int = 5,
    upper_pct: float = 0.02,
    lower_pct: float = 0.02,
    cv_splits: int = 3,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Exécute la pipeline label_primaire de bout en bout."""

    df = _load_dataset(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Colonne cible {target_column} absente du dataset")
    if price_column not in df.columns:
        raise ValueError(f"Colonne de prix {price_column} absente du dataset")

    train_df, test_df = _split_train_test(df)
    X_train = _prepare_features(train_df, target_column)
    X_test = _prepare_features(test_df, target_column)
    y_train = train_df[target_column]
    y_test = test_df[target_column]

    model = _optimize_model(model_name, X_train, y_train, cv_splits)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, predictions)
    logger.info("RMSE sur le split test (%s): %.4f", model_name, test_rmse)

    test_labels = _apply_triple_barrier(test_df[price_column].reset_index(drop=True), horizon, upper_pct, lower_pct)

    output_dir = output_dir or LABEL_PRIMAIRE_RESULTS_DIR
    _attach_labels_and_save(df.reset_index(drop=True), train_df.reset_index(drop=True), test_df.reset_index(drop=True), test_labels, output_dir)
    return df


def _parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", help="Chemin personnalisé vers le dataset label_primaire")
    parser.add_argument("--model", default="lightgbm", help="Nom du modèle à utiliser")
    parser.add_argument("--target-column", default="target", help="Nom de la colonne cible pour l'entraînement")
    parser.add_argument("--price-column", default="target", help="Colonne utilisée pour le calcul des labels triple-barrière")
    parser.add_argument("--horizon", type=int, default=5, help="Fenêtre temporelle pour le triple-barrier")
    parser.add_argument(
        "--upper-pct",
        type=float,
        default=0.02,
        help="Seuil supérieur (en pourcentage) pour la barrière haute",
    )
    parser.add_argument(
        "--lower-pct",
        type=float,
        default=0.02,
        help="Seuil inférieur (en pourcentage) pour la barrière basse",
    )
    parser.add_argument("--cv-splits", type=int, default=3, help="Nombre de splits pour la validation temporelle")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Répertoire de sortie pour les fichiers annotés (Parquet et CSV)",
    )
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parsed = _parse_args(argv or sys.argv[1:])

    run_label_primaire_pipeline(
        data_path=parsed.data_path,
        model_name=parsed.model,
        target_column=parsed.target_column,
        price_column=parsed.price_column,
        horizon=parsed.horizon,
        upper_pct=parsed.upper_pct,
        lower_pct=parsed.lower_pct,
        cv_splits=parsed.cv_splits,
        output_dir=Path(parsed.output_dir) if parsed.output_dir else None,
    )


if __name__ == "__main__":
    main()
