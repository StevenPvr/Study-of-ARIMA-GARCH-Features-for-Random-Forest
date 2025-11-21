# S&P 500 Forecasting: ARIMA-EGARCH + LightGBM Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Objectif

Évaluer l'impact d'un **log-sigma conditionnel** extrait d'un pipeline ARIMA-EGARCH sur la prévision de volatilité S&P 500 via LightGBM, avec des safeguards anti-leakage rigoureux.

**Question de recherche** : La prévision de volatilité conditionnelle issue d'ARIMA-EGARCH améliore-t-elle la capacité prévisionnelle d'un LightGBM à prédire la log-volatilité des actions ?

### Hypothèses testées

- **H1** : Les insights ARIMA-EGARCH apportent un gain de prévision significatif par rapport aux indicateurs techniques seuls
- **H2** : Le log-sigma conditionnel seul surpasse les baselines naïves (persistance, random)
- **H3** : Le modèle EGARCH capture correctement la dynamique de volatilité (log-QLIKE, VaR backtests)

## Architecture du Pipeline

```
DATA ──────► ARIMA ──────► EGARCH ──────► LIGHTGBM
500+ tickers  (0,0,0)     (1,1) Skew-t    8 variants
12 ans        Résidus     σ² forecasts    Ablation
```

| Étape | Description | Output |
|-------|-------------|--------|
| **DATA** | 500+ tickers S&P 500 (2013-2024), nettoyage, split 80/20 chronologique | `data_tickers_full.parquet` |
| **ARIMA** | ARIMA(0,0,0) avec constante - extraction résidus démoyennisés | `dataset_garch.csv` |
| **EGARCH** | EGARCH(1,1) Skew-t - modélisation volatilité conditionnelle | `data_tickers_full_insights.parquet` |
| **LightGBM** | Prédiction log_volatility J+1 avec 8 datasets d'ablation | Modèles + métriques |

## Résultats Clés

### Performance des modèles LightGBM (Test Set: 213,021 observations)

| Dataset | RMSE | MAE | R² | Conclusion |
|---------|------|-----|-----|------------|
| **(1) Complet** | **0.0109** | **0.0062** | **0.765** | Meilleur modèle |
| (2) Sans insight | 0.0113 | 0.0065 | 0.749 | Référence sans EGARCH |
| (6) Log-volatilité seule | 0.0160 | 0.0095 | 0.497 | AR baseline |
| (7) Log-vol + insights | 0.0153 | 0.0091 | 0.538 | AR + EGARCH |
| (5) Insights seuls | 0.0221 | 0.0142 | 0.037 | Échec H2 |
| *Baseline persistance* | 0.0185 | 0.0107 | 0.320 | $\hat{y}_{t+1} = y_t$ |

### Conclusions

- **H1 validée** : Dataset complet vs sans insight → RMSE -3.1%, R² +1.54 pts (DM test p < 0.01)
- **H2 rejetée** : Insights seuls (R² = 0.037) < Persistance (R² = 0.320)
- **H3 non rejetée** : EGARCH bien calibré (log-QLIKE -8.41, VaR backtests OK)

### Modèle EGARCH retenu

- **Spécification** : EGARCH(1,1) avec distribution Skew-t (ν=7.30, λ=-0.30)
- **Paramètres** : ω=-0.31, α=0.23, γ=-0.18, β=0.97
- **Validation** : Ljung-Box OK, ARCH-LM OK, Engle-Ng OK, Nyblom stability OK

## Quick Start

### Installation

```bash
git clone <repo>
cd S&P500_Forecasting
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Exécution complète

```bash
python src/main_global.py  # Auto-découverte et exécution des 22 modules
```

### Exécution par étape

```bash
# Data Pipeline (3 modules)
python src/data_fetching/main.py
python src/data_cleaning/main.py
python src/data_preparation/main.py

# ARIMA Pipeline (4 modules)
python src/arima/data_visualisation/main.py
python src/arima/stationnarity_check/main.py
python src/arima/training_arima/main.py
python src/arima/evaluation_arima/main.py

# GARCH Pipeline (7 modules)
python src/garch/garch_data_visualisation/main.py
python src/garch/garch_numerical_test/main.py
python src/garch/structure_garch/main.py
python src/garch/garch_params/main.py
python src/garch/training_garch/main.py
python src/garch/garch_diagnostic/main.py
python src/garch/garch_eval/main.py --forecast-mode no_refit

# LightGBM Pipeline (8 modules)
python src/lightgbm/data_preparation/main.py
python src/lightgbm/correlation/main.py
python src/lightgbm/optimisation/main.py
python src/lightgbm/training/main.py
python src/lightgbm/eval/main.py
python src/lightgbm/data_leakage_checkup/main.py
python src/lightgbm/baseline/main.py
python src/lightgbm/permutation/main.py
```

## Méthodologie

### Safeguards Anti-Leakage

| Mécanisme | Description |
|-----------|-------------|
| **Split chronologique** | 80% train (2013-2022) / 20% test (2022-2024) |
| **TimeSeriesSplit** | Cross-validation temporelle (pas KFold) |
| **Causalité features** | Features à t utilisent données ≤ t-1 uniquement |
| **Forecast mode** | `no_refit` (défaut) - paramètres EGARCH frozen pour zéro leakage |
| **Tests automatisés** | R² threshold, temporal order checks |

### 8 Variants Datasets (Ablation Studies)

| # | Dataset | Tech | Log-vol* lags | Calendar | Insights EGARCH |
|---|---------|------|---------------|----------|-----------------|
| 1 | Complete | ✓ | ✓ | ✓ | ✓ |
| 2 | Sans insight | ✓ | ✓ | ✓ | ✗ |
| 3 | Technical only | ✓ | ✗ | ✓ | ✗ |
| 4 | Technical + insights | ✓ | ✗ | ✓ | ✓ |
| 5 | Insights only | ✗ | ✗ | ✗ | ✓ |
| 6 | Log-volatilité* only | ✗ | ✓ | ✗ | ✗ |
| 7 | Log-vol* + insights | ✗ | ✓ | ✗ | ✓ |
| 8 | Without sigma2 | ✓ | ✓ | ✓ | Partiel |

*Log-volatilité = log de la volatilité réalisée (target) avec lags 1, 2, 3

### Optimisation Hyperparamètres

**EGARCH** (Optuna, 100 trials) :

- Ordres : o∈{1,2}, p∈{1,2}
- Distributions : Student-t, Skew-t
- Refit freq : {1, 5, 15, 21, 63} jours
- Windows : expanding, rolling (500-2000 obs)
- Critère : log-QLIKE + validation walk-forward

**LightGBM** (Optuna, 100 trials/dataset) :

- num_leaves, learning_rate, max_depth, min_child_samples
- feature_fraction, bagging_fraction, reg_alpha/lambda
- Critère : RMSE via TimeSeriesSplit 5-fold

## Structure du Projet

```
├── data/                    # Datasets (CSV/Parquet)
│   ├── dataset.csv          # Données brutes yfinance
│   ├── data_tickers_full.parquet
│   └── data_tickers_full_insights.parquet  # Avec features EGARCH
│
├── results/                 # Modèles + métriques
│   ├── arima/              # training/, evaluation/, outputs/
│   ├── garch/              # structure/, estimation/, training/, diagnostic/, evaluation/
│   └── lightgbm/           # optimization/, models/, evaluation/, ablation/
│
├── plots/                   # Visualisations
│   ├── arima/              # ACF/PACF, stationnarité, résidus
│   ├── garch/              # Diagnostics, VaR backtests
│   └── lightgbm/           # SHAP, permutation importance, corrélations
│
├── src/
│   ├── data_fetching/      # Wikipedia scraping, yfinance download
│   ├── data_cleaning/      # Validation, nettoyage
│   ├── data_preparation/   # Split train/test, log returns
│   ├── arima/              # 4 modules: visualisation, stationnarité, training, evaluation
│   ├── garch/              # 7 modules: visualisation → évaluation
│   ├── lightgbm/           # 8 modules: data_prep → permutation
│   ├── utils/              # 11 modules utilitaires
│   ├── visualization/      # Plotting utilities
│   ├── constants.py        # Constantes centralisées (529 lignes)
│   ├── path.py             # Chemins centralisés (217 lignes)
│   └── main_global.py      # Orchestrateur pipeline complet
│
├── tests/                   # 99 fichiers de tests pytest
├── documentation/           # Méthodologies détaillées
└── main.tex                 # Document LaTeX complet
```

## Features

### Features Techniques (laggées 1, 2, 3 jours)

| Catégorie | Features |
|-----------|----------|
| **Returns** | `log_return`, `abs_ret`, `ret_sq` |
| **Volume** | `log_volume`, `log_volume_rel_ma_5`, `log_volume_zscore_20` |
| **Turnover** | `log_turnover`, `turnover_rel_ma_5` |
| **Momentum** | `obv` (On-Balance Volume), `atr` (Average True Range 14j) |
| **Calendrier** | `day_of_week`, `month`, `is_month_end`, `day_in_month_norm` (non laggés) |

### Feature ARIMA-EGARCH

| Feature | Description |
|---------|-------------|
| `log_sigma_garch` | Log de l'écart-type conditionnel EGARCH prédit (log σ̂_{t+1\|t}) |

### Target

| Variable | Description |
|----------|-------------|
| `log_volatility` | Log de la volatilité réalisée sur 5 jours : log(1 + √(Σr²)) |

## Tests

```bash
# Tous les tests
pytest

# Avec couverture
pytest --cov=src

# Tests spécifiques
pytest tests/garch/                    # GARCH uniquement
pytest tests/lightgbm/data_preparation/ # Data prep LightGBM
pytest -v tests/lightgbm/data_preparation/test_data_leakage_fix.py
```

## Constantes Clés

```python
# Reproductibilité
DEFAULT_RANDOM_STATE = 42

# Split données
LIGHTGBM_TRAIN_TEST_SPLIT_RATIO = 0.8  # 80% train, 20% test

# ARIMA
ARIMA_DEFAULT_ORDER = (0, 0, 0)        # White noise model
ARIMA_DEFAULT_TREND = "c"              # Constante uniquement
ARIMA_DEFAULT_REFIT_EVERY = 21         # ~1 mois trading

# GARCH
GARCH_MIN_WINDOW_SIZE = 250            # ~1 an trading
GARCH_EVAL_FORECAST_MODE_DEFAULT = "no_refit"
GARCH_OPTIMIZATION_N_TRIALS = 100

# LightGBM
LIGHTGBM_LAG_WINDOWS = (1, 2, 3)
LIGHTGBM_OPTIMIZATION_N_TRIALS = 100
LEAKAGE_R2_THRESHOLD = 0.1             # Seuil détection leakage
```

## Évaluation et Interprétabilité

### Métriques

- **Performance** : RMSE, MAE, MSE, R²
- **EGARCH** : log-QLIKE, VaR backtests (1%, 5%), Mincer-Zarnowitz calibration
- **Comparaisons** : Diebold-Mariano test (HAC), Bootstrap R² (5000 reps)

### Interprétabilité

- **SHAP Analysis** : TreeExplainer pour impact marginal des features
- **Block Permutation Importance** : Dégradation R² après permutation temporelle (200 reps)

### Résultat clé SHAP

`log_sigma_garch` se positionne :

- **Rang 13/55** dans le dataset complet
- **Rang 9/45** dans technical + insights
- **Rang 3/8** dans log-vol + insights (devant lag-2 de log_volatility)

## Documentation

| Fichier | Description |
|---------|-------------|
| `CLAUDE.md` | Instructions pour Claude Code |
| `main.tex` | Document LaTeX complet avec toutes les analyses |
| `documentation/METHODOLOGIE_PRINCIPALE.md` | Vue d'ensemble méthodologique |
| `src/arima/METHODOLOGIE_ARIMA.txt` | Pipeline ARIMA |
| `src/garch/METHODOLOGIE_GARCH.txt` | Pipeline GARCH (7 modules) |
| `src/lightgbm/methodologie_lightgbm.txt` | Pipeline LightGBM (8 modules) |

## Limitations

- **Méthodologiques** : ARIMA(0,0,0) minimaliste, horizon J+1 uniquement, seed unique (42)
- **Empiriques** : S&P 500 2013-2024, pas de test sur crise 2008 ou autres marchés
- **Données** : Survivorship bias, imputation valeurs manquantes à 0

## Extensions Futures

1. Comparer GJR-GARCH, FIGARCH, HAR-RV
2. Tester XGBoost, CatBoost, LSTM/GRU
3. Horizons 5, 10, 21 jours
4. Autres marchés (EURO STOXX 50, FTSE 100)
5. Backtest VaR opérationnelle avec coûts de transaction

## Code Quality

```bash
# Format
black .
ruff format .

# Lint
ruff check .

# Type checking
mypy src/
```

## License

MIT License - See LICENSE file
