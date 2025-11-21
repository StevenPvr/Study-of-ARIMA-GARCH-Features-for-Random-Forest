# MÉTHODOLOGIE GÉNÉRALE - S&P 500 FORECASTING

## Objectif du projet
Évaluer l'impact des features ARIMA-GARCH sur la prévision volatilité S&P 500 via LightGBM avec safeguards anti-leakage rigoureux.

## Hypothèse centrale
ARIMA-GARCH capturent patterns temporels et volatilité conditionnelle qui améliorent prédictions LightGBM vs indicateurs techniques seuls.

## Architecture pipeline (4 étapes séquentielles)

```
DATA → ARIMA → GARCH → LIGHTGBM
```

### 1. DATA PIPELINE
**Input**: Wikipedia + Yahoo Finance
**Output**: `data_tickers_full.parquet`, `weighted_log_returns_split.csv`
**Durée**: 2013-2024 (12 ans)

**Étapes**:
- Scraping tickers S&P 500 (Wikipedia)
- Téléchargement OHLCV parallèle (yfinance)
- Validation + nettoyage (doublons, valeurs manquantes)
- Split temporel 80/20 (TRAIN: dates antérieures, TEST: postérieures)
- Calcul log returns: log(close_t / close_{t-1})

**Safeguards anti-leakage**:
- Split chronologique strict (pas shuffle)
- Features fenêtres roulantes calculées AVANT split
- Suppression (window_size-1) premières obs TEST (contamination)

### 2. ARIMA PIPELINE
**Input**: `weighted_log_returns_split.csv`
**Output**: `dataset_garch.csv` (résidus ARIMA)
**Modèle**: ARIMA(0,0,0) avec constante

**Décision conception CRITIQUE**:
- Pas de capacité prévision (ordre 0,0,0)
- Objectif: Extraction résidus démoyennisés pour GARCH
- Formule: y_t = μ + ε_t

**Modules**:
1. **data_visualisation**: ACF/PACF, tests stationnarité, décomposition saisonnière
2. **stationnarity_check**: ADF, KPSS, Zivot-Andrews
3. **training_arima**: Fit ARIMA(0,0,0) sur TRAIN (refit 21j)
4. **evaluation_arima**: Rolling forecast TEST + backtest full-series TRAIN only

**Safeguards**: Backtest `include_test=False`, walk-forward jamais données futures

### 3. GARCH PIPELINE
**Input**: `dataset_garch.csv` (résidus ARIMA)
**Output**: `data_tickers_full_insights.parquet` (σ² forecasts)
**Modèle**: EGARCH avec Student-t ou Skew-t

**Formule**: log(σ²_{t+1}) = ω + β·log(σ²_t) + α·|z_t| + γ·z_t

**Modules**:
1. **garch_data_visualisation**: ACF résidus², effets ARCH
2. **garch_numerical_test**: Tests hétéroscédasticité (Ljung-Box, ARCH-LM)
3. **structure_garch**: Détection formelle effets ARCH
4. **garch_params**: MLE estimation + Optuna optimization (180 combos)
5. **training_garch**: Train EGARCH (expanding + refit périodique)
6. **garch_diagnostic**: Validation résidus standardisés
7. **garch_eval**: Full-sample forecasts + évaluation TEST

**Grille optimisation**:
- Ordres: {(1,1), (1,2), (2,1), (2,2)}
- Distributions: {student, skewt}
- Refit freq: {1, 5, 15, 21, 63} jours
- Windows: {expanding, rolling}

**Modes forecast**:
- **no_refit** (DEFAULT): Paramètres frozen (zéro leakage)
- **hybrid**: Refit schedule activé (use with caution, logged)

**Safeguards**:
- Optimisation: TRAIN only (60% train, 30% val, 10% test interne)
- Évaluation: TEST holdout (20% full data)
- Forecast mode logged (audit trail)

### 4. LIGHTGBM PIPELINE
**Input**: `data_tickers_full_insights.parquet` (avec σ²_garch)
**Output**: Modèles + métriques + SHAP analysis
**Target**: log_volatility (J+1)

**7 variants datasets (ablation)**:
1. Complete: Technical + insights + target lags
2. Without Insights: Technical + target lags
3. Without Sigma2: Complete - sigma2_garch
4. Sigma Plus Base: GARCH σ² + log_volatility + lags
5. Log Volatility Only: Target + lags (autorégressif)
6. Technical Only: Technical (pas target lags)
7. Technical + Insights: Technical + ARIMA-GARCH (pas target lags)

**Modules**:
1. **data_preparation**: Feature engineering + 7 variants
2. **correlation**: Heatmaps corrélations
3. **optimisation**: Optuna tuning (TimeSeriesSplit)
4. **training**: Train modèles (80% TRAIN)
5. **eval**: TEST (20%) + SHAP + Diebold-Mariano
6. **data_leakage_checkup**: Tests anti-leakage automatisés
7. **baseline**: Persistence, mean, random walk
8. **permutation**: Feature importance (model-agnostic, 200 repeats)

**Safeguards**:
- Features t: données ≤ t-1 uniquement
- TimeSeriesSplit (pas KFold)
- R² threshold test (>0.1 = suspect)
- TEST jamais visible pendant optimisation

## Safeguards Anti-Leakage Généraux

**Séparation temporelle**:
- Split 80/20 chronologique (TRAIN antérieur, TEST postérieur)
- Optimisation/validation: TRAIN uniquement
- Évaluation finale: TEST holdout

**Validation croisée**:
- TimeSeriesSplit (ordre temporel respecté)
- Walk-forward (expanding/rolling windows)
- Jamais shuffle/KFold

**Feature engineering**:
- Features t: données passées uniquement (≤ t-1)
- Moving averages: backward-looking
- Target shift: prédire t+1 depuis t

**Tests automatisés**:
- R² threshold (leakage detection)
- Temporal order checks
- Lag alignment validation

## Fichiers principaux

**Données**:
- `data/dataset_filtered.parquet`: OHLCV nettoyé
- `data/weighted_log_returns_split.csv`: Séries temporelles (split)
- `data/data_tickers_full_insights.parquet`: Dataset ML final

**Résultats**:
- `results/arima/`: Modèles + évaluation ARIMA
- `results/garch/`: Estimation + forecasts GARCH
- `results/lightgbm/`: Modèles + métriques + SHAP

## Exécution complète
```bash
python src/main_global.py  # Auto-découverte et exécution ordonnée
```

## Constantes clés
- `DEFAULT_RANDOM_STATE`: 42 (reproductibilité)
- `TRAIN_TEST_SPLIT_RATIO`: 0.8
- `ARIMA_DEFAULT_ORDER`: (0,0,0)
- `GARCH_MIN_WINDOW_SIZE`: 250
- `LIGHTGBM_LAG_WINDOWS`: (1, 2, 3)
- `LEAKAGE_R2_THRESHOLD`: 0.1
