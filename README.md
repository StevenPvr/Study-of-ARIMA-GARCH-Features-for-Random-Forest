# S&P 500 Forecasting: Impact des Features ARIMA-GARCH sur LightGBM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Description

Ce projet évalue l'impact de l'ajout de features issues d'un modèle **ARIMA-GARCH** sur les performances de prévision d'un modèle **LightGBM** pour les rendements du S&P 500.

### Objectif Principal

Tester l'hypothèse selon laquelle les modèles ARIMA-GARCH capturent des patterns temporels et de volatilité conditionnelle qui peuvent enrichir un modèle LightGBM et améliorer ses prévisions de rendements.

### Approche Méthodologique

Le projet suit une architecture modulaire en pipelines séquentiels :

1. **Pipeline de Données** : Collecte, nettoyage et agrégation pondérée par liquidité
2. **Pipeline ARIMA/SARIMA** : Modélisation des rendements avec optimisation et évaluation
3. **Pipeline GARCH** : Modélisation de la volatilité conditionnelle (EGARCH)
4. **Pipeline LightGBM** : Modèle ML avec features ARIMA-GARCH
5. **Benchmark** : Comparaison avec baselines de volatilité

## 🚀 Installation

### Prérequis

- Python 3.10 ou supérieur
- pip ou conda

### Installation des dépendances

```bash
# Cloner le repository
git clone <repository-url>
cd S&P500_Forecasting*

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -e .

# Pour le développement (optionnel)
pip install -e ".[dev]"
```

### Dépendances principales

- `pandas` : Manipulation de données
- `numpy` : Calculs numériques
- `statsmodels` : Modèles ARIMA/SARIMA
- `scikit-learn` : LightGBM et métriques
- `optuna` : Optimisation bayésienne
- `shap` : Interprétabilité ML
- `yfinance` : Téléchargement de données
- `matplotlib`, `seaborn` : Visualisations

## 📁 Structure du Projet

```
S&P500_Forecasting*/
├── data/                    # Données brutes et intermédiaires
│   ├── dataset.csv          # Données historiques S&P 500
│   ├── weighted_log_returns.csv
│   └── lightgbm_dataset_*.csv     # Datasets pour LightGBM
├── src/
│   ├── data_fetching/       # Collecte des données
│   ├── data_cleaning/       # Nettoyage et validation
│   ├── data_conversion/     # Agrégation pondérée
│   ├── data_preparation/    # Split train/test
│   ├── arima/               # Pipeline ARIMA/SARIMA
│   │   ├── optimisation_arima/
│   │   ├── training_arima/
│   │   └── evaluation_arima/
│   ├── garch/               # Pipeline GARCH
│   │   ├── structure_garch/
│   │   ├── garch_params/
│   │   ├── training_garch/
│   │   ├── garch_diagnostic/
│   │   ├── garch_eval/
│   │   └── rolling_garch/
│   ├── lightgbm/        # Pipeline LightGBM
│   │   ├── data_preparation/
│   │   ├── optimisation/
│   │   ├── training/
│   │   ├── eval/
│   │   └── ablation/
│   └── benchmark/            # Comparaison avec baselines
├── results/                 # Résultats et modèles entraînés
│   ├── models/              # Modèles sauvegardés
│   ├── garch/               # Résultats GARCH
│   └── lightgbm/       # Résultats LightGBM
├── plots/                   # Visualisations
├── documentation/           # Documentation méthodologique
│   └── methodologie.txt     # Documentation détaillée
├── requirements.txt         # Dépendances principales
├── requirements-dev.txt    # Dépendances développement
└── README.md               # Ce fichier
```

## 🎯 Utilisation

### Exécution du Pipeline Complet

Pour exécuter l'ensemble du pipeline de bout en bout :

```bash
python src/main_global.py
```

Ce script découvre automatiquement tous les modules et les exécute dans le bon ordre :

1. Collecte et préparation des données
2. Optimisation et entraînement ARIMA
3. Estimation et entraînement GARCH
4. Génération des features pour LightGBM
5. Optimisation et entraînement LightGBM
6. Évaluation et benchmark

### Exécution Module par Module

Vous pouvez également exécuter chaque module individuellement :

```bash
# Pipeline de données
python src/data_fetching/main.py
python src/data_cleaning/main.py
python src/data_conversion/main.py
python src/data_preparation/main.py

# Pipeline ARIMA
python src/arima/optimisation_arima/main.py
python src/arima/training_arima/main.py
python src/arima/evaluation_arima/main.py

# Pipeline GARCH
python src/garch/structure_garch/main.py
python src/garch/garch_params/main.py
python src/garch/training_garch/main.py
python src/garch/garch_diagnostic/main.py
python src/garch/garch_eval/main.py
python src/garch/rolling_garch/main.py

# Pipeline LightGBM
python src/lightgbm/data_preparation/main.py
python src/lightgbm/optimisation/main.py
python src/lightgbm/training/main.py
python src/lightgbm/eval/main.py
python src/lightgbm/ablation/main.py

# Benchmark
python src/benchmark/main_vol_backtest.py
```

### Tests

Exécuter les tests unitaires :

```bash
# Tous les tests
pytest

# Tests spécifiques
pytest src/arima/optimisation_arima/test_optimisation_arima.py
pytest src/lightgbm/training/test_training.py

# Avec couverture (si pytest-cov installé)
pytest --cov=src
```

## 📊 Résultats Principaux

### Données

- **Période** : 2013-01-01 à 2024-12-31 (12 ans)
- **Tickers** : ~500 composantes du S&P 500
- **Split** : 80% train / 20% test (temporel)

### Modèles

- **ARIMA/SARIMA** : Optimisation sur grille d'hyperparamètres, sélection par AIC/BIC
- **EGARCH(1,1)** : Estimation pour distributions Normale, Student-t, Skew-t
- **LightGBM** : Optimisation bayésienne avec Optuna, walk-forward CV

### Features LightGBM

- **Features ARIMA-GARCH** :
  - `arima_pred_return` : Prévisions ARIMA
  - `arima_residual_return` : Résidus ARIMA
  - `sigma2_garch` : Variance conditionnelle
  - `sigma_garch` : Volatilité conditionnelle
  - `std_resid_garch` : Résidus standardisés GARCH

- **Indicateurs techniques** :
  - RSI (14), SMA (20), EMA (20), MACD

- **Features laggées** : Windows [1, 5, 10, 20] jours

### Métriques d'Évaluation

- **ARIMA** : MSE, RMSE, MAE, MAPE, Ljung-Box test
- **GARCH** : MSE, MAE, QLIKE, MZ regression, VaR coverage
- **LightGBM** : MSE, RMSE, MAE, R², Feature importances, SHAP

## 📈 Visualisations

Les visualisations sont générées automatiquement dans le dossier `plots/` :

- **ARIMA** : ACF/PACF, résidus, prévisions rolling
- **GARCH** : Structure ARCH, diagnostics, prévisions de volatilité
- **LightGBM** : Corrélations, SHAP values, importance des features
- **Benchmark** : Comparaison des prévisions de volatilité

## 📚 Documentation

Pour une documentation méthodologique détaillée, consultez :

```
documentation/methodologie.txt
```

Cette documentation couvre :

- L'architecture complète du projet
- La méthodologie de chaque pipeline
- Les formules et équations utilisées
- La structure des données
- Les hypothèses testées

## 🔬 Hypothèses Testées

1. **H1** : Les features ARIMA-GARCH améliorent les prévisions du LightGBM
   - Testé via comparaison Complete vs Without Insights

2. **H2** : La variance conditionnelle (sigma2_garch) est une feature importante
   - Testé via ablation study (Complete vs Without Sigma2)

3. **H3** : Le modèle ARIMA-GARCH capture des patterns de volatilité significatifs
   - Testé via diagnostics GARCH (ARCH-LM, Ljung-Box)

4. **H4** : Le modèle EGARCH est supérieur aux baselines simples
   - Testé via volatility backtest (EWMA, Rolling, ARCH, HAR)

## 🔄 Reproductibilité

Le projet garantit la reproductibilité via :

- **Random state fixe** : 42 (DEFAULT_RANDOM_STATE)
- **Période de données fixe** : 2013-2024 (pas de mise à jour automatique)
- **Split temporel fixe** : 80/20
- **Seeds fixés** pour numpy, pandas, sklearn, optuna

## ⚙️ Configuration

Les constantes et paramètres par défaut sont définis dans :

```
src/constants.py
```

Principaux paramètres configurables :

- Période de données
- Split train/test ratio
- Hyperparamètres d'optimisation
- Fenêtres de lag
- Fréquence de refit (rolling forecasts)

## 🐛 Dépannage

### Erreurs courantes

1. **ModuleNotFoundError** : Vérifier que toutes les dépendances sont installées

   ```bash
   pip install -r requirements.txt
   ```

2. **Fichiers manquants** : Exécuter les pipelines dans l'ordre

   ```bash
   python src/main_global.py
   ```

3. **Erreurs de mémoire** : Réduire la taille des datasets ou optimiser les paramètres

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

### Standards de code

- Suivre PEP 8
- Utiliser type hints
- Ajouter des docstrings (style Google/NumPy)
- Écrire des tests unitaires
- Vérifier avec `black` et `ruff` avant de commit

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👤 Auteur

Projet développé dans le cadre d'une étude sur l'impact des features ARIMA-GARCH sur les modèles de machine learning pour la prévision financière.

## 🙏 Remerciements

- Bibliothèques open-source utilisées : pandas, statsmodels, scikit-learn, optuna, shap
- Données : yfinance pour les données historiques du S&P 500

## 📧 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur le repository.

---

**Note** : Ce projet est à des fins éducatives et de recherche. Les résultats ne constituent pas des conseils financiers.
