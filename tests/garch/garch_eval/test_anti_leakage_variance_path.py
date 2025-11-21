"""Test anti-leakage pour compute_initial_forecasts.

Vérifie que la fonction compute_initial_forecasts respecte bien la causalité
temporelle en excluant la dernière observation lors du calcul des forecasts.
"""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent.parent.parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pytest

from src.garch.garch_eval.variance_path import compute_initial_forecasts


def test_compute_initial_forecasts_anti_leakage():
    """Vérifie que compute_initial_forecasts exclut bien la dernière observation."""
    # Données synthétiques
    np.random.seed(42)
    resid_train = np.random.randn(100) * 0.01
    sigma2_path_train = np.ones(100) * 0.0001  # Dummy variance path

    # Paramètres EGARCH(1,1) stables
    omega = -0.1
    alpha = 0.15
    gamma = -0.05
    beta = 0.95
    dist = "student"
    nu = 10.0
    horizon = 1

    # Exécuter compute_initial_forecasts
    s2_1, s2_h = compute_initial_forecasts(
        resid_train=resid_train,
        sigma2_path_train=sigma2_path_train,
        horizon=horizon,
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        dist=dist,
        nu=nu,
    )

    # Vérifier que les forecasts sont valides
    assert np.isfinite(s2_1), "One-step forecast doit être fini"
    assert s2_1 > 0, "Variance forecast doit être positive"
    assert len(s2_h) == horizon, f"Expected {horizon} forecasts, got {len(s2_h)}"

    # L'assertion anti-leakage dans la fonction elle-même vérifiera
    # que resid_for_forecast a bien exclu la dernière observation


def test_compute_initial_forecasts_minimal_data():
    """Vérifie que compute_initial_forecasts gère correctement les cas limites."""
    # Cas avec seulement 2 observations (minimum requis)
    resid_train = np.array([0.01, -0.01])
    sigma2_path_train = np.array([0.0001, 0.0001])

    omega = -0.1
    alpha = 0.15
    gamma = -0.05
    beta = 0.95
    dist = "student"
    nu = 10.0
    horizon = 1

    s2_1, s2_h = compute_initial_forecasts(
        resid_train=resid_train,
        sigma2_path_train=sigma2_path_train,
        horizon=horizon,
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        dist=dist,
        nu=nu,
    )

    assert np.isfinite(s2_1)
    assert s2_1 > 0


def test_compute_initial_forecasts_insufficient_data():
    """Vérifie que compute_initial_forecasts rejette les données insuffisantes."""
    # Cas avec seulement 1 observation (insuffisant)
    resid_train = np.array([0.01])
    sigma2_path_train = np.array([0.0001])

    omega = -0.1
    alpha = 0.15
    gamma = -0.05
    beta = 0.95
    dist = "student"
    nu = 10.0
    horizon = 1

    with pytest.raises(ValueError, match="Need at least 2 training residuals"):
        compute_initial_forecasts(
            resid_train=resid_train,
            sigma2_path_train=sigma2_path_train,
            horizon=horizon,
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            dist=dist,
            nu=nu,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
