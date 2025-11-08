"""Classic evaluation metrics for GARCH volatility forecasts.

Includes:
- Variance forecast losses: QLIKE, MSE, MAE
- Mincer-Zarnowitz regression on e_t^2 vs sigma_t^2
- VaR backtests: Kupiec POF and Christoffersen independence + combined

All functions are small, typed, and dependency-light (SciPy optional for p-values).
"""

from __future__ import annotations

import json

import numpy as np

from src.constants import (
    GARCH_CALIBRATION_EPS,
    GARCH_EVAL_DEFAULT_ALPHAS,
    GARCH_EVAL_EPSILON,
    GARCH_EVAL_HALF,
    GARCH_EVAL_METRICS_FILE,
    GARCH_EVAL_MIN_ALPHA,
)
from src.garch.garch_eval.utils import (
    chi2_sf,
    compute_all_metrics,
    compute_mz_pvalues,
    filter_test_data,
    load_dataset_for_metrics,
    load_test_resid_sigma2,
)
from src.utils import get_logger

logger = get_logger(__name__)


def qlike_loss(e: np.ndarray, sigma2: np.ndarray) -> float:
    """Return average QLIKE loss: log(sigma2) + e^2 / sigma2.

    Args:
    ----
        e: Residuals aligned to sigma2 (length n)
        sigma2: Conditional variance sequence (positive; length n)

    Returns:
    -------
        Mean QLIKE over finite pairs.

    """
    e = np.asarray(e, dtype=float).ravel()
    s2 = np.asarray(sigma2, dtype=float).ravel()
    m = np.isfinite(e) & np.isfinite(s2) & (s2 > 0)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.log(s2[m]) + (e[m] ** 2) / s2[m]))


def mse_mae_variance(e: np.ndarray, sigma2: np.ndarray) -> dict[str, float]:
    """Compute MSE and MAE between realized e^2 and forecast sigma^2."""
    e = np.asarray(e, dtype=float).ravel()
    s2 = np.asarray(sigma2, dtype=float).ravel()
    m = np.isfinite(e) & np.isfinite(s2)
    if not np.any(m):
        return {"mse": float("nan"), "mae": float("nan")}
    y = e[m] ** 2
    f = s2[m]
    return {
        "mse": float(np.mean((y - f) ** 2)),
        "mae": float(np.mean(np.abs(y - f))),
    }


def mincer_zarnowitz(e: np.ndarray, sigma2: np.ndarray) -> dict[str, float]:
    """Run Mincer-Zarnowitz regression: e^2 = c + b * sigma^2 + u.

    Returns intercept c, slope b, R^2 and (optionally) p-values when SciPy is available.
    """
    e = np.asarray(e, dtype=float).ravel()
    s2 = np.asarray(sigma2, dtype=float).ravel()
    m = np.isfinite(e) & np.isfinite(s2)
    y = (e[m] ** 2).astype(float)
    x = s2[m].astype(float)
    min_obs = 2
    if y.size < min_obs:
        return {"intercept": float("nan"), "slope": float("nan"), "r2": float("nan")}

    x_mat = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(x_mat, y, rcond=None)
    y_hat = x_mat @ beta
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - y_hat) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    out: dict[str, float] = {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "r2": float(r2),
    }
    out.update(compute_mz_pvalues(beta, x_mat, ss_res))
    return out


def mz_calibration_params(e: np.ndarray, sigma2: np.ndarray) -> dict[str, float]:
    """Return intercept and slope (a,b) from MZ regression to calibrate variance.

    Why: If b>1 (like in your diagnostics), the raw sigma² underestimates
    realized variance. Typically use multiplicative calibration: h_adj = b * h
    (intercept usually not significant and can cause instability).
    """
    mz = mincer_zarnowitz(e, sigma2)
    return {
        "intercept": float(mz.get("intercept", np.nan)),
        "slope": float(mz.get("slope", np.nan)),
        "p_intercept": float(mz.get("p_intercept", np.nan)),
        "p_slope": float(mz.get("p_slope", np.nan)),
    }


def apply_mz_calibration(
    sigma2: np.ndarray,
    intercept: float,
    slope: float,
    *,
    eps: float = GARCH_CALIBRATION_EPS,
    use_intercept: bool = False,
) -> np.ndarray:
    """Apply MZ calibration: h_adj = max(eps, a + b * h) or h_adj = b * h.

    Args:
    ----
        sigma2: Variance array to calibrate.
        intercept: MZ regression intercept.
        slope: MZ regression slope.
        eps: Minimum variance threshold.
        use_intercept: If False, use multiplicative calibration only (slope * h).
                      If True, use full additive calibration (intercept + slope * h).

    Returns:
    -------
        Calibrated variance array.

    """
    s2 = np.asarray(sigma2, dtype=float)
    # Multiplicative calibration only (more stable) if use_intercept is False
    h_adj = intercept + slope * s2 if use_intercept else slope * s2
    return np.asarray(np.maximum(eps, h_adj), dtype=float)


def _build_var_series(
    sigma2: np.ndarray,
    alpha: float,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
) -> np.ndarray:
    """Return VaR_t series at level alpha with zero mean: VaR = q_alpha * sigma_t."""
    from src.garch.garch_eval.utils import build_var_series

    return build_var_series(sigma2, alpha, dist, nu, lambda_skew)


def empirical_quantiles(z: np.ndarray, alphas: list[float]) -> dict[float, float]:
    """Return empirical left-tail quantiles of standardized residuals.

    Args:
    ----
        z: Standardized residuals (train) ~ iid under well-specified model.
        alphas: Tail probabilities (e.g., [0.01, 0.05]).

    """
    zz = np.asarray(z, dtype=float)
    zz = zz[np.isfinite(zz)]
    out: dict[float, float] = {}
    for a in alphas:
        a_f = float(a)
        a_f = min(max(a_f, GARCH_EVAL_MIN_ALPHA), GARCH_EVAL_HALF)
        out[a] = float(np.quantile(zz, a_f)) if zz.size else float("nan")
    return out


def kupiec_pof_test(hits: np.ndarray, alpha: float) -> dict[str, float]:
    """Kupiec Proportion-of-Failures (POF) test.

    Args:
    ----
        hits: 1 if return < VaR, else 0.
        alpha: Target tail probability (e.g. 0.01).

    Returns:
    -------
        Dict with n, x, hit_rate, lr_uc, p_value.

    """
    h = np.asarray(hits, dtype=float).ravel()
    m = np.isfinite(h)
    h = h[m]
    n = int(h.size)
    x = int(np.sum(h > GARCH_EVAL_HALF))
    if n == 0:
        return {
            "n": 0.0,
            "x": 0.0,
            "hit_rate": float("nan"),
            "lr_uc": float("nan"),
            "p_value": float("nan"),
        }
    phat = x / max(1, n)

    # Likelihood ratio for unconditional coverage
    def _lnp(p: float) -> float:
        p = min(max(p, GARCH_EVAL_EPSILON), 1 - GARCH_EVAL_EPSILON)
        return (n - x) * np.log(1 - p) + x * np.log(p)

    lr_uc = -2.0 * (_lnp(alpha) - _lnp(phat))
    p_val = chi2_sf(float(lr_uc), df=1)
    return {
        "n": float(n),
        "x": float(x),
        "hit_rate": float(phat),
        "lr_uc": float(lr_uc),
        "p_value": float(p_val),
    }


def christoffersen_ind_test(hits: np.ndarray) -> dict[str, float]:
    """Christoffersen independence test (first-order Markov).

    Returns LR_ind and p-value (df=1) and the transition counts.
    """
    h = (np.asarray(hits, dtype=float).ravel() > GARCH_EVAL_HALF).astype(int)
    if h.size <= 1:
        return {
            "lr_ind": float("nan"),
            "p_value": float("nan"),
            "n00": 0.0,
            "n01": 0.0,
            "n10": 0.0,
            "n11": 0.0,
        }
    n00 = np.sum((h[1:] == 0) & (h[:-1] == 0))
    n01 = np.sum((h[1:] == 1) & (h[:-1] == 0))
    n10 = np.sum((h[1:] == 0) & (h[:-1] == 1))
    n11 = np.sum((h[1:] == 1) & (h[:-1] == 1))
    n0 = max(1, n00 + n01)
    n1 = max(1, n10 + n11)
    p01 = n01 / n0
    p11 = n11 / n1
    p = (n01 + n11) / max(1, n00 + n01 + n10 + n11)

    # Likelihood ratio statistic for independence
    def _ll(p0: float, p1: float) -> float:
        p0 = min(max(p0, GARCH_EVAL_EPSILON), 1 - GARCH_EVAL_EPSILON)
        p1 = min(max(p1, GARCH_EVAL_EPSILON), 1 - GARCH_EVAL_EPSILON)
        return n00 * np.log(1 - p0) + n01 * np.log(p0) + n10 * np.log(1 - p1) + n11 * np.log(p1)

    l1 = _ll(p01, p11)
    l0 = _ll(p, p)
    lr_ind = -2.0 * (l0 - l1)
    p_val = chi2_sf(float(lr_ind), df=1)
    return {
        "lr_ind": float(lr_ind),
        "p_value": float(p_val),
        "n00": float(n00),
        "n01": float(n01),
        "n10": float(n10),
        "n11": float(n11),
    }


def var_backtest_metrics(
    e: np.ndarray,
    sigma2: np.ndarray,
    *,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
    alphas: list[float],
) -> dict[str, dict[str, float]]:
    """Compute VaR backtests (Kupiec/Christoffersen) for multiple alphas."""
    e = np.asarray(e, dtype=float).ravel()
    s2 = np.asarray(sigma2, dtype=float).ravel()
    m = np.isfinite(e) & np.isfinite(s2) & (s2 > 0)
    out: dict[str, dict[str, float]] = {}
    for a in alphas:
        var_t = _build_var_series(s2[m], a, dist, nu, lambda_skew)
        hits = (e[m] < var_t).astype(int)
        kup = kupiec_pof_test(hits, a)
        ind = christoffersen_ind_test(hits)
        out[str(a)] = {
            "n": kup["n"],
            "violations": kup["x"],
            "hit_rate": kup["hit_rate"],
            "lr_uc": kup["lr_uc"],
            "p_uc": kup["p_value"],
            "lr_ind": ind["lr_ind"],
            "p_ind": ind["p_value"],
            "lr_cc": float(kup["lr_uc"] + ind["lr_ind"]),
            "p_cc": chi2_sf(float(kup["lr_uc"] + ind["lr_ind"]), df=2),
        }
    return out


def compute_classic_metrics_from_artifacts(
    *,
    params: dict[str, float],
    model_name: str,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
    alphas: list[float] | None = None,
    apply_mz_calibration: bool = False,
) -> dict[str, object]:
    """Compute classic GARCH metrics on the test split and return a summary dict.

    Args:
    ----
        params: Model parameters dictionary.
        model_name: Model name.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).
        alphas: VaR alpha levels (default: [0.01, 0.05]).
        apply_mz_calibration: Whether to apply MZ calibration to variances.

    Returns:
    -------
        Dictionary with all computed metrics.

    """
    if alphas is None:
        alphas = list(GARCH_EVAL_DEFAULT_ALPHAS)

    # Load dataset
    dataset_df = load_dataset_for_metrics()

    # Load test residuals and variance
    e_test, s2_test = load_test_resid_sigma2(
        params,
        dataset_df,
        model_name=model_name,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
    )

    # Filter test data
    e_test, s2_test = filter_test_data(e_test, s2_test)

    # Compute all metrics with optional MZ calibration
    return compute_all_metrics(
        e_test,
        s2_test,
        dist,
        nu,
        alphas,
        lambda_skew=lambda_skew,
        use_mz_calibration=apply_mz_calibration,
    )


def save_metrics_json(payload: dict[str, object]) -> None:
    """Persist metrics to JSON path in constants."""
    GARCH_EVAL_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with GARCH_EVAL_METRICS_FILE.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved GARCH evaluation metrics to: %s", GARCH_EVAL_METRICS_FILE)


__all__ = [
    "qlike_loss",
    "mse_mae_variance",
    "mincer_zarnowitz",
    "kupiec_pof_test",
    "christoffersen_ind_test",
    "var_backtest_metrics",
    "compute_classic_metrics_from_artifacts",
    "save_metrics_json",
]
