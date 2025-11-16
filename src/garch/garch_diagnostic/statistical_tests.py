"""Statistical test utilities for GARCH diagnostics.

Contains functions for computing Ljung-Box test statistics.
"""

from __future__ import annotations

import numpy as np

from src.garch.garch_diagnostic.autocorrelation import autocorr


def compute_ljung_box_statistics(
    series: np.ndarray,
    lags: int,
) -> dict[str, list[int] | list[float]]:
    """Compute Ljung-Box test statistics for a given series."""
    lags_list = list(range(1, int(lags) + 1))
    r = autocorr(series, max(lags_list))
    n = float(np.sum(np.isfinite(series)))
    q_stats = []
    p_values = []
    try:
        from scipy.stats import chi2  # type: ignore

        has_scipy = True
    except ImportError:  # pragma: no cover - optional
        has_scipy = False
    s = 0.0
    for h in lags_list:
        rk = r[h]
        s += (rk * rk) / max(1.0, (n - h))
        q = n * (n + 2.0) * s
        q_stats.append(float(q))
        if has_scipy:
            # Survival function is numerically stable for upper tail
            p = float(chi2.sf(q, df=h))  # type: ignore[attr-defined]
        else:
            p = float("nan")
        p_values.append(p)
    return {"lags": lags_list, "lb_stat": q_stats, "lb_pvalue": p_values}
