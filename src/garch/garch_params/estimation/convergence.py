"""Convergence tracking for EGARCH MLE estimation.

This module tracks optimizer convergence rates and diagnostics:
- Convergence status tracking
- Iteration counts
- Final log-likelihood values
- Convergence rate computation
- Warnings for low convergence rates

Critical for academic papers: convergence rates must be reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ConvergenceResult:
    """Convergence result for a single estimation.

    Attributes:
        converged: Whether optimizer converged successfully.
        n_iterations: Number of optimizer iterations (if available).
        final_loglik: Final log-likelihood value.
        message: Optimizer message (if available).
    """

    converged: bool
    n_iterations: int | None
    final_loglik: float
    message: str | None = None

    def __str__(self) -> str:
        """String representation."""
        parts = [f"converged={self.converged}"]
        if self.n_iterations is not None:
            parts.append(f"n_iter={self.n_iterations}")
        parts.append(f"loglik={self.final_loglik:.2f}")
        if self.message:
            parts.append(f"msg='{self.message}'")
        return ", ".join(parts)


class ConvergenceTracker:
    """Track convergence across multiple estimations."""

    def __init__(self) -> None:
        """Initialize convergence tracker."""
        self.results: list[ConvergenceResult] = []

    def add_result(self, result: ConvergenceResult) -> None:
        """Add convergence result.

        Args:
            result: Convergence result to track.
        """
        self.results.append(result)

    def add_from_optimizer_result(self, opt_result: Any) -> None:
        """Add result from SciPy optimizer result.

        Args:
            opt_result: SciPy optimization result object.
        """
        converged = bool(opt_result.success) if hasattr(opt_result, "success") else False
        n_iterations = int(opt_result.nit) if hasattr(opt_result, "nit") else None
        final_loglik = float(-opt_result.fun) if hasattr(opt_result, "fun") else float("nan")
        message = str(opt_result.message) if hasattr(opt_result, "message") else None

        result = ConvergenceResult(
            converged=converged,
            n_iterations=n_iterations,
            final_loglik=final_loglik,
            message=message,
        )
        self.add_result(result)

    def compute_convergence_rate(self) -> float:
        """Compute convergence rate as percentage.

        Returns:
            Convergence rate in range [0, 1].
        """
        if not self.results:
            return 0.0
        n_converged = sum(1 for r in self.results if r.converged)
        return n_converged / len(self.results)

    def get_summary(self) -> dict[str, Any]:
        """Get convergence summary statistics.

        Returns:
            Dictionary with convergence statistics.
        """
        if not self.results:
            return {
                "n_estimations": 0,
                "n_converged": 0,
                "convergence_rate": 0.0,
                "mean_iterations": None,
                "mean_loglik": None,
            }

        n_converged = sum(1 for r in self.results if r.converged)
        convergence_rate = n_converged / len(self.results)

        iterations = [r.n_iterations for r in self.results if r.n_iterations is not None]
        mean_iterations = sum(iterations) / len(iterations) if iterations else None

        logliks = [r.final_loglik for r in self.results if r.converged]
        mean_loglik = sum(logliks) / len(logliks) if logliks else None

        return {
            "n_estimations": len(self.results),
            "n_converged": n_converged,
            "convergence_rate": convergence_rate,
            "mean_iterations": mean_iterations,
            "mean_loglik": mean_loglik,
        }

    def log_summary(self, min_rate: float = 0.95) -> None:
        """Log convergence summary with warnings if rate is low.

        Args:
            min_rate: Minimum acceptable convergence rate.
        """
        summary = self.get_summary()

        if summary["n_estimations"] == 0:
            logger.warning("No estimations tracked")
            return

        conv_rate = summary["convergence_rate"]
        logger.info(
            "Convergence summary: %d/%d succeeded (%.1f%%)",
            summary["n_converged"],
            summary["n_estimations"],
            conv_rate * 100,
        )

        if summary["mean_iterations"] is not None:
            logger.info("Mean iterations: %.1f", summary["mean_iterations"])

        if conv_rate < min_rate:
            logger.warning(
                "Low convergence rate: %.1f%% < %.1f%% threshold",
                conv_rate * 100,
                min_rate * 100,
            )
            logger.warning("Consider:")
            logger.warning("  - Adjusting initial values")
            logger.warning("  - Increasing maxiter")
            logger.warning("  - Checking data quality")

    def get_failed_results(self) -> list[ConvergenceResult]:
        """Get all failed convergence results.

        Returns:
            List of failed results.
        """
        return [r for r in self.results if not r.converged]

    def clear(self) -> None:
        """Clear all tracked results."""
        self.results.clear()


def extract_convergence_info(opt_result: Any) -> ConvergenceResult:
    """Extract convergence information from optimizer result.

    Args:
        opt_result: SciPy optimization result object.

    Returns:
        Convergence result.
    """
    converged = bool(opt_result.success) if hasattr(opt_result, "success") else False
    n_iterations = int(opt_result.nit) if hasattr(opt_result, "nit") else None
    final_loglik = float(-opt_result.fun) if hasattr(opt_result, "fun") else float("nan")
    message = str(opt_result.message) if hasattr(opt_result, "message") else None

    return ConvergenceResult(
        converged=converged,
        n_iterations=n_iterations,
        final_loglik=final_loglik,
        message=message,
    )
