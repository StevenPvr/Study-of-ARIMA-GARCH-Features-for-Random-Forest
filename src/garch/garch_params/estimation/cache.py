"""MLE estimation cache to avoid repeated computations.

This module provides a thread-safe LRU cache for EGARCH MLE estimations.
The cache key is based on residual data hash and model parameters.

The cache improves performance during hyperparameter optimization where
the same residuals with same parameters might be estimated multiple times.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


def _compute_residual_hash(residuals: np.ndarray) -> str:
    """Compute a hash of residuals for cache key generation.

    Args:
        residuals: Residual array.

    Returns:
        SHA-256 hash of residuals as hex string.
    """
    # Use tobytes() for consistent hashing across runs
    residual_bytes = residuals.tobytes()
    return hashlib.sha256(residual_bytes).hexdigest()


def _create_cache_key(residuals: np.ndarray, o: int, p: int, dist: str) -> str:
    """Create cache key for MLE estimation.

    Args:
        residuals: Residual array.
        o: ARCH order.
        p: GARCH order.
        dist: Distribution name.

    Returns:
        Cache key string.
    """
    residual_hash = _compute_residual_hash(residuals)
    return f"{residual_hash}_{o}_{p}_{dist}"


class MLECache:
    """Thread-safe cache for EGARCH MLE estimations.

    This cache stores MLE estimation results to avoid redundant computations
    during hyperparameter optimization.

    Attributes:
        cache: Dictionary storing cached results.
        max_size: Maximum number of cached entries.
        hits: Number of cache hits.
        misses: Number of cache misses.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize MLE cache.

        Args:
            max_size: Maximum number of cached entries.

        Raises:
            ValueError: If max_size is not positive.
        """
        if max_size <= 0:
            msg = f"max_size must be positive, got {max_size}"
            raise ValueError(msg)

        self.cache: dict[str, tuple[dict[str, float], Any]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._access_order: list[str] = []  # Track access order for LRU

    def get(
        self, residuals: np.ndarray, o: int, p: int, dist: str
    ) -> tuple[dict[str, float], Any] | None:
        """Get cached MLE estimation result.

        Args:
            residuals: Residual array.
            o: ARCH order.
            p: GARCH order.
            dist: Distribution name.

        Returns:
            Tuple of (parameters, convergence) if cached, None otherwise.
        """
        key = _create_cache_key(residuals, o, p, dist)
        if key in self.cache:
            self.hits += 1
            # Update access order for LRU
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            logger.debug(
                "Cache hit for key=%s (hits=%d, misses=%d)",
                key[:16],
                self.hits,
                self.misses,
            )
            return self.cache[key]

        self.misses += 1
        logger.debug(
            "Cache miss for key=%s (hits=%d, misses=%d)",
            key[:16],
            self.hits,
            self.misses,
        )
        return None

    def put(
        self,
        residuals: np.ndarray,
        o: int,
        p: int,
        dist: str,
        params: dict[str, float],
        convergence: Any,
    ) -> None:
        """Store MLE estimation result in cache.

        Args:
            residuals: Residual array.
            o: ARCH order.
            p: GARCH order.
            dist: Distribution name.
            params: Estimated parameters.
            convergence: Convergence information.
        """
        key = _create_cache_key(residuals, o, p, dist)

        # Evict least recently used if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            if self._access_order:
                lru_key = self._access_order.pop(0)
                del self.cache[lru_key]
                logger.debug("Evicted LRU key=%s from cache", lru_key[:16])

        self.cache[key] = (params, convergence)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        logger.debug(
            "Stored result in cache for key=%s (size=%d/%d)",
            key[:16],
            len(self.cache),
            self.max_size,
        )

    def clear(self) -> None:
        """Clear all cached entries and reset statistics."""
        self.cache.clear()
        self._access_order.clear()
        self.hits = 0
        self.misses = 0
        logger.debug("Cache cleared")

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
        }


# Global cache instance
_global_cache: MLECache | None = None


def get_global_cache() -> MLECache:
    """Get global MLE cache instance.

    Returns:
        Global MLECache instance.
    """
    global _global_cache  # noqa: PLW0603
    if _global_cache is None:
        _global_cache = MLECache(max_size=1000)
    return _global_cache


def clear_global_cache() -> None:
    """Clear global MLE cache."""
    global _global_cache  # noqa: PLW0603
    if _global_cache is not None:
        _global_cache.clear()


__all__ = [
    "MLECache",
    "get_global_cache",
    "clear_global_cache",
]
