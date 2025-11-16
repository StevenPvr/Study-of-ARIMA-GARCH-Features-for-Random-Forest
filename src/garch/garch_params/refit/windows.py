"""Window management for EGARCH refit.

This module provides window management for periodic refit:
- Expanding windows (use all data from start)
- Rolling windows (use fixed-size window)
- Window bounds computation
- Validation

All windows maintain temporal causality (no look-ahead).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Window:
    """Training window specification.

    Attributes:
        start: Window start index (inclusive).
        end: Window end index (exclusive).
        size: Window size (number of observations).
    """

    start: int
    end: int

    @property
    def size(self) -> int:
        """Compute window size."""
        return self.end - self.start

    def validate(self) -> None:
        """Validate window bounds.

        Raises:
            ValueError: If window is invalid.
        """
        if self.start < 0:
            msg = f"Window start ({self.start}) must be >= 0"
            raise ValueError(msg)
        if self.end <= self.start:
            msg = f"Window end ({self.end}) must be > start ({self.start})"
            raise ValueError(msg)

    def __str__(self) -> str:
        """String representation."""
        return f"Window[{self.start}:{self.end}] (size={self.size})"


class ExpandingWindow:
    """Expanding window manager.

    Expands from fixed start to increasing end positions.
    Use all data from start to current position.
    """

    def __init__(self, start: int = 0) -> None:
        """Initialize expanding window.

        Args:
            start: Fixed start position.
        """
        self.start = start

    def compute_window(self, current_pos: int) -> Window:
        """Compute expanding window up to current position.

        Args:
            current_pos: Current position (exclusive end).

        Returns:
            Window from start to current position.

        Raises:
            ValueError: If current_pos <= start.
        """
        if current_pos <= self.start:
            msg = f"current_pos ({current_pos}) must be > start ({self.start})"
            raise ValueError(msg)

        window = Window(start=self.start, end=current_pos)
        window.validate()
        return window

    def __str__(self) -> str:
        """String representation."""
        return f"ExpandingWindow(start={self.start})"


class RollingWindow:
    """Rolling window manager.

    Maintains fixed-size window that slides forward.
    More recent data only.
    """

    def __init__(self, window_size: int) -> None:
        """Initialize rolling window.

        Args:
            window_size: Fixed window size.

        Raises:
            ValueError: If window_size <= 0.
        """
        if window_size <= 0:
            msg = f"window_size ({window_size}) must be > 0"
            raise ValueError(msg)
        self.window_size = window_size

    def compute_window(self, current_pos: int) -> Window:
        """Compute rolling window ending at current position.

        Args:
            current_pos: Current position (exclusive end).

        Returns:
            Window of size window_size ending at current_pos.

        Raises:
            ValueError: If current_pos <= 0.
        """
        if current_pos <= 0:
            msg = f"current_pos ({current_pos}) must be > 0"
            raise ValueError(msg)

        start = max(0, current_pos - self.window_size)
        window = Window(start=start, end=current_pos)
        window.validate()
        return window

    def __str__(self) -> str:
        """String representation."""
        return f"RollingWindow(size={self.window_size})"


def create_window_manager(
    window_type: str, window_size: int | None = None, start: int = 0
) -> ExpandingWindow | RollingWindow:
    """Create window manager based on configuration.

    Args:
        window_type: Window type ('expanding' or 'rolling').
        window_size: Rolling window size (required if window_type='rolling').
        start: Start position for expanding window.

    Returns:
        Window manager (ExpandingWindow or RollingWindow).

    Raises:
        ValueError: If window_type is invalid or window_size missing for rolling.
    """
    if window_type == "expanding":
        return ExpandingWindow(start=start)
    if window_type == "rolling":
        if window_size is None:
            msg = "window_size required for rolling window"
            raise ValueError(msg)
        return RollingWindow(window_size=window_size)
    msg = f"Invalid window_type: {window_type}. Must be 'expanding' or 'rolling'."
    raise ValueError(msg)
