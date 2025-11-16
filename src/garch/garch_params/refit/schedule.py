"""Refit scheduling for EGARCH models.

This module provides refit scheduling logic:
- Periodic refit (fixed frequency)
- Refit position computation
- Schedule validation

All schedules maintain temporal causality.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RefitSchedule:
    """Refit schedule specification.

    Attributes:
        frequency: Refit every N steps.
        start_pos: First refit position.
        end_pos: Last position (exclusive).
    """

    frequency: int
    start_pos: int
    end_pos: int

    def __post_init__(self) -> None:
        """Validate schedule parameters."""
        if self.frequency <= 0:
            msg = f"frequency ({self.frequency}) must be > 0"
            raise ValueError(msg)
        if self.start_pos < 0:
            msg = f"start_pos ({self.start_pos}) must be >= 0"
            raise ValueError(msg)
        if self.end_pos <= self.start_pos:
            msg = f"end_pos ({self.end_pos}) must be > start_pos ({self.start_pos})"
            raise ValueError(msg)

    def get_refit_positions(self) -> list[int]:
        """Get all refit positions.

        Returns:
            List of positions where refit should occur.
        """
        positions = list(range(self.start_pos, self.end_pos, self.frequency))
        return positions

    def should_refit(self, position: int) -> bool:
        """Check if refit should occur at position.

        Args:
            position: Current position.

        Returns:
            True if refit should occur.
        """
        if position < self.start_pos:
            return False
        if position >= self.end_pos:
            return False
        return (position - self.start_pos) % self.frequency == 0

    def count_refits(self) -> int:
        """Count number of refits in schedule.

        Returns:
            Number of refit events.
        """
        return len(self.get_refit_positions())

    def __str__(self) -> str:
        """String representation."""
        return (
            f"RefitSchedule(freq={self.frequency}, "
            f"start={self.start_pos}, end={self.end_pos}, "
            f"n_refits={self.count_refits()})"
        )


def create_periodic_schedule(
    frequency: int,
    start_pos: int,
    end_pos: int,
) -> RefitSchedule:
    """Create periodic refit schedule.

    Args:
        frequency: Refit every N steps.
        start_pos: First refit position.
        end_pos: Last position (exclusive).

    Returns:
        Refit schedule.

    Raises:
        ValueError: If parameters are invalid.
    """
    return RefitSchedule(frequency=frequency, start_pos=start_pos, end_pos=end_pos)


def compute_next_refit_pos(
    current_pos: int,
    frequency: int,
    start_pos: int,
) -> int:
    """Compute next refit position after current position.

    Args:
        current_pos: Current position.
        frequency: Refit frequency.
        start_pos: First refit position.

    Returns:
        Next refit position.

    Raises:
        ValueError: If parameters are invalid.
    """
    if frequency <= 0:
        msg = f"frequency ({frequency}) must be > 0"
        raise ValueError(msg)
    if current_pos < start_pos:
        return start_pos

    steps_since_start = current_pos - start_pos
    steps_since_last_refit = steps_since_start % frequency
    steps_to_next_refit = frequency - steps_since_last_refit

    return current_pos + steps_to_next_refit


def is_refit_position(
    position: int,
    frequency: int,
    start_pos: int,
) -> bool:
    """Check if position is a refit position.

    Args:
        position: Position to check.
        frequency: Refit frequency.
        start_pos: First refit position.

    Returns:
        True if position is a refit position.
    """
    if position < start_pos:
        return False
    return (position - start_pos) % frequency == 0
