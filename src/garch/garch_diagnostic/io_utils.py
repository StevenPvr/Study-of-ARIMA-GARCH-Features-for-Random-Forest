"""I/O utilities for GARCH diagnostics.

Contains functions for saving diagnostic results to JSON files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils import get_logger, save_json_pretty

logger = get_logger(__name__)


def save_diagnostics_json(
    data: dict[str, Any],
    output_file: Path,
    description: str = "diagnostics",
) -> None:
    """Save diagnostics dictionary to JSON file.

    Delegates to src.utils.save_json_pretty() for consistency.

    Args:
        data: Dictionary to save as JSON.
        output_file: Path to output file.
        description: Description for logging (e.g., "distribution diagnostics").
    """
    save_json_pretty(data, output_file)
    logger.info("Saved %s to: %s", description, output_file)


def validate_dict_field(
    data: dict[str, Any],
    field_name: str,
) -> None:
    """Validate that a field in a dictionary is itself a dict.

    Args:
        data: Dictionary containing the field.
        field_name: Name of the field to validate.

    Raises:
        TypeError: If the field is not a dict.
        KeyError: If the field is missing.
    """
    if field_name not in data:
        msg = f"Missing required field: {field_name}"
        raise KeyError(msg)

    field_value = data[field_name]
    if not isinstance(field_value, dict):
        msg = f"{field_name} must be a dict, got {type(field_value).__name__}"
        raise TypeError(msg)
