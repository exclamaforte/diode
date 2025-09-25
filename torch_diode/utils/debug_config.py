"""
Debug configuration for torch_diode.

This module provides debug flags and utilities for development and testing.
"""


# Debug flags
# ENABLE_TYPE_ASSERTS = os.environ.get("TORCH_DIODE_ENABLE_TYPE_ASSERTS", "false").lower() in ("true", "1", "yes")
ENABLE_TYPE_ASSERTS = True


def type_assert(condition: bool, message: str = "Type assertion failed") -> None:
    """Assert a type condition if type assertions are enabled.

    Args:
        condition: Boolean condition to check
        message: Error message if assertion fails

    Raises:
        AssertionError: If condition is False and type assertions are enabled
    """
    if ENABLE_TYPE_ASSERTS and not condition:
        raise AssertionError(message)


def get_debug_flags() -> dict:
    """Get current debug flag settings.

    Returns:
        Dictionary of debug flags and their values
    """
    return {
        "ENABLE_TYPE_ASSERTS": ENABLE_TYPE_ASSERTS,
    }


def set_debug_flag(flag_name: str, value: bool) -> None:
    """Set a debug flag programmatically.

    Args:
        flag_name: Name of the debug flag
        value: Value to set
    """
    global ENABLE_TYPE_ASSERTS

    if flag_name == "ENABLE_TYPE_ASSERTS":
        ENABLE_TYPE_ASSERTS = value
    else:
        raise ValueError(f"Unknown debug flag: {flag_name}")
