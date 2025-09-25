"""
Global pytest configuration for all Diode tests.

This file automatically enables debug features for all test runs.
"""

import os

import pytest

# Ensure type asserts are enabled for all tests
os.environ["TORCH_DIODE_ENABLE_TYPE_ASSERTS"] = "true"

# Also enable it programmatically in case the environment variable is not read
# early enough
from torch_diode.utils.debug_config import set_debug_flag

set_debug_flag("ENABLE_TYPE_ASSERTS", True)


@pytest.fixture(autouse=True)
def enable_debug_flags():
    """Fixture that automatically enables debug flags for all tests."""
    # Set the debug flag at the start of each test
    set_debug_flag("ENABLE_TYPE_ASSERTS", True)
    yield
    # Optional: reset after each test (commented out to keep consistent across
    # tests)
    # set_debug_flag("ENABLE_TYPE_ASSERTS", False)


def pytest_configure(config):
    """Called after command line options have been parsed."""
    # Ensure debug flags are enabled during test collection
    os.environ["TORCH_DIODE_ENABLE_TYPE_ASSERTS"] = "true"
    set_debug_flag("ENABLE_TYPE_ASSERTS", True)


def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    print("Debug flags enabled for all tests:")
    from torch_diode.utils.debug_config import get_debug_flags

    flags = get_debug_flags()
    for flag, value in flags.items():
        print(f"  {flag}: {value}")
