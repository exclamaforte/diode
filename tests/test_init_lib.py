"""
Tests for diode.__init___lib module.
"""

import pytest

# Enable debug flags for testing
try:
    from torch_diode.utils.debug_config import set_debug_flag

    set_debug_flag("ENABLE_TYPE_ASSERTS", True)
except ImportError:
    pass  # In case debug_config is not available yet
import logging
from unittest.mock import Mock, patch

import torch_diode.__init___lib as init_lib


class TestInstallDiodeIntegrations:
    """Test the install_diode_integrations function."""

    @patch("torch_diode.__init___lib.discover_and_register_integrations")
    @patch("torch_diode.__init___lib.integrate_all")
    def test_successful_installation(self, mock_integrate_all, mock_discover):
        """Test successful integration installation."""
        # Mock the functions
        mock_discover.return_value = {"integration1": True, "integration2": True}
        mock_integrate_all.return_value = {"integration1": True, "integration2": True}

        # Mock torch import
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: Mock()
            if name == "torch"
            else __import__(name, *args),
        ):
            # Call the function
            result = init_lib.install_diode_integrations()

            # Verify results
            assert result == {"integration1": True, "integration2": True}
            mock_discover.assert_called_once()
            mock_integrate_all.assert_called_once()

    @patch("torch_diode.__init___lib.discover_and_register_integrations")
    @patch("torch_diode.__init___lib.integrate_all")
    def test_installation_with_fallback_disabled(
        self, mock_integrate_all, mock_discover
    ):
        """Test installation with fallback disabled."""
        mock_discover.side_effect = Exception("Discovery failed")

        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: Mock()
            if name == "torch"
            else __import__(name, *args),
        ):
            with pytest.raises(Exception, match="Discovery failed"):
                init_lib.install_diode_integrations(enable_fallback=False)

    @patch("torch_diode.__init___lib.discover_and_register_integrations")
    @patch("torch_diode.__init___lib.integrate_all")
    def test_installation_with_fallback_enabled(
        self, mock_integrate_all, mock_discover
    ):
        """Test installation with fallback enabled (default)."""
        mock_discover.side_effect = Exception("Discovery failed")

        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: Mock()
            if name == "torch"
            else __import__(name, *args),
        ):
            result = init_lib.install_diode_integrations()
            assert result == {}

    def test_installation_without_pytorch(self):
        """Test installation when PyTorch is not available."""
        # Mock torch import to raise ImportError
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: ImportError()
            if name == "torch"
            else __import__(name, *args),
        ):
            result = init_lib.install_diode_integrations()
            assert result == {}

    @patch("torch_diode.__init___lib.discover_and_register_integrations")
    @patch("torch_diode.__init___lib.integrate_all")
    def test_partial_success(self, mock_integrate_all, mock_discover):
        """Test with partial integration success."""
        mock_discover.return_value = {"int1": True, "int2": True, "int3": True}
        mock_integrate_all.return_value = {"int1": True, "int2": False, "int3": True}

        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: Mock()
            if name == "torch"
            else __import__(name, *args),
        ):
            result = init_lib.install_diode_integrations()

            assert result == {"int1": True, "int2": False, "int3": True}


class TestGetDiodeStatus:
    """Test the get_diode_status function."""

    @patch("torch_diode.__init___lib.get_integration_status")
    def test_get_status(self, mock_get_status):
        """Test getting diode status."""
        expected_status = {
            "integrations": {"int1": True, "int2": False},
            "registry_size": 2,
            "active_integrations": 1,
        }
        mock_get_status.return_value = expected_status

        result = init_lib.get_diode_status()

        assert result == expected_status
        mock_get_status.assert_called_once()

    @patch("torch_diode.__init___lib.get_integration_status")
    def test_get_status_empty(self, mock_get_status):
        """Test getting empty status."""
        mock_get_status.return_value = {}

        result = init_lib.get_diode_status()

        assert result == {}


class TestModuleExports:
    """Test that the module exports the expected functions and classes."""

    def test_all_exports_exist(self):
        """Test that all items in __all__ are available."""
        for item in init_lib.__all__:
            assert hasattr(init_lib, item), f"Missing export: {item}"

    def test_version_attribute(self):
        """Test that version is defined."""
        assert hasattr(init_lib, "__version__")
        assert init_lib.__version__ == "0.1.0"

    def test_logger_defined(self):
        """Test that logger is defined."""
        assert hasattr(init_lib, "logger")
        assert isinstance(init_lib.logger, logging.Logger)
