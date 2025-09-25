"""
Enhanced tests for diode.__init___lib module to improve coverage.
"""

import logging
import tempfile
from unittest.mock import patch

import pytest

# Enable debug flags for testing
try:
    from torch_diode.utils.debug_config import set_debug_flag

    set_debug_flag("ENABLE_TYPE_ASSERTS", True)
except ImportError:
    pass  # In case debug_config is not available yet

# Import the functions we want to test
from torch_diode.__init___lib import (
    get_diode_status,
    install_diode_integrations,
)


class TestInitLibEnhanced:
    """Enhanced test class for __init___lib module."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_install_diode_integrations_success(self):
        """Test install_diode_integrations with successful installation."""
        with patch(
            "torch_diode.__init___lib.discover_and_register_integrations"
        ) as mock_discover:
            with patch("torch_diode.__init___lib.integrate_all") as mock_integrate:
                mock_discover.return_value = {
                    "integration1": True,
                    "integration2": True,
                }
                mock_integrate.return_value = {
                    "integration1": True,
                    "integration2": False,
                }

                result = install_diode_integrations(enable_fallback=False)

                assert result == {"integration1": True, "integration2": False}
                mock_discover.assert_called_once()
                mock_integrate.assert_called_once()

    def test_install_diode_integrations_pytorch_not_available(self):
        """Test install_diode_integrations when PyTorch is not available."""
        with patch(
            "torch_diode.__init___lib.discover_and_register_integrations",
            side_effect=ImportError("No module named 'torch'"),
        ):
            result = install_diode_integrations(enable_fallback=False)

            assert result == {}

    def test_install_diode_integrations_exception_with_fallback(self):
        """Test install_diode_integrations exception handling with fallback enabled."""
        with patch(
            "torch_diode.__init___lib.discover_and_register_integrations",
            side_effect=Exception("General error"),
        ):
            result = install_diode_integrations(enable_fallback=True)

            assert result == {}

    def test_install_diode_integrations_exception_without_fallback(self):
        """Test install_diode_integrations exception handling with fallback disabled."""
        with patch(
            "torch_diode.__init___lib.discover_and_register_integrations",
            side_effect=Exception("General error"),
        ):
            with pytest.raises(Exception, match="General error"):
                install_diode_integrations(enable_fallback=False)

    def test_install_diode_integrations_torch_import_logging(self):
        """Test install_diode_integrations logs appropriately when torch is available."""
        with patch(
            "torch_diode.__init___lib.discover_and_register_integrations"
        ) as mock_discover:
            with patch("torch_diode.__init___lib.integrate_all") as mock_integrate:
                with patch("logging.Logger.info") as mock_log_info:
                    mock_discover.return_value = {"integration1": True}
                    mock_integrate.return_value = {"integration1": True}

                    install_diode_integrations()

                    # Check that appropriate log messages were called
                    assert any(
                        "PyTorch detected" in str(call)
                        for call in mock_log_info.call_args_list
                    )
                    assert any(
                        "Manual integration installation complete" in str(call)
                        for call in mock_log_info.call_args_list
                    )

    def test_install_diode_integrations_partial_success_logging(self):
        """Test install_diode_integrations logs correctly with partial success."""
        with patch(
            "torch_diode.__init___lib.discover_and_register_integrations"
        ) as mock_discover:
            with patch("torch_diode.__init___lib.integrate_all") as mock_integrate:
                with patch("logging.Logger.info") as mock_log_info:
                    mock_discover.return_value = {
                        "integration1": True,
                        "integration2": True,
                        "integration3": True,
                    }
                    mock_integrate.return_value = {
                        "integration1": True,
                        "integration2": False,
                        "integration3": True,
                    }

                    install_diode_integrations()

                    # Check that the log message shows correct counts
                    log_messages = [str(call) for call in mock_log_info.call_args_list]
                    success_message = next(
                        msg
                        for msg in log_messages
                        if "Manual integration installation complete" in msg
                    )
                    assert "2/3 integrations successful" in success_message

    def test_install_diode_integrations_import_error_logging(self):
        """Test install_diode_integrations logs import error correctly."""
        with patch(
            "torch_diode.__init___lib.discover_and_register_integrations",
            side_effect=ImportError("No module named 'torch'"),
        ):
            with patch("logging.Logger.error") as mock_log_error:
                result = install_diode_integrations()

                assert result == {}
                mock_log_error.assert_called_once()
                assert "PyTorch not available" in str(mock_log_error.call_args)

    def test_install_diode_integrations_general_error_logging(self):
        """Test install_diode_integrations logs general errors correctly with fallback disabled."""
        with patch(
            "torch_diode.__init___lib.discover_and_register_integrations",
            side_effect=RuntimeError("Connection failed"),
        ):
            with patch("logging.Logger.error") as mock_log_error:
                with pytest.raises(RuntimeError, match="Connection failed"):
                    install_diode_integrations(enable_fallback=False)

                mock_log_error.assert_called_once()
                error_message = str(mock_log_error.call_args)
                assert "Manual integration installation failed" in error_message
                assert "Connection failed" in error_message

    def test_get_diode_status_calls_integration_status(self):
        """Test get_diode_status calls get_integration_status correctly."""
        mock_status = {"integration1": {"status": "active", "info": "test"}}

        with patch(
            "torch_diode.__init___lib.get_integration_status", return_value=mock_status
        ) as mock_get_status:
            result = get_diode_status()

            assert result == mock_status
            mock_get_status.assert_called_once()

    def test_version_attribute_exists(self):
        """Test that __version__ attribute exists and is correct."""
        import torch_diode.__init___lib as init_lib

        assert hasattr(init_lib, "__version__")
        assert init_lib.__version__ == "0.1.0"

    def test_all_exports_exist(self):
        """Test that all exports in __all__ actually exist."""
        import torch_diode.__init___lib as init_lib

        for name in init_lib.__all__:
            assert hasattr(init_lib, name), f"Export {name} does not exist in module"

    def test_all_exports_are_callable_or_classes(self):
        """Test that all exports are either callable functions or classes."""
        import torch_diode.__init___lib as init_lib

        for name in init_lib.__all__:
            obj = getattr(init_lib, name)
            # Should be either callable (function) or a class
            assert callable(obj) or isinstance(
                obj, type
            ), f"Export {name} is neither callable nor a class"

    def test_logger_setup(self):
        """Test that logger is properly set up."""
        import torch_diode.__init___lib as init_lib

        assert hasattr(init_lib, "logger")
        assert isinstance(init_lib.logger, logging.Logger)
        assert init_lib.logger.name == "torch_diode.__init___lib"

    def test_import_paths_exist(self):
        """Test that all imported functions/classes actually exist."""
        # Test that the imports work
        from torch_diode.integration.base_integration import (
            BaseIntegration,
            IntegrationRegistry,
            ModelPointer,
            discover_and_register_integrations,
            get_integration_registry,
            get_integration_status,
            integrate_all,
        )

        # All imports should be callable or classes
        assert callable(discover_and_register_integrations)
        assert callable(integrate_all)
        assert callable(get_integration_status)
        assert callable(get_integration_registry)
        assert isinstance(IntegrationRegistry, type)
        assert isinstance(BaseIntegration, type)
        assert isinstance(ModelPointer, type)

    def test_install_diode_integrations_no_results(self):
        """Test install_diode_integrations when integration functions return empty results."""
        with patch(
            "torch_diode.__init___lib.discover_and_register_integrations"
        ) as mock_discover:
            with patch("torch_diode.__init___lib.integrate_all") as mock_integrate:
                mock_discover.return_value = {}
                mock_integrate.return_value = {}

                result = install_diode_integrations()

                assert result == {}

    def test_install_diode_integrations_all_failed(self):
        """Test install_diode_integrations when all integrations fail."""
        with patch(
            "torch_diode.__init___lib.discover_and_register_integrations"
        ) as mock_discover:
            with patch("torch_diode.__init___lib.integrate_all") as mock_integrate:
                with patch("logging.Logger.info") as mock_log_info:
                    mock_discover.return_value = {
                        "integration1": True,
                        "integration2": True,
                    }
                    mock_integrate.return_value = {
                        "integration1": False,
                        "integration2": False,
                    }

                    result = install_diode_integrations()

                    assert result == {"integration1": False, "integration2": False}

                    # Check that the log message shows correct counts (0 successful)
                    log_messages = [str(call) for call in mock_log_info.call_args_list]
                    success_message = next(
                        msg
                        for msg in log_messages
                        if "Manual integration installation complete" in msg
                    )
                    assert "0/2 integrations successful" in success_message

    def test_install_diode_integrations_single_integration_success(self):
        """Test install_diode_integrations with single successful integration."""
        with patch(
            "torch_diode.__init___lib.discover_and_register_integrations"
        ) as mock_discover:
            with patch("torch_diode.__init___lib.integrate_all") as mock_integrate:
                with patch("logging.Logger.info") as mock_log_info:
                    mock_discover.return_value = {"only_integration": True}
                    mock_integrate.return_value = {"only_integration": True}

                    result = install_diode_integrations()

                    assert result == {"only_integration": True}

                    # Check that the log message shows correct counts
                    log_messages = [str(call) for call in mock_log_info.call_args_list]
                    success_message = next(
                        msg
                        for msg in log_messages
                        if "Manual integration installation complete" in msg
                    )
                    assert "1/1 integrations successful" in success_message

    def test_install_diode_integrations_with_default_parameters(self):
        """Test install_diode_integrations with default parameter values."""
        with patch(
            "torch_diode.__init___lib.discover_and_register_integrations"
        ) as mock_discover:
            with patch("torch_diode.__init___lib.integrate_all") as mock_integrate:
                mock_discover.return_value = {"integration1": True}
                mock_integrate.return_value = {"integration1": True}

                # Call without specifying enable_fallback (should default to True)
                result = install_diode_integrations()

                assert result == {"integration1": True}

    def test_docstring_content(self):
        """Test that the module has appropriate docstring content."""
        import torch_diode.__init___lib as init_lib

        assert init_lib.__doc__ is not None
        assert "library version" in init_lib.__doc__.lower()
        assert "auto-registration" in init_lib.__doc__.lower()

    def test_install_diode_integrations_docstring(self):
        """Test that install_diode_integrations has proper docstring."""
        assert install_diode_integrations.__doc__ is not None
        doc = install_diode_integrations.__doc__
        assert "Manually install all Diode integrations" in doc
        assert "enable_fallback" in doc
        assert "Returns:" in doc

    def test_get_diode_status_docstring(self):
        """Test that get_diode_status has proper docstring."""
        assert get_diode_status.__doc__ is not None
        doc = get_diode_status.__doc__
        assert "comprehensive status information" in doc
        assert "Returns:" in doc
