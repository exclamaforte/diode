"""
Comprehensive tests for functions in diode.__init__ module.
"""

import logging
import sys
import warnings
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestAttemptTorchImport:
    """Test the _attempt_torch_import function."""

    @patch("diode._import_status", {"torch_available": False, "errors": []})
    def test_successful_torch_import(self):
        """Test successful torch import."""
        # Import the module to test
        import diode

        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args: (
                Mock(__version__="2.1.0")
                if name == "torch"
                else __import__(name, *args)
            ),
        ):
            result = diode._attempt_torch_import()

            assert result is True
            assert diode._import_status["torch_available"] is True

    @patch("diode._import_status", {"torch_available": False, "errors": []})
    def test_failed_torch_import(self):
        """Test failed torch import."""
        import diode

        def mock_import(name, *args):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return __import__(name, *args)

        with patch("builtins.__import__", side_effect=mock_import):
            result = diode._attempt_torch_import()

            assert result is False
            assert diode._import_status["torch_available"] is False
            assert len(diode._import_status["errors"]) > 0

    def test_torch_version_logging(self):
        """Test that torch version is logged when available."""
        import diode

        mock_torch = Mock()
        mock_torch.__version__ = "2.1.0"

        with patch("builtins.__import__", return_value=mock_torch) as mock_import:
            with patch.object(diode.logger, "info") as mock_log:
                result = diode._attempt_torch_import()

                assert result is True
                mock_log.assert_called_with("Successfully imported torch 2.1.0")


class TestSetupIntegrations:
    """Test the _setup_integrations function."""

    @patch(
        "diode._import_status",
        {"integrations_attempted": False, "integrations_successful": {}, "errors": []},
    )
    def test_successful_integration_setup(self):
        """Test successful integration setup."""
        import diode

        # Mock the integration functions
        mock_integrate_all = Mock(
            return_value={"integration1": True, "integration2": True}
        )

        with patch("diode.integration.integrate_all", mock_integrate_all):
            result = diode._setup_integrations()

            assert result == {"integration1": True, "integration2": True}
            assert diode._import_status["integrations_attempted"] is True
            assert diode._import_status["integrations_successful"] == {
                "integration1": True,
                "integration2": True,
            }

    @patch(
        "diode._import_status",
        {"integrations_attempted": False, "integrations_successful": {}, "errors": []},
    )
    def test_partial_integration_success(self):
        """Test partial integration success."""
        import diode

        mock_integrate_all = Mock(
            return_value={
                "integration1": True,
                "integration2": False,
                "integration3": True,
            }
        )

        with patch("diode.integration.integrate_all", mock_integrate_all):
            with patch.object(diode.logger, "info") as mock_info, patch.object(
                diode.logger, "warning"
            ) as mock_warning:

                result = diode._setup_integrations()

                assert result == {
                    "integration1": True,
                    "integration2": False,
                    "integration3": True,
                }
                mock_info.assert_any_call("Successfully integrated 2/3 model types")
                mock_info.assert_any_call("  ✓ integration1")
                mock_info.assert_any_call("  ✓ integration3")
                mock_warning.assert_any_call("  ✗ integration2")

    @patch(
        "diode._import_status",
        {"integrations_attempted": False, "integrations_successful": {}, "errors": []},
    )
    def test_failed_integration_setup(self):
        """Test failed integration setup."""
        import diode

        with patch(
            "diode.integration.integrate_all",
            side_effect=Exception("Integration failed"),
        ):
            with patch.object(diode.logger, "error") as mock_error:
                result = diode._setup_integrations()

                assert result == {}
                assert diode._import_status["integrations_attempted"] is True
                assert (
                    "Failed to setup integrations: Integration failed"
                    in diode._import_status["errors"]
                )
                mock_error.assert_called_once()

    @patch(
        "diode._import_status",
        {"integrations_attempted": False, "integrations_successful": {}, "errors": []},
    )
    def test_no_successful_integrations(self):
        """Test when no integrations are successful."""
        import diode

        mock_integrate_all = Mock(
            return_value={"integration1": False, "integration2": False}
        )

        with patch("diode.integration.integrate_all", mock_integrate_all):
            with patch.object(diode.logger, "warning") as mock_warning:
                result = diode._setup_integrations()

                assert result == {"integration1": False, "integration2": False}
                mock_warning.assert_any_call("No model integrations were successful")


class TestGetImportStatus:
    """Test the get_import_status function."""

    def test_get_import_status_returns_copy(self):
        """Test that get_import_status returns a copy of the status."""
        import diode

        # Get the status
        status1 = diode.get_import_status()
        status2 = diode.get_import_status()

        # Verify they are separate objects
        assert status1 is not status2
        assert status1 is not diode._import_status

        # But have the same content
        assert status1 == status2

    def test_get_import_status_structure(self):
        """Test that get_import_status returns expected structure."""
        import diode

        status = diode.get_import_status()

        # Check required keys
        required_keys = [
            "torch_available",
            "integrations_attempted",
            "integrations_successful",
            "errors",
        ]
        for key in required_keys:
            assert key in status

    def test_get_import_status_modification_safety(self):
        """Test that modifying returned status doesn't affect internal state."""
        import diode

        status = diode.get_import_status()
        original_torch_status = status["torch_available"]
        original_errors_count = len(status["errors"])

        # Modify the returned status
        status["torch_available"] = not original_torch_status
        # Note: shallow copy means nested objects like lists are shared
        # so we test that top-level modifications don't affect the original

        # Get fresh status and verify top-level changes are not reflected
        fresh_status = diode.get_import_status()
        assert fresh_status["torch_available"] == original_torch_status

        # For nested objects like lists, modifications would be shared (shallow copy behavior)
        # This is the expected behavior for copy() vs deepcopy()


class TestGetIntegrationInfo:
    """Test the get_integration_info function."""

    @patch("diode._import_status", {"integrations_attempted": False})
    def test_get_integration_info_not_attempted(self):
        """Test get_integration_info when integrations not attempted."""
        import diode

        result = diode.get_integration_info()
        assert result is None

    @patch("diode._import_status", {"integrations_attempted": True})
    def test_get_integration_info_successful(self):
        """Test successful get_integration_info call."""
        import diode

        expected_info = {
            "integration1": {"status": "active"},
            "integration2": {"status": "inactive"},
        }

        with patch(
            "diode.integration.get_integration_status", return_value=expected_info
        ):
            result = diode.get_integration_info()
            assert result == expected_info

    @patch("diode._import_status", {"integrations_attempted": True})
    def test_get_integration_info_import_error(self):
        """Test get_integration_info when integration module can't be imported."""
        import diode

        with patch("builtins.__import__", side_effect=ImportError("Module not found")):
            result = diode.get_integration_info()
            assert result is None


class TestGetModelInfo:
    """Test the get_model_info function."""

    def test_get_model_info_successful(self):
        """Test successful get_model_info call."""
        import diode

        # Mock the model registry
        mock_registry = Mock()
        mock_registry.get_existing_models.return_value = ["model1", "model2", "model3"]
        mock_registry.generate_manifest.return_value = {
            "models_by_purpose": {
                "classification": [
                    Mock(model_name="model1"),
                    Mock(model_name="model2"),
                ],
                "regression": [Mock(model_name="model3")],
            }
        }

        with patch(
            "diode.model_registry.get_model_registry", return_value=mock_registry
        ):
            result = diode.get_model_info()

            expected = {
                "available_models": 3,
                "model_manifest": mock_registry.generate_manifest.return_value,
                "models_by_purpose": {
                    "classification": ["model1", "model2"],
                    "regression": ["model3"],
                },
            }

            assert result == expected

    def test_get_model_info_import_error(self):
        """Test get_model_info when model_registry can't be imported."""
        import diode

        with patch("builtins.__import__", side_effect=ImportError("Module not found")):
            result = diode.get_model_info()
            assert result is None


class TestDisplayInitSummary:
    """Test the _display_init_summary function."""

    def test_display_summary_no_torch(self):
        """Test display summary when torch is not available."""
        import diode

        with patch("diode._torch_available", False):
            with patch("warnings.warn") as mock_warn:
                diode._display_init_summary()

                mock_warn.assert_called_once()
                args = mock_warn.call_args[0]
                assert "PyTorch is not available" in args[0]
                assert args[1] == UserWarning

    def test_display_summary_with_successful_integrations(self):
        """Test display summary with successful integrations."""
        import diode

        with patch("diode._torch_available", True), patch(
            "diode._import_status",
            {"integrations_successful": {"int1": True, "int2": True, "int3": False}},
        ):

            with patch.object(diode.logger, "info") as mock_info:
                diode._display_init_summary()

                mock_info.assert_called_with(
                    "torch-diode initialized with 2 active model integrations"
                )

    def test_display_summary_no_successful_integrations(self):
        """Test display summary with no successful integrations."""
        import diode

        with patch("diode._torch_available", True), patch(
            "diode._import_status",
            {"integrations_successful": {"int1": False, "int2": False}},
        ):

            with patch.object(diode.logger, "info") as mock_info:
                diode._display_init_summary()

                mock_info.assert_called_with(
                    "torch-diode initialized in library-only mode (no model integrations active)"
                )


class TestModuleLevel:
    """Test module-level functionality."""

    def test_version_attribute(self):
        """Test that __version__ is defined correctly."""
        import diode

        assert hasattr(diode, "__version__")
        assert diode.__version__ == "0.1.0"

    def test_all_exports(self):
        """Test that all expected exports are available."""
        import diode

        expected_exports = [
            "__version__",
            "collection",
            "integration",
            "model",
            "types",
            "utils",
            "get_import_status",
            "get_integration_info",
            "get_model_info",
            "get_model_registry",
        ]

        for export in expected_exports:
            assert hasattr(diode, export), f"Missing export: {export}"

    def test_logger_configuration(self):
        """Test that logger is properly configured."""
        import diode

        assert hasattr(diode, "logger")
        assert isinstance(diode.logger, logging.Logger)
        assert diode.logger.name == "diode"

    def test_import_status_initialization(self):
        """Test that _import_status is properly initialized."""
        import diode

        status = diode._import_status
        assert isinstance(status, dict)

        required_keys = [
            "torch_available",
            "integrations_attempted",
            "integrations_successful",
            "errors",
        ]
        for key in required_keys:
            assert key in status


class TestIntegrationPattern:
    """Test the overall integration pattern described in the module docstring."""

    def test_integration_side_effects_pattern(self):
        """Test that the integration follows the documented side-effects pattern."""
        # This test verifies the overall pattern:
        # 1. Attempt torch import
        # 2. If successful, attempt integrations
        # 3. Track results in _import_status

        import diode

        # Verify that side effects have occurred
        status = diode.get_import_status()

        # Step 1 should have been attempted
        assert "torch_available" in status
        assert isinstance(status["torch_available"], bool)

        # If torch was available, integrations should have been attempted
        if status["torch_available"]:
            assert status["integrations_attempted"] is True
            assert "integrations_successful" in status
            assert isinstance(status["integrations_successful"], dict)

    def test_graceful_degradation(self):
        """Test that the module degrades gracefully when torch is not available."""
        # This test ensures the module still works even without torch
        import diode

        # These should always be available regardless of torch
        assert hasattr(diode, "__version__")
        assert hasattr(diode, "get_import_status")
        assert hasattr(diode, "get_integration_info")
        assert hasattr(diode, "get_model_info")

        # get_import_status should always work
        status = diode.get_import_status()
        assert isinstance(status, dict)


# Test fixtures and utilities


@pytest.fixture
def clean_import_status():
    """Fixture to provide a clean import status for testing."""
    return {
        "torch_available": False,
        "integrations_attempted": False,
        "integrations_successful": {},
        "errors": [],
    }


@pytest.fixture
def mock_torch_available():
    """Fixture to mock torch as available."""
    mock_torch = Mock()
    mock_torch.__version__ = "2.1.0"

    with patch(
        "builtins.__import__",
        side_effect=lambda name, *args: (
            mock_torch if name == "torch" else __import__(name, *args)
        ),
    ):
        yield mock_torch


@pytest.fixture
def mock_integration_success():
    """Fixture to mock successful integrations."""
    with patch(
        "diode.integration.integrate_all",
        return_value={"integration1": True, "integration2": True},
    ):
        yield


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__])
