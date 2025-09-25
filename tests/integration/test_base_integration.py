"""
Tests for base integration system.

This module contains tests for the BaseIntegration class and IntegrationRegistry
that provide the framework for integrating trained models with PyTorch interfaces.
"""

import os

# Enable debug flags for testing
try:
    from torch_diode.utils.debug_config import set_debug_flag

    set_debug_flag("ENABLE_TYPE_ASSERTS", True)
except ImportError:
    pass  # In case debug_config is not available yet
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from torch_diode.integration.base_integration import (
    BaseIntegration,
    IntegrationRegistry,
    ModelPointer,
    discover_and_register_integrations,
    get_integration_registry,
    get_integration_status,
    integrate_all,
    register_integration,
)


class TestModelPointer:
    """Test the ModelPointer class."""

    def test_init_basic(self):
        """Test basic initialization of ModelPointer."""
        pointer = ModelPointer(
            model_name="test_model.pt",
            relative_path="matmul",
            model_purpose="test_purpose",
            interface_name="test_interface",
        )

        assert pointer.model_name == "test_model.pt"
        assert pointer.relative_path == "matmul"
        assert pointer.model_purpose == "test_purpose"
        assert pointer.interface_name == "test_interface"
        assert pointer.description == "Model for test_purpose"
        assert pointer.version == "1.0"
        assert pointer.dependencies == []

    def test_init_with_optional_params(self):
        """Test initialization with all optional parameters."""
        pointer = ModelPointer(
            model_name="advanced_model.pt",
            relative_path="conv",
            model_purpose="conv_prediction",
            interface_name="conv_interface",
            description="Advanced convolution model",
            version="2.1",
            dependencies=["torch", "numpy"],
        )

        assert pointer.description == "Advanced convolution model"
        assert pointer.version == "2.1"
        assert pointer.dependencies == ["torch", "numpy"]

    @patch.object(Path, "exists")
    def test_exists(self, mock_exists):
        """Test the exists method."""
        mock_exists.return_value = True

        pointer = ModelPointer(
            model_name="existing_model.pt",
            relative_path=".",
            model_purpose="test",
            interface_name="test",
        )

        assert pointer.exists() == True
        mock_exists.assert_called_once()

    @patch.object(Path, "stat")
    @patch.object(Path, "exists")
    def test_get_size_mb(self, mock_exists, mock_stat):
        """Test the get_size_mb method."""
        mock_exists.return_value = True
        mock_stat_result = Mock()
        mock_stat_result.st_size = 1024 * 1024 * 5  # 5 MB
        mock_stat.return_value = mock_stat_result

        pointer = ModelPointer(
            model_name="large_model.pt",
            relative_path=".",
            model_purpose="test",
            interface_name="test",
        )

        assert pointer.get_size_mb() == 5.0

    @patch.object(Path, "exists")
    def test_get_size_mb_nonexistent(self, mock_exists):
        """Test get_size_mb for non-existent file."""
        mock_exists.return_value = False

        pointer = ModelPointer(
            model_name="missing_model.pt",
            relative_path=".",
            model_purpose="test",
            interface_name="test",
        )

        assert pointer.get_size_mb() == 0.0

    def test_full_path_root_directory(self):
        """Test full_path for root directory relative path."""
        pointer = ModelPointer(
            model_name="root_model.pt",
            relative_path=".",
            model_purpose="test",
            interface_name="test",
        )

        # Should contain trained_models at the end
        full_path = pointer.full_path
        assert str(full_path).endswith("trained_models/root_model.pt")

    def test_full_path_subdirectory(self):
        """Test full_path for subdirectory relative path."""
        pointer = ModelPointer(
            model_name="sub_model.pt",
            relative_path="subdir",
            model_purpose="test",
            interface_name="test",
        )

        full_path = pointer.full_path
        assert str(full_path).endswith("trained_models/subdir/sub_model.pt")

    def test_repr(self):
        """Test string representation."""
        with patch.object(ModelPointer, "exists", return_value=True):
            pointer = ModelPointer(
                model_name="repr_model.pt",
                relative_path=".",
                model_purpose="test_purpose",
                interface_name="test",
            )

            repr_str = repr(pointer)
            assert "ModelPointer" in repr_str
            assert "repr_model.pt" in repr_str
            assert "test_purpose" in repr_str
            assert "exists=True" in repr_str


class ConcreteIntegration(BaseIntegration):
    """Concrete implementation of BaseIntegration for testing."""

    def create_dummy_function(self):
        """Create a dummy function for testing."""
        return Mock()

    def load_model(self, model_pointer):
        """Load a model (mock implementation)."""
        return Mock()

    def register_model(self, model, model_pointer):
        """Register a model (mock implementation)."""
        return True

    def enable_configs(self):
        """Enable configs (mock implementation)."""
        return True


class TestBaseIntegration:
    """Test the BaseIntegration class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model_pointers = [
            ModelPointer(
                model_name="test_model1.pt",
                relative_path=".",
                model_purpose="test1",
                interface_name="test_interface",
            ),
            ModelPointer(
                model_name="test_model2.pt",
                relative_path="subdir",
                model_purpose="test2",
                interface_name="test_interface",
                dependencies=["torch"],
            ),
        ]

    def test_init(self):
        """Test initialization of BaseIntegration."""
        integration = ConcreteIntegration(
            name="test_integration",
            interface_name="test_interface",
            model_pointers=self.model_pointers,
            enable_fallback=False,
        )

        assert integration.name == "test_integration"
        assert integration.interface_name == "test_interface"
        assert integration.model_pointers == self.model_pointers
        assert integration.enable_fallback == False
        assert integration.loaded_models == {}
        assert integration.registration_status == {}
        assert integration.integration_status == "not_started"

    def test_check_dependencies(self):
        """Test dependency checking."""
        integration = ConcreteIntegration(
            name="test_integration",
            interface_name="test_interface",
            model_pointers=self.model_pointers,
        )

        with patch("builtins.__import__") as mock_import:

            def side_effect(module_name):
                if module_name == "torch":
                    return Mock()
                raise ImportError(f"No module named '{module_name}'")

            mock_import.side_effect = side_effect

            deps = integration.check_dependencies()
            assert deps["torch"] == True

    @patch.object(ModelPointer, "exists")
    def test_get_available_models(self, mock_exists):
        """Test getting available models."""
        # First model exists, second doesn't
        mock_exists.side_effect = [True, False]

        integration = ConcreteIntegration(
            name="test_integration",
            interface_name="test_interface",
            model_pointers=self.model_pointers,
        )

        available = integration.get_available_models()
        assert len(available) == 1
        assert available[0].model_name == "test_model1.pt"

    @patch("torch._inductor.choices.InductorChoices")
    @patch("torch._inductor.virtualized.V")
    def test_register_dummy_success(self, mock_v, mock_inductor_choices):
        """Test successful dummy registration."""
        integration = ConcreteIntegration(
            name="test_integration",
            interface_name="test_interface",
            model_pointers=self.model_pointers,
        )

        # Mock V object to simulate successful registration
        mock_dummy = Mock()

        # Mock the choices attribute and set_choices_handler method
        mock_v.choices = None  # Initially no handler

        def mock_set_handler(handler):
            mock_v.choices = handler

        mock_v.set_choices_handler = mock_set_handler

        result = integration.register_dummy(mock_dummy)

        assert result == True

    @patch(
        "torch._inductor.virtualized.V",
        side_effect=ImportError("No module named 'torch._inductor'"),
    )
    def test_register_dummy_import_error(self, mock_v):
        """Test dummy registration with import error."""
        integration = ConcreteIntegration(
            name="test_integration",
            interface_name="test_interface",
            model_pointers=self.model_pointers,
        )

        result = integration.register_dummy(Mock())
        assert result == False

    @patch("torch._inductor.choices.InductorChoices")
    @patch("torch._inductor.virtualized.V")
    def test_register_dummy_already_registered(self, mock_v, mock_inductor_choices):
        """Test dummy registration when another handler is already registered."""
        # Mock an existing handler that's not InductorChoices
        mock_existing_handler = Mock()
        mock_existing_handler.__class__.__name__ = "DiodeInductorChoices"
        type(mock_existing_handler).__name__ = "DiodeInductorChoices"
        mock_v._choices_handler = mock_existing_handler

        integration = ConcreteIntegration(
            name="test_integration",
            interface_name="test_interface",
            model_pointers=self.model_pointers,
        )

        result = integration.register_dummy(Mock())
        assert result == False

    @patch.object(ConcreteIntegration, "enable_configs")
    @patch.object(ConcreteIntegration, "register_model")
    @patch.object(ConcreteIntegration, "load_model")
    @patch.object(ConcreteIntegration, "register_dummy")
    @patch.object(ConcreteIntegration, "check_dependencies")
    @patch.object(ConcreteIntegration, "get_available_models")
    def test_integrate_success(
        self,
        mock_get_available,
        mock_check_deps,
        mock_register_dummy,
        mock_load_model,
        mock_register_model,
        mock_enable_configs,
    ):
        """Test successful integration."""
        # Set up mocks
        mock_check_deps.return_value = {"torch": True}
        mock_register_dummy.return_value = True
        mock_get_available.return_value = self.model_pointers[:1]  # One available model
        mock_load_model.return_value = Mock()
        mock_register_model.return_value = True
        mock_enable_configs.return_value = True

        integration = ConcreteIntegration(
            name="test_integration",
            interface_name="test_interface",
            model_pointers=self.model_pointers,
        )

        result = integration.integrate()

        assert result == True
        assert integration.integration_status == "success"
        assert len(integration.loaded_models) == 1

    @patch.object(ConcreteIntegration, "register_dummy")
    @patch.object(ConcreteIntegration, "check_dependencies")
    def test_integrate_interface_unavailable(
        self,
        mock_check_deps,
        mock_register_dummy,
    ):
        """Test integration when interface is unavailable."""
        mock_check_deps.return_value = {"torch": True}
        mock_register_dummy.return_value = False

        integration = ConcreteIntegration(
            name="test_integration",
            interface_name="test_interface",
            model_pointers=self.model_pointers,
        )

        result = integration.integrate()

        assert result == False
        assert integration.integration_status == "interface_unavailable"

    @patch.object(ConcreteIntegration, "register_dummy")
    @patch.object(ConcreteIntegration, "check_dependencies")
    @patch.object(ConcreteIntegration, "get_available_models")
    def test_integrate_no_models(
        self,
        mock_get_available,
        mock_check_deps,
        mock_register_dummy,
    ):
        """Test integration when no models are available."""
        mock_check_deps.return_value = {"torch": True}
        mock_register_dummy.return_value = True
        mock_get_available.return_value = []  # No available models

        integration = ConcreteIntegration(
            name="test_integration",
            interface_name="test_interface",
            model_pointers=self.model_pointers,
            enable_fallback=False,
        )

        result = integration.integrate()

        assert result == False
        assert integration.integration_status == "no_models"

    @patch.object(ConcreteIntegration, "enable_configs")
    @patch.object(ConcreteIntegration, "register_model")
    @patch.object(ConcreteIntegration, "load_model")
    @patch.object(ConcreteIntegration, "register_dummy")
    @patch.object(ConcreteIntegration, "check_dependencies")
    @patch.object(ConcreteIntegration, "get_available_models")
    def test_integrate_config_failed_with_fallback(
        self,
        mock_get_available,
        mock_check_deps,
        mock_register_dummy,
        mock_load_model,
        mock_register_model,
        mock_enable_configs,
    ):
        """Test integration when config enabling fails but fallback is enabled."""
        mock_check_deps.return_value = {"torch": True}
        mock_register_dummy.return_value = True
        mock_get_available.return_value = self.model_pointers[:1]
        mock_load_model.return_value = Mock()
        mock_register_model.return_value = True
        mock_enable_configs.return_value = False  # Config enabling fails

        integration = ConcreteIntegration(
            name="test_integration",
            interface_name="test_interface",
            model_pointers=self.model_pointers,
            enable_fallback=True,
        )

        result = integration.integrate()

        assert result == True  # Should succeed due to fallback
        assert integration.integration_status == "config_failed"

    def test_get_status(self):
        """Test getting integration status."""
        with patch.object(
            ConcreteIntegration, "get_available_models"
        ) as mock_available:
            mock_available.return_value = self.model_pointers[:1]

            integration = ConcreteIntegration(
                name="test_integration",
                interface_name="test_interface",
                model_pointers=self.model_pointers,
            )

            integration.loaded_models["test_model1.pt"] = Mock()
            integration.registration_status["test_model1.pt"] = True

            status = integration.get_status()

            assert status["name"] == "test_integration"
            assert status["interface_name"] == "test_interface"
            assert status["models_available"] == 1
            assert status["models_loaded"] == 1
            assert status["registration_status"]["test_model1.pt"] == True


class TestIntegrationRegistry:
    """Test the IntegrationRegistry class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = IntegrationRegistry()
        self.integration1 = ConcreteIntegration(
            name="integration1",
            interface_name="interface1",
            model_pointers=[ModelPointer("model1.pt", ".", "purpose1", "interface1")],
        )
        self.integration2 = ConcreteIntegration(
            name="integration2",
            interface_name="interface2",
            model_pointers=[ModelPointer("model2.pt", ".", "purpose2", "interface2")],
        )

    def test_init(self):
        """Test registry initialization."""
        assert len(self.registry.integrations) == 0
        assert len(self.registry.execution_order) == 0

    def test_register_integration(self):
        """Test registering an integration."""
        self.registry.register(self.integration1)

        assert "integration1" in self.registry.integrations
        assert self.registry.integrations["integration1"] == self.integration1
        assert "integration1" in self.registry.execution_order

    def test_register_integration_with_order(self):
        """Test registering integrations with specific execution order."""
        self.registry.register(self.integration1, execute_order=2)
        self.registry.register(self.integration2, execute_order=1)

        # integration2 should come first due to lower order
        assert self.registry.execution_order[0] == "integration2"
        assert self.registry.execution_order[1] == "integration1"

    @patch.object(ConcreteIntegration, "integrate")
    def test_integrate_all(self, mock_integrate):
        """Test integrating all registered integrations."""
        mock_integrate.return_value = True

        self.registry.register(self.integration1)
        self.registry.register(self.integration2)

        results = self.registry.integrate_all()

        assert results["integration1"] == True
        assert results["integration2"] == True
        assert mock_integrate.call_count == 2

    @patch.object(ConcreteIntegration, "integrate")
    def test_integrate_all_with_exception(self, mock_integrate):
        """Test integrate_all when one integration raises exception."""
        mock_integrate.side_effect = [Exception("Test error"), True]

        self.registry.register(self.integration1)
        self.registry.register(self.integration2)

        results = self.registry.integrate_all()

        assert results["integration1"] == False
        assert results["integration2"] == True

    @patch.object(ConcreteIntegration, "get_status")
    def test_get_status_report(self, mock_get_status):
        """Test getting status report for all integrations."""
        mock_get_status.return_value = {"status": "test"}

        self.registry.register(self.integration1)
        self.registry.register(self.integration2)

        report = self.registry.get_status_report()

        assert "integration1" in report
        assert "integration2" in report
        assert report["integration1"]["status"] == "test"


class TestGlobalFunctions:
    """Test global registry functions."""

    def setup_method(self):
        """Reset global registry for each test."""
        # Clear the global registry
        global_registry = get_integration_registry()
        global_registry.integrations.clear()
        global_registry.execution_order.clear()

    def test_get_integration_registry(self):
        """Test getting the global registry."""
        registry = get_integration_registry()
        assert isinstance(registry, IntegrationRegistry)

    def test_register_integration(self):
        """Test registering integration with global function."""
        integration = ConcreteIntegration(
            name="global_test",
            interface_name="test",
            model_pointers=[],
        )

        register_integration(integration)

        registry = get_integration_registry()
        assert "global_test" in registry.integrations

    @patch.object(IntegrationRegistry, "integrate_all")
    def test_integrate_all(self, mock_integrate_all):
        """Test global integrate_all function."""
        mock_integrate_all.return_value = {"test": True}

        results = integrate_all()

        assert results == {"test": True}
        mock_integrate_all.assert_called_once()

    @patch.object(IntegrationRegistry, "get_status_report")
    def test_get_integration_status(self, mock_get_status):
        """Test global get_integration_status function."""
        mock_get_status.return_value = {"test": {"status": "success"}}

        status = get_integration_status()

        assert status == {"test": {"status": "success"}}
        mock_get_status.assert_called_once()


class TestDiscoverAndRegisterIntegrations:
    """Test the discovery and registration function."""

    def setup_method(self):
        """Reset global registry for each test."""
        global_registry = get_integration_registry()
        global_registry.integrations.clear()
        global_registry.execution_order.clear()

    @patch("importlib.import_module")
    def test_discover_and_register_success(self, mock_import_module):
        """Test successful discovery and registration."""
        # Mock module with factory function
        mock_module = Mock()
        mock_integration = ConcreteIntegration(
            name="discovered_integration",
            interface_name="test",
            model_pointers=[],
        )
        mock_module.create_matmul_integration.return_value = mock_integration
        mock_import_module.return_value = mock_module

        results = discover_and_register_integrations()

        assert "matmul_integration" in results
        assert results["matmul_integration"] == True

        # Check that integration was registered
        registry = get_integration_registry()
        assert "discovered_integration" in registry.integrations

    @patch("importlib.import_module")
    def test_discover_and_register_import_error(self, mock_import_module):
        """Test discovery with import error."""
        mock_import_module.side_effect = ImportError("Module not found")

        results = discover_and_register_integrations()

        assert "matmul_integration" in results
        assert results["matmul_integration"] == False

    @patch("importlib.import_module")
    def test_discover_and_register_no_factory_function(self, mock_import_module):
        """Test discovery when module has no factory function."""
        mock_module = Mock()
        # Remove the expected factory function
        if hasattr(mock_module, "create_matmul_integration"):
            delattr(mock_module, "create_matmul_integration")
        mock_import_module.return_value = mock_module

        results = discover_and_register_integrations()

        assert "matmul_integration" in results
        assert results["matmul_integration"] == False

    @patch("importlib.import_module")
    def test_discover_and_register_factory_exception(self, mock_import_module):
        """Test discovery when factory function raises exception."""
        mock_module = Mock()
        mock_module.create_matmul_integration.side_effect = Exception("Factory error")
        mock_import_module.return_value = mock_module

        results = discover_and_register_integrations()

        assert "matmul_integration" in results
        assert results["matmul_integration"] == False


if __name__ == "__main__":
    pytest.main([__file__])
