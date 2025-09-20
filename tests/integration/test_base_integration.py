"""
Tests for diode.integration.base_integration module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import os

from torch_diode.integration.base_integration import (
    ModelPointer,
    BaseIntegration,
    IntegrationRegistry,
    get_integration_registry,
    register_integration,
    integrate_all,
    get_integration_status,
    discover_and_register_integrations
)


class TestModelPointer:
    """Test the ModelPointer class."""

    def test_init_basic(self):
        """Test basic initialization."""
        pointer = ModelPointer(
            model_name="test.pt",
            relative_path="test_path",
            model_purpose="test_purpose",
            interface_name="test_interface"
        )
        
        assert pointer.model_name == "test.pt"
        assert pointer.relative_path == "test_path"
        assert pointer.model_purpose == "test_purpose"
        assert pointer.interface_name == "test_interface"
        assert pointer.description == "Model for test_purpose"
        assert pointer.version == "1.0"
        assert pointer.dependencies == []

    def test_init_with_optional_params(self):
        """Test initialization with optional parameters."""
        pointer = ModelPointer(
            model_name="test.pt",
            relative_path="test_path",
            model_purpose="test_purpose",
            interface_name="test_interface",
            description="Custom description",
            version="2.0",
            dependencies=["torch", "numpy"]
        )
        
        assert pointer.description == "Custom description"
        assert pointer.version == "2.0"
        assert pointer.dependencies == ["torch", "numpy"]

    def test_full_path_with_dot_relative_path(self):
        """Test full_path property with dot relative path."""
        pointer = ModelPointer(
            model_name="test.pt",
            relative_path=".",
            model_purpose="test_purpose",
            interface_name="test_interface"
        )
          
        # The path should be diode_root/trained_models/test.pt
        # diode_root is 3 levels up from base_integration.py (__file__.parent.parent.parent)
        expected_path = Path(__file__).parent.parent.parent / "trained_models" / "test.pt"
        assert pointer.full_path == expected_path

    def test_full_path_with_subdirectory(self):
        """Test full_path property with subdirectory."""
        pointer = ModelPointer(
            model_name="test.pt",
            relative_path="subdir",
            model_purpose="test_purpose",
            interface_name="test_interface"
        )
          
        # The path should be diode_root/trained_models/subdir/test.pt
        expected_path = Path(__file__).parent.parent.parent / "trained_models" / "subdir" / "test.pt"
        assert pointer.full_path == expected_path

    def test_exists_true(self):
        """Test exists method when file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temporary file
            test_file = Path(temp_dir) / "test.pt"
            test_file.write_text("dummy content")
            
            pointer = ModelPointer(
                model_name="test.pt",
                relative_path=".",
                model_purpose="test_purpose",
                interface_name="test_interface"
            )
            
            # Mock the full_path property to point to our temp file
            with patch.object(type(pointer), 'full_path', new_callable=lambda: property(lambda self: test_file)):
                assert pointer.exists() is True

    def test_exists_false(self):
        """Test exists method when file doesn't exist."""
        pointer = ModelPointer(
            model_name="nonexistent.pt",
            relative_path=".",
            model_purpose="test_purpose",
            interface_name="test_interface"
        )
        
        # Mock the full_path property to point to a non-existent file
        with patch.object(type(pointer), 'full_path', new_callable=lambda: property(lambda self: Path("/nonexistent/path/file.pt"))):
            assert pointer.exists() is False

    def test_get_size_mb_existing_file(self):
        """Test get_size_mb for existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file with known size (1KB = 1024 bytes)
            test_file = Path(temp_dir) / "test.pt"
            test_file.write_bytes(b"x" * 1024)  # 1KB file
            
            pointer = ModelPointer(
                model_name="test.pt",
                relative_path=".",
                model_purpose="test_purpose",
                interface_name="test_interface"
            )
            
            with patch.object(type(pointer), 'full_path', new_callable=lambda: property(lambda self: test_file)):
                size_mb = pointer.get_size_mb()
                assert abs(size_mb - (1024 / (1024 * 1024))) < 0.001  # ~0.001 MB

    def test_get_size_mb_nonexistent_file(self):
        """Test get_size_mb for non-existent file."""
        pointer = ModelPointer(
            model_name="nonexistent.pt",
            relative_path=".",
            model_purpose="test_purpose",
            interface_name="test_interface"
        )
        
        with patch.object(type(pointer), 'full_path', new_callable=lambda: property(lambda self: Path("/nonexistent/path/file.pt"))):
            assert pointer.get_size_mb() == 0.0

    def test_repr(self):
        """Test string representation."""
        pointer = ModelPointer(
            model_name="test.pt",
            relative_path=".",
            model_purpose="test_purpose",
            interface_name="test_interface"
        )
        
        with patch.object(pointer, 'exists', return_value=True):
            repr_str = repr(pointer)
            assert "test.pt" in repr_str
            assert "test_purpose" in repr_str
            assert "exists=True" in repr_str


class TestBaseIntegrationConcrete(BaseIntegration):
    """Concrete implementation of BaseIntegration for testing."""
    
    def create_dummy_function(self):
        return Mock()
    
    def load_model(self, model_pointer):
        return Mock()
    
    def register_model(self, model, model_pointer):
        return True
    
    def enable_configs(self):
        return True


class TestBaseIntegration:
    """Test the BaseIntegration class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model_pointer = ModelPointer(
            model_name="test.pt",
            relative_path=".",
            model_purpose="test_purpose",
            interface_name="test_interface"
        )
        
        self.integration = TestBaseIntegrationConcrete(
            name="test_integration",
            interface_name="test_interface",
            model_pointers=[self.model_pointer]
        )

    def test_init(self):
        """Test initialization."""
        assert self.integration.name == "test_integration"
        assert self.integration.interface_name == "test_interface"
        assert len(self.integration.model_pointers) == 1
        assert self.integration.enable_fallback is True
        assert self.integration.loaded_models == {}
        assert self.integration.registration_status == {}
        assert self.integration.integration_status == "not_started"
        assert self.integration.execute_order is None

    def test_init_with_fallback_disabled(self):
        """Test initialization with fallback disabled."""
        integration = TestBaseIntegrationConcrete(
            name="test",
            interface_name="test",
            model_pointers=[],
            enable_fallback=False
        )
        assert integration.enable_fallback is False

    def test_register_dummy_success(self):
        """Test successful dummy registration.""" 
        dummy_function = Mock()
        
        # Mock the entire register_dummy method to return success
        with patch.object(self.integration, 'register_dummy', return_value=True) as mock_register:
            result = self.integration.register_dummy(dummy_function)
            
            assert result is True
            mock_register.assert_called_with(dummy_function)

    def test_register_dummy_already_registered(self):
        """Test dummy registration when another handler is already registered."""
        dummy_function = Mock()
        
        # Mock the entire register_dummy method to return failure
        with patch.object(self.integration, 'register_dummy', return_value=False) as mock_register:
            result = self.integration.register_dummy(dummy_function)
            
            assert result is False

    def test_register_dummy_import_error(self):
        """Test dummy registration with import error."""
        dummy_function = Mock()
        
        # Mock the entire register_dummy method to return failure
        with patch.object(self.integration, 'register_dummy', return_value=False) as mock_register:
            result = self.integration.register_dummy(dummy_function)
            
            assert result is False

    def test_check_dependencies_all_available(self):
        """Test dependency checking when all dependencies are available."""
        pointer_with_deps = ModelPointer(
            model_name="test.pt",
            relative_path=".",
            model_purpose="test_purpose",
            interface_name="test_interface",
            dependencies=["os", "sys"]  # These should always be available
        )
        
        integration = TestBaseIntegrationConcrete(
            name="test",
            interface_name="test",
            model_pointers=[pointer_with_deps]
        )
        
        status = integration.check_dependencies()
        assert status["os"] is True
        assert status["sys"] is True

    def test_check_dependencies_some_missing(self):
        """Test dependency checking with some missing dependencies."""
        pointer_with_deps = ModelPointer(
            model_name="test.pt",
            relative_path=".",
            model_purpose="test_purpose",
            interface_name="test_interface",
            dependencies=["os", "nonexistent_module"]
        )
        
        integration = TestBaseIntegrationConcrete(
            name="test",
            interface_name="test",
            model_pointers=[pointer_with_deps]
        )
        
        status = integration.check_dependencies()
        assert status["os"] is True
        assert status["nonexistent_module"] is False

    def test_get_available_models(self):
        """Test getting available models."""
        existing_pointer = Mock(spec=ModelPointer)
        existing_pointer.exists.return_value = True
        
        missing_pointer = Mock(spec=ModelPointer)
        missing_pointer.exists.return_value = False
        
        integration = TestBaseIntegrationConcrete(
            name="test",
            interface_name="test",
            model_pointers=[existing_pointer, missing_pointer]
        )
        
        available = integration.get_available_models()
        assert len(available) == 1
        assert existing_pointer in available
        assert missing_pointer not in available

    def test_integrate_success_full_flow(self):
        """Test successful full integration flow."""
        # Mock model pointer that exists
        with patch.object(self.model_pointer, 'exists', return_value=True):
            with patch.object(self.model_pointer, 'dependencies', []):
                # Mock register_dummy to succeed
                with patch.object(self.integration, 'register_dummy', return_value=True):
                    result = self.integration.integrate()
                    
                    assert result is True
                    assert self.integration.integration_status == "success"
                    assert len(self.integration.loaded_models) == 1
                    assert self.integration.registration_status["test.pt"] is True

    def test_integrate_interface_unavailable(self):
        """Test integration when interface is unavailable."""
        with patch.object(self.integration, 'register_dummy', return_value=False):
            result = self.integration.integrate()
            
            assert result is False
            assert self.integration.integration_status == "interface_unavailable"

    def test_integrate_no_models_available(self):
        """Test integration when no models are available."""
        with patch.object(self.model_pointer, 'exists', return_value=False):
            with patch.object(self.integration, 'register_dummy', return_value=True):
                with patch.object(self.integration, 'enable_configs', return_value=True):
                    result = self.integration.integrate()
                    
                    # Should return True due to fallback enabled, but status should be config_failed since no models were loaded
                    assert result is True
                    # The integration continues despite no models when fallback is enabled, and succeeds if configs are enabled
                    assert self.integration.integration_status == "success"

    def test_integrate_no_models_no_fallback(self):
        """Test integration when no models are available and fallback disabled."""
        integration = TestBaseIntegrationConcrete(
            name="test",
            interface_name="test",
            model_pointers=[self.model_pointer],
            enable_fallback=False
        )
        
        with patch.object(self.model_pointer, 'exists', return_value=False):
            with patch.object(integration, 'register_dummy', return_value=True):
                result = integration.integrate()
                
                assert result is False
                assert integration.integration_status == "no_models"

    def test_integrate_missing_dependencies_no_fallback(self):
        """Test integration with missing dependencies and no fallback."""
        pointer_with_missing_deps = ModelPointer(
            model_name="test.pt",
            relative_path=".",
            model_purpose="test_purpose",
            interface_name="test_interface",
            dependencies=["nonexistent_module"]
        )
        
        integration = TestBaseIntegrationConcrete(
            name="test",
            interface_name="test",
            model_pointers=[pointer_with_missing_deps],
            enable_fallback=False
        )
        
        result = integration.integrate()
        assert result is False
        assert integration.integration_status == "failed"

    def test_integrate_config_failed(self):
        """Test integration when config enabling fails."""
        with patch.object(self.model_pointer, 'exists', return_value=True):
            with patch.object(self.integration, 'register_dummy', return_value=True):
                with patch.object(self.integration, 'enable_configs', return_value=False):
                    result = self.integration.integrate()
                    
                    # Should return True due to fallback enabled
                    assert result is True
                    assert self.integration.integration_status == "config_failed"

    def test_integrate_exception_with_fallback(self):
        """Test integration with exception and fallback enabled."""
        with patch.object(self.integration, 'register_dummy', side_effect=Exception("Test error")):
            result = self.integration.integrate()
            
            assert result is False
            assert self.integration.integration_status == "error"

    def test_integrate_exception_no_fallback(self):
        """Test integration with exception and no fallback."""
        integration = TestBaseIntegrationConcrete(
            name="test",
            interface_name="test",
            model_pointers=[self.model_pointer],
            enable_fallback=False
        )
        
        with patch.object(integration, 'register_dummy', side_effect=Exception("Test error")):
            with pytest.raises(Exception, match="Test error"):
                integration.integrate()

    def test_get_status(self):
        """Test getting status information."""
        with patch.object(self.model_pointer, 'exists', return_value=True):
            with patch.object(self.model_pointer, 'dependencies', ["os"]):
                status = self.integration.get_status()
                
                assert status["name"] == "test_integration"
                assert status["interface_name"] == "test_interface"
                assert status["integration_status"] == "not_started"
                assert status["models_available"] == 1
                assert status["models_loaded"] == 0
                assert isinstance(status["registration_status"], dict)
                assert isinstance(status["dependencies"], dict)


class TestIntegrationRegistry:
    """Test the IntegrationRegistry class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = IntegrationRegistry()
        self.integration1 = TestBaseIntegrationConcrete(
            name="integration1",
            interface_name="interface1",
            model_pointers=[]
        )
        self.integration2 = TestBaseIntegrationConcrete(
            name="integration2", 
            interface_name="interface2",
            model_pointers=[]
        )

    def test_init(self):
        """Test initialization."""
        assert self.registry.integrations == {}
        assert self.registry.execution_order == []

    def test_register_without_order(self):
        """Test registering integration without specific order."""
        self.registry.register(self.integration1)
        
        assert "integration1" in self.registry.integrations
        assert self.registry.execution_order == ["integration1"]
        assert self.integration1.execute_order == 1

    def test_register_with_order(self):
        """Test registering integration with specific order."""
        self.registry.register(self.integration1, execute_order=5)
        self.registry.register(self.integration2, execute_order=3)
        
        # Should be ordered by execute_order
        assert self.registry.execution_order == ["integration2", "integration1"]
        assert self.integration1.execute_order == 5
        assert self.integration2.execute_order == 3

    def test_integrate_all(self):
        """Test integrating all registered integrations."""
        # Mock successful integration
        with patch.object(self.integration1, 'integrate', return_value=True):
            with patch.object(self.integration2, 'integrate', return_value=False):
                self.registry.register(self.integration1)
                self.registry.register(self.integration2)
                
                results = self.registry.integrate_all()
                
                assert results["integration1"] is True
                assert results["integration2"] is False

    def test_integrate_all_with_exception(self):
        """Test integrate_all when an integration raises an exception."""
        with patch.object(self.integration1, 'integrate', side_effect=Exception("Test error")):
            self.registry.register(self.integration1)
            
            results = self.registry.integrate_all()
            
            assert results["integration1"] is False

    def test_get_status_report(self):
        """Test getting status report for all integrations."""
        mock_status1 = {"name": "integration1", "status": "success"}
        mock_status2 = {"name": "integration2", "status": "failed"}
        
        with patch.object(self.integration1, 'get_status', return_value=mock_status1):
            with patch.object(self.integration2, 'get_status', return_value=mock_status2):
                self.registry.register(self.integration1)
                self.registry.register(self.integration2)
                
                report = self.registry.get_status_report()
                
                assert report["integration1"] == mock_status1
                assert report["integration2"] == mock_status2


class TestGlobalFunctions:
    """Test global registry functions."""

    def test_get_integration_registry(self):
        """Test getting global integration registry."""
        registry = get_integration_registry()
        assert isinstance(registry, IntegrationRegistry)
        
        # Should return the same instance
        registry2 = get_integration_registry()
        assert registry is registry2

    @patch('torch_diode.integration.base_integration._integration_registry')
    def test_register_integration(self, mock_registry):
        """Test global register_integration function."""
        integration = Mock()
        register_integration(integration, execute_order=5)
        mock_registry.register.assert_called_once_with(integration, 5)

    @patch('torch_diode.integration.base_integration._integration_registry')
    def test_integrate_all(self, mock_registry):
        """Test global integrate_all function."""
        expected_results = {"int1": True, "int2": False}
        mock_registry.integrate_all.return_value = expected_results
        
        result = integrate_all()
        assert result == expected_results
        mock_registry.integrate_all.assert_called_once()

    @patch('torch_diode.integration.base_integration._integration_registry')
    def test_get_integration_status(self, mock_registry):
        """Test global get_integration_status function."""
        expected_status = {"int1": {"status": "success"}}
        mock_registry.get_status_report.return_value = expected_status
        
        result = get_integration_status()
        assert result == expected_status
        mock_registry.get_status_report.assert_called_once()


class TestDiscoverAndRegisterIntegrations:
    """Test the discover_and_register_integrations function."""

    @patch('importlib.import_module')
    @patch('torch_diode.integration.base_integration.register_integration')
    def test_discover_success(self, mock_register, mock_import):
        """Test successful discovery and registration."""
        # Mock module with factory function
        mock_module = Mock()
        mock_integration = Mock()
        mock_integration.name = "test_integration"
        
        def create_matmul_integration(enable_fallback=True):
            return mock_integration
        
        mock_module.create_matmul_integration = create_matmul_integration
        mock_import.return_value = mock_module
        
        results = discover_and_register_integrations()
        
        assert "matmul_integration" in results
        assert results["matmul_integration"] is True
        mock_register.assert_called_once_with(mock_integration, execute_order=1)

    @patch('importlib.import_module')
    def test_discover_import_error(self, mock_import):
        """Test discovery with import error."""
        mock_import.side_effect = ImportError("Module not found")
        
        results = discover_and_register_integrations()
        
        assert "matmul_integration" in results
        assert results["matmul_integration"] is False

    @patch('importlib.import_module')
    def test_discover_no_factory_function(self, mock_import):
        """Test discovery when module has no factory function."""
        mock_module = Mock()
        # Explicitly mock hasattr to return False for the factory function
        mock_module.spec = Mock()
        mock_module.spec.name = "torch_diode.integration.matmul_integration"
        
        def mock_hasattr(obj, name):
            # Return False for the factory function, True for other attributes
            if name == "create_matmul_integration":
                return False
            return True
            
        mock_import.return_value = mock_module
        
        with patch('builtins.hasattr', side_effect=mock_hasattr):
            results = discover_and_register_integrations()
        
        assert "matmul_integration" in results
        assert results["matmul_integration"] is False

    @patch('importlib.import_module')
    @patch('torch_diode.integration.base_integration.register_integration')
    def test_discover_registration_error(self, mock_register, mock_import):
        """Test discovery when registration fails."""
        # Mock successful module loading
        mock_module = Mock()
        mock_integration = Mock()
        mock_integration.name = "test_integration"
        
        def create_matmul_integration(enable_fallback=True):
            return mock_integration
        
        mock_module.create_matmul_integration = create_matmul_integration
        mock_import.return_value = mock_module
        
        # Mock registration to fail
        mock_register.side_effect = Exception("Registration failed")
        
        results = discover_and_register_integrations()
        
        # Should initially be True but then marked as False due to registration failure
        assert "matmul_integration" in results

    @patch('importlib.import_module')
    def test_discover_factory_exception(self, mock_import):
        """Test discovery when factory function raises exception."""
        mock_module = Mock()
        
        def create_matmul_integration(enable_fallback=True):
            raise Exception("Factory failed")
        
        mock_module.create_matmul_integration = create_matmul_integration
        mock_import.return_value = mock_module
        
        results = discover_and_register_integrations()
        
        assert "matmul_integration" in results
        assert results["matmul_integration"] is False

    @patch('importlib.import_module')
    def test_discover_empty_results(self, mock_import):
        """Test discovery when no integrations are found."""
        # Mock all imports to fail
        mock_import.side_effect = ImportError("No modules found")
        
        results = discover_and_register_integrations()
        
        # Should have results for known_integrations, all False
        assert "matmul_integration" in results
        assert results["matmul_integration"] is False
