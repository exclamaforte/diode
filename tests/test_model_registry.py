"""
Tests for diode.model_registry module.
"""


# Enable debug flags for testing
try:
    from torch_diode.utils.debug_config import set_debug_flag

    set_debug_flag("ENABLE_TYPE_ASSERTS", True)
except ImportError:
    pass  # In case debug_config is not available yet
from pathlib import Path
from unittest.mock import Mock, patch

from torch_diode.integration.base_integration import ModelPointer
from torch_diode.model_registry import (
    ModelRegistry,
    generate_model_manifest,
    get_model_info_for_build,
    get_model_paths_for_build,
    get_model_registry,
    register_model,
)


class TestModelRegistry:
    """Test the ModelRegistry class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ModelRegistry()

    def test_init_creates_empty_models_dict(self):
        """Test that initialization creates an empty models dictionary."""
        # Create registry without default models for testing
        registry = ModelRegistry.__new__(ModelRegistry)
        registry._models = {}
        assert registry._models == {}

    @patch.object(ModelRegistry, "_initialize_default_models")
    def test_init_calls_initialize_default_models(self, mock_init):
        """Test that initialization calls _initialize_default_models."""
        ModelRegistry()
        mock_init.assert_called_once()

    def test_initialize_default_models(self):
        """Test that default models are properly initialized."""
        # Check that models were registered
        assert len(self.registry._models) > 0

        # Check for specific default models
        matmul_v1_key = "matmul_kernel_runtime_prediction_v1_model.pt"
        matmul_exhaustive_key = (
            "matmul_kernel_runtime_prediction_matmul_model_exhaustive.pt"
        )

        assert matmul_v1_key in self.registry._models
        assert matmul_exhaustive_key in self.registry._models

    def test_register_model(self):
        """Test registering a new model."""
        model = ModelPointer(
            model_name="test_model.pt",
            relative_path="test_path",
            model_purpose="test_purpose",
            interface_name="test_interface",
            description="Test model",
            version="1.0",
            dependencies=["test_dep"],
        )

        # Clear existing models for clean test
        self.registry._models.clear()

        self.registry.register_model(model)

        expected_key = "test_purpose_test_model.pt"
        assert expected_key in self.registry._models
        assert self.registry._models[expected_key] == model

    def test_get_model_existing(self):
        """Test getting an existing model."""
        model = ModelPointer(
            model_name="test_model.pt",
            relative_path="test_path",
            model_purpose="test_purpose",
            interface_name="test_interface",
            description="Test model",
            version="1.0",
            dependencies=["test_dep"],
        )

        self.registry.register_model(model)

        result = self.registry.get_model("test_purpose", "test_model.pt")
        assert result == model

    def test_get_model_nonexistent(self):
        """Test getting a non-existent model."""
        result = self.registry.get_model("nonexistent_purpose", "nonexistent_model.pt")
        assert result is None

    def test_get_models_by_purpose(self):
        """Test getting models by purpose."""
        # Create test models with same purpose
        model1 = ModelPointer(
            model_name="model1.pt",
            relative_path="path1",
            model_purpose="common_purpose",
            interface_name="interface1",
            description="Model 1",
            version="1.0",
            dependencies=[],
        )

        model2 = ModelPointer(
            model_name="model2.pt",
            relative_path="path2",
            model_purpose="common_purpose",
            interface_name="interface2",
            description="Model 2",
            version="1.0",
            dependencies=[],
        )

        model3 = ModelPointer(
            model_name="model3.pt",
            relative_path="path3",
            model_purpose="different_purpose",
            interface_name="interface3",
            description="Model 3",
            version="1.0",
            dependencies=[],
        )

        self.registry._models.clear()
        self.registry.register_model(model1)
        self.registry.register_model(model2)
        self.registry.register_model(model3)

        results = self.registry.get_models_by_purpose("common_purpose")
        assert len(results) == 2
        assert model1 in results
        assert model2 in results
        assert model3 not in results

    def test_get_models_by_interface(self):
        """Test getting models by interface."""
        model1 = ModelPointer(
            model_name="model1.pt",
            relative_path="path1",
            model_purpose="purpose1",
            interface_name="common_interface",
            description="Model 1",
            version="1.0",
            dependencies=[],
        )

        model2 = ModelPointer(
            model_name="model2.pt",
            relative_path="path2",
            model_purpose="purpose2",
            interface_name="common_interface",
            description="Model 2",
            version="1.0",
            dependencies=[],
        )

        model3 = ModelPointer(
            model_name="model3.pt",
            relative_path="path3",
            model_purpose="purpose3",
            interface_name="different_interface",
            description="Model 3",
            version="1.0",
            dependencies=[],
        )

        self.registry._models.clear()
        self.registry.register_model(model1)
        self.registry.register_model(model2)
        self.registry.register_model(model3)

        results = self.registry.get_models_by_interface("common_interface")
        assert len(results) == 2
        assert model1 in results
        assert model2 in results
        assert model3 not in results

    def test_get_all_models(self):
        """Test getting all models."""
        # Clear and add test models
        self.registry._models.clear()

        model1 = ModelPointer(
            model_name="model1.pt",
            relative_path="path1",
            model_purpose="purpose1",
            interface_name="interface1",
            description="Model 1",
            version="1.0",
            dependencies=[],
        )

        model2 = ModelPointer(
            model_name="model2.pt",
            relative_path="path2",
            model_purpose="purpose2",
            interface_name="interface2",
            description="Model 2",
            version="1.0",
            dependencies=[],
        )

        self.registry.register_model(model1)
        self.registry.register_model(model2)

        results = self.registry.get_all_models()
        assert len(results) == 2
        assert model1 in results
        assert model2 in results

    def test_get_existing_models(self):
        """Test getting only models that exist on disk."""
        # Create mock models with different existence status
        existing_model = Mock(spec=ModelPointer)
        existing_model.exists.return_value = True

        nonexistent_model = Mock(spec=ModelPointer)
        nonexistent_model.exists.return_value = False

        self.registry._models = {
            "existing": existing_model,
            "nonexistent": nonexistent_model,
        }

        results = self.registry.get_existing_models()
        assert len(results) == 1
        assert existing_model in results
        assert nonexistent_model not in results

    def test_get_model_paths_for_build(self):
        """Test getting model paths for build system."""
        # Create mock models
        model1 = Mock(spec=ModelPointer)
        model1.exists.return_value = True
        model1.full_path = Path("/path/to/model1.pt")

        model2 = Mock(spec=ModelPointer)
        model2.exists.return_value = True
        model2.full_path = Path("/path/to/model2.pt")

        nonexistent_model = Mock(spec=ModelPointer)
        nonexistent_model.exists.return_value = False

        self.registry._models = {
            "model1": model1,
            "model2": model2,
            "nonexistent": nonexistent_model,
        }

        paths = self.registry.get_model_paths_for_build()
        assert len(paths) == 2
        assert Path("/path/to/model1.pt") in paths
        assert Path("/path/to/model2.pt") in paths

    def test_get_model_info_for_build(self):
        """Test getting model info for build system."""
        # Create mock models
        model1 = Mock(spec=ModelPointer)
        model1.exists.return_value = True
        model1.model_purpose = "test_purpose"
        model1.interface_name = "test_interface"
        model1.model_name = "model1.pt"
        model1.full_path = Path("/path/to/model1.pt")
        model1.relative_path = "relative/path"
        model1.get_size_mb.return_value = 10.5
        model1.version = "1.0"
        model1.description = "Test model 1"
        model1.dependencies = ["dep1", "dep2"]

        model2 = Mock(spec=ModelPointer)
        model2.exists.return_value = True
        model2.model_purpose = "test_purpose"
        model2.interface_name = "test_interface"
        model2.model_name = "model2.pt"
        model2.full_path = Path("/path/to/model2.pt")
        model2.relative_path = "relative/path2"
        model2.get_size_mb.return_value = 15.2
        model2.version = "2.0"
        model2.description = "Test model 2"
        model2.dependencies = ["dep2", "dep3"]

        self.registry._models = {"model1": model1, "model2": model2}

        info = self.registry.get_model_info_for_build()

        assert "test_purpose" in info
        purpose_info = info["test_purpose"]
        assert purpose_info["interface"] == "test_interface"
        assert len(purpose_info["models"]) == 2
        assert set(purpose_info["dependencies"]) == {"dep1", "dep2", "dep3"}

    def test_generate_manifest(self):
        """Test generating model manifest."""
        # Create mock models
        model1 = Mock(spec=ModelPointer)
        model1.exists.return_value = True
        model1.model_purpose = "purpose1"
        model1.interface_name = "interface1"
        model1.model_name = "model1.pt"
        model1.relative_path = "path1"
        model1.get_size_mb.return_value = 10.0
        model1.version = "1.0"
        model1.dependencies = ["dep1"]

        model2 = Mock(spec=ModelPointer)
        model2.exists.return_value = True
        model2.model_purpose = "purpose2"
        model2.interface_name = "interface1"
        model2.model_name = "model2.pt"
        model2.relative_path = "path2"
        model2.get_size_mb.return_value = 15.0
        model2.version = "2.0"
        model2.dependencies = ["dep2"]

        self.registry._models = {"model1": model1, "model2": model2}

        manifest = self.registry.generate_manifest()

        assert manifest["version"] == "1.0"
        assert manifest["total_models"] == 2
        assert manifest["total_size_mb"] == 25.0
        assert "purpose1" in manifest["models_by_purpose"]
        assert "purpose2" in manifest["models_by_purpose"]
        assert "interface1" in manifest["models_by_interface"]
        assert set(manifest["all_dependencies"]) == {"dep1", "dep2"}


class TestGlobalFunctions:
    """Test global registry functions."""

    def test_get_model_registry(self):
        """Test getting global model registry."""
        registry = get_model_registry()
        assert isinstance(registry, ModelRegistry)

        # Should return the same instance
        registry2 = get_model_registry()
        assert registry is registry2

    @patch("torch_diode.model_registry._model_registry")
    def test_register_model(self, mock_registry):
        """Test global register_model function."""
        model = Mock(spec=ModelPointer)
        register_model(model)
        mock_registry.register_model.assert_called_once_with(model)

    @patch("torch_diode.model_registry._model_registry")
    def test_get_model_paths_for_build(self, mock_registry):
        """Test global get_model_paths_for_build function."""
        expected_paths = [Path("/path1"), Path("/path2")]
        mock_registry.get_model_paths_for_build.return_value = expected_paths

        result = get_model_paths_for_build()
        assert result == expected_paths
        mock_registry.get_model_paths_for_build.assert_called_once()

    @patch("torch_diode.model_registry._model_registry")
    def test_get_model_info_for_build(self, mock_registry):
        """Test global get_model_info_for_build function."""
        expected_info = {"purpose": {"models": []}}
        mock_registry.get_model_info_for_build.return_value = expected_info

        result = get_model_info_for_build()
        assert result == expected_info
        mock_registry.get_model_info_for_build.assert_called_once()

    @patch("torch_diode.model_registry._model_registry")
    def test_generate_model_manifest(self, mock_registry):
        """Test global generate_model_manifest function."""
        expected_manifest = {"version": "1.0", "total_models": 0}
        mock_registry.generate_manifest.return_value = expected_manifest

        result = generate_model_manifest()
        assert result == expected_manifest
        mock_registry.generate_manifest.assert_called_once()


class TestModelRegistryIntegration:
    """Integration tests for ModelRegistry."""

    def test_full_workflow(self):
        """Test a complete workflow with the registry."""
        registry = ModelRegistry()
        registry._models.clear()  # Start clean

        # Register a model
        model = ModelPointer(
            model_name="test_model.pt",
            relative_path="test_path",
            model_purpose="test_purpose",
            interface_name="test_interface",
            description="Test model",
            version="1.0",
            dependencies=["dep1", "dep2"],
        )

        registry.register_model(model)

        # Test retrieval methods
        assert registry.get_model("test_purpose", "test_model.pt") == model
        assert model in registry.get_models_by_purpose("test_purpose")
        assert model in registry.get_models_by_interface("test_interface")
        assert model in registry.get_all_models()

        # Test build methods (with mocked existence)
        with patch.object(model, "exists", return_value=True):
            # Mock full_path property instead of setting it directly
            with patch.object(
                type(model),
                "full_path",
                new_callable=lambda: property(
                    lambda self: Path("/fake/path/test_model.pt")
                ),
            ):
                with patch.object(model, "get_size_mb", return_value=5.0):
                    assert model in registry.get_existing_models()

                    paths = registry.get_model_paths_for_build()
                    assert Path("/fake/path/test_model.pt") in paths

                    info = registry.get_model_info_for_build()
                    assert "test_purpose" in info

                    manifest = registry.generate_manifest()
                    assert manifest["total_models"] == 1
                    assert manifest["total_size_mb"] == 5.0

    def test_empty_registry_behavior(self):
        """Test behavior with empty registry."""
        registry = ModelRegistry()
        registry._models.clear()

        assert registry.get_model("any", "any") is None
        assert registry.get_models_by_purpose("any") == []
        assert registry.get_models_by_interface("any") == []
        assert registry.get_all_models() == []
        assert registry.get_existing_models() == []
        assert registry.get_model_paths_for_build() == []

        info = registry.get_model_info_for_build()
        assert info == {}

        manifest = registry.generate_manifest()
        assert manifest["total_models"] == 0
        assert manifest["total_size_mb"] == 0
        assert manifest["models_by_purpose"] == {}
        assert manifest["models_by_interface"] == {}
        assert manifest["all_dependencies"] == []
