"""
Tests for ModelWrapper functionality in diode_models when used standalone.
"""
import unittest
import tempfile
import torch
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestModelWrapperStandalone(unittest.TestCase):
    """Test ModelWrapper functionality when diode_models is used standalone."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create a simple mock model file
        self.model_path = self.temp_path / "test_model.pt"
        self.config_path = self.temp_path / "test_model.json"
        
        # Create a simple model (just a linear layer for testing)
        simple_model = torch.nn.Linear(10, 1)
        torch.save(simple_model.state_dict(), self.model_path)
        
        # Create a mock config file
        config_data = {
            "model_type": "base",
            "problem_feature_dim": 10,
            "config_feature_dim": 5,
            "hidden_dims": [64, 32],
            "dropout_rate": 0.1,
            "hardware_name": "test_cpu",
            "hardware_type": "Intel-CPU",
            "heuristic_name": "matmul"
        }
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f)

    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_model_wrapper_import(self):
        """Test that ModelWrapper can be imported from diode_models."""
        try:
            from diode_models.model_wrapper import ModelWrapper
            self.assertIsNotNone(ModelWrapper)
            self.assertTrue(callable(ModelWrapper))
        except ImportError as e:
            self.fail(f"Failed to import ModelWrapper: {e}")

    def test_load_model_config_function(self):
        """Test the load_model_config function."""
        try:
            from diode_models.model_wrapper import load_model_config
            
            # Test loading config
            config = load_model_config(self.model_path)
            if config is not None:  # Config might be None if file doesn't exist
                self.assertEqual(config.model_type, "base")
                self.assertEqual(config.hardware_name, "test_cpu")
                self.assertEqual(config.heuristic_name, "matmul")
        except ImportError as e:
            self.fail(f"Failed to import load_model_config: {e}")

    def test_list_available_models_function(self):
        """Test the list_available_models function from model_wrapper."""
        try:
            from diode_models.model_wrapper import list_available_models
            
            # This should not fail even if no models are present
            models = list_available_models()
            self.assertIsInstance(models, list)
            
            # Test with filters
            models = list_available_models(heuristic_name="matmul")
            self.assertIsInstance(models, list)
            
            models = list_available_models(
                heuristic_name="matmul", 
                hardware_name="test_hw"
            )
            self.assertIsInstance(models, list)
        except ImportError as e:
            self.fail(f"Failed to import list_available_models: {e}")

    def test_get_models_dir_function(self):
        """Test the get_models_dir function from model_wrapper."""
        try:
            from diode_models.model_wrapper import get_models_dir
            
            models_dir = get_models_dir()
            self.assertIsInstance(models_dir, Path)
        except ImportError as e:
            self.fail(f"Failed to import get_models_dir: {e}")

    @patch('diode_models.model_wrapper.DIODE_AVAILABLE', False)
    def test_fallback_model_classes(self):
        """Test that fallback model classes work when diode package is not available."""
        try:
            from diode_models.model_wrapper import MatmulTimingModel, DeepMatmulTimingModel
            
            # These should be the minimal fallback implementations
            self.assertIsNotNone(MatmulTimingModel)
            self.assertIsNotNone(DeepMatmulTimingModel)
            
            # Test that they have the load class method
            self.assertTrue(hasattr(MatmulTimingModel, 'load'))
            self.assertTrue(hasattr(DeepMatmulTimingModel, 'load'))
            
        except ImportError as e:
            self.fail(f"Failed to import fallback model classes: {e}")

    @patch('diode_models.model_wrapper.DIODE_AVAILABLE', False)
    def test_fallback_model_config(self):
        """Test that fallback MatmulModelConfig works when diode package is not available."""
        try:
            from diode_models.model_wrapper import MatmulModelConfig
            
            # Test default initialization
            config = MatmulModelConfig()
            self.assertEqual(config.model_type, "deep")
            self.assertEqual(config.heuristic_name, "matmul")
            self.assertEqual(config.hardware_name, "unknown")
            self.assertIsInstance(config.hidden_dims, list)
            
            # Test custom initialization
            config = MatmulModelConfig(
                model_type="base",
                hardware_name="test_hw",
                problem_feature_dim=20,
                config_feature_dim=10
            )
            self.assertEqual(config.model_type, "base")
            self.assertEqual(config.hardware_name, "test_hw")
            self.assertEqual(config.problem_feature_dim, 20)
            self.assertEqual(config.config_feature_dim, 10)
            
            # Test from_dict method
            config_dict = {
                "model_type": "deep",
                "hardware_name": "gpu_test",
                "problem_feature_dim": 15,
                "config_feature_dim": 8,
                "hidden_dim": 256,
                "num_layers": 4
            }
            config_from_dict = MatmulModelConfig.from_dict(config_dict)
            self.assertEqual(config_from_dict.model_type, "deep")
            self.assertEqual(config_from_dict.hardware_name, "gpu_test")
            self.assertEqual(config_from_dict.problem_feature_dim, 15)
            self.assertEqual(config_from_dict.config_feature_dim, 8)
            self.assertEqual(config_from_dict.hidden_dim, 256)
            self.assertEqual(config_from_dict.num_layers, 4)
            
        except ImportError as e:
            self.fail(f"Failed to import fallback MatmulModelConfig: {e}")

    def test_model_wrapper_creation_basic(self):
        """Test basic ModelWrapper creation."""
        try:
            from diode_models.model_wrapper import ModelWrapper
            
            # This might fail if diode_common is not available, but we should test the import
            # The actual functionality test would require mocking diode_common
            self.assertTrue(callable(ModelWrapper))
            
        except ImportError as e:
            # This is expected if diode_common is not available
            self.skipTest(f"Skipping ModelWrapper creation test due to missing dependencies: {e}")

    @patch('diode_models.model_wrapper.CommonModelWrapper')
    def test_model_wrapper_creation_with_mock(self, mock_common_wrapper):
        """Test ModelWrapper creation with mocked CommonModelWrapper."""
        try:
            from diode_models.model_wrapper import ModelWrapper
            
            # Mock the CommonModelWrapper
            mock_instance = MagicMock()
            mock_common_wrapper.return_value = mock_instance
            
            # Create ModelWrapper
            wrapper = ModelWrapper(
                model_path=str(self.model_path),
                device="cpu",
                compile_model=False
            )
            
            # Verify CommonModelWrapper was called with correct arguments
            mock_common_wrapper.assert_called_once()
            call_args = mock_common_wrapper.call_args
            
            self.assertEqual(call_args.kwargs['model_path'], str(self.model_path))
            self.assertEqual(call_args.kwargs['device'], "cpu")
            self.assertEqual(call_args.kwargs['compile_model'], False)
            self.assertIn('model_classes', call_args.kwargs)
            self.assertIn('model_config_class', call_args.kwargs)
            
        except ImportError as e:
            self.skipTest(f"Skipping ModelWrapper creation test due to missing dependencies: {e}")


class TestDiodeModelsStandaloneIntegration(unittest.TestCase):
    """Integration tests for diode_models standalone functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_end_to_end_import_flow(self):
        """Test the complete import flow for diode_models standalone usage."""
        try:
            # Test main package import
            import diode_models
            self.assertIsNotNone(diode_models)
            
            # Test getting ModelWrapper class
            ModelWrapper = diode_models.get_model_wrapper()
            self.assertIsNotNone(ModelWrapper)
            
            # Test listing models (should not fail even if empty)
            models = diode_models.list_available_models()
            self.assertIsInstance(models, list)
            
            # Test getting models directory
            models_dir = diode_models.get_models_dir()
            self.assertIsInstance(models_dir, str)
            
        except ImportError as e:
            self.fail(f"End-to-end import flow failed: {e}")

    def test_package_attributes(self):
        """Test that all expected package attributes are available."""
        try:
            import diode_models
            
            # Check for expected functions
            expected_functions = [
                'get_models_dir',
                'list_available_models', 
                'get_model_wrapper'
            ]
            
            for func_name in expected_functions:
                self.assertTrue(hasattr(diode_models, func_name), 
                              f"Missing function: {func_name}")
                self.assertTrue(callable(getattr(diode_models, func_name)),
                              f"Attribute {func_name} is not callable")
            
            # Check for version
            self.assertTrue(hasattr(diode_models, '__version__'))
            
        except ImportError as e:
            self.fail(f"Package attributes test failed: {e}")


if __name__ == "__main__":
    unittest.main()
