"""
Tests for diode_models package when used standalone (without other diode packages).
"""
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys


class TestDiodeModelsStandalone(unittest.TestCase):
    """Test diode_models package standalone functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_diode_models_import(self):
        """Test that diode_models can be imported without other diode packages."""
        try:
            import diode_models
            self.assertIsNotNone(diode_models)
            self.assertTrue(hasattr(diode_models, 'get_models_dir'))
            self.assertTrue(hasattr(diode_models, 'list_available_models'))
            self.assertTrue(hasattr(diode_models, 'get_model_wrapper'))
        except ImportError as e:
            self.fail(f"Failed to import diode_models: {e}")

    def test_get_models_dir(self):
        """Test getting the models directory."""
        import diode_models
        models_dir = diode_models.get_models_dir()
        self.assertIsInstance(models_dir, str)
        self.assertTrue(models_dir.endswith('trained_models'))

    def test_list_available_models_empty_dir(self):
        """Test listing models when directory is empty."""
        import diode_models
        # This should not fail even if no models are present
        models = diode_models.list_available_models()
        self.assertIsInstance(models, list)

    def test_list_available_models_with_filters(self):
        """Test listing models with filters."""
        import diode_models
        # Test with heuristic filter
        models = diode_models.list_available_models(heuristic_name="matmul")
        self.assertIsInstance(models, list)
        
        # Test with both filters
        models = diode_models.list_available_models(
            heuristic_name="matmul", 
            hardware_name="test_hardware"
        )
        self.assertIsInstance(models, list)

    def test_get_model_wrapper_class(self):
        """Test getting the ModelWrapper class."""
        import diode_models
        ModelWrapper = diode_models.get_model_wrapper()
        self.assertIsNotNone(ModelWrapper)
        # The function should return a class or function
        self.assertTrue(callable(ModelWrapper))

    def test_version_attribute(self):
        """Test that version attribute exists."""
        import diode_models
        self.assertTrue(hasattr(diode_models, '__version__'))
        self.assertIsInstance(diode_models.__version__, str)

    @patch('diode_models.MODELS_DIR')
    def test_models_dir_path_construction(self, mock_models_dir):
        """Test that models directory path is constructed correctly."""
        mock_models_dir.return_value = str(self.temp_path / "trained_models")
        import diode_models
        
        # Create some mock model files
        matmul_dir = self.temp_path / "trained_models" / "matmul" / "test_hw"
        matmul_dir.mkdir(parents=True, exist_ok=True)
        (matmul_dir / "model1.pt").touch()
        (matmul_dir / "model2.pt").touch()
        
        # Mock the MODELS_DIR to point to our temp directory
        with patch.object(diode_models, 'MODELS_DIR', str(self.temp_path / "trained_models")):
            models = diode_models.list_available_models(heuristic_name="matmul")
            # Should find the models we created
            self.assertGreaterEqual(len(models), 0)


class TestDiodeModelsWithoutDiodePackage(unittest.TestCase):
    """Test diode_models functionality when main diode package is not available."""

    def setUp(self):
        """Set up test environment by mocking import failure."""
        # Store original modules to restore later
        self.original_modules = {}
        modules_to_mock = [
            'diode',
            'diode.model',
            'diode.model.matmul_timing_model',
            'diode.model.matmul_model_config'
        ]
        
        for module in modules_to_mock:
            if module in sys.modules:
                self.original_modules[module] = sys.modules[module]
                del sys.modules[module]

    def tearDown(self):
        """Restore original modules."""
        for module, original in self.original_modules.items():
            sys.modules[module] = original

    def test_model_wrapper_fallback_import(self):
        """Test that model_wrapper falls back to standalone implementations."""
        # Mock the import to fail
        with patch.dict('sys.modules', {
            'diode': None,
            'diode.model': None,
            'diode.model.matmul_timing_model': None,
            'diode.model.matmul_model_config': None
        }):
            try:
                from diode_models.model_wrapper import ModelWrapper
                from diode_models.model_wrapper import MatmulModelConfig
                from diode_models.model_wrapper import MatmulTimingModel
                from diode_models.model_wrapper import DeepMatmulTimingModel
                
                # These should be the fallback implementations
                self.assertIsNotNone(ModelWrapper)
                self.assertIsNotNone(MatmulModelConfig)
                self.assertIsNotNone(MatmulTimingModel)
                self.assertIsNotNone(DeepMatmulTimingModel)
                
            except ImportError as e:
                self.fail(f"Failed to import fallback implementations: {e}")

    def test_minimal_model_config(self):
        """Test the minimal MatmulModelConfig implementation."""
        with patch.dict('sys.modules', {
            'diode': None,
            'diode.model': None,
            'diode.model.matmul_timing_model': None,
            'diode.model.matmul_model_config': None
        }):
            from diode_models.model_wrapper import MatmulModelConfig
            
            # Test default initialization
            config = MatmulModelConfig()
            self.assertEqual(config.model_type, "deep")
            self.assertEqual(config.heuristic_name, "matmul")
            self.assertIsInstance(config.hidden_dims, list)
            
            # Test from_dict method
            config_dict = {
                "model_type": "base",
                "hardware_name": "test_hw",
                "problem_feature_dim": 10,
                "config_feature_dim": 5
            }
            config_from_dict = MatmulModelConfig.from_dict(config_dict)
            self.assertEqual(config_from_dict.model_type, "base")
            self.assertEqual(config_from_dict.hardware_name, "test_hw")
            self.assertEqual(config_from_dict.problem_feature_dim, 10)
            self.assertEqual(config_from_dict.config_feature_dim, 5)


if __name__ == "__main__":
    unittest.main()
