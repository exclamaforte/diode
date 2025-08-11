"""
Integration tests for the diode-models package.
"""

import os
import unittest
import torch
import tempfile
import shutil
from pathlib import Path

# Import from diode package
from diode.model.matmul_timing_model import MatmulTimingModel, DeepMatmulTimingModel


class TestDiodeModelsIntegration(unittest.TestCase):
    """Integration tests for the diode-models package."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test models
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.temp_dir.name)
        
        # Create a simple model for testing
        self.problem_feature_dim = 3
        self.config_feature_dim = 2
        
        # Create a MatmulTimingModel
        self.base_model = MatmulTimingModel(
            problem_feature_dim=self.problem_feature_dim,
            config_feature_dim=self.config_feature_dim,
            hidden_dims=[64, 32, 16],
            dropout_rate=0.1,
        )
        
        # Create a DeepMatmulTimingModel
        self.deep_model = DeepMatmulTimingModel(
            problem_feature_dim=self.problem_feature_dim,
            config_feature_dim=self.config_feature_dim,
            hidden_dim=32,
            num_layers=3,
            dropout_rate=0.1,
        )
        
        # Save the models
        self.base_model_path = self.model_dir / "base_model.pt"
        self.deep_model_path = self.model_dir / "deep_model.pt"
        
        self.base_model.save(str(self.base_model_path))
        self.deep_model.save(str(self.deep_model_path))
        
        # Create test inputs
        self.problem_features = torch.randn(5, self.problem_feature_dim)
        self.config_features = torch.randn(5, self.config_feature_dim)
        
        # Create a mock trained_models directory in the diode-models package
        self.diode_models_dir = Path("/home/gabeferns/diode/diode-models/diode_models/trained_models")
        os.makedirs(self.diode_models_dir, exist_ok=True)
        
        # Copy the models to the diode-models package
        shutil.copy(self.base_model_path, self.diode_models_dir / "base_model.pt")
        shutil.copy(self.deep_model_path, self.diode_models_dir / "deep_model.pt")

    def tearDown(self):
        """Clean up the test environment."""
        self.temp_dir.cleanup()
        
        # Clean up the mock trained_models directory
        if self.diode_models_dir.exists():
            shutil.rmtree(self.diode_models_dir)

    def test_import_diode_models(self):
        """Test importing the diode_models package."""
        try:
            import diode_models
            self.assertTrue(hasattr(diode_models, "get_models_dir"))
            self.assertTrue(hasattr(diode_models, "list_available_models"))
            self.assertTrue(hasattr(diode_models, "get_model_wrapper"))
        except ImportError:
            self.skipTest("diode_models package not installed")

    def test_get_model_wrapper(self):
        """Test getting the ModelWrapper class."""
        try:
            import diode_models
            ModelWrapper = diode_models.get_model_wrapper()
            self.assertTrue(hasattr(ModelWrapper, "predict"))
            self.assertTrue(hasattr(ModelWrapper, "list_available_models"))
        except ImportError:
            self.skipTest("diode_models package not installed")

    def test_list_available_models(self):
        """Test listing available models."""
        try:
            import diode_models
            models = diode_models.list_available_models()
            self.assertGreaterEqual(len(models), 2)  # At least the two models we copied
        except ImportError:
            self.skipTest("diode_models package not installed")

    def test_load_and_run_model(self):
        """Test loading and running a model."""
        try:
            import diode_models
            ModelWrapper = diode_models.get_model_wrapper()
            
            # Get the first available model
            models = diode_models.list_available_models()
            if not models:
                self.skipTest("No models available")
            
            # Load the model
            wrapper = ModelWrapper(
                model_path=models[0],
                device="cpu",
                compile_model=False,
            )
            
            # Run inference
            predictions = wrapper.predict(self.problem_features, self.config_features)
            
            # Check the output shape
            self.assertEqual(predictions.shape, (5, 1))
        except ImportError:
            self.skipTest("diode_models package not installed")


if __name__ == "__main__":
    unittest.main()
