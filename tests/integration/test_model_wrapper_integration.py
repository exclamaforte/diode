"""
Integration tests for the model wrapper.
"""

import json
# Enable debug flags for testing
try:
    from torch_diode.utils.debug_config import set_debug_flag
    set_debug_flag("ENABLE_TYPE_ASSERTS", True)
except ImportError:
    pass  # In case debug_config is not available yet
import os
import tempfile
import unittest
from pathlib import Path

import torch
from torch_diode.model.matmul_timing_model import (
    DeepMatmulTimingModel,
    MatmulTimingModel,
)

from torch_diode.model.model_wrapper import ModelWrapper


class TestModelWrapperIntegration(unittest.TestCase):
    """Integration tests for the model wrapper."""

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

        # Create model checkpoints with the expected structure
        base_checkpoint = {
            "model_state_dict": self.base_model.state_dict(),
            "problem_feature_dim": self.problem_feature_dim,
            "config_feature_dim": self.config_feature_dim,
            "hidden_dims": [64, 32, 16],
            "dropout_rate": 0.1,
        }

        deep_checkpoint = {
            "model_state_dict": self.deep_model.state_dict(),
            "problem_feature_dim": self.problem_feature_dim,
            "config_feature_dim": self.config_feature_dim,
            "hidden_dim": 32,
            "num_layers": 3,
            "dropout_rate": 0.1,
        }

        torch.save(base_checkpoint, self.base_model_path)
        torch.save(deep_checkpoint, self.deep_model_path)

        # Create corresponding config files
        base_config = {
            "model_type": "base",
            "problem_feature_dim": self.problem_feature_dim,
            "config_feature_dim": self.config_feature_dim,
            "hidden_dims": [64, 32, 16],
            "dropout_rate": 0.1,
        }

        deep_config = {
            "model_type": "deep",
            "problem_feature_dim": self.problem_feature_dim,
            "config_feature_dim": self.config_feature_dim,
            "hidden_dim": 32,
            "num_layers": 3,
            "dropout_rate": 0.1,
        }

        with open(self.base_model_path.with_suffix(".json"), "w") as f:
            json.dump(base_config, f)

        with open(self.deep_model_path.with_suffix(".json"), "w") as f:
            json.dump(deep_config, f)

        # Create test inputs
        self.problem_features = torch.randn(5, self.problem_feature_dim)
        self.config_features = torch.randn(5, self.config_feature_dim)

    def tearDown(self):
        """Clean up the test environment."""
        self.temp_dir.cleanup()

    def test_load_base_model(self):
        """Test loading a base model."""
        # Load the model using the wrapper
        wrapper = ModelWrapper(
            model_path=str(self.base_model_path),
            device="cpu",
            compile_model=False,
        )

        # Check that the model was loaded correctly
        self.assertIsInstance(wrapper.model, MatmulTimingModel)
        self.assertEqual(wrapper.model.problem_feature_dim, self.problem_feature_dim)
        self.assertEqual(wrapper.model.config_feature_dim, self.config_feature_dim)

        # Run inference
        predictions = wrapper.predict(self.problem_features, self.config_features)

        # Check the output shape
        self.assertEqual(predictions.shape, (5, 1))

    def test_load_deep_model(self):
        """Test loading a deep model."""
        # Load the model using the wrapper
        wrapper = ModelWrapper(
            model_path=str(self.deep_model_path),
            device="cpu",
            compile_model=False,
        )

        # Check that the model was loaded correctly
        self.assertIsInstance(wrapper.model, DeepMatmulTimingModel)
        self.assertEqual(wrapper.model.problem_feature_dim, self.problem_feature_dim)
        self.assertEqual(wrapper.model.config_feature_dim, self.config_feature_dim)

        # Run inference
        predictions = wrapper.predict(self.problem_features, self.config_features)

        # Check the output shape
        self.assertEqual(predictions.shape, (5, 1))

    def test_compile_model(self):
        """Test compiling a model."""
        # Skip this test if torch.compile is not available
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile not available")

        # Load the model using the wrapper with compilation
        wrapper = ModelWrapper(
            model_path=str(self.base_model_path),
            device="cpu",
            compile_model=True,
        )

        # Run inference
        predictions = wrapper.predict(self.problem_features, self.config_features)

        # Check the output shape
        self.assertEqual(predictions.shape, (5, 1))

        # Verify that the model was actually compiled
        # The compiled model should still be wrapped in the model wrapper
        self.assertIsNotNone(wrapper.model)

        # Test that the compiled model produces consistent results
        predictions2 = wrapper.predict(self.problem_features, self.config_features)
        torch.testing.assert_close(predictions, predictions2, rtol=1e-5, atol=1e-5)

        # Verify that we're actually using the compiled model when compile_model=True
        self.assertTrue(hasattr(wrapper, 'compiled_model'))
        self.assertIsNotNone(wrapper.compiled_model)

    def test_list_available_models(self):
        """Test listing available models."""
        # Import the list_available_models function
        from torch_diode.model.model_wrapper import list_available_models

        # List models in the temporary directory
        models = list_available_models(str(self.model_dir))

        # Check that both models are found
        self.assertEqual(len(models), 2)
        self.assertTrue(any(str(self.base_model_path) in model for model in models))
        self.assertTrue(any(str(self.deep_model_path) in model for model in models))


if __name__ == "__main__":
    unittest.main()
