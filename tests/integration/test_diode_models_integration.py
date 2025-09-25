"""
Integration tests for the diode model functionality (formerly diode_models package).
"""

import os

# Enable debug flags for testing
try:
    from torch_diode.utils.debug_config import set_debug_flag

    set_debug_flag("ENABLE_TYPE_ASSERTS", True)
except ImportError:
    pass  # In case debug_config is not available yet
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from torch_diode.model.matmul_model_config import MatmulModelConfig
from torch_diode.model.matmul_model_trainer import (
    get_model_save_path,
    load_model_with_config,
    save_model_with_config,
)

# Import from torch_diode package
from torch_diode.model.matmul_timing_model import (
    DeepMatmulTimingModel,
    MatmulTimingModel,
)
from torch_diode.model.model_wrapper import ModelWrapper, list_available_models


class TestDiodeModelsIntegration(unittest.TestCase):
    """Integration tests for the diode model functionality."""

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

        # Create model configs
        self.base_config = MatmulModelConfig(
            model_type="base",
            problem_feature_dim=self.problem_feature_dim,
            config_feature_dim=self.config_feature_dim,
            hidden_dims=[64, 32, 16],
            dropout_rate=0.1,
            hardware_name="test_cpu",
            hardware_type="Intel-CPU",
            heuristic_name="matmul",
        )

        self.deep_config = MatmulModelConfig(
            model_type="deep",
            problem_feature_dim=self.problem_feature_dim,
            config_feature_dim=self.config_feature_dim,
            hidden_dim=32,
            num_layers=3,
            dropout_rate=0.1,
            hardware_name="test_gpu",
            hardware_type="NVIDIA-H100",
            heuristic_name="matmul",
        )

        # Save the models with configs
        self.base_model_path = self.model_dir / "base_model"
        self.deep_model_path = self.model_dir / "deep_model"

        save_model_with_config(self.base_model, self.base_config, self.base_model_path)
        save_model_with_config(self.deep_model, self.deep_config, self.deep_model_path)

        # Create test inputs
        self.problem_features = torch.randn(5, self.problem_feature_dim)
        self.config_features = torch.randn(5, self.config_feature_dim)

        # Create a mock directory structure in the diode package
        self.trained_models_dir = Path("/home/gabeferns/diode/trained_models")

        # Create the directory structure for the new format
        self.matmul_cpu_dir = self.trained_models_dir / "matmul" / "test_cpu"
        self.matmul_gpu_dir = self.trained_models_dir / "matmul" / "test_gpu"

        os.makedirs(self.matmul_cpu_dir, exist_ok=True)
        os.makedirs(self.matmul_gpu_dir, exist_ok=True)

        # Copy the models to the trained_models directory with the new structure
        shutil.copy(
            self.base_model_path.with_suffix(".pt"),
            self.matmul_cpu_dir / "matmul_test_cpu_base.pt",
        )
        shutil.copy(
            self.deep_model_path.with_suffix(".pt"),
            self.matmul_gpu_dir / "matmul_test_gpu_deep.pt",
        )

        # Copy the configs
        shutil.copy(
            self.base_model_path.with_suffix(".json"),
            self.matmul_cpu_dir / "matmul_test_cpu_base.json",
        )
        shutil.copy(
            self.deep_model_path.with_suffix(".json"),
            self.matmul_gpu_dir / "matmul_test_gpu_deep.json",
        )

    def tearDown(self):
        """Clean up the test environment."""
        self.temp_dir.cleanup()

        # Clean up the mock directory structure
        if (self.trained_models_dir / "matmul").exists():
            shutil.rmtree(self.trained_models_dir / "matmul")

    def test_import_diode_models(self):
        """Test importing the diode model functionality."""
        # Test that we can import the merged functionality from torch_diode
        from torch_diode.model.model_wrapper import ModelWrapper, list_available_models

        self.assertTrue(callable(ModelWrapper))
        self.assertTrue(callable(list_available_models))

    def test_get_model_wrapper(self):
        """Test getting the ModelWrapper class."""
        # ModelWrapper is now directly available from torch_diode.model.model_wrapper
        self.assertTrue(callable(ModelWrapper))

    def test_list_available_models(self):
        """Test listing available models."""
        models = list_available_models(models_dir=str(self.trained_models_dir))
        self.assertGreaterEqual(len(models), 2)  # At least the two models we copied

        # Test filtering by heuristic
        matmul_models = list_available_models(
            models_dir=str(self.trained_models_dir), heuristic_name="matmul"
        )
        self.assertGreaterEqual(len(matmul_models), 2)

        # Test filtering by hardware
        cpu_models = list_available_models(
            models_dir=str(self.trained_models_dir),
            heuristic_name="matmul",
            hardware_name="test_cpu",
        )
        self.assertGreaterEqual(len(cpu_models), 1)

        gpu_models = list_available_models(
            models_dir=str(self.trained_models_dir),
            heuristic_name="matmul",
            hardware_name="test_gpu",
        )
        self.assertGreaterEqual(len(gpu_models), 1)

    def test_load_and_run_model(self):
        """Test loading and running a model."""
        # Get models by hardware
        cpu_models = list_available_models(
            models_dir=str(self.trained_models_dir),
            heuristic_name="matmul",
            hardware_name="test_cpu",
        )
        if not cpu_models:
            self.skipTest("No CPU models available")

        # Load the model
        wrapper = ModelWrapper(
            model_path=cpu_models[0],
            device="cpu",
            compile_model=False,
        )

        # Run inference
        predictions = wrapper.predict(self.problem_features, self.config_features)

        # Check the output shape
        self.assertEqual(predictions.shape, (5, 1))

        # Check that the config was loaded
        self.assertIsNotNone(wrapper.config)
        self.assertEqual(wrapper.config.hardware_name, "test_cpu")
        self.assertEqual(wrapper.config.heuristic_name, "matmul")

    def test_save_and_load_model_with_config(self):
        """Test saving and loading a model with its configuration."""
        # Create a new model and config
        model = MatmulTimingModel(
            problem_feature_dim=self.problem_feature_dim,
            config_feature_dim=self.config_feature_dim,
            hidden_dims=[32, 16],
            dropout_rate=0.2,
        )

        config = MatmulModelConfig(
            model_type="base",
            problem_feature_dim=self.problem_feature_dim,
            config_feature_dim=self.config_feature_dim,
            hidden_dims=[32, 16],
            dropout_rate=0.2,
            hardware_name="test_tpu",
            heuristic_name="matmul",
        )

        # Save the model with its config
        save_path = self.model_dir / "test_model"
        save_model_with_config(model, config, save_path)

        # Check that the files were created
        self.assertTrue(save_path.with_suffix(".pt").exists())
        self.assertTrue(save_path.with_suffix(".json").exists())

        # Load the model and config
        loaded_model, loaded_config = load_model_with_config(save_path, device="cpu")

        # Check that the config was loaded correctly
        self.assertEqual(loaded_config.model_type, "base")
        self.assertEqual(loaded_config.problem_feature_dim, self.problem_feature_dim)
        self.assertEqual(loaded_config.config_feature_dim, self.config_feature_dim)
        self.assertEqual(loaded_config.hidden_dims, [32, 16])
        self.assertEqual(loaded_config.dropout_rate, 0.2)
        self.assertEqual(loaded_config.hardware_name, "test_tpu")
        self.assertEqual(loaded_config.heuristic_name, "matmul")

        # Run inference with the loaded model
        with torch.no_grad():
            predictions = loaded_model(self.problem_features, self.config_features)

        # Check the output shape
        self.assertEqual(predictions.shape, (5, 1))

    def test_get_model_save_path(self):
        """Test getting the model save path."""
        # Test with default base_dir
        save_path = get_model_save_path(
            heuristic_name="matmul",
            hardware_name="test_hardware",
            model_name="test_model",
            base_dir=self.model_dir,
        )

        expected_path = self.model_dir / "matmul" / "test_hardware" / "test_model"
        self.assertEqual(save_path, expected_path)

        # Check that the directory was created
        self.assertTrue((self.model_dir / "matmul" / "test_hardware").exists())


if __name__ == "__main__":
    unittest.main()
