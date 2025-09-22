"""
Unit tests for the DiodeInductorChoices integration.

NOTE: These tests are for the old API that used methods like _extract_features_from_kernel_inputs,
_convert_ktc_to_config, _predict_config_performance, and _finalize_mm_configs.
The current implementation uses _finalize_template_configs and delegates to kernel_conversions module.
These tests are marked as deprecated but kept for reference.
"""

import os
# Enable debug flags for testing
try:
    from torch_diode.utils.debug_config import set_debug_flag
    set_debug_flag("ENABLE_TYPE_ASSERTS", True)
except ImportError:
    pass  # In case debug_config is not available yet
import sys
import tempfile
import unittest
import unittest.mock as mock
from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional

import torch
import pytest

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from torch_diode.integration.inductor_integration import (
    create_diode_choices,
    DiodeInductorChoices,
    install_diode_choices,
)
from torch_diode.model.matmul_timing_model import MatmulTimingModel
from torch_diode.types.matmul_types import MMShape, TritonGEMMConfig


class MockKernelInputs:
    """Mock KernelInputs class for testing."""

    def __init__(self, tensors: List, device_type: str = "cuda"):
        self._tensors = tensors
        self._device_type = device_type

    def nodes(self):
        return self._tensors

    def device_type(self):
        return self._device_type

    def output_layout(self, flexible=True):
        return MockLayout()


class MockTensor:
    """Mock tensor for testing."""

    def __init__(self, size: tuple, dtype: torch.dtype):
        self._size = size
        self._dtype = dtype

    def get_size(self):
        return self._size

    def get_dtype(self):
        return self._dtype


class MockLayout:
    """Mock layout for testing."""

    def __init__(self):
        self.size = (128, 64)
        self.stride = (64, 1)
        self.dtype = torch.float16


class MockConfig:
    """Mock config for testing."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class MockTemplate:
    """Mock template for testing."""

    def __init__(self, uid: str):
        self.uid = uid


class MockKernelTemplateChoice:
    """Mock KernelTemplateChoice for testing."""

    def __init__(self, template, config, choice=None):
        self.template = template
        self.config = config
        self.choice = choice or self


class MockModelWrapper:
    """Mock ModelWrapper for testing."""

    def __init__(self, predictions: Optional[List[float]] = None):
        self.predictions = predictions or [0.1, 0.2, 0.15, 0.25]
        self.prediction_idx = 0

    def predict(self, problem_features, config_features):
        # Return different predictions for each call
        pred = self.predictions[self.prediction_idx % len(self.predictions)]
        self.prediction_idx += 1
        return torch.tensor([[pred]])


class TestDiodeInductorChoices(unittest.TestCase):
    """
    Tests for the DiodeInductorChoices class.
    """

    def setUp(self):
        """Set up test environment."""
        self.device = "cpu"  # Use CPU for tests
        self.temp_model_path = None

        # Create a temporary model for testing
        self._create_temp_model()

    def tearDown(self):
        """Clean up test environment."""
        if self.temp_model_path and os.path.exists(self.temp_model_path):
            os.unlink(self.temp_model_path)

    def _create_temp_model(self):
        """Create a temporary model file for testing."""
        # Create a simple model
        model = MatmulTimingModel(
            problem_feature_dim=4,
            config_feature_dim=6,
            hidden_dims=[32, 16],
            dropout_rate=0.1,
        )

        # Save to temporary file
        fd, self.temp_model_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        model.save(self.temp_model_path)

    def test_initialization_with_model_path(self):
        """Test DiodeInductorChoices initialization with model path."""
        choices = DiodeInductorChoices(
            model_path=self.temp_model_path,
            device=self.device,
            top_k_configs=3,
            performance_threshold=1.1,
        )

        self.assertEqual(choices.model_path, self.temp_model_path)
        self.assertEqual(choices.device, self.device)
        self.assertEqual(choices.top_k_configs, 3)
        self.assertEqual(choices.performance_threshold, 1.1)
        self.assertFalse(choices.enable_fallback)
        self.assertTrue(choices._model_loaded)
        self.assertIsNotNone(choices.model_wrapper)

    def test_initialization_without_model_path(self):
        """Test DiodeInductorChoices initialization without model path."""
        # Mock _find_default_model to return None to ensure no model is found
        with mock.patch.object(
            DiodeInductorChoices, "_find_default_model", return_value=None
        ):
            choices = DiodeInductorChoices(device=self.device)

            self.assertIsNone(choices.model_path)
            self.assertEqual(choices.device, self.device)
            self.assertEqual(choices.top_k_configs, 3)
            self.assertEqual(choices.performance_threshold, 1.1)
            self.assertFalse(choices._model_loaded)
            self.assertIsNone(choices.model_wrapper)

    def test_find_default_model(self):
        """Test finding default model."""
        # Mock _find_default_model to return None to ensure no model is found by default
        with mock.patch.object(
            DiodeInductorChoices, "_find_default_model", return_value=None
        ):
            choices = DiodeInductorChoices(device=self.device)

            # Test with non-existent files
            result = choices._find_default_model()
            self.assertIsNone(result)

        # Test with existing file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Patch os.path.exists to return True for our test file
            with mock.patch("os.path.exists") as mock_exists:

                def exists_side_effect(path):
                    return path == "matmul_model_exhaustive.pt"

                mock_exists.side_effect = exists_side_effect

                choices = DiodeInductorChoices(device=self.device)
                result = choices._find_default_model()
                self.assertEqual(result, "matmul_model_exhaustive.pt")
        finally:
            os.unlink(tmp_path)

    def test_extract_features_from_kernel_inputs_mm(self):
        """Test feature extraction for matrix multiplication."""
        choices = DiodeInductorChoices(device=self.device)

        # Create mock tensors for mm operation
        tensor_a = MockTensor((128, 64), torch.float16)
        tensor_b = MockTensor((64, 32), torch.float16)
        kernel_inputs = MockKernelInputs([tensor_a, tensor_b])

        result = choices._extract_features_from_kernel_inputs(kernel_inputs, "mm")

        self.assertIsNotNone(result)
        self.assertIn("mm_shape", result)
        self.assertIn("problem_features", result)
        self.assertIn("op_name", result)

        mm_shape = result["mm_shape"]
        self.assertEqual(mm_shape.M, 128)
        self.assertEqual(mm_shape.N, 32)
        self.assertEqual(mm_shape.K, 64)
        self.assertEqual(mm_shape.B, 1)
        self.assertEqual(result["op_name"], "mm")

    def test_extract_features_from_kernel_inputs_bmm(self):
        """Test feature extraction for batch matrix multiplication."""
        choices = DiodeInductorChoices(device=self.device)

        # Create mock tensors for bmm operation
        tensor_a = MockTensor((4, 128, 64), torch.float16)
        tensor_b = MockTensor((4, 64, 32), torch.float16)
        kernel_inputs = MockKernelInputs([tensor_a, tensor_b])

        result = choices._extract_features_from_kernel_inputs(kernel_inputs, "bmm")

        self.assertIsNotNone(result)
        mm_shape = result["mm_shape"]
        self.assertEqual(mm_shape.M, 128)
        self.assertEqual(mm_shape.N, 32)
        self.assertEqual(mm_shape.K, 64)
        self.assertEqual(mm_shape.B, 4)
        self.assertEqual(result["op_name"], "bmm")

    def test_extract_features_insufficient_tensors(self):
        """Test feature extraction with insufficient tensors."""
        choices = DiodeInductorChoices(device=self.device)

        # Create kernel inputs with only one tensor
        tensor_a = MockTensor((128, 64), torch.float16)
        kernel_inputs = MockKernelInputs([tensor_a])

        result = choices._extract_features_from_kernel_inputs(kernel_inputs, "mm")
        self.assertIsNone(result)

    def test_extract_features_unsupported_operation(self):
        """Test feature extraction with unsupported operation."""
        choices = DiodeInductorChoices(device=self.device)

        tensor_a = MockTensor((128, 64), torch.float16)
        tensor_b = MockTensor((64, 32), torch.float16)
        kernel_inputs = MockKernelInputs([tensor_a, tensor_b])

        result = choices._extract_features_from_kernel_inputs(
            kernel_inputs, "unsupported"
        )
        self.assertIsNone(result)

    def test_convert_ktc_to_config(self):
        """Test conversion from KernelTemplateChoice to TritonGEMMConfig."""
        choices = DiodeInductorChoices(device=self.device)

        # Create mock KTC
        template = MockTemplate("test_template")
        config = MockConfig(
            BLOCK_M=64,
            BLOCK_N=32,
            BLOCK_K=16,
            GROUP_M=8,
            num_stages=3,
            num_warps=4,
            EVEN_K=True,
            ALLOW_TF32=False,
        )
        ktc = MockKernelTemplateChoice(template, config)

        result = choices._convert_ktc_to_config(ktc)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, TritonGEMMConfig)
        self.assertEqual(result.name, "test_template")
        self.assertEqual(result.block_m, 64)
        self.assertEqual(result.block_n, 32)
        self.assertEqual(result.block_k, 16)
        self.assertEqual(result.group_m, 8)
        self.assertEqual(result.num_stages, 3)
        self.assertEqual(result.num_warps, 4)
        self.assertTrue(result.EVEN_K)
        self.assertFalse(result.ALLOW_TF32)

    def test_predict_config_performance_no_model(self):
        """Test performance prediction without loaded model."""
        choices = DiodeInductorChoices(device=self.device)

        problem_features = torch.tensor([1.0, 2.0, 3.0, 4.0])
        configs = [
            TritonGEMMConfig(
                name="test1",
                grid=1,
                block_m=64,
                block_n=32,
                block_k=16,
                group_m=8,
                num_stages=2,
                num_warps=4,
            ),
            TritonGEMMConfig(
                name="test2",
                grid=1,
                block_m=32,
                block_n=64,
                block_k=16,
                group_m=8,
                num_stages=2,
                num_warps=4,
            ),
        ]

        predictions = choices._predict_config_performance(problem_features, configs)

        # Should return zeros when no model is loaded
        self.assertEqual(predictions, [0.0, 0.0])

    def test_predict_config_performance_with_model(self):
        """Test performance prediction with loaded model."""
        choices = DiodeInductorChoices(
            model_path=self.temp_model_path, device=self.device
        )

        # Mock the model wrapper to return predictable results
        mock_wrapper = MockModelWrapper([0.1, 0.2])
        choices.model_wrapper = mock_wrapper

        problem_features = torch.tensor([1.0, 2.0, 3.0, 4.0])
        configs = [
            TritonGEMMConfig(
                name="test1",
                grid=1,
                block_m=64,
                block_n=32,
                block_k=16,
                group_m=8,
                num_stages=2,
                num_warps=4,
            ),
            TritonGEMMConfig(
                name="test2",
                grid=1,
                block_m=32,
                block_n=64,
                block_k=16,
                group_m=8,
                num_stages=2,
                num_warps=4,
            ),
        ]

        predictions = choices._predict_config_performance(problem_features, configs)

        self.assertEqual(len(predictions), 2)
        self.assertAlmostEqual(predictions[0], 0.1, places=5)
        self.assertAlmostEqual(predictions[1], 0.2, places=5)

    def test_finalize_template_configs_no_model(self):
        """Test _finalize_template_configs without model."""
        # Mock _find_default_model to return None to ensure no model is loaded
        with mock.patch.object(
            DiodeInductorChoices, "_find_default_model", return_value=None
        ):
            choices = DiodeInductorChoices(device=self.device, enable_fallback=True)  # Enable fallback

            # Create mock choices
            template = MockTemplate("test_template")
            config = MockConfig(BLOCK_M=64, BLOCK_N=32, BLOCK_K=16)
            ktc1 = MockKernelTemplateChoice(template, config)
            ktc2 = MockKernelTemplateChoice(template, config)

            template_choices = {"test": iter([ktc1, ktc2])}
            kernel_inputs = MockKernelInputs(
                [
                    MockTensor((128, 64), torch.float16),
                    MockTensor((64, 32), torch.float16),
                ]
            )

            result = choices._finalize_template_configs(
                template_choices, kernel_inputs, None, [], "mm"
            )

            # Should return original choices when no model
            self.assertEqual(len(result), 2)
            self.assertEqual(choices.stats["fallback_no_model"], 1)

    def test_finalize_template_configs_with_model(self):
        """Test _finalize_template_configs with model."""
        choices = DiodeInductorChoices(
            model_path=self.temp_model_path,
            device=self.device,
            top_k_configs=2,
            performance_threshold=1.2,
        )

        # Mock the model wrapper
        mock_wrapper = MockModelWrapper([0.1, 0.3, 0.2])  # Different predictions
        choices.model_wrapper = mock_wrapper

        # Create mock choices
        template = MockTemplate("test_template")
        config = MockConfig(
            BLOCK_M=64, BLOCK_N=32, BLOCK_K=16, GROUP_M=8, num_stages=2, num_warps=4
        )
        ktc1 = MockKernelTemplateChoice(template, config)
        ktc2 = MockKernelTemplateChoice(template, config)
        ktc3 = MockKernelTemplateChoice(template, config)

        template_choices = {"test": iter([ktc1, ktc2, ktc3])}
        kernel_inputs = MockKernelInputs(
            [MockTensor((128, 64), torch.float16), MockTensor((64, 32), torch.float16)]
        )

        result = choices._finalize_template_configs(
            template_choices, kernel_inputs, None, [], "mm"
        )

        # Should select top configs based on predictions
        self.assertLessEqual(len(result), 2)  # Top-k limit
        self.assertEqual(choices.stats["model_selections"], 1)

    def test_finalize_template_configs_empty_choices(self):
        """Test _finalize_template_configs with empty choices."""
        choices = DiodeInductorChoices(
            model_path=self.temp_model_path, device=self.device
        )

        # Mock the conversion pipeline to return empty list
        with mock.patch('torch_diode.integration.inductor_integration.convert_and_run_inference_pipeline') as mock_pipeline:
            mock_pipeline.return_value = []

            template_choices = {"test": iter([])}
            kernel_inputs = MockKernelInputs(
                [MockTensor((128, 64), torch.float16), MockTensor((64, 32), torch.float16)]
            )

            result = choices._finalize_template_configs(
                template_choices, kernel_inputs, None, [], "mm"
            )

            self.assertEqual(len(result), 0)
            # The current implementation returns empty results from the pipeline, not from a "no_choices" stat
            self.assertEqual(choices.stats["total_calls"], 1)

    def test_statistics_tracking(self):
        """Test statistics tracking functionality."""
        choices = DiodeInductorChoices(device=self.device)

        # Initial stats should be empty
        stats = choices.get_stats()
        self.assertEqual(len(stats), 0)

        # Add some stats
        choices.stats["test_stat"] = 5
        choices.stats["another_stat"] = 10

        stats = choices.get_stats()
        self.assertEqual(stats["test_stat"], 5)
        self.assertEqual(stats["another_stat"], 10)

        # Reset stats
        choices.reset_stats()
        stats = choices.get_stats()
        self.assertEqual(len(stats), 0)

    def test_fallback_behavior(self):
        """Test fallback behavior on errors."""
        # Mock _find_default_model to return None to ensure no model is loaded
        with mock.patch.object(
            DiodeInductorChoices, "_find_default_model", return_value=None
        ):
            choices = DiodeInductorChoices(device=self.device, enable_fallback=True)

            # Create mock choices that will cause errors
            template = MockTemplate("test_template")
            config = MockConfig()  # Empty config to cause conversion errors
            ktc = MockKernelTemplateChoice(template, config)

            template_choices = {"test": iter([ktc])}
            kernel_inputs = MockKernelInputs(
                [
                    MockTensor((128, 64), torch.float16),
                    MockTensor((64, 32), torch.float16),
                ]
            )

            result = choices._finalize_template_configs(
                template_choices, kernel_inputs, None, [], "mm"
            )

            # Should fallback gracefully
            self.assertEqual(len(result), 1)
            # Check that either fallback_no_model or fallback_config_conversion is incremented
            fallback_count = choices.stats.get(
                "fallback_no_model", 0
            ) + choices.stats.get("fallback_config_conversion", 0)
            self.assertGreater(fallback_count, 0)

    def test_no_fallback_behavior(self):
        """Test behavior when fallback is disabled."""
        # Mock _find_default_model to return None and test loading invalid model
        with mock.patch.object(
            DiodeInductorChoices, "_find_default_model", return_value=None
        ):
            choices = DiodeInductorChoices(
                model_path="/nonexistent/model.pt",
                device=self.device,
                enable_fallback=False,
            )

            # This should raise an exception when model loading fails
            with self.assertRaises(Exception):
                choices._load_model()


class TestFactoryFunctions(unittest.TestCase):
    """
    Tests for factory functions.
    """

    def test_create_diode_choices(self):
        """Test create_diode_choices factory function."""
        choices = create_diode_choices(
            device="cpu", top_k_configs=5, performance_threshold=1.5
        )

        self.assertIsInstance(choices, DiodeInductorChoices)
        self.assertEqual(choices.device, "cpu")
        self.assertEqual(choices.top_k_configs, 5)
        self.assertEqual(choices.performance_threshold, 1.5)

    @mock.patch("torch._inductor.virtualized.V")
    def test_install_diode_choices(self, mock_v):
        """Test install_diode_choices function."""
        # Mock the virtualized module
        mock_v.set_choices_handler = mock.MagicMock()

        result = install_diode_choices(device="cpu", top_k_configs=3)

        # Verify that set_choices_handler was called
        mock_v.set_choices_handler.assert_called_once()
        self.assertIsInstance(result, DiodeInductorChoices)

    def test_install_diode_choices_import_error(self):
        """Test install_diode_choices with import error."""
        # Create a mock module that raises ImportError when accessed
        with mock.patch.dict("sys.modules", {"torch._inductor.virtualized": None}):
            with mock.patch(
                "torch_diode.integration.inductor_integration.torch._inductor.virtualized.V",
                side_effect=ImportError("Module not found"),
            ):
                with self.assertRaises(ImportError):
                    install_diode_choices()


class TestErrorHandling(unittest.TestCase):
    """
    Tests for error handling scenarios.
    """

    def test_model_loading_error(self):
        """Test handling of model loading errors."""
        choices = DiodeInductorChoices(
            model_path="/nonexistent/model.pt", device="cpu", enable_fallback=True
        )

        # Should handle missing model gracefully
        self.assertFalse(choices._model_loaded)
        self.assertIsNone(choices.model_wrapper)

    def test_feature_extraction_error(self):
        """Test handling of feature extraction errors."""
        choices = DiodeInductorChoices(device="cpu")

        # Test with invalid kernel inputs
        result = choices._extract_features_from_kernel_inputs(None, "mm")
        self.assertIsNone(result)

    def test_config_conversion_error(self):
        """Test handling of config conversion errors."""
        choices = DiodeInductorChoices(device="cpu")

        # Test with invalid KTC
        result = choices._convert_ktc_to_config(None)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
