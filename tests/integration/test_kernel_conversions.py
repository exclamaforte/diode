"""
Tests for kernel conversion utilities.

This module contains tests for the kernel_conversions module that handles
conversion between PyTorch Inductor types and Diode types.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest
import torch

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Enable debug flags for testing
from torch_diode.utils.debug_config import set_debug_flag

set_debug_flag("ENABLE_TYPE_ASSERTS", True)

from torch_diode.integration.kernel_conversions import (
    convert_ktc_to_triton_config,
    create_features_and_run_inference,
    extract_mmshape_from_kernel_inputs,
    select_best_configs,
)
from torch_diode.types.matmul_types import MMShape, TritonGEMMConfig


class TestMMShapeExtraction:
    """Test MMShape extraction from kernel inputs."""

    def test_extract_mmshape_basic(self):
        """Test basic MMShape extraction."""
        # Mock MMKernelInputs
        mock_kernel_inputs = Mock()
        mock_kernel_inputs.__class__.__name__ = "MMKernelInputs"
        mock_kernel_inputs.mnk_symbolic.return_value = (64, 128, 32)
        mock_kernel_inputs.dtype.return_value = torch.float32

        # Mock output layout
        mock_layout = Mock()
        mock_layout.size = (64, 128)
        mock_layout.stride = (128, 1)
        mock_layout.dtype = torch.float32
        mock_kernel_inputs.output_layout.return_value = mock_layout

        # Mock input tensors
        mock_tensor_a = Mock()
        mock_tensor_a.get_size.return_value = (64, 32)
        mock_tensor_b = Mock()
        mock_tensor_b.get_size.return_value = (32, 128)
        mock_kernel_inputs.nodes.return_value = [mock_tensor_a, mock_tensor_b]

        with patch(
            "torch_diode.integration.kernel_conversions.isinstance"
        ) as mock_isinstance:
            mock_isinstance.return_value = True

            mmshape = extract_mmshape_from_kernel_inputs(mock_kernel_inputs, "mm")

            assert mmshape is not None
            assert mmshape.M == 64
            assert mmshape.N == 128
            assert mmshape.K == 32
            assert mmshape.B == 1  # No batch for mm
            assert mmshape.M_dtype == torch.float32

    def test_extract_mmshape_batch_operation(self):
        """Test MMShape extraction for batch operations."""
        # Mock MMKernelInputs for bmm
        mock_kernel_inputs = Mock()
        mock_kernel_inputs.__class__.__name__ = "MMKernelInputs"
        mock_kernel_inputs.mnk_symbolic.return_value = (64, 128, 32)
        mock_kernel_inputs.dtype.return_value = torch.float16

        # Mock output layout
        mock_layout = Mock()
        mock_layout.size = (8, 64, 128)
        mock_layout.stride = (8192, 128, 1)
        mock_layout.dtype = torch.float16
        mock_kernel_inputs.output_layout.return_value = mock_layout

        # Mock input tensors with batch dimension
        mock_tensor_a = Mock()
        mock_tensor_a.get_size.return_value = (8, 64, 32)
        mock_tensor_b = Mock()
        mock_tensor_b.get_size.return_value = (8, 32, 128)
        mock_kernel_inputs.nodes.return_value = [mock_tensor_a, mock_tensor_b]

        with patch(
            "torch_diode.integration.kernel_conversions.isinstance"
        ) as mock_isinstance:
            mock_isinstance.return_value = True

            mmshape = extract_mmshape_from_kernel_inputs(mock_kernel_inputs, "bmm")

            assert mmshape is not None
            assert mmshape.M == 64
            assert mmshape.N == 128
            assert mmshape.K == 32
            assert mmshape.B == 8  # Batch size for bmm
            assert mmshape.M_dtype == torch.float16

    def test_extract_mmshape_invalid_input(self):
        """Test MMShape extraction with invalid input."""
        # Test with non-MMKernelInputs
        mock_kernel_inputs = Mock()
        mock_kernel_inputs.__class__.__name__ = "SomeOtherInput"

        with patch(
            "torch_diode.integration.kernel_conversions.isinstance"
        ) as mock_isinstance:
            mock_isinstance.return_value = False

            mmshape = extract_mmshape_from_kernel_inputs(mock_kernel_inputs, "mm")

            assert mmshape is None


class TestKTCToTritonConfig:
    """Test conversion from KernelTemplateChoice to TritonGEMMConfig."""

    def test_convert_ktc_basic(self):
        """Test basic KTC to TritonGEMMConfig conversion."""

        # Create a more complete mock object that avoids Mock recursion issues
        class MockParams:
            def __init__(self):
                self.kwargs = {
                    "BLOCK_M": 64,
                    "BLOCK_N": 128,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                    "num_stages": 4,
                    "num_warps": 8,
                    "EVEN_K": True,
                    "ALLOW_TF32": False,
                }

        class MockTemplate:
            def __init__(self):
                self.uid = "aten_mm_default"

        class MockKTC:
            def __init__(self):
                self.template = MockTemplate()
                self.params = MockParams()

        mock_ktc = MockKTC()

        triton_config = convert_ktc_to_triton_config(mock_ktc)

        assert triton_config is not None
        assert triton_config.name == "aten_mm_default"
        assert triton_config.block_m == 64
        assert triton_config.block_n == 128
        assert triton_config.block_k == 32
        assert triton_config.group_m == 8
        assert triton_config.num_stages == 4
        assert triton_config.num_warps == 8
        assert triton_config.EVEN_K == True
        assert triton_config.ALLOW_TF32 == False

    def test_convert_ktc_with_all_kwargs_method(self):
        """Test KTC conversion when config has all_kwargs method."""

        # Create classes that avoid Mock issues
        class MockParamsWithAllKwargs:
            def all_kwargs(self):
                return {
                    "BLOCK_M": 32,
                    "BLOCK_N": 64,
                    "BLOCK_K": 16,
                    "num_stages": 2,
                    "num_warps": 4,
                }

        class MockTemplate:
            def __init__(self):
                self.uid = "test_template"

        class MockKTC:
            def __init__(self):
                self.template = MockTemplate()
                self.params = MockParamsWithAllKwargs()

        mock_ktc = MockKTC()

        triton_config = convert_ktc_to_triton_config(mock_ktc)

        assert triton_config is not None
        assert triton_config.block_m == 32
        assert triton_config.block_n == 64
        assert triton_config.block_k == 16
        assert triton_config.num_stages == 2
        assert triton_config.num_warps == 4

    def test_convert_ktc_with_defaults(self):
        """Test KTC conversion with missing parameters using defaults."""

        # Create classes that avoid Mock issues
        class MockParamsMinimal:
            def __init__(self):
                self.kwargs = {}  # Empty kwargs should use defaults

        class MockTemplate:
            def __init__(self):
                self.uid = "minimal_template"

        class MockKTC:
            def __init__(self):
                self.template = MockTemplate()
                self.params = MockParamsMinimal()

        mock_ktc = MockKTC()

        triton_config = convert_ktc_to_triton_config(mock_ktc)

        assert triton_config is not None
        assert triton_config.name == "minimal_template"
        # Check defaults are applied
        assert triton_config.block_m == 64
        assert triton_config.block_n == 64
        assert triton_config.block_k == 32
        assert triton_config.group_m == 8
        assert triton_config.num_stages == 4
        assert triton_config.num_warps == 4


class TestFeatureCreationAndInference:
    """Test feature creation and model inference."""

    def test_create_features_and_run_inference(self):
        """Test feature creation and inference pipeline."""
        # Create test MMShape
        mmshape = MMShape(
            B=1,
            M=64,
            M_dtype=torch.float32,
            N=128,
            K=32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1, 64, 128),
            out_stride=(128, 128, 1),
        )

        # Create test configs
        configs = [
            TritonGEMMConfig(
                name="config1",
                grid=1,
                block_m=64,
                block_n=64,
                block_k=32,
                group_m=8,
                num_stages=4,
                num_warps=4,
            ),
            TritonGEMMConfig(
                name="config2",
                grid=1,
                block_m=32,
                block_n=128,
                block_k=16,
                group_m=8,
                num_stages=2,
                num_warps=8,
            ),
        ]

        # Mock model with predict_from_features method
        mock_model = Mock()
        mock_predictions = torch.tensor([[1.5], [2.1]])
        mock_model.predict_from_features.return_value = mock_predictions

        # Mock the feature creation function
        with patch(
            "torch_diode.integration.kernel_conversions.create_features_from_mmshape_and_configs"
        ) as mock_create_features:
            mock_problem_features = torch.randn(2, 7)
            mock_config_features = torch.randn(2, 5)
            mock_create_features.return_value = (
                mock_problem_features,
                mock_config_features,
            )

            predictions = create_features_and_run_inference(
                mmshape, configs, mock_model, device="cpu"
            )

            assert len(predictions) == 2
            assert pytest.approx(predictions) == [1.5, 2.1]
            mock_model.predict_from_features.assert_called_once()

    def test_create_features_with_model_wrapper_interface(self):
        """Test inference with model wrapper interface."""
        mmshape = MMShape(
            B=1,
            M=64,
            M_dtype=torch.float32,
            N=128,
            K=32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1, 64, 128),
            out_stride=(128, 128, 1),
        )
        configs = [
            TritonGEMMConfig(
                name="test",
                grid=1,
                block_m=64,
                block_n=64,
                block_k=32,
                group_m=8,
                num_stages=4,
                num_warps=4,
            )
        ]

        # Mock model with predict method (no predict_from_features)
        mock_model = Mock()
        del mock_model.predict_from_features  # Remove the attribute
        mock_model.predict.return_value = torch.tensor([[0.8]])

        with patch(
            "torch_diode.integration.kernel_conversions.create_features_from_mmshape_and_configs"
        ) as mock_create_features:
            mock_create_features.return_value = (torch.randn(1, 7), torch.randn(1, 5))

            predictions = create_features_and_run_inference(
                mmshape, configs, mock_model
            )

            assert len(predictions) == 1
            assert pytest.approx(predictions) == [0.8]
            mock_model.predict.assert_called_once()


class TestConfigSelection:
    """Test configuration selection based on predictions."""

    def test_select_best_configs_basic(self):
        """Test basic config selection."""
        # Create mock choices
        choices = [Mock() for _ in range(4)]
        for i, choice in enumerate(choices):
            choice.name = f"choice_{i}"

        # Predictions (lower is better)
        predictions = [2.0, 1.0, 1.5, 3.0]  # choice_1 is best at 1.0

        selected = select_best_configs(choices, predictions, top_k=2)

        # Should select only choice_1 (1.0) since 1.5 > 1.0 * 1.1 = 1.1
        assert len(selected) == 1
        assert selected[0] == choices[1]  # Best prediction

    def test_select_best_configs_with_threshold(self):
        """Test config selection with performance threshold."""
        choices = [Mock() for _ in range(3)]
        predictions = [1.0, 1.05, 2.0]  # 1.05 is within 1.1x of 1.0, but 2.0 is not

        selected = select_best_configs(choices, predictions, top_k=3)

        # Should select first two (1.0 and 1.05), but not 2.0 (2.0 > 1.0 * 1.1)
        assert len(selected) == 2
        assert selected[0] == choices[0]
        assert selected[1] == choices[1]

    def test_select_best_configs_ensure_one_choice(self):
        """Test that at least one choice is always selected."""
        choices = [Mock()]
        predictions = [5.0]  # Even if prediction is bad

        selected = select_best_configs(choices, predictions, top_k=3)

        # Should still select the one choice available
        assert len(selected) == 1
        assert selected[0] == choices[0]

    def test_select_best_configs_mismatch_lengths(self):
        """Test handling of mismatched choices and predictions lengths."""
        choices = [Mock(), Mock()]
        predictions = [1.0]  # Different length

        selected = select_best_configs(choices, predictions, top_k=3)

        # Should return all original choices when lengths don't match
        assert len(selected) == 2
        assert selected == choices


class TestFullPipeline:
    """Test the complete conversion and inference pipeline."""


if __name__ == "__main__":
    pytest.main([__file__])
