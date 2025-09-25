"""
Tests for inductor integration module.

This module contains tests for the DiodeInductorChoices class that implements
_finalize_template_configs for model-based kernel selection.
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


from torch_diode.integration.inductor_integration import (
    DiodeInductorChoices,
    create_diode_choices,
    install_diode_choices,
)

# Import ExternKernelChoice for testing
try:
    from torch._inductor.select_algorithm import ExternKernelChoice
except ImportError:
    # Create a mock for testing
    class ExternKernelChoice:
        pass


class TestDiodeInductorChoices:
    """Test the DiodeInductorChoices class."""

    def test_init_with_model_path(self):
        """Test initialization with model path."""
        with patch.object(DiodeInductorChoices, "_load_model") as mock_load:
            mock_load.return_value = True

            choices = DiodeInductorChoices(
                model_path="/path/to/model.pt",
                device="cuda",
                top_k_configs=5,
                performance_threshold=1.2,
            )

            assert choices.model_path == "/path/to/model.pt"
            assert choices.device == "cuda"
            assert choices.top_k_configs == 5
            assert choices.performance_threshold == 1.2
            assert not choices.enable_fallback

    def test_init_without_model_path(self):
        """Test initialization without model path."""
        with patch.object(DiodeInductorChoices, "_find_default_model") as mock_find:
            mock_find.return_value = None

            choices = DiodeInductorChoices(device="cpu")

            assert choices.device == "cpu"
            assert choices.model_path is None
            assert not choices._model_loaded

    def test_load_model_success(self):
        """Test successful model loading."""
        with patch(
            "torch_diode.integration.inductor_integration.ModelWrapper"
        ) as mock_wrapper_class:
            with patch.object(
                DiodeInductorChoices, "_find_default_model", return_value=None
            ):
                mock_wrapper = Mock()
                mock_wrapper_class.return_value = mock_wrapper

                choices = DiodeInductorChoices()
                choices.model_path = "/path/to/model.pt"
                choices._model_loaded = False  # Reset since init might have set it

                result = choices._load_model()

                assert result
                assert choices._model_loaded
                assert choices.model_wrapper == mock_wrapper
                mock_wrapper_class.assert_called_with(
                    model_path="/path/to/model.pt",
                    device=choices.device,
                    compile_model=False,
                )

    def test_load_model_failure(self):
        """Test model loading failure with fallback enabled."""
        with patch(
            "torch_diode.integration.inductor_integration.ModelWrapper"
        ) as mock_wrapper_class:
            mock_wrapper_class.side_effect = Exception("Load failed")

            choices = DiodeInductorChoices(
                model_path="/path/to/model.pt",
                enable_fallback=True,
            )

            assert not choices._model_loaded
            assert choices.model_wrapper is None

    def test_load_model_failure_no_fallback(self):
        """Test model loading failure without fallback."""
        with patch(
            "torch_diode.integration.inductor_integration.ModelWrapper"
        ) as mock_wrapper_class:
            with patch("os.path.exists", return_value=True):  # Mock file exists
                mock_wrapper_class.side_effect = Exception("Load failed")

                with pytest.raises(Exception, match="Load failed"):
                    DiodeInductorChoices(
                        model_path="/path/to/model.pt",
                        enable_fallback=False,
                    )

    def test_finalize_template_configs_no_model(self):
        """Test _finalize_template_configs when no model is loaded."""
        choices = DiodeInductorChoices(enable_fallback=True)  # Enable fallback for test
        choices._model_loaded = False
        choices.model_wrapper = None

        # Mock template choices
        mock_choice1, mock_choice2 = Mock(), Mock()
        template_choices = {
            "template1": iter([mock_choice1]),
            "template2": iter([mock_choice2]),
        }

        result = choices._finalize_template_configs(
            template_choices=template_choices,
            kernel_inputs=Mock(),
            templates=[Mock()],
            op_name="mm",
        )

        # Should fall back to flattened choices
        assert len(result) == 2
        assert mock_choice1 in result
        assert mock_choice2 in result
        assert choices.stats["fallback_no_model"] == 1

    def test_finalize_template_configs_success(self):
        """Test successful _finalize_template_configs with model."""
        mock_wrapper = Mock()
        choices = DiodeInductorChoices(model_path="/path/to/model.pt")
        choices._model_loaded = True
        choices.model_wrapper = mock_wrapper

        # Mock the pipeline function
        mock_selected_choices = [Mock(), Mock()]
        with patch(
            "torch_diode.integration.inductor_integration.convert_and_run_inference_pipeline"
        ) as mock_pipeline:
            with patch.object(
                choices, "_create_unified_predictor"
            ) as mock_create_predictor:
                mock_predictor = Mock()
                mock_create_predictor.return_value = mock_predictor
                mock_pipeline.return_value = mock_selected_choices

                template_choices = {"template1": iter([Mock(), Mock(), Mock()])}
                kernel_inputs = Mock()
                templates = [Mock()]

                result = choices._finalize_template_configs(
                    template_choices=template_choices,
                    kernel_inputs=kernel_inputs,
                    templates=templates,
                    op_name="mm",
                    kwarg_overrides={"template1": {"key": "value"}},
                )

                assert result == mock_selected_choices
                assert choices.stats["model_selections"] == 1

                mock_pipeline.assert_called_once_with(
                    kernel_inputs=kernel_inputs,
                    templates=templates,
                    op_name="mm",
                    model=mock_predictor,
                    device=choices.device,
                    top_k=choices.top_k_configs,
                    performance_threshold=choices.performance_threshold,
                    kwarg_overrides={"template1": {"key": "value"}},
                )

    def test_finalize_template_configs_pipeline_failure(self):
        """Test _finalize_template_configs when pipeline returns no choices."""
        mock_wrapper = Mock()
        choices = DiodeInductorChoices(enable_fallback=True)
        choices._model_loaded = True
        choices.model_wrapper = mock_wrapper

        # Mock pipeline to return empty list
        with patch(
            "torch_diode.integration.inductor_integration.convert_and_run_inference_pipeline"
        ) as mock_pipeline:
            with patch.object(
                choices, "_create_unified_predictor"
            ) as mock_create_predictor:
                mock_create_predictor.return_value = Mock()
                mock_pipeline.return_value = []

                mock_choice = Mock()
                template_choices = {"template1": iter([mock_choice])}

                result = choices._finalize_template_configs(
                    template_choices=template_choices,
                    kernel_inputs=Mock(),
                    templates=[Mock()],
                    op_name="mm",
                )

                # Should fall back to original choices
                assert len(result) == 1
                assert mock_choice in result

    def test_finalize_template_configs_exception_handling(self):
        """Test _finalize_template_configs exception handling."""
        mock_wrapper = Mock()
        choices = DiodeInductorChoices(enable_fallback=True)
        choices._model_loaded = True
        choices.model_wrapper = mock_wrapper

        # Mock pipeline to raise exception
        with patch(
            "torch_diode.integration.inductor_integration.convert_and_run_inference_pipeline"
        ) as mock_pipeline:
            mock_pipeline.side_effect = Exception("Pipeline error")

            mock_choice = Mock()
            template_choices = {"template1": iter([mock_choice])}

            result = choices._finalize_template_configs(
                template_choices=template_choices,
                kernel_inputs=Mock(),
                templates=[Mock()],
                op_name="mm",
            )

            # Should fall back to original choices
            assert len(result) == 1
            assert mock_choice in result
            assert choices.stats["fallback_error"] == 1

    def test_finalize_template_configs_exception_no_fallback(self):
        """Test _finalize_template_configs exception handling without fallback."""
        mock_wrapper = Mock()
        choices = DiodeInductorChoices(enable_fallback=False)
        choices._model_loaded = True
        choices.model_wrapper = mock_wrapper

        # Mock pipeline to raise exception
        with patch(
            "torch_diode.integration.inductor_integration.convert_and_run_inference_pipeline"
        ) as mock_pipeline:
            mock_pipeline.side_effect = Exception("Pipeline error")

            template_choices = {"template1": iter([Mock()])}

            with pytest.raises(Exception, match="Pipeline error"):
                choices._finalize_template_configs(
                    template_choices=template_choices,
                    kernel_inputs=Mock(),
                    templates=[Mock()],
                    op_name="mm",
                )

    def test_finalize_template_configs_no_model_no_fallback(self):
        """Test _finalize_template_configs when no model is loaded and fallback disabled."""
        choices = DiodeInductorChoices(enable_fallback=False)  # Disable fallback
        choices._model_loaded = False
        choices.model_wrapper = None

        # Mock template choices
        template_choices = {
            "template1": iter([Mock()]),
        }

        with pytest.raises(
            RuntimeError, match="No model available and fallback is disabled"
        ):
            choices._finalize_template_configs(
                template_choices=template_choices,
                kernel_inputs=Mock(),
                templates=[Mock()],
                op_name="mm",
            )

    def test_create_unified_predictor_success(self):
        """Test successful creation of unified predictor."""
        mock_wrapper = Mock()
        choices = DiodeInductorChoices()
        choices._model_loaded = True
        choices.model_wrapper = mock_wrapper

        predictor = choices._create_unified_predictor()

        assert predictor is not None
        assert hasattr(predictor, "predict_from_features")

    def test_create_unified_predictor_no_model(self):
        """Test unified predictor creation when no model is loaded."""
        choices = DiodeInductorChoices()
        choices._model_loaded = False
        choices.model_wrapper = None

        predictor = choices._create_unified_predictor()

        assert predictor is None

    def test_stats_tracking(self):
        """Test statistics tracking."""
        with patch.object(
            DiodeInductorChoices, "_find_default_model", return_value=None
        ):
            choices = DiodeInductorChoices()

            # Initial stats should be empty dict, but get_stats returns dict
            stats = choices.get_stats()
            assert stats.get("total_calls", 0) == 0
            assert stats.get("model_selections", 0) == 0
            assert stats.get("fallback_no_model", 0) == 0

            # Manually increment some stats
            choices.stats["total_calls"] += 1
            choices.stats["model_selections"] += 1

            stats = choices.get_stats()
            assert stats["total_calls"] == 1
            assert stats["model_selections"] == 1

            # Reset stats
            choices.reset_stats()
            stats = choices.get_stats()
            assert stats.get("total_calls", 0) == 0
            assert stats.get("model_selections", 0) == 0


class TestCreateDiodeChoices:
    """Test the factory function for creating DiodeInductorChoices."""

    def test_create_diode_choices_with_path(self):
        """Test factory creation from model path."""
        with patch.object(
            DiodeInductorChoices, "__init__", return_value=None
        ) as mock_init:
            create_diode_choices(
                model_path="/path/to/model.pt",
                device="cuda",
                top_k_configs=5,
            )

            mock_init.assert_called_once_with(
                model_path="/path/to/model.pt",
                device="cuda",
                top_k_configs=5,
            )

    def test_create_diode_choices_defaults(self):
        """Test factory creation with defaults."""
        with patch.object(
            DiodeInductorChoices, "__init__", return_value=None
        ) as mock_init:
            create_diode_choices(device="cpu")

            mock_init.assert_called_once_with(
                model_path=None,
                device="cpu",
            )


class TestExhaustiveConfigGeneration:
    """Test the exhaustive config generation functionality."""

    def test_generate_exhaustive_configs_for_templates_success(self):
        """Test successful exhaustive config generation."""
        choices = DiodeInductorChoices(enable_fallback=True)

        # Mock the required modules and dependencies from torch._inductor where they're imported
        with patch(
            "torch._inductor.template_heuristics.get_template_heuristic"
        ) as mock_get_heuristic:
            with patch(
                "torch._inductor.kernel_template_choice.make_ktc_generator"
            ) as mock_make_ktc:
                with patch(
                    "torch._inductor.kernel_inputs.MMKernelInputs"
                ):
                    # Setup mock template and kernel inputs
                    mock_template = Mock()
                    mock_template.uid = "mm"
                    mock_template.name = "mm"

                    mock_kernel_inputs = Mock()
                    mock_kernel_inputs.device_type = "cuda"
                    mock_kernel_inputs.mnk_symbolic.return_value = (1024, 1024, 1024)
                    mock_kernel_inputs.dtype.return_value = torch.float16
                    mock_kernel_inputs.output_layout.return_value = Mock()

                    # Setup mock heuristic
                    mock_heuristic = Mock()
                    mock_heuristic.get_exhaustive_mm_configs.return_value = (
                        lambda m, n, k, dtype_size, op_name: [
                            Mock(
                                BLOCK_M=64,
                                BLOCK_N=64,
                                BLOCK_K=32,
                                num_stages=2,
                                num_warps=4,
                            ),
                            Mock(
                                BLOCK_M=128,
                                BLOCK_N=128,
                                BLOCK_K=32,
                                num_stages=3,
                                num_warps=8,
                            ),
                            # Add more mock configs to simulate exhaustive generation
                        ]
                        * 500
                    )  # This should give us ~1000 configs
                    mock_heuristic.adjust_kernel_inputs.return_value = Mock()

                    mock_get_heuristic.return_value = mock_heuristic

                    # Setup mock KTC generator to return multiple choices
                    mock_ktc_choices = [
                        Mock(template=mock_template, config=Mock()) for _ in range(1000)
                    ]
                    mock_make_ktc.return_value = mock_ktc_choices

                    # Make kernel_inputs pass isinstance check by setting it as the correct type
                    type(mock_kernel_inputs).__name__ = "MMKernelInputs"

                    # Call the method
                    result = choices._generate_exhaustive_configs_for_templates(
                        templates=[mock_template],
                        kernel_inputs=mock_kernel_inputs,
                        op_name="mm",
                        kwarg_overrides={},
                    )

                    # Verify we got a substantial number of configs (around 1000)
                    assert len(result) >= 900, (
                        f"Expected ~1000 configs, got {len(result)}"
                    )
                    assert len(result) <= 1100, (
                        f"Expected ~1000 configs, got {len(result)}"
                    )

                    # Verify the heuristic was called correctly
                    mock_get_heuristic.assert_called_once_with("mm", "cuda", "mm")
                    mock_heuristic.get_exhaustive_mm_configs.assert_called_once()
                    mock_heuristic.adjust_kernel_inputs.assert_called_once_with(
                        mock_kernel_inputs, "mm"
                    )

                    # Verify make_ktc_generator was called
                    mock_make_ktc.assert_called_once()

    def test_generate_exhaustive_configs_for_templates_extern_kernel_skip(self):
        """Test that ExternKernelChoice templates are skipped."""
        choices = DiodeInductorChoices(enable_fallback=True)

        # Create mock ExternKernelChoice
        mock_extern_template = Mock(spec=ExternKernelChoice)
        mock_extern_template.uid = "extern_mm"

        mock_kernel_inputs = Mock()
        mock_kernel_inputs.device_type = "cuda"

        result = choices._generate_exhaustive_configs_for_templates(
            templates=[mock_extern_template],
            kernel_inputs=mock_kernel_inputs,
            op_name="mm",
            kwarg_overrides={},
        )

        # Should return empty list since ExternKernelChoice is skipped
        assert result == []

    def test_generate_exhaustive_configs_for_templates_no_exhaustive_method(self):
        """Test behavior when heuristic doesn't have get_exhaustive_mm_configs method."""
        choices = DiodeInductorChoices(enable_fallback=True)

        with patch(
            "torch._inductor.template_heuristics.get_template_heuristic"
        ) as mock_get_heuristic:
            mock_template = Mock()
            mock_template.uid = "mm"

            mock_kernel_inputs = Mock()
            mock_kernel_inputs.device_type = "cuda"

            # Setup mock heuristic without get_exhaustive_mm_configs method
            mock_heuristic = Mock()
            del mock_heuristic.get_exhaustive_mm_configs  # Ensure method doesn't exist
            mock_get_heuristic.return_value = mock_heuristic

            result = choices._generate_exhaustive_configs_for_templates(
                templates=[mock_template],
                kernel_inputs=mock_kernel_inputs,
                op_name="mm",
                kwarg_overrides={},
            )

            # Should return empty list
            assert result == []

    def test_generate_exhaustive_configs_for_templates_error_handling(self):
        """Test error handling in exhaustive config generation."""
        choices = DiodeInductorChoices(enable_fallback=True)

        with patch(
            "torch._inductor.template_heuristics.get_template_heuristic"
        ) as mock_get_heuristic:
            mock_get_heuristic.side_effect = Exception("Heuristic error")

            mock_template = Mock()
            mock_template.uid = "mm"

            mock_kernel_inputs = Mock()
            mock_kernel_inputs.device_type = "cuda"

            result = choices._generate_exhaustive_configs_for_templates(
                templates=[mock_template],
                kernel_inputs=mock_kernel_inputs,
                op_name="mm",
                kwarg_overrides={},
            )

            # Should return empty list on error
            assert result == []

    def test_generate_exhaustive_configs_no_device_type(self):
        """Test that assertion fails when device_type is None."""
        choices = DiodeInductorChoices(enable_fallback=True)

        mock_template = Mock()
        mock_template.uid = "mm"

        mock_kernel_inputs = Mock()
        mock_kernel_inputs.device_type = None  # This should cause assertion failure

        with pytest.raises(AssertionError, match="requires a valid device type"):
            choices._generate_exhaustive_configs_for_templates(
                templates=[mock_template],
                kernel_inputs=mock_kernel_inputs,
                op_name="mm",
                kwarg_overrides={},
            )

    def test_finalize_template_configs_non_mm_operation(self):
        """Test that non-mm operations fall back to superclass."""
        choices = DiodeInductorChoices(enable_fallback=True)

        # Mock superclass method
        mock_super_result = [Mock(), Mock()]
        with patch.object(
            DiodeInductorChoices.__bases__[0],
            "_finalize_template_configs",
            return_value=mock_super_result,
        ) as mock_super:
            template_choices = {"template1": iter([Mock()])}
            kernel_inputs = Mock()
            templates = [Mock()]

            result = choices._finalize_template_configs(
                template_choices=template_choices,
                kernel_inputs=kernel_inputs,
                templates=templates,
                op_name="addmm",  # Non-mm operation
                kwarg_overrides={},
            )

            # Should call superclass method
            mock_super.assert_called_once_with(
                template_choices, kernel_inputs, templates, "addmm", {}
            )
            assert result == mock_super_result
            assert choices.stats["fallback_non_mm_op"] == 1

    def test_finalize_template_configs_base_and_nonbase_separation(self):
        """Test that base and non-base templates are separated correctly."""
        mock_wrapper = Mock()
        choices = DiodeInductorChoices(enable_fallback=True)
        choices._model_loaded = True
        choices.model_wrapper = mock_wrapper

        # Create mock templates
        mock_base_template = Mock()
        mock_base_template.name = "mm"
        mock_base_template.uid = "mm"

        mock_nonbase_template1 = Mock()
        mock_nonbase_template1.name = "extern_mm"
        mock_nonbase_template1.uid = "extern_mm"

        mock_nonbase_template2 = Mock()
        mock_nonbase_template2.name = "mm_persistent_tma"
        mock_nonbase_template2.uid = "mm_persistent_tma"

        templates = [mock_base_template, mock_nonbase_template1, mock_nonbase_template2]

        # Mock the methods
        with patch.object(
            choices,
            "_generate_exhaustive_configs_for_templates",
            return_value=[Mock(), Mock()],
        ) as mock_gen:
            with patch.object(
                choices, "_run_model_inference_on_configs", return_value=[Mock()]
            ) as mock_inference:
                with patch.object(
                    DiodeInductorChoices.__bases__[0],
                    "_finalize_template_configs",
                    return_value=[Mock(), Mock()],
                ) as mock_super:
                    template_choices = {
                        "mm": iter([Mock()]),
                        "extern_mm": iter([Mock()]),
                        "mm_persistent_tma": iter([Mock()]),
                    }
                    kernel_inputs = Mock()

                    result = choices._finalize_template_configs(
                        template_choices=template_choices,
                        kernel_inputs=kernel_inputs,
                        templates=templates,
                        op_name="mm",
                        kwarg_overrides={},
                    )

                    # Verify base template was processed through model
                    mock_gen.assert_called_once_with(
                        [mock_base_template], kernel_inputs, "mm", {}
                    )
                    mock_inference.assert_called_once()

                    # Verify non-base templates were passed through superclass
                    mock_super.assert_called_once()

                    # Should have results from both paths
                    assert (
                        len(result) >= 2
                    )  # At least one from model + two from superclass


class TestInstallDiodeChoices:
    """Test the installation function."""

    @patch("torch._inductor.virtualized.V")
    def test_install_with_model_path(self, mock_v):
        """Test installation with model path."""
        with patch(
            "torch_diode.integration.inductor_integration.create_diode_choices"
        ) as mock_create:
            mock_choices = Mock()
            mock_create.return_value = mock_choices

            result = install_diode_choices(
                model_path="/path/to/model.pt",
                device="cpu",
                top_k_configs=5,
            )

            mock_create.assert_called_once_with(
                model_path="/path/to/model.pt",
                device="cpu",
                top_k_configs=5,
            )
            # Verify that set_choices_handler was called correctly
            mock_v.set_choices_handler.assert_called_once_with(mock_choices)
            assert result == mock_choices

    @patch("torch._inductor.virtualized.V")
    def test_install_with_defaults(self, mock_v):
        """Test installation with default settings."""
        with patch(
            "torch_diode.integration.inductor_integration.create_diode_choices"
        ) as mock_create:
            mock_choices = Mock()
            mock_create.return_value = mock_choices

            result = install_diode_choices(device="cuda")

            mock_create.assert_called_once_with(
                model_path=None,
                device="cuda",
            )
            # Verify that set_choices_handler was called correctly
            mock_v.set_choices_handler.assert_called_once_with(mock_choices)
            assert result == mock_choices

    def test_install_import_error(self):
        """Test installation when PyTorch Inductor is not available."""
        # Mock the entire torch._inductor.virtualized module to not exist
        import sys

        original_modules = sys.modules.copy()

        # Remove torch._inductor.virtualized from sys.modules if it exists
        if "torch._inductor.virtualized" in sys.modules:
            del sys.modules["torch._inductor.virtualized"]

        # Mock import to raise ImportError for this specific module
        def mock_import(name, *args, **kwargs):
            if name == "torch._inductor.virtualized":
                raise ImportError("No module named 'torch._inductor.virtualized'")
            return __import__(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                with pytest.raises(ImportError):
                    install_diode_choices()
        finally:
            # Restore original sys.modules
            sys.modules.clear()
            sys.modules.update(original_modules)

    @patch("torch._inductor.virtualized.V")
    def test_install_general_error(self, mock_v):
        """Test installation with general error."""
        # Mock create_diode_choices to raise an exception during creation
        with patch(
            "torch_diode.integration.inductor_integration.create_diode_choices"
        ) as mock_create:
            mock_create.side_effect = Exception("General error")

            with pytest.raises(Exception, match="General error"):
                install_diode_choices()


if __name__ == "__main__":
    pytest.main([__file__])
