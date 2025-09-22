"""
Integration with PyTorch Inductor's choices.py for model-based config selection.

This module provides functionality to integrate trained Diode models with PyTorch Inductor's
_finalize_template_configs method to enable model-based selection of optimal kernel configurations.
"""

import logging
import os
from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Union

import torch
import triton

# Import PyTorch Inductor types (these would be available when integrated)
try:
    from torch._inductor.choices import InductorChoices
    from torch._inductor.codegen.common import KernelTemplate
    from torch._inductor.ir import Layout
    from torch._inductor.kernel_inputs import KernelInputs
    from torch._inductor.kernel_template_choice import KernelTemplateChoice
    from torch._inductor.select_algorithm import ExternKernelChoice
except ImportError:
    # For development/testing when not in inductor environment
    KernelInputs = Any
    KernelTemplateChoice = Any
    KernelTemplate = Any
    ExternKernelChoice = Any
    Layout = Any
    InductorChoices = Any

# Import Diode components
from ..model.matmul_inference import UnifiedMatmulPredictor
from ..model.model_wrapper import ModelWrapper
from ..types.matmul_types import MMShape, TritonGEMMConfig
from ..utils.debug_config import type_assert
from ..utils.feature_extraction import (
    extract_config_features_compat,
    extract_problem_features_compat,
)
from .kernel_conversions import generate_exhaustive_triton_template_configs
from .kernel_conversions import (
    convert_triton_config_to_triton_gemm_config,
    create_features_and_run_inference,
    extract_mmshape_from_kernel_inputs,
    select_best_configs,
    convert_triton_configs_to_ktc,
)


logger = logging.getLogger(__name__)


class DiodeInductorChoices(InductorChoices):
    """
    Extended InductorChoices class that uses Diode models for config selection.

    This class overrides the _finalize_template_configs method to run model inference
    on available configurations and select the best ones based on predicted timing.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        top_k_configs: int = 5,
        enable_fallback: bool = False,
        performance_threshold: float = 1.1,  # Allow configs within 10% of best prediction
        **kwargs,
    ):
        """
        Initialize the DiodeInductorChoices.

        Args:
            model_path: Path to the trained Diode model. If None, will try to find a default model.
            device: Device to run the model on
            top_k_configs: Maximum number of configurations to return after model filtering
            enable_fallback: Whether to fall back to default behavior if model fails
            performance_threshold: Ratio threshold for including configs (1.0 = only best, 1.1 = within 10% of best)
            **kwargs: Additional arguments passed to parent class
        """
        type_assert(
            model_path is None or isinstance(model_path, str),
            f"Expected str or None for model_path, got {type(model_path)}",
        )
        type_assert(
            isinstance(device, str), f"Expected str for device, got {type(device)}"
        )
        type_assert(
            isinstance(top_k_configs, int),
            f"Expected int for top_k_configs, got {type(top_k_configs)}",
        )
        type_assert(
            top_k_configs > 0, f"top_k_configs must be positive, got {top_k_configs}"
        )
        type_assert(
            isinstance(enable_fallback, bool),
            f"Expected bool for enable_fallback, got {type(enable_fallback)}",
        )
        type_assert(
            isinstance(performance_threshold, (int, float)),
            f"Expected numeric type for performance_threshold, got {type(performance_threshold)}",
        )
        type_assert(
            performance_threshold > 0,
            f"performance_threshold must be positive, got {performance_threshold}",
        )
        super().__init__(**kwargs)

        self.model_path = model_path or self._find_default_model()
        self.device = device
        self.top_k_configs = top_k_configs
        self.enable_fallback = enable_fallback
        self.performance_threshold = performance_threshold
        self.model_wrapper = None
        self._model_loaded = False

        # Statistics for monitoring
        self.stats = defaultdict(int)
        # Pre-initialize known statistics keys for better reporting
        self.stats.update(
            {
                "total_calls": 0,
                "fallback_non_mm_op": 0,
                "fallback_no_model": 0,
                "fallback_empty_result": 0,
                "fallback_error": 0,
                "model_selections": 0,
                "configs_filtered": 0,
            }
        )

        # Load model if path is provided
        if self.model_path and os.path.exists(self.model_path):
            self._load_model()

    def _find_default_model(self) -> Optional[str]:
        """Try to find a default model in common locations."""
        possible_paths = [
            "matmul_model_exhaustive.pt",
            "trained_models/matmul_model_exhaustive.pt",
            os.path.expanduser("~/diode/matmul_model_exhaustive.pt"),
            os.path.join(
                os.path.dirname(__file__), "..", "..", "matmul_model_exhaustive.pt"
            ),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found default Diode model at: {path}")
                return path

        logger.warning(
            "No default Diode model found. Model-based selection will be disabled."
        )
        return None

    def _load_model(self) -> bool:
        """Load the Diode model for inference."""
        if self._model_loaded or not self.model_path:
            return self._model_loaded

        try:
            logger.info(f"Loading Diode model from: {self.model_path}")
            self.model_wrapper = ModelWrapper(
                model_path=self.model_path,
                device=self.device,
                compile_model=False,  # Disable compilation to avoid dynamic shape issues
            )
            self._model_loaded = True
            logger.info("Diode model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load Diode model: {e}")
            if not self.enable_fallback:
                raise
            return False

    def _finalize_template_configs(
        self,
        template_choices: Dict[str, Generator[KernelTemplateChoice, None, None]],
        kernel_inputs: KernelInputs,
        templates: List[Union[KernelTemplate, ExternKernelChoice]],
        op_name: str,
        kwarg_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[KernelTemplateChoice]:
        """
        Override _finalize_template_configs to use model-based selection.

        This method implements the workflow described by the user:
        1. Only use model for "mm" operations, fall back to superclass for others
        2. For "mm", separate base templates from non-base templates
        3. Use model inference only on base templates (mm_template, aten_mm)
        4. Pass through non-base templates (persistent TMA, Blackwell TMA, decompose K, etc.)

        Args:
            template_choices: Dictionary mapping template UIDs to generators (IGNORED)
            kernel_inputs: MMKernelInputs containing input tensor nodes and matrix indices
            templates: List of template objects (KernelTemplate or ExternKernelChoice) in use
            op_name: Operation name (e.g., "bmm", "baddbmm", "addmm")
            kwarg_overrides: Optional dict of kwargs to override for each template heuristic

        Returns:
            Filtered list of KernelTemplateChoice objects based on model predictions
        """
        # Track statistics
        self.stats["total_calls"] += 1

        # V1 model is only trained on "mm" operations
        if op_name != "mm":
            self.stats["fallback_non_mm_op"] += 1
            logger.debug(f"Op {op_name} is not 'mm', falling back to superclass")
            return super()._finalize_template_configs(
                template_choices, kernel_inputs, templates, op_name, kwarg_overrides
            )

        # If no model is loaded or available, fall back to default behavior
        if not self._model_loaded or not self.model_wrapper:
            self.stats["fallback_no_model"] += 1
            logger.debug("No model available, using default config selection")
            if self.enable_fallback:
                return super()._finalize_template_configs(
                    template_choices, kernel_inputs, templates, op_name, kwarg_overrides
                )
            else:
                raise RuntimeError("No model available and fallback is disabled")

        # Separate base templates from non-base templates
        base_templates = []
        non_base_templates = []

        # Templates that should go through model inference (base templates)
        base_template_names = {
            "triton::mm",
        }

        for template in templates:
            template_uid = getattr(template, "uid", "unknown") or getattr(template, "name", None)
            if template_uid in base_template_names:
                base_templates.append(template)
            else:
                # Everything else goes to non-base (including extern_mm, persistent TMA, etc.)
                non_base_templates.append(template)

        selected_choices = []

        # Process base templates through model inference if any exist
        if base_templates:
            logger.debug(
                f"Processing {len(base_templates)} base templates through model"
            )

            # Generate exhaustive configs from scratch for base templates
            base_configs = generate_exhaustive_triton_template_configs(
                kernel_inputs, base_templates, op_name, kwarg_overrides
            )

            if base_configs:
                # Use model inference directly on the generated configs
                final_triton_configs = self._run_model_inference_on_configs(
                    configs=base_configs,
                    base_templates=base_templates,
                    kernel_inputs=kernel_inputs,
                    op_name=op_name,
                )
                # TODO(gabe) This smashes the configs into a single list, solve mapping of this list
                # to template later.
                model_selected_choices = convert_triton_configs_to_ktc(
                    final_triton_configs,
                    base_templates[0],
                    kernel_inputs,
                    op_name,
                )
                for template_generator in model_selected_choices:
                    templates = list(template_generator)
                    if len(templates) == 0:
                        logger.error("Model returned no configs")
                    selected_choices.extend(templates)

        # Pass through non-base templates without model filtering
        if non_base_templates:
            logger.debug(
                f"Passing through {len(non_base_templates)} non-base templates"
            )
            # Use the superclass method to generate configs for non-base templates
            non_base_template_choices = {}
            for template in non_base_templates:
                template_uid = getattr(template, "uid", str(template))
                if template_uid in template_choices:
                    non_base_template_choices[template_uid] = template_choices[
                        template_uid
                    ]

            if non_base_template_choices:
                non_base_selected = super()._finalize_template_configs(
                    non_base_template_choices,
                    kernel_inputs,
                    non_base_templates,
                    op_name,
                    kwarg_overrides,
                )
                selected_choices.extend(non_base_selected)

        # If pipeline returns no choices, fall back to original choices if fallback enabled
        if not selected_choices and self.enable_fallback:
            self.stats["fallback_empty_result"] += 1
            logger.debug(
                "Pipeline returned no configs, falling back to original choices"
            )
            return super()._finalize_template_configs(
                template_choices, kernel_inputs, templates, op_name, kwarg_overrides
            )

        # Log the number of configs returned vs expected
        num_returned = len(selected_choices)
        if num_returned > self.top_k_configs:
            logger.info(
                f"Got {num_returned} configs, top_k_configs is {self.top_k_configs}!"
            )
            # Truncate to top_k_configs
            selected_choices = selected_choices[: self.top_k_configs]
            logger.info(f"Truncated to {len(selected_choices)} configs")

        self.stats["model_selections"] += 1
        try:
            original_count = sum(
                1 for choices in template_choices.values() for _ in choices
            )
        except:
            # If we can't count original choices, just use the current count
            original_count = len(base_configs) if "base_configs" in locals() else 0

        self.stats["configs_filtered"] += max(
            0, original_count - len(selected_choices)
        )

        logger.info(
            f"   - _finalize_template_configs returned {len(selected_choices)} configs (expected ≤ {self.top_k_configs})"
        )
        logger.debug(
            f"Model selected {len(selected_choices)} configs for {op_name} "
            f"({len(base_templates)} base, {len(non_base_templates)} non-base templates)"
        )
        return selected_choices

    def _generate_exhaustive_configs_for_templates(
        self,
        templates: List[Union[KernelTemplate, ExternKernelChoice]],
        kernel_inputs: KernelInputs,
        op_name: str,
        kwarg_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[KernelTemplateChoice]:
        """
        Generate exhaustive configs from scratch for the given templates.

        This ignores any input template_choices and generates exhaustive configs
        by calling the working implementation from kernel_conversions module.

        Args:
            templates: List of template objects (KernelTemplate or ExternKernelChoice) in use
            kernel_inputs: MMKernelInputs containing input tensor nodes and matrix indices
            op_name: Operation name (e.g., "mm")
            kwarg_overrides: Optional dict of kwargs to override for each template heuristic

        Returns:
            List of KernelTemplateChoice objects with exhaustive configs
        """
        try:

            logger.debug(
                f"Generating exhaustive configs for {len(templates)} templates"
            )
            choices = generate_exhaustive_triton_template_configs(
                kernel_inputs, templates, op_name, kwarg_overrides
            )

            logger.debug(f"Generated {len(choices)} exhaustive configs")
            return choices

        except Exception as e:
            logger.error(f"Error generating exhaustive configs: {e}")
            return []

    def _run_model_inference_on_configs(
        self,
        configs,
        base_templates: List[List[Union[KernelTemplate, ExternKernelChoice]]],
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> List[KernelTemplateChoice]:
        """
        Run model inference on pre-generated configs and return the best ones.

        Args:
            configs: List of pre-generated KernelTemplateChoice objects
            kernel_inputs: MMKernelInputs containing input tensor nodes and matrix indices
            op_name: Operation name (e.g., "mm")

        Returns:
            Filtered list of KernelTemplateChoice objects based on model predictions
        """
        if not configs:
            logger.warning("No configs provided for model inference")
            return []

        # Extract MMShape from kernel inputs
        mmshape = extract_mmshape_from_kernel_inputs(kernel_inputs, op_name)
        if mmshape is None:
            logger.warning("Could not extract MMShape, returning original configs")
            return configs

        # Convert configs to TritonGEMMConfig objects
        triton_configs = []
        valid_choices = []

        for template_generator, template in zip(configs, base_templates):
            for choice in template_generator:
                config = convert_triton_config_to_triton_gemm_config(choice, template.uid)
                if config is not None:
                    triton_configs.append(config)
                    valid_choices.append(choice)

        if not triton_configs:
            logger.warning(
                "Could not convert any configs to TritonGEMMConfig, returning original"
            )
            return configs

        logger.debug(f"Converted {len(triton_configs)} choices to configs")

        # Create features and run inference
        runtimes = create_features_and_run_inference(
            mmshape, triton_configs, self._create_unified_predictor(), self.device
        )

        if not runtimes or all(p == 0.0 for p in runtimes):
            logger.warning("Model inference failed, returning original configs")
            return configs

        # Select best configurations based on predictions
        selected_choices = select_best_configs(
            valid_choices,
            runtimes,
            self.top_k_configs,
        )

        logger.info(
            f"Model selected {len(selected_choices)}/{len(configs)} configs for {op_name}"
        )

        return selected_choices

    def _create_unified_predictor(self) -> Optional[UnifiedMatmulPredictor]:
        """Create a UnifiedMatmulPredictor from the loaded model wrapper."""
        if not self._model_loaded or not self.model_wrapper:
            return None

        try:
            # Create a simple wrapper that adapts the model_wrapper to UnifiedMatmulPredictor interface
            class ModelWrapperAdapter:
                def __init__(self, model_wrapper):
                    self.model_wrapper = model_wrapper

                def predict_from_features(self, problem_features, config_features):
                    # Combine features as expected by the model wrapper
                    combined_features = torch.cat(
                        [problem_features, config_features], dim=1
                    )
                    return self.model_wrapper.inference(combined_features)

            return ModelWrapperAdapter(self.model_wrapper)
        except Exception as e:
            logger.error(f"Failed to create unified predictor: {e}")
            return None

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about model usage."""
        return dict(self.stats)

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats.clear()

    # Deprecated methods for backward compatibility with old tests
    def _extract_features_from_kernel_inputs(self, kernel_inputs, op_name):
        """
        DEPRECATED: Extract features from kernel inputs.
        This method is deprecated - the current implementation uses kernel_conversions module.
        """
        try:
            from ..utils.feature_extraction import extract_problem_features_compat

            if not kernel_inputs:
                return None

            # Handle different kernel input interfaces
            if hasattr(kernel_inputs, "nodes"):
                nodes = kernel_inputs.nodes()
            elif hasattr(kernel_inputs, "_tensors"):
                nodes = kernel_inputs._tensors
            else:
                return None

            if len(nodes) < 2:
                return None

            if op_name not in ["mm", "bmm", "addmm", "baddbmm"]:
                return None

            # Extract tensor shapes
            tensor_a = nodes[0]
            tensor_b = nodes[1]

            # Handle different tensor interfaces
            if hasattr(tensor_a, "get_size") and hasattr(tensor_b, "get_size"):
                size_a = tensor_a.get_size()
                size_b = tensor_b.get_size()
            elif hasattr(tensor_a, "_size") and hasattr(tensor_b, "_size"):
                size_a = tensor_a._size
                size_b = tensor_b._size
            else:
                return None

            # Determine matrix dimensions based on operation
            if op_name in ["mm", "addmm"]:
                if len(size_a) < 2 or len(size_b) < 2:
                    return None
                M, K = size_a[-2], size_a[-1]
                K2, N = size_b[-2], size_b[-1]
                B = 1
            elif op_name in ["bmm", "baddbmm"]:
                if len(size_a) < 3 or len(size_b) < 3:
                    return None
                B, M, K = size_a[-3], size_a[-2], size_a[-1]
                B2, K2, N = size_b[-3], size_b[-2], size_b[-1]
            else:
                return None

            if K != K2:
                return None

            from ..types.matmul_types import MMShape

            # For the mock tensors, use the provided dtypes or defaults
            tensor_a_dtype = getattr(tensor_a, "_dtype", torch.float16)
            tensor_b_dtype = getattr(tensor_b, "_dtype", torch.float16)

            mm_shape = MMShape(
                M=M,
                N=N,
                K=K,
                B=B,
                M_dtype=tensor_a_dtype,
                K_dtype=tensor_a_dtype,
                out_dtype=tensor_a_dtype,  # Assume output has same dtype as input A
                out_size=(B, M, N),
                out_stride=(M * N, N, 1),  # Default row-major stride
            )

            # Create problem features
            problem_features = extract_problem_features(mm_shape)

            return {
                "mm_shape": mm_shape,
                "problem_features": problem_features,
                "op_name": op_name,
            }
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _convert_ktc_to_config(self, ktc):
        """
        DEPRECATED: Convert KernelTemplateChoice to TritonGEMMConfig.
        This method is deprecated - the current implementation uses kernel_conversions module.
        """
        try:
            from ..types.matmul_types import TritonGEMMConfig

            if not ktc or not hasattr(ktc, "template") or not hasattr(ktc, "config"):
                return None

            template = ktc.template
            config = ktc.config

            # Extract template name
            template_name = getattr(template, "uid", "unknown")

            # Extract config parameters
            kwargs = getattr(config, "kwargs", {})

            return TritonGEMMConfig(
                name=template_name,
                grid=1,  # Default grid
                block_m=kwargs.get("BLOCK_M", 64),
                block_n=kwargs.get("BLOCK_N", 64),
                block_k=kwargs.get("BLOCK_K", 32),
                group_m=kwargs.get("GROUP_M", 8),
                num_stages=kwargs.get("num_stages", 2),
                num_warps=kwargs.get("num_warps", 4),
                EVEN_K=kwargs.get("EVEN_K", True),
                ALLOW_TF32=kwargs.get("ALLOW_TF32", True),
            )
        except Exception:
            return None

    def _predict_config_performance(self, problem_features, configs):
        """
        DEPRECATED: Predict config performance.
        This method is deprecated - the current implementation uses kernel_conversions module.
        """
        if not self._model_loaded or not self.model_wrapper:
            return [0.0] * len(configs)

        try:
            from ..utils.feature_extraction import extract_config_features_compat

            predictions = []
            for config in configs:
                # Extract config features
                config_features = extract_config_features_compat(config)

                # Ensure both features are 2D for batching
                if problem_features.dim() == 1:
                    problem_features_batch = problem_features.unsqueeze(0)
                else:
                    problem_features_batch = problem_features

                if config_features.dim() == 1:
                    config_features_batch = config_features.unsqueeze(0)
                else:
                    config_features_batch = config_features

                # Combine features
                combined_features = torch.cat(
                    [problem_features_batch, config_features_batch], dim=1
                )

                # Get prediction using the model wrapper's method
                # Try different possible method names
                if hasattr(self.model_wrapper, "predict"):
                    # For MockModelWrapper used in tests
                    prediction = self.model_wrapper.predict(
                        problem_features_batch, config_features_batch
                    )
                elif hasattr(self.model_wrapper, "inference"):
                    try:
                        prediction = self.model_wrapper.inference(combined_features)
                    except Exception as e:
                        # If inference fails due to dimension mismatch, try to pad features
                        logger.debug(
                            f"Inference failed: {e}, trying to handle dimension mismatch"
                        )
                        try:
                            # For real model wrappers, check expected dimensions
                            expected_dim = getattr(self.model_wrapper, "model", None)
                            if expected_dim and hasattr(
                                expected_dim, "problem_feature_dim"
                            ):
                                prob_dim = expected_dim.problem_feature_dim
                                conf_dim = expected_dim.config_feature_dim

                                # Pad or truncate features to match expected dimensions
                                if problem_features_batch.shape[1] < prob_dim:
                                    pad_prob = (
                                        prob_dim - problem_features_batch.shape[1]
                                    )
                                    problem_features_batch = torch.cat(
                                        [
                                            problem_features_batch,
                                            torch.zeros(
                                                problem_features_batch.shape[0],
                                                pad_prob,
                                            ),
                                        ],
                                        dim=1,
                                    )
                                elif problem_features_batch.shape[1] > prob_dim:
                                    problem_features_batch = problem_features_batch[
                                        :, :prob_dim
                                    ]

                                if config_features_batch.shape[1] < conf_dim:
                                    pad_conf = conf_dim - config_features_batch.shape[1]
                                    config_features_batch = torch.cat(
                                        [
                                            config_features_batch,
                                            torch.zeros(
                                                config_features_batch.shape[0], pad_conf
                                            ),
                                        ],
                                        dim=1,
                                    )
                                elif config_features_batch.shape[1] > conf_dim:
                                    config_features_batch = config_features_batch[
                                        :, :conf_dim
                                    ]

                                combined_features = torch.cat(
                                    [problem_features_batch, config_features_batch],
                                    dim=1,
                                )
                                prediction = self.model_wrapper.inference(
                                    combined_features
                                )
                            else:
                                # Fall back to zero prediction
                                predictions.append(0.0)
                                continue
                        except Exception:
                            # If all attempts fail, fall back to zero
                            predictions.append(0.0)
                            continue
                elif hasattr(self.model_wrapper, "__call__"):
                    prediction = self.model_wrapper(combined_features)
                else:
                    # If we can't find the right method, return 0.0
                    predictions.append(0.0)
                    continue

                predictions.append(float(prediction.squeeze()))

            return predictions
        except Exception as e:
            logger.debug(f"Prediction failed: {e}")
            return [0.0] * len(configs)


# Factory function for easy integration
def create_diode_choices(
    model_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs,
) -> DiodeInductorChoices:
    """
    Factory function to create a DiodeInductorChoices instance.

    Args:
        model_path: Path to the trained Diode model
        device: Device to run the model on
        **kwargs: Additional arguments passed to DiodeInductorChoices

    Returns:
        DiodeInductorChoices instance
    """
    return DiodeInductorChoices(model_path=model_path, device=device, **kwargs)


# Integration helper function
def install_diode_choices(
    model_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs,
) -> DiodeInductorChoices:
    """
    Install Diode model-based choices as the default choice handler.

    This function sets up the Diode choices handler in the PyTorch Inductor
    virtualized environment.

    Args:
        model_path: Path to the trained Diode model
        device: Device to run the model on
        **kwargs: Additional arguments passed to DiodeInductorChoices

    Returns:
        The installed DiodeInductorChoices instance

    Raises:
        ImportError: If PyTorch Inductor is not available
        Exception: If installation fails
    """
    try:
        from torch._inductor.virtualized import V

        # Create the Diode choices handler
        diode_choices = create_diode_choices(
            model_path=model_path,
            device=device,
            **kwargs,
        )

        # Install as the choices handler using the standard PyTorch Inductor API
        V.set_choices_handler(diode_choices)
        logger.info("DiodeInductorChoices installed successfully")

        return diode_choices

    except ImportError:
        logger.error("Could not import PyTorch Inductor virtualized module")
        raise
    except Exception as e:
        logger.error(f"Failed to install DiodeInductorChoices: {e}")
        raise


def example_usage():
    """Example of how to use the Diode integration."""

    # Option 1: Create directly
    choices = create_diode_choices(
        model_path="path/to/your/model.pt", device="cuda", top_k_configs=5
    )

    # Option 2: Install as default choice handler
    install_diode_choices(
        model_path="path/to/your/model.pt", device="cuda", top_k_configs=5
    )

    # After installation, all torch.compile operations will use the Diode model
    # for kernel configuration selection

    print("Diode integration example completed")


if __name__ == "__main__":
    # Run example usage
    example_usage()
