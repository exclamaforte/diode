"""
Integration with PyTorch Inductor's choices.py for model-based config selection.

This module provides functionality to integrate trained Diode models with PyTorch Inductor's
_finalize_mm_configs method to enable model-based selection of optimal kernel configurations.
"""

import logging
import os
from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Union

import torch

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
from diode.model.model_wrapper import ModelWrapper
from diode.types.matmul_types import MMShape, TritonGEMMConfig

# Feature extraction functions (placeholder implementations)
# TODO: Move proper implementations to diode.collection.matmul_data_utils when available
def extract_problem_features(mm_shape, op_name):
    """Placeholder for feature extraction when utils are not available."""
    import torch

    # Convert dimensions to integers (handle symbolic dimensions)
    B = int(mm_shape.B) if hasattr(mm_shape.B, "__int__") else 1
    M = int(mm_shape.M) if hasattr(mm_shape.M, "__int__") else 64
    N = int(mm_shape.N) if hasattr(mm_shape.N, "__int__") else 64
    K = int(mm_shape.K) if hasattr(mm_shape.K, "__int__") else 64

    # Return same feature set as the real function (17 features)
    features = [
        B,
        M,
        N,
        K,
        # dtype features (simplified)
        32.0,
        32.0,
        32.0,  # Assume float32 for dtypes
        # Log-transformed features
        torch.log(torch.tensor(max(1, B), dtype=torch.float32)).item(),
        torch.log(torch.tensor(max(1, M), dtype=torch.float32)).item(),
        torch.log(torch.tensor(max(1, N), dtype=torch.float32)).item(),
        torch.log(torch.tensor(max(1, K), dtype=torch.float32)).item(),
        # Derived features
        M * K,  # Input matrix size
        K * N,  # Weight matrix size
        M * N,  # Output matrix size
        torch.log(torch.tensor(max(1, M * K), dtype=torch.float32)).item(),
        torch.log(torch.tensor(max(1, K * N), dtype=torch.float32)).item(),
        torch.log(torch.tensor(max(1, M * N), dtype=torch.float32)).item(),
    ]
    return features

def extract_config_features(config):
    """Placeholder for config feature extraction when utils are not available."""
    import torch

    # Return same feature set as the real function (19 features)
    features = [
        config.grid,
        config.block_m,
        config.block_n,
        config.block_k,
        config.group_m,
        config.num_stages,
        config.num_warps,
        int(getattr(config, "EVEN_K", False)),
        int(getattr(config, "ALLOW_TF32", False)),
        int(getattr(config, "USE_FAST_ACCUM", False)),
        # Log-transformed features
        torch.log(torch.tensor(max(1, config.block_m), dtype=torch.float32)).item(),
        torch.log(torch.tensor(max(1, config.block_n), dtype=torch.float32)).item(),
        torch.log(torch.tensor(max(1, config.block_k), dtype=torch.float32)).item(),
        # Derived features
        config.block_m * config.block_k,
        config.block_k * config.block_n,
        config.block_m * config.block_n,
        torch.log(
            torch.tensor(
                max(1, config.block_m * config.block_k), dtype=torch.float32
            )
        ).item(),
        torch.log(
            torch.tensor(
                max(1, config.block_k * config.block_n), dtype=torch.float32
            )
        ).item(),
        torch.log(
            torch.tensor(
                max(1, config.block_m * config.block_n), dtype=torch.float32
            )
        ).item(),
    ]
    return features


logger = logging.getLogger(__name__)


class DiodeInductorChoices(InductorChoices):
    """
    Extended InductorChoices class that uses Diode models for config selection.

    This class overrides the _finalize_mm_configs method to run model inference
    on available configurations and select the best ones based on predicted timing.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        top_k_configs: int = 3,
        enable_fallback: bool = True,
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
        if self._model_loaded:
            return True

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

    def _extract_features_from_kernel_inputs(
        self, kernel_inputs: KernelInputs, op_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract problem features from KernelInputs.

        Args:
            kernel_inputs: The kernel inputs containing tensor information
            op_name: Operation name (e.g., "mm", "addmm", "bmm")

        Returns:
            Dictionary of problem features or None if extraction fails
        """
        try:
            # Get input tensors from kernel_inputs
            input_tensors = kernel_inputs.nodes()

            if len(input_tensors) < 2:
                logger.warning(
                    f"Insufficient tensors for {op_name}: {len(input_tensors)}"
                )
                return None

            # Extract tensor information
            tensor_a = input_tensors[0]
            tensor_b = input_tensors[1]

            # Get shapes and dtypes
            shape_a = tensor_a.get_size()
            shape_b = tensor_b.get_size()
            dtype_a = tensor_a.get_dtype()
            dtype_b = tensor_b.get_dtype()

            # Get output layout information
            output_layout = kernel_inputs.output_layout(flexible=False)
            out_size = output_layout.size
            out_stride = output_layout.stride
            out_dtype = output_layout.dtype

            # Create MMShape object for feature extraction
            # For matrix multiplication, we need to determine M, N, K dimensions
            if op_name in ["mm", "addmm"]:
                # Regular matrix multiplication: A(M,K) x B(K,N) = C(M,N)
                if len(shape_a) >= 2 and len(shape_b) >= 2:
                    M = shape_a[-2]
                    K = shape_a[-1]
                    N = shape_b[-1]
                    B = 1  # No batch dimension for mm
                else:
                    logger.warning(
                        f"Invalid shapes for {op_name}: {shape_a}, {shape_b}"
                    )
                    return None
            elif op_name == "bmm":
                # Batch matrix multiplication: A(B,M,K) x B(B,K,N) = C(B,M,N)
                if len(shape_a) >= 3 and len(shape_b) >= 3:
                    B = shape_a[-3]
                    M = shape_a[-2]
                    K = shape_a[-1]
                    N = shape_b[-1]
                else:
                    logger.warning(
                        f"Invalid shapes for {op_name}: {shape_a}, {shape_b}"
                    )
                    return None
            else:
                logger.warning(f"Unsupported operation: {op_name}")
                return None

            # Convert tensor sizes to integers (handle sympy expressions)
            try:
                B = int(B) if hasattr(B, "__int__") else 1
                M = int(M) if hasattr(M, "__int__") else 1
                N = int(N) if hasattr(N, "__int__") else 1
                K = int(K) if hasattr(K, "__int__") else 1
            except (TypeError, ValueError) as e:
                logger.warning(f"Could not convert tensor dimensions to int: {e}")
                return None

            # Create MMShape
            mm_shape = MMShape(
                B=B,
                M=M,
                M_dtype=dtype_a,
                N=N,
                K=K,
                K_dtype=dtype_b,
                out_dtype=out_dtype,
                out_size=tuple(
                    int(s) if hasattr(s, "__int__") else 1 for s in out_size
                ),
                out_stride=tuple(
                    int(s) if hasattr(s, "__int__") else 1 for s in out_stride
                ),
            )

            # Extract problem features using diode utility
            problem_features = extract_problem_features(mm_shape, op_name)

            return {
                "mm_shape": mm_shape,
                "problem_features": problem_features,
                "op_name": op_name,
            }

        except Exception as e:
            logger.error(f"Error extracting features from kernel inputs: {e}")
            return None

    def _convert_ktc_to_config(
        self, ktc: KernelTemplateChoice
    ) -> Optional[TritonGEMMConfig]:
        """
        Convert a KernelTemplateChoice to a TritonGEMMConfig.

        Args:
            ktc: KernelTemplateChoice object

        Returns:
            TritonGEMMConfig object or None if conversion fails
        """
        try:
            # Extract template and config information from KernelTemplateChoice
            template = ktc.template
            config = ktc.config

            # Get template name
            template_name = getattr(template, "uid", "unknown")

            # Extract Triton config parameters
            # These would typically be in the config object
            if hasattr(config, "kwargs"):
                kwargs = config.kwargs
            else:
                kwargs = {}

            # Extract standard Triton GEMM parameters
            block_m = kwargs.get("BLOCK_M", 64)
            block_n = kwargs.get("BLOCK_N", 64)
            block_k = kwargs.get("BLOCK_K", 32)
            group_m = kwargs.get("GROUP_M", 8)
            num_stages = kwargs.get("num_stages", 4)
            num_warps = kwargs.get("num_warps", 4)

            # Create TritonGEMMConfig
            triton_config = TritonGEMMConfig(
                name=template_name,
                grid=1,  # Grid will be computed dynamically
                block_m=int(block_m),
                block_n=int(block_n),
                block_k=int(block_k),
                group_m=int(group_m),
                num_stages=int(num_stages),
                num_warps=int(num_warps),
                EVEN_K=kwargs.get("EVEN_K", False),
                ALLOW_TF32=kwargs.get("ALLOW_TF32", False),
                USE_FAST_ACCUM=kwargs.get("USE_FAST_ACCUM", False),
                ACC_TYPE=kwargs.get("ACC_TYPE", "tl.float32"),
            )

            return triton_config

        except Exception as e:
            logger.error(f"Error converting KTC to TritonGEMMConfig: {e}")
            return None

    def _predict_config_performance(
        self, problem_features: torch.Tensor, configs: List[TritonGEMMConfig]
    ) -> List[float]:
        """
        Predict performance for a list of configurations.

        Args:
            problem_features: Tensor of problem features
            configs: List of TritonGEMMConfig objects

        Returns:
            List of predicted log execution times
        """
        if not self._model_loaded or not self.model_wrapper:
            return [0.0] * len(configs)

        try:
            predictions = []

            for config in configs:
                # Extract config features
                config_features = extract_config_features(config)

                # Convert to tensor with explicit shape check
                config_tensor = torch.tensor(
                    config_features, dtype=torch.float32, device=self.device
                )

                # Ensure both tensors have correct dimensions
                if problem_features.numel() == 0 or config_tensor.numel() == 0:
                    logger.warning("Empty feature tensors, skipping prediction")
                    predictions.append(0.0)
                    continue

                # Add batch dimension
                problem_batch = problem_features.unsqueeze(0)
                config_batch = config_tensor.unsqueeze(0)

                # Check that tensors have expected dimensions
                expected_problem_dim = getattr(self.model_wrapper, "model", None)
                if hasattr(expected_problem_dim, "problem_feature_dim"):
                    if (
                        problem_batch.size(1)
                        != expected_problem_dim.problem_feature_dim
                    ):
                        logger.warning(
                            f"Problem feature dimension mismatch: expected {expected_problem_dim.problem_feature_dim}, got {problem_batch.size(1)}"
                        )
                        predictions.append(0.0)
                        continue

                # Run prediction with error handling for compilation issues
                with torch.no_grad():
                    try:
                        prediction = self.model_wrapper.predict(
                            problem_batch, config_batch
                        )
                        predictions.append(float(prediction.item()))
                    except RuntimeError as runtime_e:
                        if "must have same reduction dim" in str(runtime_e):
                            logger.warning(
                                f"Dimension mismatch in model prediction, likely due to compilation: {runtime_e}"
                            )
                            # Try with uncompiled model if available
                            try:
                                if hasattr(self.model_wrapper, "model"):
                                    raw_prediction = self.model_wrapper.model(
                                        problem_batch, config_batch
                                    )
                                    predictions.append(float(raw_prediction.item()))
                                else:
                                    predictions.append(0.0)
                            except Exception:
                                predictions.append(0.0)
                        else:
                            logger.error(f"Runtime error in prediction: {runtime_e}")
                            predictions.append(0.0)

            return predictions

        except Exception as e:
            logger.error(f"Error predicting config performance: {e}")
            return [0.0] * len(configs)

    def _finalize_mm_configs(
        self,
        template_choices: Dict[str, Generator[KernelTemplateChoice, None, None]],
        kernel_inputs: KernelInputs,
        layout: Any,
        templates: List[Union[KernelTemplate, ExternKernelChoice]],
        op_name: str,
        kwarg_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[KernelTemplateChoice]:
        """
        Override of the base _finalize_mm_configs method to use model-based selection.

        Args:
            template_choices: Dictionary mapping template UIDs to generators of KernelTemplateChoice objects
            kernel_inputs: MMKernelInputs containing input tensor nodes and matrix indices
            layout: Output layout
            templates: List of template objects (KernelTemplate or ExternKernelChoice) in use
            op_name: Operation name (e.g., "bmm", "baddbmm", "addmm")
            kwarg_overrides: Optional dict of kwargs to override for each template heuristic

        Returns:
            Filtered list of KernelTemplateChoice objects based on model predictions
        """
        # Track statistics
        self.stats["total_calls"] += 1

        # First, get all choices using the default method
        choices: List[KernelTemplateChoice] = []
        for choice_gen in template_choices.values():
            choices.extend(choice_gen)

        # If no model is loaded or available, fall back to default behavior
        if not self._model_loaded or not self.model_wrapper:
            self.stats["fallback_no_model"] += 1
            logger.debug("No model available, using default config selection")
            return choices

        # If no choices available, return empty list
        if not choices:
            self.stats["no_choices"] += 1
            return choices

        try:
            # Extract problem features
            feature_data = self._extract_features_from_kernel_inputs(
                kernel_inputs, op_name
            )
            if not feature_data:
                self.stats["fallback_feature_extraction"] += 1
                logger.debug(
                    "Could not extract features, using default config selection"
                )
                return choices

            problem_features = torch.tensor(
                feature_data["problem_features"],
                dtype=torch.float32,
                device=self.device,
            )

            # Convert choices to configs
            configs = []
            valid_choices = []

            for choice in choices:
                config = self._convert_ktc_to_config(choice)
                if config:
                    configs.append(config)
                    valid_choices.append(choice)

            if not configs:
                self.stats["fallback_config_conversion"] += 1
                logger.debug(
                    "Could not convert any choices to configs, using default selection"
                )
                return choices

            # Predict performance for all configs
            predictions = self._predict_config_performance(problem_features, configs)

            if not predictions or all(p == 0.0 for p in predictions):
                self.stats["fallback_prediction"] += 1
                logger.debug("Model prediction failed, using default config selection")
                return choices

            # Sort choices by predicted performance (lower is better for log execution time)
            sorted_indices = sorted(
                range(len(predictions)), key=lambda i: predictions[i]
            )

            # Select top configurations within performance threshold
            best_prediction = predictions[sorted_indices[0]]
            selected_choices = []

            for idx in sorted_indices:
                if len(selected_choices) >= self.top_k_configs:
                    break

                # Include if within performance threshold
                if predictions[idx] <= best_prediction * self.performance_threshold:
                    selected_choices.append(valid_choices[idx])

            # Ensure we have at least one choice
            if not selected_choices and valid_choices:
                selected_choices = [valid_choices[sorted_indices[0]]]

            self.stats["model_selections"] += 1
            self.stats["configs_filtered"] += len(choices) - len(selected_choices)

            logger.debug(
                f"Model selected {len(selected_choices)}/{len(choices)} configs for {op_name} "
                f"(best prediction: {best_prediction:.3f})"
            )

            return selected_choices

        except Exception as e:
            self.stats["fallback_error"] += 1
            logger.error(f"Error in model-based config selection: {e}")

            if self.enable_fallback:
                logger.debug("Falling back to default config selection")
                return choices
            else:
                raise

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about model usage."""
        return dict(self.stats)

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats.clear()

    def get_base_mm_configs(self, *args, **kwargs):
        """
        Compatibility method for PyTorch Inductor.
        This method is expected by the Inductor framework but we override 
        _finalize_mm_configs instead.
        """
        # Return empty list since we handle config selection in _finalize_mm_configs
        return []

    def get_persistent_mm_configs(self, *args, **kwargs):
        """
        Compatibility method for PyTorch Inductor.
        This method is expected by the Inductor framework but we override 
        _finalize_mm_configs instead.
        """
        # Return empty list since we handle config selection in _finalize_mm_configs
        return []

    def get_extra_mm_configs(self, *args, **kwargs):
        """
        Compatibility method for PyTorch Inductor.
        This method is expected by the Inductor framework but we override 
        _finalize_mm_configs instead.
        """
        # Return empty list since we handle config selection in _finalize_mm_configs
        return []


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
) -> None:
    """
    Install Diode model-based choices as the default choice handler.

    This function sets up the Diode choices handler in the PyTorch Inductor
    virtualized environment.

    Args:
        model_path: Path to the trained Diode model
        device: Device to run the model on
        **kwargs: Additional arguments passed to DiodeInductorChoices
    """
    try:
        from torch._inductor.virtualized import V

        diode_choices = create_diode_choices(
            model_path=model_path, device=device, **kwargs
        )
        V.set_choices_handler(diode_choices)

        logger.info("Diode model-based choices installed successfully")

    except ImportError:
        logger.error("Could not import PyTorch Inductor virtualized module")
        raise
    except Exception as e:
        logger.error(f"Failed to install Diode choices: {e}")
        raise


# Example usage function
def example_usage():
    """Example of how to use the Diode integration."""

    # Option 1: Direct usage with explicit model path
    choices = DiodeInductorChoices(
        model_path="path/to/your/model.pt",
        device="cuda",
        top_k_configs=3,
        performance_threshold=1.1,
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
