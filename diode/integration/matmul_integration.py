"""
Matmul kernel integration for PyTorch Inductor.

This module implements the BaseIntegration interface for matmul kernel selection,
providing the specific implementation of the integration pattern for matrix multiplication
kernel optimization.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .base_integration import BaseIntegration, ModelPointer

logger = logging.getLogger(__name__)


class MatmulIntegration(BaseIntegration):
    """Integration for matmul kernel runtime prediction models."""

    def __init__(
        self,
        model_pointers: Optional[List[ModelPointer]] = None,
        enable_fallback: bool = True,
        **kwargs,
    ):
        """
        Initialize the matmul integration.

        Args:
            model_pointers: Optional list of model pointers. If None, uses default models.
            enable_fallback: Whether to enable fallback when models fail to load
            **kwargs: Additional arguments passed to parent class
        """
        # Use provided model pointers or define default ones
        if model_pointers is None:
            model_pointers = [
                ModelPointer(
                    model_name="v1_model.pt",
                    relative_path="matmul_kernel_runtime_prediction",
                    model_purpose="matmul_kernel_runtime_prediction",
                    interface_name="torch._inductor.choices",
                    description="Matrix multiplication kernel runtime prediction model v1",
                    version="1.0",
                    dependencies=["torch._inductor", "torch._inductor.choices"],
                ),
                # Add fallback model pointer for the model in the root
                ModelPointer(
                    model_name="matmul_model_exhaustive.pt",
                    relative_path=".",  # Root of trained_models
                    model_purpose="matmul_kernel_runtime_prediction",
                    interface_name="torch._inductor.choices",
                    description="Matrix multiplication kernel runtime prediction model (exhaustive)",
                    version="1.0",
                    dependencies=["torch._inductor", "torch._inductor.choices"],
                ),
            ]

        super().__init__(
            name="matmul_kernel_prediction",
            interface_name="torch._inductor.choices",
            model_pointers=model_pointers,
            enable_fallback=enable_fallback,
            **kwargs,
        )

    def create_dummy_function(self) -> Any:
        """Create a dummy choices handler to test interface availability."""
        try:
            from torch._inductor.choices import InductorChoices

            class DummyInductorChoices(InductorChoices):
                """Dummy choices handler for testing interface availability."""

                def __init__(self):
                    super().__init__()
                    self._is_dummy = True

                def get_base_mm_configs(self, *args, **kwargs):
                    """Dummy implementation."""
                    return []

                def get_persistent_mm_configs(self, *args, **kwargs):
                    """Dummy implementation."""
                    return []

                def get_extra_mm_configs(self, *args, **kwargs):
                    """Dummy implementation."""
                    return []

            return DummyInductorChoices()

        except ImportError:
            logger.debug("torch._inductor.choices not available")
            return None

    def load_model(self, model_pointer: ModelPointer) -> Any:
        """Load a matmul model from a model pointer."""
        model_path = model_pointer.full_path

        if not model_path.exists():
            # Try alternative locations
            alternative_paths = [
                Path(__file__).parent.parent.parent / model_pointer.model_name,
                Path(__file__).parent.parent / "data" / model_pointer.model_name,
            ]

            for alt_path in alternative_paths:
                if alt_path.exists():
                    model_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Model not found: {model_pointer.model_name}")

        logger.info(f"Loading matmul model from: {model_path}")

        # Determine the appropriate loading method based on model name and content
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_pointer.model_name == "v1_model.pt":
            # This is a V1 model - but it's saved as raw state_dict, not a full checkpoint
            from ..model.matmul_model_v1 import MatmulModelV1

            # Create the model with default V1 parameters
            model = MatmulModelV1(
                problem_feature_dim=7,  # dtype_size, dim_m, dim_n, dim_k, total_gb, total_gflop, flops_per_byte
                config_feature_dim=5,  # config_block_k, config_block_m, config_block_n, config_num_stages, config_num_warps
                hidden_layer_widths=[256, 256, 256, 256, 256, 256],
                kernel_overhead=0.00541,
                dropout_rate=0.0,
            )

            # Load the state_dict manually since it's not a full checkpoint
            state_dict = torch.load(str(model_path), map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()

            # Create a simple wrapper that provides the interface needed by the integration
            class SimpleModelWrapper:
                def __init__(self, model, device):
                    self.model = model
                    self.device = device

                def encode(
                    self, m: int, n: int, k: int, dtype: torch.dtype, configs: list
                ):
                    """Encode inputs for the model - delegate to ModelWrapper from matmul_model_v1."""
                    from ..model.matmul_model_v1 import ModelWrapper

                    wrapper = ModelWrapper(
                        model_path=None
                    )  # No path needed for encoding
                    return wrapper.encode(m, n, k, dtype, configs)

                def inference(self, inp_tensor: torch.Tensor) -> torch.Tensor:
                    """Run inference using the loaded model."""
                    with torch.no_grad():
                        # Split input into problem and config features as done in ModelWrapper
                        problem_features = inp_tensor[
                            :, :7
                        ]  # first 7 features are problem features
                        config_features = inp_tensor[
                            :, 7:
                        ]  # remaining are config features
                        return self.model(problem_features, config_features)

                def decode(self, ret_tensor: torch.Tensor) -> torch.Tensor:
                    """Decode model output."""
                    return ret_tensor

            return SimpleModelWrapper(model, device)
        else:
            # For other models, try the original ModelWrapper approach
            from ..model.model_wrapper import ModelWrapper

            model_wrapper = ModelWrapper(
                model_path=str(model_path),
                device=device,
                compile_model=False,  # Disable compilation to avoid dynamic shape issues
            )
            return model_wrapper

    def register_model(self, model: Any, model_pointer: ModelPointer) -> bool:
        """Register a loaded matmul model with the inductor choices system."""
        try:
            from torch._inductor.virtualized import V

            from .inductor_integration import create_diode_choices

            # Create diode choices handler with the loaded model
            model_path = str(model_pointer.full_path)
            diode_choices = create_diode_choices(
                model_path=model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                top_k_configs=3,
                performance_threshold=1.1,
                enable_fallback=self.enable_fallback,
            )

            # Set as the choices handler
            V.set_choices_handler(diode_choices)

            logger.info(f"Registered matmul model: {model_pointer.model_name}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to register matmul model {model_pointer.model_name}: {e}"
            )
            return False

    def enable_configs(self) -> bool:
        """Enable PyTorch configs that engage matmul model-based selection."""
        try:
            # Enable fast autotune which will use our model-based choices
            torch._inductor.config.autotune = True
            torch._inductor.config.mm_autotune = True
            torch._inductor.config.max_autotune_gemm_search_space = 10

            # Enable model-based kernel selection
            if hasattr(torch._inductor.config, "enable_diode_choices"):
                torch._inductor.config.enable_diode_choices = True

            logger.info("Enabled PyTorch Inductor configs for matmul model integration")
            return True

        except Exception as e:
            logger.error(f"Failed to enable PyTorch configs: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available and loaded matmul models."""
        available_models = self.get_available_models()

        model_info = {
            "available_models": [
                {
                    "name": pointer.model_name,
                    "path": str(pointer.full_path),
                    "size_mb": pointer.get_size_mb(),
                    "version": pointer.version,
                    "description": pointer.description,
                }
                for pointer in available_models
            ],
            "loaded_models": list(self.loaded_models.keys()),
            "registration_status": self.registration_status.copy(),
        }

        return model_info


def create_matmul_integration(enable_fallback: bool = True) -> MatmulIntegration:
    """
    Factory function to create a matmul integration.

    Args:
        enable_fallback: Whether to enable fallback when models fail to load

    Returns:
        MatmulIntegration instance
    """
    return MatmulIntegration(enable_fallback=enable_fallback)
