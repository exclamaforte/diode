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
from torch_diode.utils.debug_config import type_assert

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
        type_assert(model_pointers is None or isinstance(model_pointers, list), f"model_pointers must be list or None, got {type(model_pointers)}")
        type_assert(isinstance(enable_fallback, bool), f"enable_fallback must be bool, got {type(enable_fallback)}")
        if model_pointers is not None:
            type_assert(all(isinstance(mp, ModelPointer) for mp in model_pointers), "All items in model_pointers must be ModelPointer instances")
        
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
        type_assert(hasattr(self, 'name'), "MatmulIntegration must have name attribute")
        
        try:
            from torch._inductor.choices import InductorChoices

            class DummyInductorChoices(InductorChoices):
                """Dummy choices handler for testing interface availability."""

                def __init__(self):
                    super().__init__()
                    self._is_dummy = True

            return DummyInductorChoices()

        except ImportError:
            logger.debug("torch._inductor.choices not available")
            return None

    def load_model(self, model_pointer: ModelPointer) -> Any:
        """Load a matmul model from a model pointer."""
        type_assert(isinstance(model_pointer, ModelPointer), f"model_pointer must be ModelPointer, got {type(model_pointer)}")
        type_assert(hasattr(model_pointer, 'full_path'), "model_pointer must have full_path property")
        type_assert(hasattr(model_pointer, 'model_name'), "model_pointer must have model_name attribute")
        
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

        # Load all models using the consistent ModelWrapper approach
        device = "cuda" if torch.cuda.is_available() else "cpu"

        from ..model.model_wrapper import ModelWrapper

        model_wrapper = ModelWrapper(
            model_path=str(model_path),
            device=device,
            compile_model=False,  # Disable compilation to avoid dynamic shape issues
        )
        return model_wrapper

    def register_model(self, model: Any, model_pointer: ModelPointer) -> bool:
        """Register a loaded matmul model with the inductor choices system."""
        type_assert(model is not None, "model cannot be None")
        type_assert(isinstance(model_pointer, ModelPointer), f"model_pointer must be ModelPointer, got {type(model_pointer)}")
        type_assert(hasattr(model_pointer, 'full_path'), "model_pointer must have full_path property")
        type_assert(hasattr(model_pointer, 'model_name'), "model_pointer must have model_name attribute")
        type_assert(hasattr(self, 'enable_fallback'), "MatmulIntegration must have enable_fallback attribute")
        
        try:
            from torch._inductor.virtualized import V

            from .inductor_integration import install_diode_choices

            # Install the diode choices handler with the loaded model
            model_path = str(model_pointer.full_path)
            diode_choices = install_diode_choices(
                model_path=model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                top_k_configs=5,
                performance_threshold=1.1,
                enable_fallback=self.enable_fallback,
            )

            logger.info(
                f"Registered matmul model with enhanced choices: {model_pointer.model_name}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to register matmul model {model_pointer.model_name}: {e}"
            )
            return False

    def enable_configs(self) -> bool:
        """Enable PyTorch configs that engage matmul model-based selection."""
        type_assert(hasattr(self, 'name'), "MatmulIntegration must have name attribute")
        
        try:
            # Enable max_autotune which will use our model-based choices
            torch._inductor.config.max_autotune = True

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
    type_assert(isinstance(enable_fallback, bool), f"enable_fallback must be bool, got {type(enable_fallback)}")
    return MatmulIntegration(enable_fallback=enable_fallback)
