"""
Model wrapper for loading and running inference on trained models.

This module imports the common ModelWrapper from diode_common and extends it
with diode_models-specific functionality.
"""

import os
import torch
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass

# Import from the common module
from diode_common.model_wrapper import ModelWrapper as CommonModelWrapper
from diode_common.model_wrapper import load_model_config as common_load_model_config
from diode_common.model_wrapper import list_available_models as common_list_available_models

# Import the model classes directly if available
try:
    from diode.model.matmul_timing_model import MatmulTimingModel, DeepMatmulTimingModel
    from diode.model.matmul_model_config import MatmulModelConfig
    DIODE_AVAILABLE = True
except ImportError:
    DIODE_AVAILABLE = False
    # Define minimal versions of the model classes for standalone use
    class MatmulTimingModel(torch.nn.Module):
        """Minimal version of MatmulTimingModel for standalone use."""
        @classmethod
        def load(cls, path: str, device: str = 'cpu'):
            """Load a model from a file."""
            return torch.load(path, map_location=device)

    class DeepMatmulTimingModel(torch.nn.Module):
        """Minimal version of DeepMatmulTimingModel for standalone use."""
        @classmethod
        def load(cls, path: str, device: str = 'cpu'):
            """Load a model from a file."""
            return torch.load(path, map_location=device)
    
    # Define a minimal version of MatmulModelConfig for standalone use
    @dataclass
    class MatmulModelConfig:
        """Minimal version of MatmulModelConfig for standalone use."""
        model_type: str = "deep"
        problem_feature_dim: int = 0
        config_feature_dim: int = 0
        hidden_dims: Optional[List[int]] = None
        hidden_dim: int = 128
        num_layers: int = 3
        dropout_rate: float = 0.1
        hardware_name: str = "unknown"
        hardware_type: str = "unknown"  # More granular hardware type (e.g., "NVIDIA-H100")
        heuristic_name: str = "matmul"
        op_name: Optional[str] = None
        
        def __post_init__(self):
            if self.hidden_dims is None:
                self.hidden_dims = [128, 64, 32]
        
        @classmethod
        def from_dict(cls, config_dict: Dict[str, Any]) -> "MatmulModelConfig":
            """Create config from dictionary."""
            return cls(**config_dict)


def get_models_dir() -> Path:
    """
    Get the directory containing the trained models.
    
    Returns:
        Path to the models directory
    """
    # Use the default directory in the package
    package_dir = Path(__file__).parent
    return package_dir


def list_available_models(
    heuristic_name: Optional[str] = None,
    hardware_name: Optional[str] = None,
) -> List[str]:
    """
    List all available models in the models directory.
    
    Args:
        heuristic_name: Filter models by heuristic name (e.g., "matmul")
        hardware_name: Filter models by hardware name (e.g., "NVIDIA-H100", "AMD-MI250", "Intel-CPU")
    
    Returns:
        List of model file paths
    """
    models_dir = get_models_dir()
    return common_list_available_models(models_dir, heuristic_name, hardware_name)


def load_model_config(model_path: Union[str, Path]) -> Optional[MatmulModelConfig]:
    """
    Load the configuration for a model.
    
    Args:
        model_path: Path to the model file (with .pt extension)
    
    Returns:
        Model configuration if available, None otherwise
    """
    return common_load_model_config(model_path, MatmulModelConfig)


def ModelWrapper(
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    compile_model: bool = True,
    compile_options: Optional[Dict[str, Any]] = None,
) -> CommonModelWrapper:
    """
    Create a ModelWrapper instance with diode_models-specific model classes.
    
    Args:
        model_path: Path to the trained model file
        device: Device to load the model to
        compile_model: Whether to compile the model using torch.compile
        compile_options: Options to pass to torch.compile
    
    Returns:
        ModelWrapper instance configured for diode_models
    """
    # Define the model classes dictionary
    model_classes = {
        "base": MatmulTimingModel,
        "deep": DeepMatmulTimingModel
    }
    
    # Create and return the common ModelWrapper with diode_models-specific classes
    return CommonModelWrapper(
        model_path=model_path,
        device=device,
        compile_model=compile_model,
        compile_options=compile_options,
        model_classes=model_classes,
        model_config_class=MatmulModelConfig
    )
