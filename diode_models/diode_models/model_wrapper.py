"""
Model wrapper for loading and running inference on trained models.

This module imports the common ModelWrapper from diode_common and extends it
with diode_models-specific functionality.
"""

import os
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass

# Import from the common module
from diode_common.model_wrapper import ModelWrapper as CommonModelWrapper
from diode_common.model_wrapper import load_model_config as common_load_model_config

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
        hidden_dims: List[int] = None
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

logger = logging.getLogger(__name__)

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
    
    # Start with the base directory
    search_dir = models_dir
    
    # If heuristic is specified, add it to the path
    if heuristic_name:
        search_dir = search_dir / heuristic_name
        
        # If hardware is also specified, add it to the path
        if hardware_name:
            search_dir = search_dir / hardware_name
    
    # Find all .pt files in the directory and its subdirectories
    model_files = []
    if search_dir.exists():
        for path in search_dir.glob("**/*.pt"):
            model_files.append(str(path))
    
    return model_files


def load_model_config(model_path: Union[str, Path]) -> Optional[MatmulModelConfig]:
    """
    Load the configuration for a model.
    
    Args:
        model_path: Path to the model file (with .pt extension)
    
    Returns:
        Model configuration if available, None otherwise
    """
    return common_load_model_config(model_path, MatmulModelConfig)


class ModelWrapper(CommonModelWrapper):
    """
    Wrapper for loading and running inference on trained models.
    
    This class extends the common ModelWrapper with diode_models-specific functionality.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        compile_model: bool = True,
        compile_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_path: Path to the trained model file
            device: Device to load the model to
            compile_model: Whether to compile the model using torch.compile
            compile_options: Options to pass to torch.compile
        """
        # Define the model classes dictionary
        model_classes = {
            "base": MatmulTimingModel,
            "deep": DeepMatmulTimingModel
        }
        
        # Call the parent class constructor
        super().__init__(
            model_path=model_path,
            device=device,
            compile_model=compile_model,
            compile_options=compile_options,
            model_classes=model_classes,
            model_config_class=MatmulModelConfig
        )
    
    @staticmethod
    def list_available_models(models_dir: str = None) -> List[str]:
        """
        List all available models in the models directory.
        
        Args:
            models_dir: Directory containing the models. If None, use the default
                        directory in the package.
        
        Returns:
            List of model file paths
        """
        if models_dir is None:
            # Use the default directory in the package
            package_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(package_dir, "trained_models")
        
        # Find all .pt files in the directory and its subdirectories
        model_files = []
        for root, _, files in os.walk(models_dir):
            for file in files:
                if file.endswith(".pt"):
                    model_files.append(os.path.join(root, file))
        
        return model_files
