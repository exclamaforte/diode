"""
Model wrapper for loading and running inference on trained models.

This module imports the common ModelWrapper from diode_common and extends it
with diode-specific functionality.
"""

import os
import torch
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from diode.model.matmul_timing_model import MatmulTimingModel, DeepMatmulTimingModel
from diode.model.matmul_model_config import MatmulModelConfig

# Import from the common module
from diode_common.model_wrapper import ModelWrapper as CommonModelWrapper
from diode_common.model_wrapper import load_model_config as common_load_model_config

logger = logging.getLogger(__name__)

def load_model_config(model_path):
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
    
    This class extends the common ModelWrapper with diode-specific functionality.
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
            models_dir = os.path.join(os.path.dirname(os.path.dirname(package_dir)), "trained_models")
        
        # Find all .pt files in the directory and its subdirectories
        model_files = []
        for root, _, files in os.walk(models_dir):
            for file in files:
                if file.endswith(".pt"):
                    model_files.append(os.path.join(root, file))
        
        return model_files
