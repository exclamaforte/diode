"""
Model wrapper for loading and running inference on trained models.

This module provides diode-specific functionality by using the common ModelWrapper
from diode_common with diode-specific model classes.
"""

import os
import torch
from typing import List, Optional, Any, Dict

from diode.model.matmul_timing_model import MatmulTimingModel, DeepMatmulTimingModel
from diode.model.matmul_model_config import MatmulModelConfig

# Import from the common module
from diode_common.diode_common.model_wrapper import ModelWrapper as CommonModelWrapper
from diode_common.diode_common.model_wrapper import load_model_config as common_load_model_config
from diode_common.diode_common.model_wrapper import list_available_models as common_list_available_models


def load_model_config(model_path):
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
    Create a ModelWrapper instance with diode-specific model classes.
    
    Args:
        model_path: Path to the trained model file
        device: Device to load the model to
        compile_model: Whether to compile the model using torch.compile
        compile_options: Options to pass to torch.compile
    
    Returns:
        ModelWrapper instance configured for diode models
    """
    # Define the model classes dictionary
    model_classes = {
        "base": MatmulTimingModel,
        "deep": DeepMatmulTimingModel
    }
    
    # Create and return the common ModelWrapper with diode-specific classes
    return CommonModelWrapper(
        model_path=model_path,
        device=device,
        compile_model=compile_model,
        compile_options=compile_options,
        model_classes=model_classes,
        model_config_class=MatmulModelConfig
    )


def list_available_models(models_dir: Optional[str] = None) -> List[str]:
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
    
    return common_list_available_models(models_dir)
