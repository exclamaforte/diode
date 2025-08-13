"""
diode_models package.

This package provides pre-trained models for the diode package.
"""

import os
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional

# Define the path to the trained models
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")

def get_models_dir() -> str:
    """
    Get the path to the trained models directory.
    
    Returns:
        Path to the trained models directory
    """
    return MODELS_DIR

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
    # Start with the base directory
    search_dir = Path(MODELS_DIR)
    
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

def get_model_wrapper():
    """
    Get the ModelWrapper class.
    
    This function tries to import the ModelWrapper class in the following order:
    1. From the diode package
    2. From the local copy
    
    Returns:
        The ModelWrapper class
    """
    try:
        # Try to import from the diode package
        from diode.model.model_wrapper import ModelWrapper
        return ModelWrapper
    except ImportError:
        # Fall back to the local copy
        from .model_wrapper import ModelWrapper
        return ModelWrapper

# Version of the package
__version__ = "0.1.0"
