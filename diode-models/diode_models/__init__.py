"""
diode-models package.

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

def list_available_models() -> List[str]:
    """
    List all available models in the models directory.
    
    Returns:
        List of model file paths
    """
    # Find all .pt files in the directory and its subdirectories
    model_files = []
    for root, _, files in os.walk(MODELS_DIR):
        for file in files:
            if file.endswith(".pt"):
                model_files.append(os.path.join(root, file))
    
    return model_files

def get_model_wrapper():
    """
    Get the ModelWrapper class.
    
    This function imports the ModelWrapper class from the diode package.
    If the diode package is not installed, it falls back to a local copy.
    
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
