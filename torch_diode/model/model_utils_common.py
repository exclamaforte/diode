"""
Common utilities for matrix multiplication timing models.
"""

import torch
import torch.nn as nn
import logging
import os

logger = logging.getLogger(__name__)


def init_model_weights(module: nn.Module) -> None:
    """
    Initialize weights to avoid NaN issues during training.
    This function is shared across all timing models.
    
    Args:
        module: The module to initialize weights for
    """
    if isinstance(module, nn.Linear):
        # Use Xavier/Glorot initialization for better gradient flow
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def safe_create_directory(path: str) -> None:
    """
    Safely create a directory for the given file path.
    
    Args:
        path: File path that may contain a directory component
    """
    directory = os.path.dirname(path)
    if directory:  # Only create directory if path has a directory component
        os.makedirs(directory, exist_ok=True)


def load_model_checkpoint(path: str, device: str = 'cpu') -> dict:
    """
    Load a model checkpoint from file.
    
    Args:
        path: Path to load the model from
        device: Device to load the model to
        
    Returns:
        The loaded checkpoint dictionary
    """
    return torch.load(path, map_location=device)


def save_model_checkpoint(model: nn.Module, checkpoint_data: dict, path: str) -> None:
    """
    Save a model checkpoint to file.
    
    Args:
        model: The model to save
        checkpoint_data: Additional data to save with the model
        path: Path to save the model to
    """
    safe_create_directory(path)
    
    # Prepare the checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        **checkpoint_data
    }
    
    # Save the model
    torch.save(checkpoint, path)
    
    logger.info(f"Model saved to {path}")