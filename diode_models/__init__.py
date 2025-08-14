"""
diode_models package.

This package provides pre-trained models for the diode package.
"""

# Import all the functions and attributes from the nested package
from .diode_models import (
    MODELS_DIR,
    get_models_dir,
    list_available_models,
    get_model_wrapper,
    __version__
)

# Make them available at the top level
__all__ = [
    'MODELS_DIR',
    'get_models_dir', 
    'list_available_models',
    'get_model_wrapper',
    '__version__'
]
