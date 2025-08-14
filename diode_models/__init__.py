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

# Import the model_wrapper module to make it available at the top level
from .diode_models import model_wrapper

# Import specific functions and classes from model_wrapper for direct access
from .diode_models.model_wrapper import (
    ModelWrapper,
    load_model_config,
    list_available_models as model_wrapper_list_available_models,
    get_models_dir as model_wrapper_get_models_dir,
    MatmulTimingModel,
    DeepMatmulTimingModel,
    MatmulModelConfig
)

# Make them available at the top level
__all__ = [
    'MODELS_DIR',
    'get_models_dir', 
    'list_available_models',
    'get_model_wrapper',
    'model_wrapper',
    'ModelWrapper',
    'load_model_config',
    'MatmulTimingModel',
    'DeepMatmulTimingModel',
    'MatmulModelConfig',
    '__version__'
]
