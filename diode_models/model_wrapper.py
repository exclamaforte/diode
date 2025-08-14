"""
Model wrapper module for diode_models package.

This module re-exports all functionality from the nested model_wrapper module
to provide a clean import interface.
"""

# Re-export everything from the nested model_wrapper module
from .diode_models.model_wrapper import *
from .diode_models.model_wrapper import (
    ModelWrapper,
    load_model_config,
    list_available_models,
    get_models_dir,
    MatmulTimingModel,
    DeepMatmulTimingModel,
    MatmulModelConfig,
    CommonModelWrapper,
    DIODE_AVAILABLE
)

# Make sure __all__ includes everything that should be publicly available
__all__ = [
    'ModelWrapper',
    'load_model_config',
    'list_available_models',
    'get_models_dir',
    'MatmulTimingModel',
    'DeepMatmulTimingModel',
    'MatmulModelConfig',
    'CommonModelWrapper',
    'DIODE_AVAILABLE'
]