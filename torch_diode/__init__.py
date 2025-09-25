"""
Framework for training ML diagnostics in Torch.

Importing this module has the following side effects:
1. Attempt to import `torch`.
2. Register dummy models to the relevant `torch.compile` interfaces.
3. For each registration that is successful, it will load the actual model and register it.
4. Enable the configs in `torch.compile` that engage the models.
"""

import logging
import warnings
from typing import Any, Dict, Optional

__version__ = "0.1.0"

# Configure logging for diode
logger = logging.getLogger(__name__)

# Import status tracking
_import_status: Dict[str, Any] = {
    "torch_available": False,
    "integrations_attempted": False,
    "integrations_successful": {},
    "errors": [],
}


def _attempt_torch_import() -> bool:
    """
    Step 1: Attempt to import torch.

    Returns:
        True if torch is available, False otherwise
    """
    try:
        import torch

        _import_status["torch_available"] = True
        logger.info(f"Successfully imported torch {torch.__version__}")
        return True
    except ImportError as e:
        _import_status["torch_available"] = False
        _import_status["errors"].append(f"Failed to import torch: {e}")
        logger.warning(f"PyTorch not available: {e}")
        return False


def _setup_integrations() -> Dict[str, bool]:
    """
    Steps 2-4: Register dummy models, load actual models, enable configs.

    This implements the integration pattern described in the README:
    - Register dummy models to test interface availability
    - Load actual models for successful registrations
    - Enable PyTorch configs that engage the models

    Returns:
        Dictionary mapping integration names to success status
    """
    _import_status["integrations_attempted"] = True

    try:
        # Import integration components
        from .integration import discover_and_register_integrations, integrate_all

        logger.info("Starting integration discovery...")

        # First discover and register all available integrations
        discovery_results = discover_and_register_integrations()
        logger.info(
            f"Integration discovery completed: {len(discovery_results)} integrations discovered"
        )

        # Execute all discovered integrations
        results = integrate_all()

        # Store results
        _import_status["integrations_successful"] = results

        # Log summary
        successful_count = sum(1 for success in results.values() if success)
        total_count = len(results)

        if successful_count > 0:
            logger.info(
                f"Successfully integrated {successful_count}/{total_count} model types"
            )
            for name, success in results.items():
                if success:
                    logger.info(f"  ✓ {name}")
                else:
                    logger.warning(f"  ✗ {name}")
        else:
            logger.warning("No model integrations were successful")

        return results

    except Exception as e:
        error_msg = f"Failed to setup integrations: {e}"
        _import_status["errors"].append(error_msg)
        logger.error(error_msg, exc_info=True)
        return {}


def get_import_status() -> Dict[str, Any]:
    """
    Get the status of the torch-diode import process.

    Returns:
        Dictionary containing import status information
    """
    return _import_status.copy()


def get_integration_info() -> Optional[Dict[str, Any]]:
    """
    Get detailed information about loaded integrations.

    Returns:
        Dictionary with integration status, or None if integrations not attempted
    """
    if not _import_status["integrations_attempted"]:
        return None

    try:
        from .integration import get_integration_status

        return get_integration_status()
    except ImportError:
        return None


def get_model_info() -> Optional[Dict[str, Any]]:
    """
    Get information about available and loaded models.

    Returns:
        Dictionary with model information, or None if not available
    """
    try:
        from .model_registry import get_model_registry

        registry = get_model_registry()

        return {
            "available_models": len(registry.get_existing_models()),
            "model_manifest": registry.generate_manifest(),
            "models_by_purpose": {
                purpose: [model.model_name for model in models]
                for purpose, models in registry.generate_manifest()[
                    "models_by_purpose"
                ].items()
            },
        }
    except ImportError:
        return None


# Execute the import side effects
_torch_available = _attempt_torch_import()

if _torch_available:
    # Only attempt integrations if torch is available
    _integration_results = _setup_integrations()
else:
    logger.info("Skipping model integrations because PyTorch is not available")


# Expose public API
from . import collection, integration, model, types, utils
from .model_registry import get_model_registry
from .utils.debug_config import get_debug_flags, set_debug_flag

__all__ = [
    "__version__",
    "collection",
    "integration",
    "model",
    "types",
    "utils",
    "get_import_status",
    "get_integration_info",
    "get_model_info",
    "get_model_registry",
]


# Display initialization summary
def _display_init_summary():
    """Display a summary of the initialization process."""
    if not _torch_available:
        warnings.warn(
            "PyTorch is not available. torch-diode will work in library-only mode. "
            "Install PyTorch 2.9+ or nightly to enable automatic model integration.",
            UserWarning, stacklevel=2,
        )
        return

    successful_integrations = sum(
        1 for success in _import_status["integrations_successful"].values() if success
    )

    if successful_integrations > 0:
        logger.info(
            f"torch-diode initialized with {successful_integrations} active model integrations"
        )
    else:
        logger.info(
            "torch-diode initialized in library-only mode (no model integrations active)"
        )


# Run initialization summary
_display_init_summary()
