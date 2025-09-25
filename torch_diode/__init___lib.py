"""Framework for training ML diodeistics in Torch (library version without auto-registration)."""

__version__ = "0.1.0"

# Library version: does NOT auto-register with PyTorch Inductor
# Users must manually call integration functions if they want integration

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Import integration functions for manual use
from torch_diode.integration.base_integration import (
    BaseIntegration,
    IntegrationRegistry,
    ModelPointer,
    discover_and_register_integrations,
    get_integration_registry,
    get_integration_status,
    integrate_all,
)


def install_diode_integrations(enable_fallback: bool = True) -> Dict[str, bool]:
    """
    Manually install all Diode integrations with PyTorch.

    This function discovers, registers, and executes all available
    integrations. Call this if you want to use Diode integrations
    when using the library version.

    Args:
        enable_fallback: Whether to enable fallback when integrations fail

    Returns:
        Dictionary mapping integration names to success status
    """
    try:
        # Check if PyTorch is available
        import torch

        logger.info("PyTorch detected, proceeding with manual integration installation")

        # Discover and register all available integrations
        discover_and_register_integrations()

        # Execute all discovered integrations
        integration_results = integrate_all()

        successful_integrations = sum(integration_results.values())
        total_integrations = len(integration_results)

        logger.info(
            f"Manual integration installation complete: {successful_integrations}/{total_integrations} integrations successful"
        )

        return integration_results

    except ImportError:
        logger.error("PyTorch not available, cannot install Diode integrations")
        return {}
    except Exception as e:
        logger.error(f"Manual integration installation failed: {e}")
        if not enable_fallback:
            raise
        return {}


def get_diode_status() -> Dict[str, Any]:
    """
    Get comprehensive status information for all Diode integrations.

    Returns:
        Dictionary with status information for all integrations
    """
    return get_integration_status()


# Export the main functions for library users
__all__ = [
    "install_diode_integrations",
    "get_diode_status",
    "discover_and_register_integrations",
    "integrate_all",
    "get_integration_status",
    "get_integration_registry",
    "IntegrationRegistry",
    "BaseIntegration",
    "ModelPointer",
]
