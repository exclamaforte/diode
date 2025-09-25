# Integration package for integrating Diode models with PyTorch Inductor

from .base_integration import (
    BaseIntegration,
    IntegrationRegistry,
    ModelPointer,
    discover_and_register_integrations,
    get_integration_registry,
    get_integration_status,
    integrate_all,
    register_integration,
)
from .inductor_integration import (
    DiodeInductorChoices,
    create_diode_choices,
    install_diode_choices,
)
from .matmul_integration import (
    MatmulIntegration,
    create_matmul_integration,
)

__all__ = [
    "DiodeInductorChoices",
    "create_diode_choices",
    "install_diode_choices",
    "BaseIntegration",
    "ModelPointer",
    "IntegrationRegistry",
    "get_integration_registry",
    "register_integration",
    "integrate_all",
    "get_integration_status",
    "discover_and_register_integrations",
    "MatmulIntegration",
    "create_matmul_integration",
]
