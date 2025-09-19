# Integration package for integrating Diode models with PyTorch Inductor

from .inductor_integration import (
    DiodeInductorChoices,
    create_diode_choices,
    install_diode_choices,
)

from .base_integration import (
    BaseIntegration,
    ModelPointer,
    IntegrationRegistry,
    get_integration_registry,
    register_integration,
    integrate_all,
    get_integration_status,
    discover_and_register_integrations,
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
