"""
Base integration system for torch-diode models.

This module provides a general framework for integrating trained models with PyTorch
interfaces. It handles the multi-step integration process described in the README:
1. Register dummy models to relevant torch.compile interfaces
2. Load actual models for successful registrations
3. Enable configs that engage the models
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelPointer:
    """Represents a pointer to a trained model with metadata."""
    
    def __init__(
        self,
        model_name: str,
        relative_path: str,
        model_purpose: str,
        interface_name: str,
        description: Optional[str] = None,
        version: str = "1.0",
        dependencies: Optional[List[str]] = None,
    ):
        """
        Initialize a model pointer.
        
        Args:
            model_name: Name of the model file (e.g., "matmul_v1.pt")
            relative_path: Path relative to trained_models directory
            model_purpose: Purpose category (e.g., "matmul_kernel_runtime_prediction")
            interface_name: Name of the PyTorch interface this model targets
            description: Human-readable description of the model
            version: Model version
            dependencies: List of required dependencies for this model
        """
        self.model_name = model_name
        self.relative_path = relative_path
        self.model_purpose = model_purpose
        self.interface_name = interface_name
        self.description = description or f"Model for {model_purpose}"
        self.version = version
        self.dependencies = dependencies or []

    @property
    def full_path(self) -> Path:
        """Get the full path to the model file."""
        # Find the diode package root directory
        diode_root = Path(__file__).parent.parent.parent
        if self.relative_path == ".":
            return diode_root / "trained_models" / self.model_name
        else:
            return diode_root / "trained_models" / self.relative_path / self.model_name

    def exists(self) -> bool:
        """Check if the model file exists."""
        return self.full_path.exists()

    def get_size_mb(self) -> float:
        """Get model file size in MB."""
        if self.exists():
            return self.full_path.stat().st_size / (1024 * 1024)
        return 0.0

    def __repr__(self) -> str:
        return f"ModelPointer(name={self.model_name}, purpose={self.model_purpose}, exists={self.exists()})"


class BaseIntegration(ABC):
    """
    Base class for integrating trained models with PyTorch interfaces.
    
    This class implements the general pattern described in the README:
    1. Attempt to register dummy functions
    2. Load actual models for successful registrations
    3. Enable relevant configs
    """

    def __init__(
        self,
        name: str,
        interface_name: str,
        model_pointers: List[ModelPointer],
        enable_fallback: bool = True,
    ):
        """
        Initialize the integration.
        
        Args:
            name: Name of this integration
            interface_name: Name of the PyTorch interface being integrated
            model_pointers: List of model pointers for this integration
            enable_fallback: Whether to enable fallback when models fail to load
        """
        self.name = name
        self.interface_name = interface_name
        self.model_pointers = model_pointers
        self.enable_fallback = enable_fallback
        self.loaded_models: Dict[str, Any] = {}
        self.registration_status: Dict[str, bool] = {}
        self.integration_status = "not_started"
        self.execute_order: Optional[int] = None

    @abstractmethod
    def create_dummy_function(self) -> Any:
        """
        Create a dummy function to test interface availability.
        
        Returns:
            A dummy function that can be registered with the target interface
        """
        pass

    def register_dummy(self, dummy_function: Any) -> bool:
        """
        Register the dummy function with the target interface.
        
        Args:
            dummy_function: The dummy function to register
            
        Returns:
            True if registration succeeded, False otherwise
        """
        try:
            from torch._inductor.virtualized import V
            
            # Check if something else has already been registered
            # by verifying the existing class is literally InductorChoices and not a subclass
            try:
                from torch._inductor.runtime.runtime_utils import InductorChoices
                current_handler = getattr(V, '_choices_handler', None)
                if current_handler is not None:
                    # If the current handler's class is not literally InductorChoices, 
                    # something else has been registered
                    if type(current_handler).__name__ != 'InductorChoices' or type(current_handler) is not InductorChoices:
                        logger.debug("Another choices handler already registered, skipping dummy registration")
                        return False
            except ImportError:
                # If we can't import InductorChoices, we can't check, so proceed with registration
                pass
            
            # Test that we can set a choices handler
            original_handler = getattr(V, '_choices_handler', None)
            V.set_choices_handler(dummy_function)
            
            # Verify it was set
            current_handler = getattr(V, '_choices_handler', None)
            success = current_handler is dummy_function
            
            # Restore original handler if there was one
            if original_handler is not None:
                V.set_choices_handler(original_handler)
            elif hasattr(V, '_choices_handler'):
                delattr(V, '_choices_handler')
                
            return success
            
        except (ImportError, AttributeError) as e:
            logger.debug(f"Failed to register dummy choices handler: {e}")
            return False

    @abstractmethod
    def load_model(self, model_pointer: ModelPointer) -> Any:
        """
        Load a model from a model pointer.
        
        Args:
            model_pointer: Pointer to the model to load
            
        Returns:
            The loaded model object
        """
        pass

    @abstractmethod
    def register_model(self, model: Any, model_pointer: ModelPointer) -> bool:
        """
        Register a loaded model with the target interface.
        
        Args:
            model: The loaded model
            model_pointer: Pointer to the model being registered
            
        Returns:
            True if registration succeeded, False otherwise
        """
        pass

    @abstractmethod
    def enable_configs(self) -> bool:
        """
        Enable PyTorch configs that engage the registered models.
        
        Returns:
            True if configs were enabled successfully, False otherwise
        """
        pass

    def check_dependencies(self) -> Dict[str, bool]:
        """
        Check if all required dependencies are available.
        
        Returns:
            Dictionary mapping dependency names to availability status
        """
        all_deps = set()
        for pointer in self.model_pointers:
            all_deps.update(pointer.dependencies)
        
        status = {}
        for dep in all_deps:
            try:
                __import__(dep)
                status[dep] = True
            except ImportError:
                status[dep] = False
                
        return status

    def get_available_models(self) -> List[ModelPointer]:
        """
        Get list of model pointers for models that actually exist.
        
        Returns:
            List of model pointers for existing models
        """
        return [pointer for pointer in self.model_pointers if pointer.exists()]

    def integrate(self) -> bool:
        """
        Execute the full integration process.
        
        This follows the pattern described in the README:
        1. Check dependencies
        2. Register dummy to test interface availability
        3. Load actual models for successful registration
        4. Enable configs
        
        Returns:
            True if integration succeeded, False otherwise
        """
        logger.info(f"Starting integration for {self.name}")
        self.integration_status = "in_progress"
        
        try:
            # Step 1: Check dependencies
            dep_status = self.check_dependencies()
            missing_deps = [dep for dep, available in dep_status.items() if not available]
            
            if missing_deps:
                logger.warning(f"Missing dependencies for {self.name}: {missing_deps}")
                if not self.enable_fallback:
                    self.integration_status = "failed"
                    return False

            # Step 2: Test interface availability with dummy
            dummy_function = self.create_dummy_function()
            registration_successful = self.register_dummy(dummy_function)
            
            if not registration_successful:
                logger.info(f"Interface {self.interface_name} not available, skipping {self.name}")
                self.integration_status = "interface_unavailable"
                return False

            logger.info(f"Successfully registered dummy for {self.name}")

            # Step 3: Load and register actual models
            available_models = self.get_available_models()
            
            if not available_models:
                logger.warning(f"No models available for {self.name}")
                if not self.enable_fallback:
                    self.integration_status = "no_models"
                    return False
            
            models_loaded = 0
            for model_pointer in available_models:
                try:
                    logger.info(f"Loading model: {model_pointer.model_name}")
                    model = self.load_model(model_pointer)
                    
                    if self.register_model(model, model_pointer):
                        self.loaded_models[model_pointer.model_name] = model
                        self.registration_status[model_pointer.model_name] = True
                        models_loaded += 1
                        logger.info(f"Successfully loaded and registered: {model_pointer.model_name}")
                    else:
                        self.registration_status[model_pointer.model_name] = False
                        logger.warning(f"Failed to register model: {model_pointer.model_name}")
                        
                except Exception as e:
                    logger.error(f"Failed to load model {model_pointer.model_name}: {e}")
                    self.registration_status[model_pointer.model_name] = False

            if models_loaded == 0:
                logger.warning(f"No models could be loaded for {self.name}")
                if not self.enable_fallback:
                    self.integration_status = "model_load_failed"
                    return False

            # Step 4: Enable configs
            if self.enable_configs():
                logger.info(f"Successfully enabled configs for {self.name}")
                self.integration_status = "success"
                return True
            else:
                logger.warning(f"Failed to enable configs for {self.name}")
                self.integration_status = "config_failed"
                return self.enable_fallback

        except Exception as e:
            logger.error(f"Integration failed for {self.name}: {e}")
            self.integration_status = "error"
            if not self.enable_fallback:
                raise
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get detailed status information about this integration.
        
        Returns:
            Dictionary with status information
        """
        return {
            "name": self.name,
            "interface_name": self.interface_name,
            "integration_status": self.integration_status,
            "models_available": len(self.get_available_models()),
            "models_loaded": len(self.loaded_models),
            "registration_status": self.registration_status.copy(),
            "dependencies": self.check_dependencies(),
        }


class IntegrationRegistry:
    """Registry for managing multiple integrations."""
    
    def __init__(self):
        self.integrations: Dict[str, BaseIntegration] = {}
        self.execution_order: List[str] = []

    def register(self, integration: BaseIntegration, execute_order: Optional[int] = None) -> None:
        """
        Register an integration.
        
        Args:
            integration: The integration to register
            execute_order: Optional order for execution (lower numbers execute first)
        """
        self.integrations[integration.name] = integration
        
        if execute_order is not None:
            # Insert at appropriate position based on order
            inserted = False
            for i, name in enumerate(self.execution_order):
                existing_integration = self.integrations[name]
                if getattr(existing_integration, 'execute_order', float('inf')) > execute_order:
                    self.execution_order.insert(i, integration.name)
                    inserted = True
                    break
            if not inserted:
                self.execution_order.append(integration.name)
        else:
            self.execution_order.append(integration.name)
        
        integration.execute_order = execute_order or len(self.execution_order)

    def integrate_all(self) -> Dict[str, bool]:
        """
        Execute all registered integrations in order.
        
        Returns:
            Dictionary mapping integration names to success status
        """
        results = {}
        
        for integration_name in self.execution_order:
            integration = self.integrations[integration_name]
            logger.info(f"Executing integration: {integration_name}")
            
            try:
                results[integration_name] = integration.integrate()
            except Exception as e:
                logger.error(f"Integration {integration_name} failed with exception: {e}")
                results[integration_name] = False
        
        return results

    def get_status_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive status report for all integrations.
        
        Returns:
            Dictionary with status information for all integrations
        """
        return {
            name: integration.get_status()
            for name, integration in self.integrations.items()
        }


# Global registry instance
_integration_registry = IntegrationRegistry()


def get_integration_registry() -> IntegrationRegistry:
    """Get the global integration registry."""
    return _integration_registry


def register_integration(integration: BaseIntegration, execute_order: Optional[int] = None) -> None:
    """
    Register an integration with the global registry.
    
    Args:
        integration: The integration to register
        execute_order: Optional order for execution
    """
    _integration_registry.register(integration, execute_order)


def integrate_all() -> Dict[str, bool]:
    """Execute all registered integrations."""
    return _integration_registry.integrate_all()


def get_integration_status() -> Dict[str, Any]:
    """Get status for all integrations."""
    return _integration_registry.get_status_report()


def discover_and_register_integrations() -> Dict[str, bool]:
    """
    Discover and register all available integrations.
    
    This function automatically finds integration modules, loads them,
    and registers them with the integration registry.
    
    Returns:
        Dictionary mapping integration names to discovery success status
    """
    import importlib
    
    logger.info("Starting integration discovery...")
    
    # Known integration modules - can be extended by adding new modules
    known_integrations = [
        "matmul_integration",
        # Add new integration modules here as they are created
        # "conv_integration",
        # "attention_integration", 
        # etc.
    ]
    
    discovery_results = {}
    loaded_integrations = []
    
    # Attempt to load each integration module
    for module_name in known_integrations:
        logger.debug(f"Attempting to load integration: {module_name}")
        
        try:
            # Import the module
            module = importlib.import_module(f".{module_name}", package="diode.integration")
            
            # Look for a factory function following the naming convention
            factory_function_name = f"create_{module_name.replace('_integration', '')}_integration"
            
            if hasattr(module, factory_function_name):
                factory_function = getattr(module, factory_function_name)
                logger.debug(f"Found factory function: {factory_function_name}")
                
                # Create the integration with fallback enabled
                integration = factory_function(enable_fallback=True)
                loaded_integrations.append(integration)
                discovery_results[module_name] = True
                logger.info(f"Discovered integration: {integration.name}")
            else:
                logger.warning(f"No factory function '{factory_function_name}' found in {module_name}")
                discovery_results[module_name] = False
                
        except ImportError as e:
            logger.debug(f"Could not import integration module {module_name}: {e}")
            discovery_results[module_name] = False
        except Exception as e:
            logger.warning(f"Error loading integration from {module_name}: {e}")
            discovery_results[module_name] = False
    
    # Register discovered integrations with appropriate execution order
    execution_order = 1
    for integration in loaded_integrations:
        try:
            register_integration(integration, execute_order=execution_order)
            logger.info(f"Registered integration: {integration.name} (order: {execution_order})")
            execution_order += 1
        except Exception as e:
            logger.error(f"Failed to register integration {integration.name}: {e}")
            # Mark as failed in results
            for module_name, success in discovery_results.items():
                if success and module_name in integration.name:
                    discovery_results[module_name] = False
    
    total_discovered = len(loaded_integrations)
    total_attempted = len(known_integrations)
    
    logger.info(f"Integration discovery complete: {total_discovered}/{total_attempted} integrations loaded")
    
    return discovery_results
