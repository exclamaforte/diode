"""
Model registry for tracking and managing trained models in torch-diode.

This module provides centralized management of model pointers and paths,
making it easy for the build system to know which models to include.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .integration.base_integration import ModelPointer

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing all available models in torch-diode."""

    def __init__(self):
        self._models: Dict[str, ModelPointer] = {}
        self._initialize_default_models()

    def _initialize_default_models(self) -> None:
        """Initialize the registry with default model configurations."""
        
        # Matmul kernel prediction models
        matmul_models = [
            ModelPointer(
                model_name="v1_model.pt",
                relative_path="matmul_kernel_runtime_prediction",
                model_purpose="matmul_kernel_runtime_prediction",
                interface_name="torch._inductor.choices",
                description="Matrix multiplication kernel runtime prediction model v1",
                version="1.0",
                dependencies=["torch._inductor", "torch._inductor.choices"],
            ),
            ModelPointer(
                model_name="matmul_model_exhaustive.pt",
                relative_path=".",  # Root of trained_models directory
                model_purpose="matmul_kernel_runtime_prediction",
                interface_name="torch._inductor.choices",
                description="Matrix multiplication kernel runtime prediction model (exhaustive)",
                version="1.0",
                dependencies=["torch._inductor", "torch._inductor.choices"],
            ),
        ]

        for model in matmul_models:
            self.register_model(model)

    def register_model(self, model_pointer: ModelPointer) -> None:
        """
        Register a model in the registry.
        
        Args:
            model_pointer: The model pointer to register
        """
        key = f"{model_pointer.model_purpose}_{model_pointer.model_name}"
        self._models[key] = model_pointer
        logger.debug(f"Registered model: {key}")

    def get_model(self, model_purpose: str, model_name: str) -> Optional[ModelPointer]:
        """
        Get a specific model pointer.
        
        Args:
            model_purpose: Purpose category of the model
            model_name: Name of the model file
            
        Returns:
            Model pointer if found, None otherwise
        """
        key = f"{model_purpose}_{model_name}"
        return self._models.get(key)

    def get_models_by_purpose(self, model_purpose: str) -> List[ModelPointer]:
        """
        Get all models for a specific purpose.
        
        Args:
            model_purpose: Purpose category to filter by
            
        Returns:
            List of model pointers for the given purpose
        """
        return [
            model for model in self._models.values()
            if model.model_purpose == model_purpose
        ]

    def get_models_by_interface(self, interface_name: str) -> List[ModelPointer]:
        """
        Get all models that target a specific interface.
        
        Args:
            interface_name: Interface name to filter by
            
        Returns:
            List of model pointers for the given interface
        """
        return [
            model for model in self._models.values()
            if model.interface_name == interface_name
        ]

    def get_all_models(self) -> List[ModelPointer]:
        """
        Get all registered models.
        
        Returns:
            List of all model pointers
        """
        return list(self._models.values())

    def get_existing_models(self) -> List[ModelPointer]:
        """
        Get all models that actually exist on disk.
        
        Returns:
            List of model pointers for existing models
        """
        return [model for model in self._models.values() if model.exists()]

    def get_model_paths_for_build(self) -> List[Path]:
        """
        Get all model paths that should be included in builds.
        
        This method is used by the build system to determine which model files
        to include in the distribution packages.
        
        Returns:
            List of paths to model files that exist
        """
        paths = []
        for model in self.get_existing_models():
            paths.append(model.full_path)
        return paths

    def get_model_info_for_build(self) -> Dict[str, Dict]:
        """
        Get model information formatted for build system consumption.
        
        Returns:
            Dictionary mapping model purposes to model information
        """
        info = {}
        
        for model in self.get_existing_models():
            purpose = model.model_purpose
            if purpose not in info:
                info[purpose] = {
                    "models": [],
                    "interface": model.interface_name,
                    "dependencies": set(),
                }
            
            info[purpose]["models"].append({
                "name": model.model_name,
                "path": str(model.full_path),
                "relative_path": model.relative_path,
                "size_mb": model.get_size_mb(),
                "version": model.version,
                "description": model.description,
            })
            
            info[purpose]["dependencies"].update(model.dependencies)
        
        # Convert sets to lists for JSON serialization
        for purpose_info in info.values():
            purpose_info["dependencies"] = list(purpose_info["dependencies"])
        
        return info

    def generate_manifest(self) -> Dict[str, Any]:
        """
        Generate a manifest of all models for build tools.
        
        Returns:
            Dictionary containing model manifest information
        """
        existing_models = self.get_existing_models()
        
        manifest = {
            "version": "1.0",
            "total_models": len(existing_models),
            "total_size_mb": sum(model.get_size_mb() for model in existing_models),
            "models_by_purpose": {},
            "models_by_interface": {},
            "all_dependencies": set(),
        }
        
        # Group by purpose
        for model in existing_models:
            purpose = model.model_purpose
            if purpose not in manifest["models_by_purpose"]:
                manifest["models_by_purpose"][purpose] = []
            
            manifest["models_by_purpose"][purpose].append({
                "name": model.model_name,
                "relative_path": model.relative_path,
                "size_mb": model.get_size_mb(),
                "version": model.version,
            })

        # Group by interface
        for model in existing_models:
            interface = model.interface_name
            if interface not in manifest["models_by_interface"]:
                manifest["models_by_interface"][interface] = []
            
            manifest["models_by_interface"][interface].append({
                "name": model.model_name,
                "purpose": model.model_purpose,
            })
            
            manifest["all_dependencies"].update(model.dependencies)
        
        # Convert set to list for JSON serialization
        manifest["all_dependencies"] = list(manifest["all_dependencies"])
        
        return manifest


# Global registry instance
_model_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Get the global model registry."""
    return _model_registry


def register_model(model_pointer: ModelPointer) -> None:
    """Register a model with the global registry."""
    _model_registry.register_model(model_pointer)


def get_model_paths_for_build() -> List[Path]:
    """Get all model paths for the build system."""
    return _model_registry.get_model_paths_for_build()


def get_model_info_for_build() -> Dict[str, Dict]:
    """Get model information for the build system.""" 
    return _model_registry.get_model_info_for_build()


def generate_model_manifest() -> Dict[str, Any]:
    """Generate a model manifest for build tools."""
    return _model_registry.generate_manifest()
