"""
Model wrapper for loading and running inference on trained models.

This module provides functionality to:
1. Load a trained model from a file
2. Compile the model using torch.compile
3. Run inference on the model
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch_diode.model.matmul_model_config import MatmulModelConfig

from torch_diode.model.matmul_timing_model import (
    DeepMatmulTimingModel,
    MatmulTimingModel,
)

logger = logging.getLogger(__name__)


def list_available_models(
    models_dir: Optional[Union[str, Path]] = None,
    heuristic_name: Optional[str] = None,
    hardware_name: Optional[str] = None,
) -> List[str]:
    """
    List all available models in the models directory.

    Args:
        models_dir: Directory containing the models. If None, use the default
                   directory in the package.
        heuristic_name: Filter models by heuristic name (e.g., "matmul")
        hardware_name: Filter models by hardware name (e.g., "NVIDIA-H100", "AMD-MI250", "Intel-CPU")

    Returns:
        List of model file paths
    """
    if models_dir is None:
        # Use the default directory in the package
        package_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(package_dir)), "trained_models"
        )

    models_dir = Path(models_dir)

    # Start with the base directory
    search_dir = models_dir

    # If heuristic is specified, add it to the path
    if heuristic_name:
        search_dir = search_dir / heuristic_name

        # If hardware is also specified, add it to the path
        if hardware_name:
            search_dir = search_dir / hardware_name

    # Find all .pt files in the directory and its subdirectories
    model_files = []
    if search_dir.exists():
        model_files.extend(str(path) for path in search_dir.glob("**/*.pt"))

    return model_files


def load_model_config(
    model_path: Union[str, Path], model_config_class=None
) -> Optional[Any]:
    """
    Load the configuration for a model.

    Args:
        model_path: Path to the model file (with .pt extension)
        model_config_class: The class to use for the model configuration.
                           If None, MatmulModelConfig is used by default.

    Returns:
        Model configuration if available, None otherwise
    """
    if model_config_class is None:
        model_config_class = MatmulModelConfig

    model_path = Path(model_path)
    config_path_json = model_path.with_suffix(".json")

    # Try to load the configuration
    if config_path_json.exists():
        # Load from JSON
        with open(config_path_json, "r") as f:
            config_dict = json.load(f)

        if model_config_class is not None:
            return model_config_class.from_dict(config_dict)
        else:
            return config_dict

    return None


class ModelWrapper:
    """
    Wrapper for loading and running inference on trained models.

    This class provides functionality to:
    1. Load a trained model from a file
    2. Compile the model using torch.compile
    3. Run inference on the model
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        compile_model: bool = True,
        compile_options: Optional[Dict[str, Any]] = None,
        model_classes: Optional[Dict[str, Any]] = None,
        model_config_class=None,
    ):
        """
        Initialize the model wrapper.

        Args:
            model_path: Path to the trained model file
            device: Device to load the model to
            compile_model: Whether to compile the model using torch.compile
            compile_options: Options to pass to torch.compile
            model_classes: Dictionary mapping model types to model classes
            model_config_class: The class to use for the model configuration
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compile_model = compile_model
        self.compile_options = compile_options or {}

        # Use diode-specific model classes if none provided
        if model_classes is None:
            from .matmul_model_v1 import MatmulModelV1

            model_classes = {
                "base": MatmulTimingModel,
                "deep": DeepMatmulTimingModel,
                "v1": MatmulModelV1,
            }
        self.model_classes = model_classes

        # Use diode-specific config class if none provided
        if model_config_class is None:
            model_config_class = MatmulModelConfig
        self.model_config_class = model_config_class

        # Load the model configuration if available
        self.config = load_model_config(model_path, model_config_class)

        # Load the model
        self._load_model()

        # Compile the model if requested
        if compile_model:
            self._compile_model()

    @staticmethod
    def list_available_models(
        models_dir: Optional[Union[str, Path]] = None,
        heuristic_name: Optional[str] = None,
        hardware_name: Optional[str] = None,
    ) -> List[str]:
        """
        List all available models in the models directory.

        Args:
            models_dir: Directory containing the models. If None, use the default
                       directory in the package.
            heuristic_name: Filter models by heuristic name (e.g., "matmul")
            hardware_name: Filter models by hardware name (e.g., "NVIDIA-H100", "AMD-MI250", "Intel-CPU")

        Returns:
            List of model file paths
        """
        return list_available_models(models_dir, heuristic_name, hardware_name)

    def _create_model_from_config(self, config_data: Dict[str, Any]) -> torch.nn.Module:
        """
        Create a model from configuration data.

        Args:
            config_data: Dictionary containing model configuration

        Returns:
            Created model instance
        """
        # Determine model type from config_data
        if "hidden_layer_widths" in config_data and "kernel_overhead" in config_data:
            # This is a V1 model
            return self.model_classes["v1"](
                problem_feature_dim=config_data["problem_feature_dim"],
                config_feature_dim=config_data["config_feature_dim"],
                hidden_layer_widths=config_data["hidden_layer_widths"],
                kernel_overhead=config_data["kernel_overhead"],
                dropout_rate=config_data["dropout_rate"],
            )
        elif "hidden_dims" in config_data:
            # This is a base model
            return self.model_classes["base"](
                problem_feature_dim=config_data["problem_feature_dim"],
                config_feature_dim=config_data["config_feature_dim"],
                hidden_dims=config_data["hidden_dims"],
                dropout_rate=config_data["dropout_rate"],
            )
        elif "hidden_dim" in config_data:
            # This is a deep model
            return self.model_classes["deep"](
                problem_feature_dim=config_data["problem_feature_dim"],
                config_feature_dim=config_data["config_feature_dim"],
                hidden_dim=config_data["hidden_dim"],
                num_layers=config_data["num_layers"],
                dropout_rate=config_data["dropout_rate"],
            )
        else:
            raise ValueError(
                f"Unknown model type in config data: {list(config_data.keys())}"
            )

    def _load_model(self) -> None:
        """
        Load the model from the file.
        """
        # Check if the file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load the model checkpoint
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )

        # If we have a config, use it; otherwise use checkpoint data
        if self.config is not None:
            # Convert config to dict-like structure for _create_model_from_config
            config_data = {
                "problem_feature_dim": self.config.problem_feature_dim,
                "config_feature_dim": self.config.config_feature_dim,
                "dropout_rate": self.config.dropout_rate,
            }

            # Add model-type specific fields
            model_type = self.config.model_type.lower()
            if model_type == "base":
                config_data["hidden_dims"] = self.config.hidden_dims
            elif model_type == "deep":
                config_data["hidden_dim"] = self.config.hidden_dim
                config_data["num_layers"] = self.config.num_layers
            else:
                raise ValueError(f"Unknown model type in config: {model_type}")

            self.model = self._create_model_from_config(config_data)
        else:
            # Use checkpoint data directly
            self.model = self._create_model_from_config(checkpoint)

        # Load the state dict
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # For backwards compatibility with models that store state dict directly
            self.model.load_state_dict(checkpoint)

        # Move the model to the device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded from {self.model_path}")

    def _compile_model(self) -> None:
        """
        Compile the model using torch.compile.
        """
        # Check if we're on CPU - torch.compile has limited support for CPU
        # For now, skip compilation on CPU to avoid compatibility issues
        if self.device == "cpu":
            logger.info("Skipping model compilation on CPU - using eager mode")
            self.compiled_model = self.model
            return

        try:
            # Configure Inductor for compilation
            import torch._inductor.config as inductor_config

            # For CUDA devices, ensure we have proper backend configuration
            if self.device.startswith("cuda"):
                # Enable Triton for CUDA compilation
                if (
                    not hasattr(inductor_config, "max_autotune_gemm_backends")
                    or not inductor_config.max_autotune_gemm_backends
                ):
                    inductor_config.max_autotune_gemm_backends = "TRITON,ATEN"
                elif "TRITON" not in str(inductor_config.max_autotune_gemm_backends):
                    inductor_config.max_autotune_gemm_backends = (
                        f"TRITON,{inductor_config.max_autotune_gemm_backends}"
                    )

            # Use the provided compile options or defaults
            compile_options = self.compile_options.copy()
            if not compile_options:
                compile_options = {"mode": "default"}

            self.compiled_model = torch.compile(self.model, **compile_options)
            logger.info(f"Model compiled with options: {compile_options}")

        except Exception as e:
            logger.warning(f"Failed to compile model: {e}")
            logger.warning("Using uncompiled model for inference")
            self.compiled_model = self.model

    def predict(
        self,
        problem_features: torch.Tensor,
        config_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run inference on the model.

        Args:
            problem_features: Tensor of shape (batch_size, problem_feature_dim)
            config_features: Tensor of shape (batch_size, config_feature_dim)

        Returns:
            Tensor of shape (batch_size, 1) containing the predicted log execution time
        """
        # Move the inputs to the device
        problem_features = problem_features.to(self.device)
        config_features = config_features.to(self.device)

        # Run inference
        with torch.no_grad():
            if self.compile_model and hasattr(self, "compiled_model"):
                predictions = self.compiled_model(problem_features, config_features)
            else:
                predictions = self.model(problem_features, config_features)

        return predictions

    def inference(self, combined_features: torch.Tensor) -> torch.Tensor:
        """
        Run inference with combined features tensor.

        Args:
            combined_features: Tensor of shape (batch_size, problem_feature_dim + config_feature_dim)

        Returns:
            Tensor of shape (batch_size, 1) containing the predicted log execution time
        """
        combined_features = combined_features.to(self.device)

        # Split the combined features back into problem and config features
        problem_dim = getattr(self.model, "problem_feature_dim", 7)
        problem_features = combined_features[:, :problem_dim]
        config_features = combined_features[:, problem_dim:]

        return self.predict(problem_features, config_features)

    def predict_from_features(
        self, problem_features: torch.Tensor, config_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Alias for predict method to match UnifiedMatmulPredictor interface.

        Args:
            problem_features: Tensor of shape (batch_size, problem_feature_dim)
            config_features: Tensor of shape (batch_size, config_feature_dim)

        Returns:
            Tensor of shape (batch_size, 1) containing the predicted log execution time
        """
        return self.predict(problem_features, config_features)
