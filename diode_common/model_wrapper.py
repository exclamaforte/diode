"""
Model wrapper for loading and running inference on trained models.

This module provides common functionality for loading and running models
that is shared between the diode and diode-models packages.
"""

import os
import torch
import logging
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

logger = logging.getLogger(__name__)

def load_model_config(model_path: Union[str, Path], model_config_class=None) -> Optional[Any]:
    """
    Load the configuration for a model.
    
    Args:
        model_path: Path to the model file (with .pt extension)
        model_config_class: The class to use for the model configuration.
                           If None, the raw dictionary is returned.
    
    Returns:
        Model configuration if available, None otherwise
    """
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
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
        self.device = device
        self.compile_model = compile_model
        self.compile_options = compile_options or {}
        self.model_classes = model_classes
        self.model_config_class = model_config_class
        
        # Load the model configuration if available
        self.config = load_model_config(model_path, model_config_class)
        
        # Load the model
        self._load_model()
        
        # Compile the model if requested
        if compile_model:
            self._compile_model()
    
    def _load_model(self) -> None:
        """
        Load the model from the file.
        """
        # Check if the file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # If we have a config and model classes, use them to create the model
        if self.config is not None and self.model_classes is not None:
            model_type = self.config.model_type.lower()
            if model_type == "base":
                self.model = self.model_classes["base"](
                    problem_feature_dim=self.config.problem_feature_dim,
                    config_feature_dim=self.config.config_feature_dim,
                    hidden_dims=self.config.hidden_dims,
                    dropout_rate=self.config.dropout_rate,
                )
            elif model_type == "deep":
                self.model = self.model_classes["deep"](
                    problem_feature_dim=self.config.problem_feature_dim,
                    config_feature_dim=self.config.config_feature_dim,
                    hidden_dim=self.config.hidden_dim,
                    num_layers=self.config.num_layers,
                    dropout_rate=self.config.dropout_rate,
                )
            else:
                raise ValueError(f"Unknown model type in config: {model_type}")
            
            # Load the state dict
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device)["model_state_dict"])
        else:
            # No config or model classes available, load the model using the old method
            # Load the model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # If model_classes is provided, use it to create the model
            if self.model_classes is not None:
                # Determine the model type based on the checkpoint
                if "hidden_dims" in checkpoint:
                    # This is a base model
                    self.model = self.model_classes["base"](
                        problem_feature_dim=checkpoint["problem_feature_dim"],
                        config_feature_dim=checkpoint["config_feature_dim"],
                        hidden_dims=checkpoint["hidden_dims"],
                        dropout_rate=checkpoint["dropout_rate"],
                    )
                elif "hidden_dim" in checkpoint:
                    # This is a deep model
                    self.model = self.model_classes["deep"](
                        problem_feature_dim=checkpoint["problem_feature_dim"],
                        config_feature_dim=checkpoint["config_feature_dim"],
                        hidden_dim=checkpoint["hidden_dim"],
                        num_layers=checkpoint["num_layers"],
                        dropout_rate=checkpoint["dropout_rate"],
                    )
                else:
                    raise ValueError(f"Unknown model type in checkpoint: {self.model_path}")
                
                # Load the state dict
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # No model_classes provided, assume the checkpoint is the model itself
                self.model = checkpoint
        
        # Move the model to the device
        self.model = self.model.to(self.device)
        
        # Set the model to evaluation mode
        self.model.eval()
        
        logger.info(f"Model loaded from {self.model_path}")
    
    def _compile_model(self) -> None:
        """
        Compile the model using torch.compile.
        """
        try:
            self.compiled_model = torch.compile(self.model, **self.compile_options)
            logger.info(f"Model compiled with options: {self.compile_options}")
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
