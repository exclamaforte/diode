"""
Model wrapper for loading and running inference on trained models.
"""

import os
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

from diode.model.matmul_timing_model import MatmulTimingModel, DeepMatmulTimingModel

logger = logging.getLogger(__name__)

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
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_path: Path to the trained model file
            device: Device to load the model to
            compile_model: Whether to compile the model using torch.compile
            compile_options: Options to pass to torch.compile
        """
        self.model_path = model_path
        self.device = device
        self.compile_model = compile_model
        self.compile_options = compile_options or {}
        
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
        
        # Load the model checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Determine the model type based on the checkpoint
        if "hidden_dims" in checkpoint:
            # This is a MatmulTimingModel
            self.model = MatmulTimingModel(
                problem_feature_dim=checkpoint["problem_feature_dim"],
                config_feature_dim=checkpoint["config_feature_dim"],
                hidden_dims=checkpoint["hidden_dims"],
                dropout_rate=checkpoint["dropout_rate"],
            )
        elif "hidden_dim" in checkpoint:
            # This is a DeepMatmulTimingModel
            self.model = DeepMatmulTimingModel(
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
    
    @staticmethod
    def list_available_models(models_dir: str = None) -> List[str]:
        """
        List all available models in the models directory.
        
        Args:
            models_dir: Directory containing the models. If None, use the default
                        directory in the package.
        
        Returns:
            List of model file paths
        """
        if models_dir is None:
            # Use the default directory in the package
            package_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(os.path.dirname(os.path.dirname(package_dir)), "trained_models")
        
        # Find all .pt files in the directory and its subdirectories
        model_files = []
        for root, _, files in os.walk(models_dir):
            for file in files:
                if file.endswith(".pt"):
                    model_files.append(os.path.join(root, file))
        
        return model_files
