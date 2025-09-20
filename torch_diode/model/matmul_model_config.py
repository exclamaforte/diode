"""
Configuration for matrix multiplication timing prediction models.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class MatmulModelConfig:
    """Configuration for matrix multiplication model training."""
    # Model parameters
    model_type: str = "deep"  # "base" or "deep"
    problem_feature_dim: int = 0  # Will be set based on dataset
    config_feature_dim: int = 0  # Will be set based on dataset
    hidden_dims: List[int] = None  # For base model
    hidden_dim: int = 128  # For deep model
    num_layers: int = 10  # For deep model
    dropout_rate: float = 0.1
    
    # Training parameters
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    patience: int = 20
    log_transform: bool = True
    
    # Hardware and operation info
    hardware_name: str = "unknown"
    hardware_type: str = "unknown"  # More granular hardware type (e.g., "NVIDIA-H100")
    heuristic_name: str = "matmul"
    op_name: Optional[str] = None
    
    # Other parameters
    seed: int = 42
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MatmulModelConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: Path) -> None:
        """Save configuration to a JSON file.
        
        Args:
            path: Path to save the configuration to (without extension)
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save as JSON
        config_path = path.with_suffix(".json")
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved config to {config_path}")
    
    @classmethod
    def load(cls, path: Path) -> "MatmulModelConfig":
        """Load configuration from a JSON file.
        
        Args:
            path: Path to load the configuration from (without extension)
            
        Returns:
            Loaded configuration
        """
        # Try with .json extension
        config_path = path.with_suffix(".json")
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load from JSON
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        config = cls.from_dict(config_dict)
        logger.info(f"Loaded config from {config_path}")
        
        return config
