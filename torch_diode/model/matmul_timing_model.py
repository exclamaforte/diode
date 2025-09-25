"""
Neural network model for predicting matrix multiplication timing.
"""

import logging

import torch
import torch.nn as nn

from .matmul_inference import MatmulInferenceInterface
from .model_utils_common import init_model_weights, save_model_checkpoint

logger = logging.getLogger(__name__)


class MatmulTimingModel(MatmulInferenceInterface):
    """
    Neural network model for predicting matrix multiplication timing.

    This model takes matrix multiplication problem features and configuration features
    as input and predicts the execution time in log space.
    """

    def __init__(
        self,
        problem_feature_dim: int,
        config_feature_dim: int,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.2,
    ):
        """
        Initialize the model.

        Args:
            problem_feature_dim: Dimension of the problem features
            config_feature_dim: Dimension of the configuration features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        if hidden_dims is None:
            hidden_dims = [256, 512, 256, 128, 64]
        super().__init__(problem_feature_dim, config_feature_dim)

        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Create the layers
        layers = []

        # Input layer
        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.model = nn.Sequential(*layers)

        # Initialize weights to avoid NaN issues
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to avoid NaN issues during training."""
        self.apply(init_model_weights)

    def forward(
        self, problem_features: torch.Tensor, config_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            problem_features: Tensor of shape (batch_size, problem_feature_dim)
            config_features: Tensor of shape (batch_size, config_feature_dim)

        Returns:
            Tensor of shape (batch_size, 1) containing the predicted log execution time
        """
        # Concatenate the features
        x = torch.cat([problem_features, config_features], dim=1)

        # Forward pass through the model
        return self.model(x)

    def save(self, path: str) -> None:
        """
        Save the model to a file.

        Args:
            path: Path to save the model to
        """
        checkpoint_data = {
            "problem_feature_dim": self.problem_feature_dim,
            "config_feature_dim": self.config_feature_dim,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
        }
        save_model_checkpoint(self, checkpoint_data, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MatmulTimingModel":
        """
        Load a model from a file.

        Args:
            path: Path to load the model from
            device: Device to load the model to

        Returns:
            The loaded model
        """
        # Load the model
        checkpoint = torch.load(path, map_location=device)

        # Create the model
        model = cls(
            problem_feature_dim=checkpoint["problem_feature_dim"],
            config_feature_dim=checkpoint["config_feature_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout_rate=checkpoint["dropout_rate"],
        )

        # Load the state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        # Move the model to the device
        model = model.to(device)

        logger.info(f"Model loaded from {path}")

        return model


class DeepMatmulTimingModel(MatmulInferenceInterface):
    """
    A deeper neural network model for predicting matrix multiplication timing.

    This model has more layers than the base model and uses residual connections
    to help with training deeper networks.
    """

    def __init__(
        self,
        problem_feature_dim: int,
        config_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 10,
        dropout_rate: float = 0.2,
    ):
        """
        Initialize the model.

        Args:
            problem_feature_dim: Dimension of the problem features
            config_feature_dim: Dimension of the configuration features
            hidden_dim: Dimension of the hidden layers
            num_layers: Number of hidden layers
            dropout_rate: Dropout rate for regularization
        """
        super().__init__(problem_feature_dim, config_feature_dim)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
        )

        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(ResidualBlock(hidden_dim, dropout_rate))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)

        # Initialize weights to avoid NaN issues
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to avoid NaN issues during training."""
        self.apply(init_model_weights)

    def forward(
        self, problem_features: torch.Tensor, config_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            problem_features: Tensor of shape (batch_size, problem_feature_dim)
            config_features: Tensor of shape (batch_size, config_feature_dim)

        Returns:
            Tensor of shape (batch_size, 1) containing the predicted log execution time
        """
        # Concatenate the features
        x = torch.cat([problem_features, config_features], dim=1)

        # Input layer
        x = self.input_layer(x)

        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)

        # Output layer
        return self.output_layer(x)

    def save(self, path: str) -> None:
        """
        Save the model to a file.

        Args:
            path: Path to save the model to
        """
        checkpoint_data = {
            "problem_feature_dim": self.problem_feature_dim,
            "config_feature_dim": self.config_feature_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
        }
        save_model_checkpoint(self, checkpoint_data, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DeepMatmulTimingModel":
        """
        Load a model from a file.

        Args:
            path: Path to load the model from
            device: Device to load the model to

        Returns:
            The loaded model
        """
        # Load the model
        checkpoint = torch.load(path, map_location=device)

        # Create the model
        model = cls(
            problem_feature_dim=checkpoint["problem_feature_dim"],
            config_feature_dim=checkpoint["config_feature_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            dropout_rate=checkpoint["dropout_rate"],
        )

        # Load the state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        # Move the model to the device
        model = model.to(device)

        logger.info(f"Model loaded from {path}")

        return model


class ResidualBlock(nn.Module):
    """
    Residual block for the deep model.
    """

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        """
        Initialize the residual block.

        Args:
            hidden_dim: Dimension of the hidden layers
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        identity = x
        out = self.block(x)
        out += identity
        out = self.relu(out)
        return out
