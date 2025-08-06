"""Models for ML heuristics."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class HeuristicModel(nn.Module):
    """Base class for heuristic models."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        """Initialize a heuristic model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [64, 32]
        
        # Build the network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)


class BinaryHeuristicModel(HeuristicModel):
    """Heuristic model for binary classification tasks."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = None):
        """Initialize a binary heuristic model.

        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__(input_dim, 1, hidden_dims)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sigmoid activation.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1) with values in [0, 1]
        """
        return torch.sigmoid(super().forward(x))


class MultiClassHeuristicModel(HeuristicModel):
    """Heuristic model for multi-class classification tasks."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = None):
        """Initialize a multi-class heuristic model.

        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__(input_dim, num_classes, hidden_dims)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with softmax activation.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, num_classes) with softmax probabilities
        """
        logits = super().forward(x)
        return torch.softmax(logits, dim=1)
