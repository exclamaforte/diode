"""
Neural network model for predicting matrix multiplication timing (V1).
"""

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# from pyre_extensions import assert_is_instance
# Fallback for environments without pyre_extensions
try:
    from pyre_extensions import assert_is_instance
except ImportError:
    def assert_is_instance(obj, expected_type):
        """Fallback implementation for assert_is_instance when pyre_extensions is not available."""
        if not isinstance(obj, expected_type):
            raise TypeError(f"Expected {expected_type}, got {type(obj)}")
        return obj

from .model_utils_common import (
    init_model_weights,
    save_model_checkpoint,
)

logger = logging.getLogger(__name__)


class MatmulModelV1(nn.Module):
    """
    Neural network model for predicting matrix multiplication timing (V1).

    This model is designed for modeling runtime when there is a constant overhead of
    `kernel_overhead` and the non-overhead runtime tends to be easier to model
    on a log scale.
    """

    def __init__(
        self,
        problem_feature_dim: int,
        config_feature_dim: int,
        hidden_layer_widths: List[int] = [256, 256, 256, 256, 256, 256],
        kernel_overhead: float = 0.00541,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Initialize the model.

        Args:
            problem_feature_dim: Dimension of the problem features
            config_feature_dim: Dimension of the configuration features
            hidden_layer_widths: Hidden layer widths
            kernel_overhead: Overhead of the kernel, assumed to be constant. The
                default of 0.00541 is the lowest runtime seen in Triton H100 data.
            dropout_rate: Dropout rate for regularization (kept for interface compatibility)
        """
        super().__init__()

        self.problem_feature_dim = problem_feature_dim
        self.config_feature_dim = config_feature_dim
        self.input_dim = problem_feature_dim + config_feature_dim
        self.hidden_layer_widths = hidden_layer_widths
        self.kernel_overhead = kernel_overhead
        self.dropout_rate = dropout_rate

        self.log_kernel_overhead: float = torch.log(
            torch.tensor(kernel_overhead)
        ).item()

        all_layer_widths = list(hidden_layer_widths) + [1]
        all_input_widths = [self.input_dim] + list(hidden_layer_widths)
        layers: List[nn.Module] = []

        for n_in, n_out in zip(all_input_widths, all_layer_widths, strict=True):
            layers.append(nn.Linear(n_in, n_out))
            layers.append(nn.BatchNorm1d(n_out))
            layers.append(nn.ReLU())

        self.linear_relu_stack = nn.Sequential(*layers[:-2])

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

        # Handle batch size of 1 by temporarily switching to eval mode
        original_training = self.training
        if x.size(0) == 1 and self.training:
            self.eval()
          
        try:
            # Forward pass through the model
            log_base_pred = self.linear_relu_stack(x)
            log_overhead_tsr = torch.full_like(
                input=log_base_pred, fill_value=self.log_kernel_overhead
            )
            return torch.logsumexp(
                torch.stack([log_base_pred, log_overhead_tsr], dim=-1), dim=-1
            )
        finally:
            # Restore original training mode
            self.train(original_training)

    def save(self, path: str) -> None:
        """
        Save the model to a file.

        Args:
            path: Path to save the model to
        """
        checkpoint_data = {
            "problem_feature_dim": self.problem_feature_dim,
            "config_feature_dim": self.config_feature_dim,
            "hidden_layer_widths": self.hidden_layer_widths,
            "kernel_overhead": self.kernel_overhead,
            "dropout_rate": self.dropout_rate,
        }
        save_model_checkpoint(self, checkpoint_data, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MatmulModelV1":
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
            hidden_layer_widths=checkpoint["hidden_layer_widths"],
            kernel_overhead=checkpoint.get("kernel_overhead", 0.00541),
            dropout_rate=checkpoint.get("dropout_rate", 0.0),
        )

        # Load the state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        # Move the model to the device
        model = model.to(device)

        logger.info(f"Model loaded from {path}")

        return model


# Legacy class name for backwards compatibility
NeuralNetwork = MatmulModelV1


def get_nn_x(
    df: pd.DataFrame, mean: torch.Tensor | None = None, std: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize the data and convert it to a tensor."""
    x_df = df[
        [
            "dtype_size",
            "dim_m",
            "dim_n",
            "dim_k",
            "total_gb",
            "total_gflop",
            "flops_per_byte",
            "config_block_k",
            "config_block_m",
            "config_block_n",
            "config_num_stages",
            "config_num_warps",
        ]
    ].copy()
    for col in x_df.columns:
        x_df[col] = np.log(x_df[col])

    x_tens = torch.from_numpy(x_df.astype(float).to_numpy()).to(device="cuda")
    if mean is None:
        mean = torch.from_numpy(
            assert_is_instance(x_df.mean(), pd.Series).to_numpy()
        ).to(device="cuda")
    if std is None:
        std = torch.from_numpy(assert_is_instance(x_df.std(), pd.Series).to_numpy()).to(
            device="cuda"
        )
    x_tens -= mean
    x_tens /= std
    return x_tens.to(torch.float32), mean, std


def get_total_gb_feature(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the total gigabytes feature from the dataframe.
    Args:
        df: DataFrame containing the necessary columns for calculation
    Returns:
        Series containing the calculated total gigabytes
    """
    # Calculate memory access in bytes
    m, n, k = df["dim_m"], df["dim_n"], df["dim_k"]
    dtype_size = df["dtype_size"] / 8  # Convert bits to bytes

    # A: m×k, B: k×n, C: m×n
    return ((m * k + k * n + m * n) * dtype_size) / 1e9  # Convert to GB


def get_total_gflop_feature(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the total gigaflops feature from the dataframe.
    Args:
        df: DataFrame containing the necessary columns for calculation
    Returns:
        Series containing the calculated total gigaflops
    """
    # For matrix multiplication, flops = 2 * m * n * k
    m, n, k = df["dim_m"], df["dim_n"], df["dim_k"]
    return (2 * m * n * k) / 1e9  # Convert to GFLOP


class ModelWrapper:
    """
    Wrapper for the neural network model that handles encoding inputs and decoding outputs.
    This class provides methods to prepare inputs for the model and interpret its outputs,
    handling the necessary standardization and feature engineering.

    Note: This is a legacy class maintained for backwards compatibility.
    For new code, consider using MatmulModelV1 directly.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize the model wrapper with the pre-trained model and standardization parameters.

        Args:
            model_path: Path to the model file. If None, uses default parameters.
        """
        import time

        start_time = time.time()

        if model_path is not None:
            # Load from file
            self.model = MatmulModelV1.load(model_path)
        else:
            # Use default configuration for backwards compatibility
            self.model = MatmulModelV1(
                problem_feature_dim=7,  # dtype_size, dim_m, dim_n, dim_k, total_gb, total_gflop, flops_per_byte
                config_feature_dim=5,  # config_block_k, config_block_m, config_block_n, config_num_stages, config_num_warps
                hidden_layer_widths=[256, 256, 256, 256, 256, 256],
            )

        self.model.eval()

        end_time = time.time()

        logger.info("NN Kernel Prediction Model loaded.")
        logger.info("Took: %s seconds", end_time - start_time)

        # Mean values for standardizing input features
        self.mean_for_standardization = torch.tensor(
            [
                2.78275084,
                8.23996746,
                7.27791873,
                7.92035942,
                -2.39558163,
                3.40679233,
                5.80237395,
                3.95781827,
                4.19478321,
                4.19098234,
                0.9045909,
                1.28331208,
            ]
        )

        # Standard deviation values for standardizing input features
        self.std_for_standardization = torch.tensor(
            [
                0.08322756,
                2.31893439,
                1.65605574,
                2.15447078,
                2.19682881,
                2.99600806,
                1.24328795,
                0.92352521,
                0.93849802,
                0.93872011,
                0.57455891,
                0.5837217,
            ]
        )

    def vec(
        self, m: int, n: int, k: int, dsize: int, config: Any
    ) -> tuple[int, int, int, int, int, int, int, int, int]:
        """
        Convert matrix multiplication parameters and config to a feature vector.
        Args:
            m: First dimension of matrix multiplication
            n: Second dimension of matrix multiplication
            k: Third dimension of matrix multiplication
            dsize: Data size in bits (e.g., 16 for float16, 32 for float32)
            config: Configuration object containing kernel parameters
        Returns:
            Tuple containing the extracted features
        """
        kwargs = config.all_kwargs()

        return (
            int(m),
            int(n),
            int(k),
            int(dsize),
            int(kwargs["BLOCK_M"]),
            int(kwargs["BLOCK_N"]),
            int(kwargs["BLOCK_K"]),
            int(kwargs["num_stages"]),
            int(kwargs["num_warps"]),
        )

    @staticmethod
    def vec_params(
        m: int,
        n: int,
        k: int,
        dsize: int,
        params: Any,  # TritonGEMMConfig type hint removed for compatibility
    ) -> tuple[int, int, int, int, int, int, int, int, int]:
        """
        Convert matrix multiplication parameters and config to a feature vector.
        Args:
            m: First dimension of matrix multiplication
            n: Second dimension of matrix multiplication
            k: Third dimension of matrix multiplication
            dsize: Data size in bits (e.g., 16 for float16, 32 for float32)
            params: Configuration object containing kernel parameters
        Returns:
            Tuple containing the extracted features
        """

        return (
            int(m),
            int(n),
            int(k),
            int(dsize),
            int(params.block_m),
            int(params.block_n),
            int(params.block_k),
            int(params.num_stages),
            int(params.num_warps),
        )

    def encode(
        self, m: int, n: int, k: int, dtype: torch.dtype, configs: list[Any]
    ) -> torch.Tensor:
        """
        Encode the matrix multiplication parameters and configs as input tensors for the model.
        Args:
            m: First dimension of matrix multiplication
            n: Second dimension of matrix multiplication
            k: Third dimension of matrix multiplication
            dtype: Data type of the matrices
            configs: List of configuration objects
        Returns:
            Tensor containing the encoded inputs ready for the model
        Raises:
            ValueError: If the dtype is not supported
        """
        # Determine data size based on dtype
        if dtype == torch.bfloat16 or dtype == torch.float16:
            dsize = 16
        elif dtype == torch.float32:
            dsize = 32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}. Add support for this dtype.")

        # Create feature dataframe
        df = pd.DataFrame(
            columns=[
                "dim_m",
                "dim_n",
                "dim_k",
                "dtype_size",
                "config_block_m",
                "config_block_n",
                "config_block_k",
                "config_num_stages",
                "config_num_warps",
            ],
            data=[self.vec(m, n, k, dsize, config) for config in configs],
        )

        # Calculate derived features
        df["total_gb"] = get_total_gb_feature(df=df).astype(np.float32)
        df["total_gflop"] = get_total_gflop_feature(df=df).astype(np.float32)
        df["flops_per_byte"] = df["total_gflop"] / df["total_gb"]

        # Reorder columns to match expected model input
        df = df[
            [
                "dtype_size",
                "dim_m",
                "dim_n",
                "dim_k",
                "total_gb",
                "total_gflop",
                "flops_per_byte",
                "config_block_k",
                "config_block_m",
                "config_block_n",
                "config_num_stages",
                "config_num_warps",
            ]
        ]

        # Standardize the input
        inp, _, _ = get_nn_x(
            df=df, mean=self.mean_for_standardization, std=self.std_for_standardization
        )

        return inp

    def inference(self, inp_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run inference on the model with the given input tensor.
        Args:
            inp_tensor: Input tensor for the model
        Returns:
            Output tensor from the model
        """
        with torch.no_grad():
            # Split the tensor into problem and config features
            problem_features = inp_tensor[
                :, :7
            ]  # dtype_size, dim_m, dim_n, dim_k, total_gb, total_gflop, flops_per_byte
            config_features = inp_tensor[:, 7:]  # config features
            return self.model(problem_features, config_features)

    def decode(self, ret_tensor: torch.Tensor) -> torch.Tensor:
        """
        Decode the model output tensor.
        Args:
            ret_tensor: Output tensor from the model
        Returns:
            Decoded tensor representing runtime predictions
        """
        return ret_tensor
