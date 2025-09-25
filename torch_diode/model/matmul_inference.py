"""
Unified inference interface for matrix multiplication timing models.

This module provides a consistent interface for inference across different
matmul model architectures and versions.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..types.matmul_types import MMShape, TritonGEMMConfig
from ..utils.debug_config import type_assert

logger = logging.getLogger(__name__)


def create_features_from_mmshape_and_configs(
    mmshape: MMShape, configs: List[TritonGEMMConfig], device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create problem and config features from MMShape and TritonGEMMConfig objects.

    This is a standalone function that can be shared across different modules
    to avoid code duplication in feature construction.

    Args:
        mmshape: MMShape object containing matrix multiplication parameters
        configs: List of TritonGEMMConfig objects
        device: Device to create tensors on

    Returns:
        Tuple of (problem_features, config_features) tensors
    """
    type_assert(
        isinstance(mmshape, MMShape), f"mmshape must be MMShape, got {type(mmshape)}"
    )
    type_assert(isinstance(configs, list), f"configs must be list, got {type(configs)}")
    type_assert(
        all(isinstance(cfg, TritonGEMMConfig) for cfg in configs),
        "All configs must be TritonGEMMConfig instances",
    )
    type_assert(isinstance(device, str), f"device must be str, got {type(device)}")

    from ..utils.feature_extraction import (
        extract_config_features,
        extract_problem_features,
    )

    # Extract problem features (same for all configs)
    problem_feat_list = extract_problem_features(mmshape, return_tensors=False)

    # Create feature tensors for each config
    problem_feats = []
    config_feats = []

    for config in configs:
        # Problem features are the same for all configs
        problem_feat = torch.tensor(problem_feat_list, dtype=torch.float32).unsqueeze(0)

        # Extract config features
        config_feat_list = extract_config_features(config, return_tensors=False)
        config_feat = torch.tensor(config_feat_list, dtype=torch.float32).unsqueeze(0)

        problem_feats.append(problem_feat)
        config_feats.append(config_feat)

    # Stack into batch tensors
    problem_features = torch.cat(problem_feats, dim=0).to(device)
    config_features = torch.cat(config_feats, dim=0).to(device)

    return problem_features, config_features


class MatmulInferenceInterface(ABC, nn.Module):
    """
    Abstract base class defining the unified inference interface for matmul models.

    This interface ensures consistency across different model architectures while
    allowing for model-specific implementations.
    """

    def __init__(
        self, problem_feature_dim: int, config_feature_dim: int, **kwargs
    ) -> None:
        """
        Initialize the model with common parameters.

        Args:
            problem_feature_dim: Dimension of the problem features
            config_feature_dim: Dimension of the configuration features
            **kwargs: Model-specific parameters
        """
        super().__init__()
        self.problem_feature_dim = problem_feature_dim
        self.config_feature_dim = config_feature_dim
        self.input_dim = problem_feature_dim + config_feature_dim

    @abstractmethod
    def forward(
        self, problem_features: torch.Tensor, config_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            problem_features: Tensor of shape (batch_size, problem_feature_dim)
            config_features: Tensor of shape (batch_size, config_feature_dim)

        Returns:
            Tensor of shape (batch_size, 1) containing predictions
        """
        pass

    def predict(
        self, problem_features: torch.Tensor, config_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Run inference on the model (wrapper around forward with no_grad).

        Args:
            problem_features: Tensor of shape (batch_size, problem_feature_dim)
            config_features: Tensor of shape (batch_size, config_feature_dim)

        Returns:
            Tensor of shape (batch_size, 1) containing predictions
        """
        with torch.no_grad():
            return self.forward(problem_features, config_features)

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to a file.

        Args:
            path: Path to save the model to
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, device: str = "cpu") -> "MatmulInferenceInterface":
        """
        Load a model from a file.

        Args:
            path: Path to load the model from
            device: Device to load the model to

        Returns:
            The loaded model
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model architecture and parameters.

        Returns:
            Dictionary containing model information
        """
        return {
            "model_class": self.__class__.__name__,
            "problem_feature_dim": self.problem_feature_dim,
            "config_feature_dim": self.config_feature_dim,
            "input_dim": self.input_dim,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }


class MatmulFeatureProcessor:
    """
    Feature processor for matrix multiplication models.

    This class handles feature extraction, preprocessing, and standardization
    for matmul inference across different model types.
    """

    def __init__(
        self,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the feature processor.

        Args:
            mean: Mean values for standardization
            std: Standard deviation values for standardization
            device: Device to perform computations on
        """
        self.device = device
        self.mean = mean.to(device) if mean is not None else None
        self.std = std.to(device) if std is not None else None

    @staticmethod
    def calculate_total_gb_feature(
        m: Union[int, torch.Tensor, np.ndarray],
        n: Union[int, torch.Tensor, np.ndarray],
        k: Union[int, torch.Tensor, np.ndarray],
        dtype_size: Union[int, torch.Tensor, np.ndarray],
    ) -> Union[float, torch.Tensor, np.ndarray]:
        """
        Calculate the total gigabytes feature for matrix multiplication.

        Args:
            m: First dimension of matrix multiplication
            n: Second dimension of matrix multiplication
            k: Third dimension of matrix multiplication
            dtype_size: Data type size in bits

        Returns:
            Total gigabytes for the operation
        """
        # Convert bits to bytes
        dtype_bytes = dtype_size / 8

        # A: m×k, B: k×n, C: m×n
        total_bytes = (m * k + k * n + m * n) * dtype_bytes
        return total_bytes / 1e9  # Convert to GB

    @staticmethod
    def calculate_total_gflop_feature(
        m: Union[int, torch.Tensor, np.ndarray],
        n: Union[int, torch.Tensor, np.ndarray],
        k: Union[int, torch.Tensor, np.ndarray],
    ) -> Union[float, torch.Tensor, np.ndarray]:
        """
        Calculate the total gigaflops feature for matrix multiplication.

        Args:
            m: First dimension of matrix multiplication
            n: Second dimension of matrix multiplication
            k: Third dimension of matrix multiplication

        Returns:
            Total gigaflops for the operation
        """
        # For matrix multiplication, flops = 2 * m * n * k
        return (2 * m * n * k) / 1e9  # Convert to GFLOP

    @staticmethod
    def get_dtype_size(dtype: torch.dtype) -> int:
        """
        Get the size in bits for a torch dtype.

        Args:
            dtype: PyTorch data type

        Returns:
            Size in bits

        Raises:
            ValueError: If dtype is not supported
        """
        if dtype in (torch.bfloat16, torch.float16):
            return 16
        elif dtype == torch.float32:
            return 32
        elif dtype == torch.float64:
            return 64
        elif dtype == torch.int8:
            return 8
        elif dtype == torch.int16:
            return 16
        elif dtype == torch.int32:
            return 32
        elif dtype == torch.int64:
            return 64
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def create_features_dataframe(
        self, m: int, n: int, k: int, dtype: torch.dtype, configs: List[Any]
    ) -> pd.DataFrame:
        """
        Create a feature dataframe from matrix multiplication parameters and configs.

        Args:
            m: First dimension of matrix multiplication
            n: Second dimension of matrix multiplication
            k: Third dimension of matrix multiplication
            dtype: Data type of the matrices
            configs: List of configuration objects with kernel parameters

        Returns:
            DataFrame containing the features
        """
        dtype_size = self.get_dtype_size(dtype)

        # Extract features from configs
        rows = []
        for config in configs:
            # Handle different config formats
            if hasattr(config, "all_kwargs"):
                # TritonGEMMConfig style
                kwargs = config.all_kwargs()
                row = {
                    "dim_m": m,
                    "dim_n": n,
                    "dim_k": k,
                    "dtype_size": dtype_size,
                    "config_block_m": kwargs["BLOCK_M"],
                    "config_block_n": kwargs["BLOCK_N"],
                    "config_block_k": kwargs["BLOCK_K"],
                    "config_num_stages": kwargs["num_stages"],
                    "config_num_warps": kwargs["num_warps"],
                }
            elif hasattr(config, "block_m"):
                # Direct attribute access style
                row = {
                    "dim_m": m,
                    "dim_n": n,
                    "dim_k": k,
                    "dtype_size": dtype_size,
                    "config_block_m": config.block_m,
                    "config_block_n": config.block_n,
                    "config_block_k": config.block_k,
                    "config_num_stages": config.num_stages,
                    "config_num_warps": config.num_warps,
                }
            else:
                raise ValueError(f"Unsupported config format: {type(config)}")

            rows.append(row)

        df = pd.DataFrame(rows)

        # Calculate derived features
        df["total_gb"] = self.calculate_total_gb_feature(
            df["dim_m"], df["dim_n"], df["dim_k"], df["dtype_size"]
        ).astype(np.float32)
        df["total_gflop"] = self.calculate_total_gflop_feature(
            df["dim_m"], df["dim_n"], df["dim_k"]
        ).astype(np.float32)
        df["flops_per_byte"] = df["total_gflop"] / df["total_gb"]

        return df

    def standardize_features(
        self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Standardize features from a dataframe.

        Args:
            df: DataFrame containing features
            feature_columns: List of column names to use as features

        Returns:
            Tuple of (standardized_tensor, mean, std)
        """
        if feature_columns is None:
            feature_columns = [
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

        # Select and copy features
        x_df = df[feature_columns].copy()

        # Apply log transformation
        for col in x_df.columns:
            x_df[col] = np.log(x_df[col])

        # Convert to tensor
        x_tensor = torch.from_numpy(x_df.astype(float).to_numpy()).to(self.device)

        # Calculate or use provided standardization parameters
        if self.mean is None:
            mean = x_tensor.mean(dim=0)
        else:
            mean = self.mean

        if self.std is None:
            std = x_tensor.std(dim=0)
        else:
            std = self.std

        # Standardize
        x_tensor = (x_tensor - mean) / std

        return x_tensor.to(torch.float32), mean, std

    def encode_for_inference(
        self, m: int, n: int, k: int, dtype: torch.dtype, configs: List[Any]
    ) -> torch.Tensor:
        """
        Encode matrix multiplication parameters and configs for inference.

        Args:
            m: First dimension of matrix multiplication
            n: Second dimension of matrix multiplication
            k: Third dimension of matrix multiplication
            dtype: Data type of the matrices
            configs: List of configuration objects

        Returns:
            Encoded tensor ready for model inference
        """
        # Create features dataframe
        df = self.create_features_dataframe(m, n, k, dtype, configs)

        # Standardize features
        encoded_tensor, _, _ = self.standardize_features(df)

        return encoded_tensor


class UnifiedMatmulPredictor:
    """
    Unified predictor class that provides a consistent interface for all matmul models.

    This class abstracts away model-specific details and provides a simple
    predict() method that works with any matmul model implementation.
    """

    def __init__(
        self,
        model: MatmulInferenceInterface,
        feature_processor: Optional[MatmulFeatureProcessor] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the unified predictor.

        Args:
            model: The matmul model implementing MatmulInferenceInterface
            feature_processor: Feature processor for encoding inputs
            device: Device to run inference on
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Use provided feature processor or create default one
        if feature_processor is not None:
            self.feature_processor = feature_processor
        else:
            self.feature_processor = MatmulFeatureProcessor(device=self.device)

    def predict_from_features(
        self, problem_features: torch.Tensor, config_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict from pre-processed feature tensors.

        Args:
            problem_features: Tensor of shape (batch_size, problem_feature_dim)
            config_features: Tensor of shape (batch_size, config_feature_dim)

        Returns:
            Tensor of shape (batch_size, 1) containing predictions
        """
        # Move tensors to device
        problem_features = problem_features.to(self.device)
        config_features = config_features.to(self.device)

        return self.model.predict(problem_features, config_features)

    def predict_from_raw_inputs(
        self, m: int, n: int, k: int, dtype: torch.dtype, configs: List[Any]
    ) -> torch.Tensor:
        """
        Predict from raw matrix multiplication parameters and configs.

        Args:
            m: First dimension of matrix multiplication
            n: Second dimension of matrix multiplication
            k: Third dimension of matrix multiplication
            dtype: Data type of the matrices
            configs: List of configuration objects

        Returns:
            Tensor of shape (batch_size, 1) containing predictions
        """
        # Encode inputs
        encoded_tensor = self.feature_processor.encode_for_inference(
            m, n, k, dtype, configs
        )

        # Split into problem and config features
        problem_features = encoded_tensor[:, : self.model.problem_feature_dim]
        config_features = encoded_tensor[:, self.model.problem_feature_dim :]

        return self.predict_from_features(problem_features, config_features)

    def predict_from_mmshape(
        self, mmshape: MMShape, configs: List[TritonGEMMConfig]
    ) -> torch.Tensor:
        """
        Predict from MMShape and TritonGEMMConfig objects directly.

        This is the high-level interface that eliminates manual feature construction.

        Args:
            mmshape: MMShape object containing matrix multiplication dimensions and types
            configs: List of TritonGEMMConfig objects

        Returns:
            Tensor of shape (batch_size, 1) containing predictions
        """
        # Create features from MMShape and configs
        problem_features, config_features = self._create_features_from_mmshape(
            mmshape, configs
        )

        return self.predict_from_features(problem_features, config_features)

    def _create_features_from_mmshape(
        self, mmshape: MMShape, configs: List[TritonGEMMConfig]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create problem and config features from MMShape and TritonGEMMConfig objects.

        This method delegates to the standalone function to avoid code duplication.

        Args:
            mmshape: MMShape object containing matrix multiplication parameters
            configs: List of TritonGEMMConfig objects

        Returns:
            Tuple of (problem_features, config_features) tensors
        """
        return create_features_from_mmshape_and_configs(mmshape, configs, self.device)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the underlying model.

        Returns:
            Dictionary containing model information
        """
        return self.model.get_model_info()
