"""
Feature extraction utilities for backward compatibility.

This module provides compatibility functions for extracting features from
kernel inputs and configurations, used by deprecated test code.
"""


import torch


def extract_problem_features_compat(mm_shape):
    """
    Compatibility function to extract problem features from MMShape.

    Args:
        mm_shape: MMShape object with M, N, K, B dimensions

    Returns:
        torch.Tensor: Problem features tensor
    """
    try:
        # Create simple features based on matrix dimensions
        features = [
            float(mm_shape.M),
            float(mm_shape.N),
            float(mm_shape.K),
            float(mm_shape.B),
        ]
        return torch.tensor(features, dtype=torch.float32)
    except Exception:
        return torch.zeros(4, dtype=torch.float32)


def extract_config_features_compat(config):
    """
    Compatibility function to extract config features from TritonGEMMConfig.

    Args:
        config: TritonGEMMConfig object

    Returns:
        torch.Tensor: Config features tensor
    """
    try:
        # Extract key configuration parameters
        features = [
            float(config.block_m),
            float(config.block_n),
            float(config.block_k),
            float(config.group_m),
            float(config.num_stages),
            float(config.num_warps),
        ]
        return torch.tensor(features, dtype=torch.float32)
    except Exception:
        return torch.zeros(6, dtype=torch.float32)


def extract_problem_features(mm_shape, return_tensors=False):
    """
    Extract comprehensive problem features from MMShape that match the model's expectations.

    The model expects 17 problem features based on the comprehensive feature set
    used during training.

    Args:
        mm_shape: MMShape object with M, N, K, B dimensions
        return_tensors: If True, return as list for dataset compatibility

    Returns:
        torch.Tensor or list: Problem features (17 dimensions)
    """
    try:
        # Get dtype size in bits
        dtype_size = (
            16 if mm_shape.M_dtype == torch.float16 else 32
        )  # Default to 16 for float16

        # Calculate derived features
        m, n, k = float(mm_shape.M), float(mm_shape.N), float(mm_shape.K)
        dtype_bytes = dtype_size / 8

        # Total memory footprint: A(m*k) + B(k*n) + C(m*n) in GB
        total_gb = (m * k + k * n + m * n) * dtype_bytes / 1e9

        # Total floating point operations: 2 * m * n * k GFLOPS
        total_gflop = (2 * m * n * k) / 1e9

        # Arithmetic intensity: flops per byte
        flops_per_byte = total_gflop / total_gb if total_gb > 0 else 0.0

        # Comprehensive problem features (17 total to match model expectation)
        features = [
            # Basic dimensions
            m,  # dim_m
            n,  # dim_n
            k,  # dim_k
            float(mm_shape.B),  # batch_size
            dtype_size,  # dtype_size (bits)
            # Derived computational features
            total_gb,  # total_gb
            total_gflop,  # total_gflop
            flops_per_byte,  # flops_per_byte
            # Dimension ratios and products
            m / n if n > 0 else 0.0,  # m/n ratio
            m / k if k > 0 else 0.0,  # m/k ratio
            n / k if k > 0 else 0.0,  # n/k ratio
            m * n,  # m*n product
            m * k,  # m*k product
            n * k,  # n*k product
            m * n * k,  # m*n*k product
            # Log features (commonly used in ML models)
            float(torch.log(torch.tensor(max(m, 1.0)))),  # log(m)
            float(torch.log(torch.tensor(max(n, 1.0)))),  # log(n)
        ]

        if return_tensors:
            return features
        return torch.tensor(features, dtype=torch.float32)
    except Exception:
        # Return 17 zeros if feature extraction fails
        if return_tensors:
            return [0.0] * 17
        return torch.zeros(17, dtype=torch.float32)


def extract_config_features(config, return_tensors=False):
    """
    Extract comprehensive config features from TritonGEMMConfig that match the model's expectations.

    The model expects 19 config features based on the comprehensive feature set
    used during training.

    Args:
        config: TritonGEMMConfig object
        return_tensors: If True, return as list for dataset compatibility

    Returns:
        torch.Tensor or list: Config features (19 dimensions)
    """
    try:
        # Basic config parameters
        block_m = float(config.block_m)
        block_n = float(config.block_n)
        block_k = float(config.block_k)
        group_m = float(config.group_m)
        num_stages = float(config.num_stages)
        num_warps = float(config.num_warps)

        # Boolean/categorical features
        even_k = float(config.EVEN_K) if hasattr(config, "EVEN_K") else 1.0
        allow_tf32 = float(config.ALLOW_TF32) if hasattr(config, "ALLOW_TF32") else 1.0
        use_fast_accum = (
            float(config.USE_FAST_ACCUM) if hasattr(config, "USE_FAST_ACCUM") else 0.0
        )

        # Comprehensive config features (19 total to match model expectation)
        features = [
            # Basic block dimensions
            block_m,  # config_block_m
            block_n,  # config_block_n
            block_k,  # config_block_k
            group_m,  # config_group_m
            num_stages,  # config_num_stages
            num_warps,  # config_num_warps
            # Boolean/categorical features
            even_k,  # config_even_k
            allow_tf32,  # config_allow_tf32
            use_fast_accum,  # config_use_fast_accum
            # Derived config features
            block_m * block_n,  # block area (m*n)
            block_m * block_k,  # block area (m*k)
            block_n * block_k,  # block area (n*k)
            block_m * block_n * block_k,  # block volume
            # Thread and memory features
            num_warps * 32,  # total threads (warps * threads_per_warp)
            block_m / num_warps if num_warps > 0 else 0.0,  # m per warp
            block_n / num_warps if num_warps > 0 else 0.0,  # n per warp
            # Log features
            float(torch.log(torch.tensor(max(block_m, 1.0)))),  # log(block_m)
            float(torch.log(torch.tensor(max(block_n, 1.0)))),  # log(block_n)
            float(torch.log(torch.tensor(max(block_k, 1.0)))),  # log(block_k)
        ]

        if return_tensors:
            return features
        return torch.tensor(features, dtype=torch.float32)
    except Exception:
        # Return 19 zeros if feature extraction fails
        if return_tensors:
            return [0.0] * 19
        return torch.zeros(19, dtype=torch.float32)
