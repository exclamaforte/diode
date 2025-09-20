"""
Feature extraction utilities for matrix multiplication configurations and problems.

This module provides common feature extraction functions used across different parts of the codebase
for consistent feature representation.
"""

import logging
from typing import List, Union

import torch

from torch_diode.types.matmul_types import MMShape, TritonGEMMConfig

logger = logging.getLogger(__name__)


def dtype_to_numeric(dtype: torch.dtype) -> float:
    """
    Convert a torch dtype to a numeric value for feature representation.

    Args:
        dtype: The torch dtype

    Returns:
        Numeric value representing the dtype (bit size as float)
    """
    dtype_map = {
        torch.float16: 16.0,
        torch.float32: 32.0,
        torch.float64: 64.0,
        torch.bfloat16: 16.0,
        torch.int8: 8.0,
        torch.int16: 16.0,
        torch.int32: 32.0,
        torch.int64: 64.0,
    }
    return dtype_map.get(dtype, 32.0)  # Default to 32.0 for unknown dtypes


def extract_problem_features(
    problem: MMShape, op_name: str = None, return_tensors: bool = False
) -> List[Union[float, torch.Tensor]]:
    """
    Extract features from an MMShape for ML model input.

    This function creates a consistent 17-dimensional feature vector from matrix
    multiplication problem specifications.

    Args:
        problem: The MMShape to extract features from
        op_name: Operation name (unused, kept for compatibility)
        return_tensors: If True, return log features as tensors; if False, convert to scalar

    Returns:
        List of 17 features: [B, M, N, K, M_dtype, K_dtype, out_dtype, 
                             log(B), log(M), log(N), log(K), 
                             M*K, K*N, M*N, log(M*K), log(K*N), log(M*N)]
    """
    # Handle symbolic dimensions and ensure numeric conversion
    try:
        # Extract basic dimensions with fallbacks for symbolic dimensions
        B = _convert_dimension(getattr(problem, "B", 1))
        M = _convert_dimension(getattr(problem, "M", 64))
        N = _convert_dimension(getattr(problem, "N", 64))
        K = _convert_dimension(getattr(problem, "K", 64))
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Failed to convert problem dimensions: {e}, using defaults")
        B, M, N, K = 1, 64, 64, 64

    # Extract dtype features
    M_dtype_num = (
        dtype_to_numeric(problem.M_dtype) if hasattr(problem, "M_dtype") else 32.0
    )
    K_dtype_num = (
        dtype_to_numeric(problem.K_dtype) if hasattr(problem, "K_dtype") else 32.0
    )
    out_dtype_num = (
        dtype_to_numeric(problem.out_dtype) if hasattr(problem, "out_dtype") else 32.0
    )

    # Calculate log features
    log_B = torch.log(torch.tensor(max(1, B), dtype=torch.float32))
    log_M = torch.log(torch.tensor(max(1, M), dtype=torch.float32))
    log_N = torch.log(torch.tensor(max(1, N), dtype=torch.float32))
    log_K = torch.log(torch.tensor(max(1, K), dtype=torch.float32))

    # Calculate derived features
    MK = M * K  # Input matrix size
    KN = K * N  # Weight matrix size
    MN = M * N  # Output matrix size

    log_MK = torch.log(torch.tensor(max(1, MK), dtype=torch.float32))
    log_KN = torch.log(torch.tensor(max(1, KN), dtype=torch.float32))
    log_MN = torch.log(torch.tensor(max(1, MN), dtype=torch.float32))

    # Build feature list
    features = [
        float(B),
        float(M),
        float(N),
        float(K),
        M_dtype_num,
        K_dtype_num,
        out_dtype_num,
        log_B if return_tensors else log_B.item(),
        log_M if return_tensors else log_M.item(),
        log_N if return_tensors else log_N.item(),
        log_K if return_tensors else log_K.item(),
        float(MK),
        float(KN),
        float(MN),
        log_MK if return_tensors else log_MK.item(),
        log_KN if return_tensors else log_KN.item(),
        log_MN if return_tensors else log_MN.item(),
    ]

    return features


def extract_config_features(
    config: TritonGEMMConfig, return_tensors: bool = False
) -> List[Union[float, torch.Tensor]]:
    """
    Extract features from a TritonGEMMConfig for ML model input.

    This function creates a consistent 19-dimensional feature vector from Triton
    kernel configuration parameters.

    Args:
        config: The TritonGEMMConfig to extract features from
        return_tensors: If True, return log features as tensors; if False, convert to scalar

    Returns:
        List of 19 features: [grid, block_m, block_n, block_k, group_m, num_stages, num_warps,
                             EVEN_K, ALLOW_TF32, USE_FAST_ACCUM,
                             log(block_m), log(block_n), log(block_k),
                             block_m*block_k, block_k*block_n, block_m*block_n,
                             log(block_m*block_k), log(block_k*block_n), log(block_m*block_n)]
    """
    # Extract basic config parameters with safe attribute access
    grid = getattr(config, "grid", 1)
    block_m = getattr(config, "block_m", 64)
    block_n = getattr(config, "block_n", 64)
    block_k = getattr(config, "block_k", 32)
    group_m = getattr(config, "group_m", 8)
    num_stages = getattr(config, "num_stages", 4)
    num_warps = getattr(config, "num_warps", 4)

    # Extract boolean flags with safe attribute access
    even_k = int(getattr(config, "EVEN_K", False))
    allow_tf32 = int(getattr(config, "ALLOW_TF32", False))
    use_fast_accum = int(getattr(config, "USE_FAST_ACCUM", False))

    # Calculate log features
    log_block_m = torch.log(torch.tensor(max(1, block_m), dtype=torch.float32))
    log_block_n = torch.log(torch.tensor(max(1, block_n), dtype=torch.float32))
    log_block_k = torch.log(torch.tensor(max(1, block_k), dtype=torch.float32))

    # Calculate derived features
    block_mk = block_m * block_k  # Block input size
    block_kn = block_k * block_n  # Block weight size
    block_mn = block_m * block_n  # Block output size

    log_block_mk = torch.log(torch.tensor(max(1, block_mk), dtype=torch.float32))
    log_block_kn = torch.log(torch.tensor(max(1, block_kn), dtype=torch.float32))
    log_block_mn = torch.log(torch.tensor(max(1, block_mn), dtype=torch.float32))

    # Build feature list
    features = [
        float(grid),
        float(block_m),
        float(block_n),
        float(block_k),
        float(group_m),
        float(num_stages),
        float(num_warps),
        float(even_k),
        float(allow_tf32),
        float(use_fast_accum),
        log_block_m if return_tensors else log_block_m.item(),
        log_block_n if return_tensors else log_block_n.item(),
        log_block_k if return_tensors else log_block_k.item(),
        float(block_mk),
        float(block_kn),
        float(block_mn),
        log_block_mk if return_tensors else log_block_mk.item(),
        log_block_kn if return_tensors else log_block_kn.item(),
        log_block_mn if return_tensors else log_block_mn.item(),
    ]

    return features


def _convert_dimension(dim) -> int:
    """
    Convert a dimension value to integer, handling symbolic dimensions.

    Args:
        dim: Dimension value (int, string, or symbolic)

    Returns:
        Integer dimension value

    Raises:
        ValueError: If dimension cannot be converted
    """
    if isinstance(dim, int):
        return dim
    elif isinstance(dim, str):
        # Handle symbolic dimensions like "s0", "s1", etc.
        if dim.startswith("s") and dim[1:].isdigit():
            # Extract numeric part or use reasonable default
            try:
                return int(dim[1:]) if dim[1:] else 64
            except ValueError:
                return 64
        else:
            raise ValueError(f"Cannot convert string dimension: {dim}")
    elif hasattr(dim, "__int__"):
        # Handle objects that can be converted to int (like sympy expressions)
        return int(dim)
    else:
        raise ValueError(f"Cannot convert dimension of type {type(dim)}: {dim}")


# Backward compatibility aliases (for existing code that might import these)
def extract_problem_features_compat(mm_shape, op_name=None):
    """Compatibility wrapper for the old function signature."""
    return extract_problem_features(mm_shape, op_name, return_tensors=False)


def extract_config_features_compat(config):
    """Compatibility wrapper for the old function signature."""
    return extract_config_features(config, return_tensors=False)
