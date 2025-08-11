import json
import logging
import typing
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from typing import Any, get_origin, Optional, TYPE_CHECKING, TypeVar, Union


if TYPE_CHECKING:
    from triton import Config as TritonConfig


try:
    from typing_extensions import Self
except ImportError:
    from typing import Self

import torch
from torch.utils._ordered_set import OrderedSet
from diode.types.matmul_types import TritonGEMMConfig, Table


# Set up logging for kernel LUT
logger = logging.getLogger(__name__)



def convert_triton_configs_to_gemm_configs(
    triton_configs: list["TritonConfig"], name_prefix: str = "triton_config"
) -> list[TritonGEMMConfig]:
    """
    Convert a list of triton.runtime.autotuner.Config objects to TritonGEMMConfig objects.

    Args:
        triton_configs: List of triton.runtime.autotuner.Config objects
        name_prefix: Prefix for generated config names (default: "triton_config")

    Returns:
        List of TritonGEMMConfig objects
    """
    gemm_configs = []

    for i, config in enumerate(triton_configs):
        # Extract kwargs which contain the block sizes
        kwargs = getattr(config, "kwargs", {})

        # Handle case where kwargs is None
        if kwargs is None:
            kwargs = {}

        # Extract required parameters from kwargs
        block_m = kwargs.get("BLOCK_M", 64)  # Default fallback values
        block_n = kwargs.get("BLOCK_N", 64)
        block_k = kwargs.get("BLOCK_K", 32)
        group_m = kwargs.get("GROUP_M", 8)

        # Extract other parameters directly from config object
        num_stages = getattr(config, "num_stages", 2)
        num_warps = getattr(config, "num_warps", 4)

        # Extract optional parameters with defaults
        even_k = kwargs.get("EVEN_K", False)
        allow_tf32 = kwargs.get("ALLOW_TF32", False)
        use_fast_accum = kwargs.get("USE_FAST_ACCUM", False)
        acc_type = kwargs.get("ACC_TYPE", "tl.float32")

        # Generate a unique name for this config
        config_name = f"{name_prefix}_{i}"

        # Create TritonGEMMConfig object
        gemm_config = TritonGEMMConfig(
            name=config_name,
            grid=1,  # Default grid value, can be adjusted based on requirements
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            group_m=group_m,
            num_stages=num_stages,
            num_warps=num_warps,
            EVEN_K=even_k,
            ALLOW_TF32=allow_tf32,
            USE_FAST_ACCUM=use_fast_accum,
            ACC_TYPE=acc_type,
        )

        gemm_configs.append(gemm_config)

    return gemm_configs


@lru_cache
def get_table(path: str) -> Optional[Table]:
    """Load a table from a file path."""
    try:
        with open(path) as f:
            table_content = f.read()
            table = Table.deserialize(table_content)
            return table
    except OSError as e:
        logger.error("Failed to read table from %s: %s", path, e)
        return None


def get_table_safe(path: str) -> Optional[Table]:
    """Safely load a table from a file path without caching."""
    try:
        with open(path) as f:
            table_content = f.read()
            table = Table.deserialize(table_content)
            return table
    except OSError as e:
        logger.error("Failed to read table from %s: %s", path, e)
        return None
