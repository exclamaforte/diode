"""
Kernel conversion utilities for integrating with PyTorch Inductor.

This module provides functions to convert between PyTorch Inductor's kernel template
choices and Diode's data structures for model inference.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import triton

from ..utils.debug_config import type_assert
from ..utils.feature_extraction import extract_problem_features

from torch._inductor.codegen.common import KernelTemplate
from torch._inductor.kernel_inputs import KernelInputs, MMKernelInputs
from torch._inductor.kernel_template_choice import (
    KernelTemplateChoice,
    make_ktc_generator,
)
from torch._inductor.select_algorithm import ExternKernelChoice
from torch._inductor.template_heuristics import get_template_heuristic
from torch._inductor.template_heuristics.params import DictKernelTemplateParams

import torch._inductor.config as inductor_config
from torch._inductor.virtualized import V

from ..types.matmul_types import MMShape, TritonGEMMConfig
from ..model.matmul_inference import create_features_from_mmshape_and_configs

logger = logging.getLogger(__name__)


def create_legacy_problem_features(
    mmshape: MMShape, num_configs: int, device: str = "cuda"
) -> torch.Tensor:
    """
    Create legacy problem features (7 dimensions) that match the trained model expectations.

    Legacy format: [dtype_size, dim_m, dim_n, dim_k, total_gb, total_gflop, flops_per_byte]

    Args:
        mmshape: MMShape object containing matrix multiplication parameters
        num_configs: Number of configs (for batch size)
        device: Device to create tensors on

    Returns:
        Tensor of shape (num_configs, 7) with legacy problem features
    """
    type_assert(isinstance(mmshape, MMShape), f"Expected MMShape, got {type(mmshape)}")
    type_assert(
        isinstance(num_configs, int),
        f"Expected int for num_configs, got {type(num_configs)}",
    )
    type_assert(num_configs > 0, f"num_configs must be positive, got {num_configs}")
    type_assert(isinstance(device, str), f"Expected str for device, got {type(device)}")
    # Get dtype size in bits
    dtype_size = 16 if mmshape.M_dtype == torch.float16 else 32

    # Calculate derived features
    m, n, k = float(mmshape.M), float(mmshape.N), float(mmshape.K)
    dtype_bytes = dtype_size / 8

    # Total memory footprint: A(m*k) + B(k*n) + C(m*n) in GB
    total_gb = (m * k + k * n + m * n) * dtype_bytes / 1e9

    # Total floating point operations: 2 * m * n * k GFLOPS
    total_gflop = (2 * m * n * k) / 1e9

    # Arithmetic intensity: flops per byte
    flops_per_byte = total_gflop / total_gb if total_gb > 0 else 0.0

    # Legacy problem features (7 total)
    features = [
        dtype_size,  # dtype_size
        m,  # dim_m
        n,  # dim_n
        k,  # dim_k
        total_gb,  # total_gb
        total_gflop,  # total_gflop
        flops_per_byte,  # flops_per_byte
    ]

    # Create batch tensor by repeating features for each config
    batch_features = torch.tensor(
        features, dtype=torch.float32, device=device
    ).unsqueeze(0)
    batch_features = batch_features.repeat(num_configs, 1)

    return batch_features


def create_legacy_config_features(
    configs: List[TritonGEMMConfig], device: str = "cuda"
) -> torch.Tensor:
    """
    Create legacy config features (5 dimensions) that match the trained model expectations.

    Legacy format: [config_block_k, config_block_m, config_block_n, config_num_stages, config_num_warps]

    Args:
        configs: List of TritonGEMMConfig objects
        device: Device to create tensors on

    Returns:
        Tensor of shape (len(configs), 5) with legacy config features
    """
    type_assert(
        isinstance(configs, list), f"Expected list for configs, got {type(configs)}"
    )
    type_assert(len(configs) > 0, "configs list cannot be empty")
    type_assert(
        all(isinstance(c, TritonGEMMConfig) for c in configs),
        "All configs must be TritonGEMMConfig instances",
    )
    type_assert(isinstance(device, str), f"Expected str for device, got {type(device)}")
    features_list = []

    for config in configs:
        # Legacy config features (5 total) - note the order matches the model expectation
        features = [
            float(config.block_k),  # config_block_k (first!)
            float(config.block_m),  # config_block_m
            float(config.block_n),  # config_block_n
            float(config.num_stages),  # config_num_stages
            float(config.num_warps),  # config_num_warps
        ]
        features_list.append(features)

    return torch.tensor(features_list, dtype=torch.float32, device=device)



def extract_mmshape_from_kernel_inputs(
    kernel_inputs: MMKernelInputs, op_name: str
) -> Optional[MMShape]:
    """
    Extract MMShape from KernelInputs.

    Args:
        kernel_inputs: The kernel inputs containing tensor information
        op_name: Operation name (e.g., "mm", "addmm", "bmm")

    Returns:
        MMShape object or None if extraction fails
    """
    type_assert(kernel_inputs is not None, "kernel_inputs cannot be None")
    type_assert(
        isinstance(op_name, str), f"Expected str for op_name, got {type(op_name)}"
    )
    if not isinstance(kernel_inputs, MMKernelInputs):
        logger.warning(f"Expected MMKernelInputs, got {type(kernel_inputs)}")
        return None

    # Get M, N, K dimensions
    M, N, K = kernel_inputs.mnk_hinted()
    dtype = kernel_inputs.dtype()

    # Get output layout information
    output_layout = kernel_inputs.output_layout()
    out_size = output_layout.size
    out_stride = output_layout.stride
    out_dtype = output_layout.dtype

    # Determine batch size based on operation
    if op_name == "bmm":
        # For batch matrix multiplication, try to get batch size
        input_tensors = kernel_inputs.nodes()
        if len(input_tensors) >= 2:
            shape_a = input_tensors[0].get_size()
            if len(shape_a) >= 3:
                B = int(shape_a[-3]) if hasattr(shape_a[-3], "__int__") else 1
            else:
                B = 1
        else:
            B = 1
    else:
        B = 1

    def resolve(val):
        return tuple(
            V.graph.sizevars.size_hints(
                val,
                fallback=torch._inductor.config.unbacked_symint_fallback,
            )
        )

    # Create MMShape
    mm_shape = MMShape(
        B=B,
        M=M,
        M_dtype=dtype,
        N=N,
        K=K,
        K_dtype=dtype,
        out_dtype=out_dtype,
        out_size=resolve(out_size),
        out_stride=resolve(out_stride),
    )

    return mm_shape



def convert_triton_config_to_triton_gemm_config(
    config: triton.runtime.autotuner.Config, template_name: str
) -> Optional[TritonGEMMConfig]:
    """
    Convert a triton.runtime.autotuner.Config to a TritonGEMMConfig.

    Args:
        config: triton.runtime.autotuner.Config object

    Returns:
        TritonGEMMConfig object or None if conversion fails
    """
    # Get kwargs from the config object
    kwargs = {}

    # Try different ways to extract config parameters
    if hasattr(config, "kwargs"):
        kwargs = config.__dict__.copy()
        kwargs.update(config.kwargs)
        del kwargs["kwargs"]
    elif hasattr(config, "all_kwargs"):
        kwargs = config.all_kwargs()
    elif hasattr(config, "__dict__"):
        # Fallback to extracting from object attributes
        kwargs = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
    else:
        logger.warning(f"Could not extract kwargs from config {type(config)}")
        return None

    # Extract standard Triton GEMM parameters with defaults
    block_m = kwargs.get("BLOCK_M", 64)
    block_n = kwargs.get("BLOCK_N", 64)
    block_k = kwargs.get("BLOCK_K", 32)
    group_m = kwargs.get("GROUP_M", 8)
    num_stages = kwargs.get("num_stages", 4)
    num_warps = kwargs.get("num_warps", 4)


    # Generate a name for this config

    # Create TritonGEMMConfig
    triton_config = TritonGEMMConfig(
        name=template_name,
        grid=1,  # Grid will be computed dynamically
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
        group_m=int(group_m),
        num_stages=int(num_stages),
        num_warps=int(num_warps),
        ALLOW_TF32=kwargs.get(
            "ALLOW_TF32", True
        ),  # Default to True for better performance
        USE_FAST_ACCUM=kwargs.get("USE_FAST_ACCUM", False),
        ACC_TYPE=kwargs.get("ACC_TYPE", "tl.float32"),
    )

    return triton_config


def convert_ktc_to_triton_config(
    ktc: KernelTemplateChoice,
) -> Optional[TritonGEMMConfig]:
    """
    Convert a KernelTemplateChoice to a TritonGEMMConfig.

    Args:
        ktc: KernelTemplateChoice object

    Returns:
        TritonGEMMConfig object or None if conversion fails
    """
    type_assert(ktc is not None, "ktc cannot be None")
    # Extract template information from KernelTemplateChoice
    template = ktc.template

    # Get template name
    template_name = getattr(template, "uid", "unknown")

    # Try different ways to extract config parameters from KernelTemplateChoice
    kwargs = {}

    # Try accessing params attribute first (newer API)
    if hasattr(ktc, "params"):
        params = ktc.params
        if hasattr(params, "to_kwargs"):
            kwargs = params.to_kwargs()
        elif hasattr(params, "kwargs"):
            kwargs = params.kwargs
        elif hasattr(params, "__dict__"):
            kwargs = params.__dict__

    # Try accessing config attribute (older API)
    elif hasattr(ktc, "config"):
        config = ktc.config
        if hasattr(config, "kwargs"):
            kwargs = config.kwargs
        elif hasattr(config, "all_kwargs"):
            kwargs = config.all_kwargs()
        elif hasattr(config, "__dict__"):
            kwargs = config.__dict__

    # Try accessing the choice directly (if it has config info)
    elif hasattr(ktc, "choice") and ktc.choice:
        choice = ktc.choice
        if hasattr(choice, "kwargs"):
            kwargs = choice.kwargs
        elif hasattr(choice, "__dict__"):
            kwargs = choice.__dict__

    # Fallback: try to extract from any attribute that looks like a config
    else:
        for attr_name in dir(ktc):
            if not attr_name.startswith("_") and not callable(
                getattr(ktc, attr_name)
            ):
                attr = getattr(ktc, attr_name)
                if hasattr(attr, "kwargs"):
                    kwargs = attr.kwargs
                    break
                elif hasattr(attr, "__dict__") and hasattr(attr, "BLOCK_M"):
                    # Looks like a config object
                    kwargs = attr.__dict__
                    break

    # Extract standard Triton GEMM parameters with defaults
    block_m = kwargs.get("BLOCK_M", 64)
    block_n = kwargs.get("BLOCK_N", 64)
    block_k = kwargs.get("BLOCK_K", 32)
    group_m = kwargs.get("GROUP_M", 8)
    num_stages = kwargs.get("num_stages", 4)
    num_warps = kwargs.get("num_warps", 4)

    # Ensure EVEN_K is properly handled
    even_k = kwargs.get(
        "EVEN_K", True
    )  # Default to True to avoid compilation errors

    # Create TritonGEMMConfig
    triton_config = TritonGEMMConfig(
        name=template_name,
        grid=1,  # Grid will be computed dynamically
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
        group_m=int(group_m),
        num_stages=int(num_stages),
        num_warps=int(num_warps),
        EVEN_K=even_k,
        ALLOW_TF32=kwargs.get(
            "ALLOW_TF32", True
        ),  # Default to True for better performance
        USE_FAST_ACCUM=kwargs.get("USE_FAST_ACCUM", False),
        ACC_TYPE=kwargs.get("ACC_TYPE", "tl.float32"),
    )

    return triton_config


def generate_exhaustive_triton_template_configs(
    kernel_inputs: KernelInputs,
    templates: List[Union[KernelTemplate, ExternKernelChoice]],
    op_name: str,
    kwarg_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[List[triton.runtime.autotuner.Config]]:
    """
    Generate exhaustive template configurations for the given inputs.

    This function reproduces the logic from the user's example to generate
    exhaustive configs from scratch, ignoring any input template_choices.

    Args:
        kernel_inputs: MMKernelInputs containing input tensor nodes and matrix indices
        templates: List of template objects (KernelTemplate or ExternKernelChoice) in use
        op_name: Operation name (e.g., "bmm", "baddbmm", "addmm")
        kwarg_overrides: Optional dict of kwargs to override for each template heuristic

    Returns:
        List of KernelTemplateChoice objects with exhaustive configs
    """
    type_assert(kernel_inputs is not None, "kernel_inputs cannot be None")
    type_assert(
        isinstance(kernel_inputs, KernelInputs),
        f"Expected MMKernelInputs, got {type(kernel_inputs)}",
    )
    type_assert(
        isinstance(templates, list),
        f"Expected list for templates, got {type(templates)}",
    )
    type_assert(
        all(
            isinstance(template, KernelTemplate)
            or isinstance(template, ExternKernelChoice)
            for template in templates
        ),
        f"Expected list of KernelTemplate or ExternKernelChoice, got {templates}",
    )
    type_assert(
        isinstance(op_name, str), f"Expected str for op_name, got {type(op_name)}"
    )
    type_assert(
        kwarg_overrides is None or isinstance(kwarg_overrides, dict),
        f"Expected dict or None for kwarg_overrides, got {type(kwarg_overrides)}",
    )

    # Extract device_type from kernel_inputs
    device_type = kernel_inputs.device_type
    assert (
        device_type is not None
    ), "generate_exhaustive_triton_template_configs requires a valid device type"

    if kwarg_overrides is None:
        kwarg_overrides = {}

    ret = []
    # Generate exhaustive configs for each template
    for template in templates:
        # Skip ExternKernelChoice - they don't use template heuristics
        if isinstance(template, ExternKernelChoice):
            continue

        # Get the appropriate template-specific heuristic
        heuristic = get_template_heuristic(template.uid, device_type, op_name)

        # Force exhaustive config generation by calling get_exhaustive_mm_configs directly
        # This bypasses the normal logic in _get_config_generator that checks config.max_autotune_gemm_search_space
        if hasattr(heuristic, "get_exhaustive_mm_configs"):
            # Extract M, N, K from kernel_inputs for config generation
            if isinstance(kernel_inputs, MMKernelInputs):
                m, n, k = kernel_inputs.mnk_symbolic()
                dtype = kernel_inputs.dtype()

                # Get exhaustive configs directly
                exhaustive_config_generator = heuristic.get_exhaustive_mm_configs()

                with inductor_config.patch(
                    max_autotune_prune_choices_based_on_shared_mem=True,
                    max_autotune_gemm_search_space="EXHAUSTIVE",
                ):
                    raw_configs = exhaustive_config_generator(
                        m, n, k, dtype_size=dtype.itemsize, op_name=op_name
                    )

                ret.append(raw_configs)
    return ret


def convert_triton_configs_to_ktc(
    raw_configs: List[triton.runtime.autotuner.Config],
    template: Union[KernelTemplate, ExternKernelChoice],
    kernel_inputs: KernelInputs,
    op_name: str,
):
    device_type = kernel_inputs.device_type
    assert (
        device_type is not None
    ), "generate_exhaustive_triton_template_configs requires a valid device type"

    ret = []
    # Generate exhaustive configs for each template
    heuristic = get_template_heuristic(template.uid, device_type, op_name)
    cs = []
    for config in raw_configs:
        if hasattr(config, "to_kwargs"):
            # Already has the right interface
            cs.append(config)
        elif hasattr(config, "__dict__"):
            # Convert object to dict and wrap
            config_dict = config.__dict__.copy()
            if "kwargs" in config_dict:
                # Has kwargs attribute
                config_dict.update(config_dict["kwargs"])
                del config_dict["kwargs"]
            config_dict["EVEN_K"] = True
            config_dict["USE_FAST_ACCUM"] = True
            config_dict["ACC_TYPE"] = "tl.float32"
            config_dict["ALLOW_TF32"] = True
            cs.append(DictKernelTemplateParams(config_dict))
        elif hasattr(config, "kwargs"):
            # Has kwargs attribute
            cs.append(DictKernelTemplateParams(config.kwargs))
        else:
            # Try to convert to dict
            try:
                config_dict = dict(config) if hasattr(config, "__iter__") else {}
                cs.append(DictKernelTemplateParams(config_dict))
            except Exception:
                logger.warning(
                    f"Could not convert config {type(config)} to parameter object"
                )
                continue

    if not cs:
        logger.warning(f"No valid configs generated for template {template.uid}")

    # Adjust kernel inputs for the heuristic
    inputs_val = heuristic.adjust_kernel_inputs(kernel_inputs, op_name)

    # Create KernelTemplateChoice generator using the same logic as get_ktc
    overrides = {}
    ktc_generator = make_ktc_generator(
        template=template,
        cs=cs,
        overrides=overrides,
        layout=kernel_inputs.output_layout(),
        inputs=inputs_val,
    )
    ret.append(ktc_generator)

    # Extend choices with all generated choices
    return ret


def create_features_and_run_inference(
    mmshape: MMShape,
    configs: List[TritonGEMMConfig],
    model,
    device: str = "cuda",
) -> List[float]:
    """
    Convert configurations to feature vectors and run model inference.

    Args:
        mmshape: MMShape object containing matrix multiplication parameters
        configs: List of TritonGEMMConfig objects
        model: The model to use for inference
        device: Device to run inference on

    Returns:
        List of predicted performance values (lower is better)
    """
    type_assert(isinstance(mmshape, MMShape), f"Expected MMShape, got {type(mmshape)}")
    type_assert(
        isinstance(configs, list), f"Expected list for configs, got {type(configs)}"
    )
    type_assert(len(configs) > 0, "configs list cannot be empty")
    type_assert(
        all(isinstance(c, TritonGEMMConfig) for c in configs),
        "All configs must be TritonGEMMConfig instances",
    )
    type_assert(model is not None, "model cannot be None")
    type_assert(isinstance(device, str), f"Expected str for device, got {type(device)}")
    # Run inference using the legacy model format (12 features: 7 problem + 5 config)
    legacy_problem_features = create_legacy_problem_features(
        mmshape, len(configs), device
    )
    legacy_config_features = create_legacy_config_features(configs, device)

    logger.debug(f"Legacy problem features shape: {legacy_problem_features.shape}")
    logger.debug(f"Legacy config features shape: {legacy_config_features.shape}")

    # Run inference
    if hasattr(model, "predict_from_features"):
        # Use the predictor interface
        predictions = model.predict_from_features(
            legacy_problem_features, legacy_config_features
        )
    elif hasattr(model, "predict"):
        # Use the model wrapper interface
        predictions = model.predict(legacy_problem_features, legacy_config_features)
    elif hasattr(model, "forward"):
        # Use raw model interface
        with torch.no_grad():
            predictions = model.forward(
                legacy_problem_features, legacy_config_features
            )
    elif hasattr(model, "inference"):
        # Use model wrapper inference interface - expects single tensor with 12 features
        combined_features = torch.cat(
            [legacy_problem_features, legacy_config_features], dim=1
        )
        logger.debug(f"Combined legacy features shape: {combined_features.shape}")
        predictions = model.inference(combined_features)
    else:
        logger.error(
            f"Model {type(model)} does not have a recognized inference interface"
        )
        return [0.0] * len(configs)

    # Convert predictions to list of floats
    if isinstance(predictions, torch.Tensor):
        # Flatten the tensor and convert to list
        return predictions.cpu().flatten().tolist()
    else:
        # Handle list of tensors or other structures
        flattened = []
        for pred in predictions:
            if isinstance(pred, torch.Tensor):
                flattened.extend(pred.cpu().flatten().tolist())
            elif isinstance(pred, (list, tuple)):
                flattened.extend(pred)
            else:
                flattened.append(float(pred))
        return flattened


def select_best_configs(
    choices: List[KernelTemplateChoice],
    predictions: List[float],
    top_k: int = 5,
) -> List[KernelTemplateChoice]:
    """
    Select the best configurations based on model predictions.

    Args:
        choices: List of KernelTemplateChoice objects
        predictions: List of predicted performance values
        top_k: Maximum number of configurations to return

    Returns:
        Filtered list of KernelTemplateChoice objects
    """
    type_assert(
        isinstance(choices, list), f"Expected list for choices, got {type(choices)}"
    )
    type_assert(
        isinstance(predictions, list),
        f"Expected list for predictions, got {type(predictions)}",
    )
    type_assert(isinstance(top_k, int), f"Expected int for top_k, got {type(top_k)}")
    type_assert(top_k > 0, f"top_k must be positive, got {top_k}")
    if not choices or not predictions or len(choices) != len(predictions):
        logger.warning(
            "Mismatch between choices and predictions, returning all choices"
        )
        return choices

    # Sort choices by predicted performance (lower is better for execution time)
    sorted_indices = sorted(range(len(predictions)), key=lambda i: predictions[i])

    # Select top configurations within performance threshold
    best_prediction = predictions[sorted_indices[0]]
    selected_choices = []

    for idx in sorted_indices:
        if len(selected_choices) >= top_k:
            break

        selected_choices.append(choices[idx])

    # Ensure we have at least one choice
    if not selected_choices and choices:
        selected_choices = [choices[sorted_indices[0]]]

    logger.debug(
        f"Selected {len(selected_choices)}/{len(choices)} configs "
        f"(best prediction: {best_prediction:.3f})"
    )

    return selected_choices
