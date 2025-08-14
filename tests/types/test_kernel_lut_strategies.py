# Owner(s): ["module: diode"]
"""
Common test strategies and data classes for kernel lookup table tests.
"""

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import triton
from hypothesis import assume, given, strategies as st
from hypothesis.strategies import composite

import torch
from diode.types.json_serializable import JSONSerializable
from diode.types.matmul_types import (
    Hardware,
    MMShape,
    Operation,
    Solution,
    Table,
    TritonGEMMConfig,
)
from diode.types.kernel_lut import convert_triton_configs_to_gemm_configs
from torch.utils._ordered_set import OrderedSet
import unittest
from unittest import TestCase


# Hypothesis strategies for generating test data


@composite
def torch_dtype_strategy(draw):
    """Generate valid torch dtypes."""
    dtypes = [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
        torch.bool,
    ]
    return draw(st.sampled_from(dtypes))


@composite
def triton_gemm_config_strategy(draw):
    """Generate TritonGEMMConfig instances."""
    return TritonGEMMConfig(
        name=draw(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            )
        ),
        grid=draw(st.integers(min_value=1, max_value=10)),
        block_m=draw(st.integers(min_value=16, max_value=128)),
        block_n=draw(st.integers(min_value=16, max_value=128)),
        block_k=draw(st.integers(min_value=16, max_value=128)),
        group_m=draw(st.integers(min_value=1, max_value=16)),
        num_stages=draw(st.integers(min_value=1, max_value=5)),
        num_warps=draw(st.integers(min_value=1, max_value=8)),
        EVEN_K=draw(st.booleans()),
        ALLOW_TF32=draw(st.booleans()),
        USE_FAST_ACCUM=draw(st.booleans()),
        ACC_TYPE=draw(st.sampled_from(["tl.float32", "tl.float16", "tl.bfloat16"])),
    )


@composite
def mm_problem_strategy(draw):
    """Generate MMShape instances."""
    B = draw(st.integers(min_value=1, max_value=10000))
    M = draw(st.integers(min_value=1, max_value=10000))
    N = draw(st.integers(min_value=1, max_value=10000))
    K = draw(st.integers(min_value=1, max_value=10000))
    return MMShape(
        B=B,
        M=M,
        M_dtype=draw(torch_dtype_strategy()),
        N=N,
        K=K,
        K_dtype=draw(torch_dtype_strategy()),
        out_dtype=draw(torch_dtype_strategy()),
        out_size=(M, N, K),
        out_stride=(N * K, K, 1),
    )


@composite
def operation_strategy(draw):
    """Generate Operation instances."""
    # Generate a small number of problem-config pairs
    num_solutions = draw(st.integers(min_value=0, max_value=3))
    solution = OrderedDict()

    for _ in range(num_solutions):
        problem = draw(mm_problem_strategy())
        config = draw(triton_gemm_config_strategy())
        # Create a Solution object containing the config
        sol = Solution(name=f"solution_{len(solution)}", config=[config])
        # Use the problem object directly as key since MMShape should be hashable
        solution[problem] = sol

    return Operation(
        name=draw(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
            )
        ),
        solution=solution,
    )


@composite
def hardware_strategy(draw):
    """Generate Hardware instances."""
    num_operations = draw(st.integers(min_value=0, max_value=3))
    operations = [draw(operation_strategy()) for _ in range(num_operations)]

    return Hardware(operation=OrderedDict((op.name, op) for op in operations))


@composite
def table_strategy(draw):
    """Generate Table instances."""
    num_hardware = draw(st.integers(min_value=0, max_value=3))
    hardware_dict = OrderedDict()

    for _ in range(num_hardware):
        hw_name = draw(
            st.text(
                min_size=1,
                max_size=30,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
            )
        )
        hardware = draw(hardware_strategy())
        hardware_dict[hw_name] = hardware

    return Table(hardware=hardware_dict)


# Test data classes for LeafType testing


@dataclass(kw_only=True)
class LeafTypeTestClass(JSONSerializable):
    """Test class that uses all LeafType values for comprehensive testing."""

    _is_leaf: bool = True

    # Test all LeafType values
    none_field: None = None
    bool_field: bool = True
    int_field: int = 42
    float_field: float = 3.14
    str_field: str = "test_string"
    ordered_dict_field: OrderedDict[str, Any] = field(
        default_factory=lambda: OrderedDict([("key1", "value1"), ("key2", 123)])
    )
    torch_dtype_field: torch.dtype = torch.float32
    list_field: list[Any] = field(default_factory=lambda: [1, "two", 3.0, True])


@dataclass(kw_only=True)
class NestedLeafTypeTestClass(JSONSerializable):
    """Test class with nested structures using LeafType values."""

    # Nested OrderedDict with various LeafType values
    nested_dict: OrderedDict[str, Any] = field(
        default_factory=lambda: OrderedDict(
            [
                ("none_val", None),
                ("bool_val", False),
                ("int_val", 999),
                ("float_val", 2.718),
                ("str_val", "nested_string"),
                ("dtype_val", torch.int64),
                ("list_val", [None, True, 42, 1.5, "item"]),
            ]
        )
    )

    # List containing various LeafType values
    mixed_list: list[Any] = field(
        default_factory=lambda: [
            None,
            True,
            123,
            4.56,
            "list_string",
            torch.bfloat16,
            OrderedDict([("nested_key", "nested_value")]),
            [1, 2, 3],
        ]
    )

    # Multiple torch.dtype fields
    dtype1: torch.dtype = torch.float16
    dtype2: torch.dtype = torch.int32
    dtype3: torch.dtype = torch.bool


@dataclass(kw_only=True)
class ComplexLeafTypeTestClass(JSONSerializable):
    """Test class with complex combinations of LeafType values."""

    # OrderedDict with nested structures
    complex_dict: OrderedDict[str, Any] = field(
        default_factory=lambda: OrderedDict(
            [
                (
                    "level1",
                    OrderedDict(
                        [
                            ("level2_none", None),
                            ("level2_bool", True),
                            ("level2_list", [torch.float64, "deep_string", 789]),
                        ]
                    ),
                ),
                ("dtypes", [torch.int8, torch.int16, torch.float32]),
                (
                    "mixed_data",
                    OrderedDict(
                        [
                            ("numbers", [1, 2.5, 3]),
                            ("flags", [True, False, True]),
                            ("types", [torch.uint8, torch.complex64]),
                        ]
                    ),
                ),
            ]
        )
    )

    # List of OrderedDicts
    dict_list: list[Any] = field(
        default_factory=lambda: [
            OrderedDict([("dict1_key", "dict1_value"), ("dict1_dtype", torch.float32)]),
            OrderedDict([("dict2_key", 42), ("dict2_dtype", torch.int64)]),
            OrderedDict([("dict3_key", None), ("dict3_dtype", torch.bool)]),
        ]
    )


# Hypothesis strategies for LeafType test classes


@composite
def leaf_type_test_class_strategy(draw):
    """Generate LeafTypeTestClass instances."""
    return LeafTypeTestClass(
        none_field=None,  # Always None
        bool_field=draw(st.booleans()),
        int_field=draw(st.integers(min_value=-1000, max_value=1000)),
        float_field=draw(
            st.floats(
                min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
            )
        ),
        str_field=draw(
            st.text(
                min_size=0,
                max_size=50,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
            )
        ),
        ordered_dict_field=draw(
            st.fixed_dictionaries(
                {
                    "key1": st.text(min_size=1, max_size=20),
                    "key2": st.integers(min_value=0, max_value=1000),
                }
            ).map(lambda d: OrderedDict(d))
        ),
        torch_dtype_field=draw(torch_dtype_strategy()),
        list_field=draw(
            st.lists(
                st.one_of(
                    st.integers(min_value=0, max_value=100),
                    st.text(min_size=1, max_size=10),
                    st.floats(
                        min_value=0.0,
                        max_value=10.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    st.booleans(),
                ),
                min_size=0,
                max_size=5,
            )
        ),
    )


@composite
def nested_leaf_type_test_class_strategy(draw):
    """Generate NestedLeafTypeTestClass instances."""
    nested_dict = OrderedDict()
    nested_dict["none_val"] = None
    nested_dict["bool_val"] = draw(st.booleans())
    nested_dict["int_val"] = draw(st.integers(min_value=0, max_value=2000))
    nested_dict["float_val"] = draw(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    nested_dict["str_val"] = draw(st.text(min_size=1, max_size=20))
    nested_dict["dtype_val"] = draw(torch_dtype_strategy())
    nested_dict["list_val"] = draw(
        st.lists(
            st.one_of(
                st.none(),
                st.booleans(),
                st.integers(min_value=0, max_value=100),
                st.floats(
                    min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False
                ),
                st.text(min_size=1, max_size=10),
            ),
            min_size=0,
            max_size=3,
        )
    )

    mixed_list = [
        None,
        draw(st.booleans()),
        draw(st.integers(min_value=0, max_value=500)),
        draw(
            st.floats(
                min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False
            )
        ),
        draw(st.text(min_size=1, max_size=15)),
        draw(torch_dtype_strategy()),
        OrderedDict([("nested_key", draw(st.text(min_size=1, max_size=10)))]),
        draw(st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=3)),
    ]

    return NestedLeafTypeTestClass(
        nested_dict=nested_dict,
        mixed_list=mixed_list,
        dtype1=draw(torch_dtype_strategy()),
        dtype2=draw(torch_dtype_strategy()),
        dtype3=draw(torch_dtype_strategy()),
    )


def run_tests():
    unittest.main()


if __name__ == "__main__":
    run_tests()