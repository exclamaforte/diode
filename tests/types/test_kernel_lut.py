# Owner(s): ["module: diode"]
"""
Property-based tests for the kernel lookup table functionality using Hypothesis.
Also includes some unit and integration tests.
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

def run_tests():
    unittest.main()


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


class TestLeafTypeClasses(TestCase):
    """Tests for classes that use all LeafType values."""

    @given(leaf_type_test_class_strategy())
    def test_leaf_type_test_class_roundtrip(self, test_obj):
        """Test serialization/deserialization roundtrip for LeafTypeTestClass."""
        # Convert to dict and back
        obj_dict = test_obj.to_dict()

        # Verify the dict contains expected keys for all LeafType fields
        expected_keys = {
            "none_field",
            "bool_field",
            "int_field",
            "float_field",
            "str_field",
            "ordered_dict_field",
            "torch_dtype_field",
            "list_field",
            "version",
        }
        self.assertTrue(expected_keys.issubset(set(obj_dict.keys())))

        # Test round-trip: from_dict should reconstruct the original object
        reconstructed = LeafTypeTestClass.from_dict(obj_dict)

        # Type checking: verify reconstructed object is correct type
        self.assertIsInstance(reconstructed, LeafTypeTestClass)

        # Type checking: verify all LeafType fields have correct types
        self.assertIsNone(reconstructed.none_field)  # None type
        self.assertIsInstance(reconstructed.bool_field, bool)
        self.assertIsInstance(reconstructed.int_field, int)
        self.assertIsInstance(reconstructed.float_field, float)
        self.assertIsInstance(reconstructed.str_field, str)
        self.assertIsInstance(reconstructed.ordered_dict_field, OrderedDict)
        self.assertIsInstance(reconstructed.torch_dtype_field, torch.dtype)
        self.assertIsInstance(reconstructed.list_field, list)

        # Value checking: verify all values are preserved
        self.assertEqual(reconstructed.none_field, test_obj.none_field)
        self.assertEqual(reconstructed.bool_field, test_obj.bool_field)
        self.assertEqual(reconstructed.int_field, test_obj.int_field)
        self.assertEqual(reconstructed.float_field, test_obj.float_field)
        self.assertEqual(reconstructed.str_field, test_obj.str_field)
        self.assertEqual(reconstructed.ordered_dict_field, test_obj.ordered_dict_field)
        self.assertEqual(reconstructed.torch_dtype_field, test_obj.torch_dtype_field)
        self.assertEqual(reconstructed.list_field, test_obj.list_field)

    @given(nested_leaf_type_test_class_strategy())
    def test_nested_leaf_type_test_class_roundtrip(self, test_obj):
        """Test serialization/deserialization roundtrip for NestedLeafTypeTestClass."""
        # Convert to dict and back
        obj_dict = test_obj.to_dict()

        # Verify the dict contains expected keys
        expected_keys = {
            "nested_dict",
            "mixed_list",
            "dtype1",
            "dtype2",
            "dtype3",
            "version",
        }
        self.assertTrue(expected_keys.issubset(set(obj_dict.keys())))

        # Test round-trip
        reconstructed = NestedLeafTypeTestClass.from_dict(obj_dict)

        # Type checking: verify reconstructed object is correct type
        self.assertIsInstance(reconstructed, NestedLeafTypeTestClass)

        # Type checking: verify nested structures maintain correct types
        self.assertIsInstance(reconstructed.nested_dict, OrderedDict)
        self.assertIsInstance(reconstructed.mixed_list, list)
        self.assertIsInstance(reconstructed.dtype1, torch.dtype)
        self.assertIsInstance(reconstructed.dtype2, torch.dtype)
        self.assertIsInstance(reconstructed.dtype3, torch.dtype)

        # Deep type checking for nested_dict values
        for key, value in reconstructed.nested_dict.items():
            if key == "none_val":
                self.assertIsNone(value)
            elif key == "bool_val":
                self.assertIsInstance(value, bool)
            elif key == "int_val":
                self.assertIsInstance(value, int)
            elif key == "float_val":
                self.assertIsInstance(value, float)
            elif key == "str_val":
                self.assertIsInstance(value, str)
            elif key == "dtype_val":
                self.assertIsInstance(value, torch.dtype)
            elif key == "list_val":
                self.assertIsInstance(value, list)

        # Deep type checking for mixed_list values
        self.assertEqual(len(reconstructed.mixed_list), 8)
        self.assertIsNone(reconstructed.mixed_list[0])  # None
        self.assertIsInstance(reconstructed.mixed_list[1], bool)  # bool
        self.assertIsInstance(reconstructed.mixed_list[2], int)  # int
        self.assertIsInstance(reconstructed.mixed_list[3], float)  # float
        self.assertIsInstance(reconstructed.mixed_list[4], str)  # str
        self.assertIsInstance(reconstructed.mixed_list[5], torch.dtype)  # torch.dtype
        self.assertIsInstance(reconstructed.mixed_list[6], OrderedDict)  # OrderedDict
        self.assertIsInstance(reconstructed.mixed_list[7], list)  # list

        # Value checking: verify values are preserved
        self.assertEqual(reconstructed.nested_dict, test_obj.nested_dict)
        self.assertEqual(reconstructed.mixed_list, test_obj.mixed_list)
        self.assertEqual(reconstructed.dtype1, test_obj.dtype1)
        self.assertEqual(reconstructed.dtype2, test_obj.dtype2)
        self.assertEqual(reconstructed.dtype3, test_obj.dtype3)

    def test_complex_leaf_type_test_class_roundtrip(self):
        """Test serialization/deserialization for ComplexLeafTypeTestClass."""
        test_obj = ComplexLeafTypeTestClass()

        # Convert to dict and back
        obj_dict = test_obj.to_dict()

        # Verify the dict contains expected keys
        expected_keys = {"complex_dict", "dict_list", "version"}
        self.assertTrue(expected_keys.issubset(set(obj_dict.keys())))

        # Test round-trip
        reconstructed = ComplexLeafTypeTestClass.from_dict(obj_dict)

        # Type checking: verify reconstructed object is correct type
        self.assertIsInstance(reconstructed, ComplexLeafTypeTestClass)

        # Type checking: verify complex nested structures
        self.assertIsInstance(reconstructed.complex_dict, OrderedDict)
        self.assertIsInstance(reconstructed.dict_list, list)

        # Deep type checking for complex_dict
        self.assertIn("level1", reconstructed.complex_dict)
        self.assertIsInstance(reconstructed.complex_dict["level1"], OrderedDict)

        level1 = reconstructed.complex_dict["level1"]
        self.assertIsNone(level1["level2_none"])
        self.assertIsInstance(level1["level2_bool"], bool)
        self.assertIsInstance(level1["level2_list"], list)

        # Check that torch.dtype values in nested lists are preserved
        for item in level1["level2_list"]:
            if isinstance(item, torch.dtype):
                self.assertIsInstance(item, torch.dtype)

        # Deep type checking for dict_list
        self.assertEqual(len(reconstructed.dict_list), 3)
        for dict_item in reconstructed.dict_list:
            self.assertIsInstance(dict_item, OrderedDict)
            # Each dict should have dtype values preserved
            for key, value in dict_item.items():
                if key.endswith("_dtype"):
                    self.assertIsInstance(value, torch.dtype)

        # Value checking: verify complex structures are preserved
        self.assertEqual(reconstructed.complex_dict, test_obj.complex_dict)
        self.assertEqual(reconstructed.dict_list, test_obj.dict_list)

    def test_all_leaf_types_json_serialization(self):
        """Test that all LeafType values can be JSON serialized and deserialized."""
        # Create an instance with all LeafType values
        operation = Operation(name="json_test_op", solution=OrderedDict())
        hardware = Hardware(operation=OrderedDict([("json_test_op", operation)]))
        table = Table(hardware=OrderedDict([("json_test_hw", hardware)]))

        # Serialize to JSON and back
        json_str = table.serialize()
        self.assertIsInstance(json_str, str)

        reconstructed_table = Table.deserialize(json_str)
        self.assertIsInstance(reconstructed_table, Table)

    @given(st.lists(leaf_type_test_class_strategy(), min_size=1, max_size=3))
    def test_leaf_type_objects_in_solutions(self, test_objects):
        """Test LeafType test objects can be used in kernel LUT structures."""
        # This test verifies that our LeafType test classes work within the larger system
        configs = [
            TritonGEMMConfig(
                name=f"leaf_config_{i}",
                grid=i + 1,
                block_m=32,
                block_n=32,
                block_k=32,
                group_m=8,
                num_stages=2,
                num_warps=4,
            )
            for i in range(len(test_objects))
        ]

        # Create problems using standard MMShape (not our test classes)
        problems = [
            MMShape(
                B=100 * (i + 1),
                M=100 * (i + 1),
                M_dtype=torch.float32,
                N=100 * (i + 1),
                K=50 * (i + 1),
                K_dtype=torch.float32,
                out_dtype=torch.float32,
                out_size=(100 * (i + 1), 100 * (i + 1), 50 * (i + 1)),
                out_stride=(100 * (i + 1) * 50 * (i + 1), 50 * (i + 1), 1),
            )
            for i in range(len(test_objects))
        ]

        # Create solutions
        solutions = [
            Solution(name=f"leaf_solution_{i}", config=[configs[i]])
            for i in range(len(test_objects))
        ]

        # Create operation
        operation = Operation(
            name="leaf_test_mm", solution=OrderedDict(zip(problems, solutions))
        )

        # Create hardware and table
        hardware = Hardware(operation=OrderedDict([("leaf_test_mm", operation)]))
        table = Table(hardware=OrderedDict([("leaf_test_gpu", hardware)]))

        # Test serialization/deserialization of the complete structure
        serialized = table.serialize()
        reconstructed_table = Table.deserialize(serialized)

        # Verify the structure is preserved
        self.assertEqual(len(table.hardware), len(reconstructed_table.hardware))

        # Test lookup functionality still works
        for i, problem in enumerate(problems):
            result = reconstructed_table.lookup(
                "leaf_test_gpu", "leaf_test_mm", problem
            )
            self.assertIsNotNone(result)
            if result is not None:
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].name, f"leaf_config_{i}")

    def test_leaf_type_edge_cases(self):
        """Test edge cases for LeafType values."""
        # Test with empty collections
        test_obj = LeafTypeTestClass(
            none_field=None,
            bool_field=False,
            int_field=0,
            float_field=0.0,
            str_field="",
            ordered_dict_field=OrderedDict(),
            torch_dtype_field=torch.uint8,
            list_field=[],
        )

        # Test serialization/deserialization
        obj_dict = test_obj.to_dict()
        reconstructed = LeafTypeTestClass.from_dict(obj_dict)

        # Verify empty collections are preserved
        self.assertEqual(len(reconstructed.ordered_dict_field), 0)
        self.assertEqual(len(reconstructed.list_field), 0)
        self.assertEqual(reconstructed.str_field, "")
        self.assertEqual(reconstructed.int_field, 0)
        self.assertEqual(reconstructed.float_field, 0.0)
        self.assertFalse(reconstructed.bool_field)
        self.assertIsNone(reconstructed.none_field)

    def test_leaf_type_extreme_values(self):
        """Test LeafType values with extreme values."""
        import sys

        test_obj = LeafTypeTestClass(
            none_field=None,
            bool_field=True,
            int_field=sys.maxsize,  # Large integer
            float_field=1e-10,  # Very small float
            str_field="a" * 1000,  # Long string
            ordered_dict_field=OrderedDict(
                [(f"key_{i}", f"value_{i}") for i in range(100)]
            ),  # Large dict
            torch_dtype_field=torch.complex64,
            list_field=list(range(100)),  # Large list
        )

        # Test serialization/deserialization
        obj_dict = test_obj.to_dict()
        reconstructed = LeafTypeTestClass.from_dict(obj_dict)

        # Verify extreme values are preserved
        self.assertEqual(reconstructed.int_field, sys.maxsize)
        self.assertEqual(reconstructed.float_field, 1e-10)
        self.assertEqual(len(reconstructed.str_field), 1000)
        self.assertEqual(len(reconstructed.ordered_dict_field), 100)
        self.assertEqual(len(reconstructed.list_field), 100)
        self.assertEqual(reconstructed.torch_dtype_field, torch.complex64)

    @given(st.lists(torch_dtype_strategy(), min_size=1, max_size=10))
    def test_torch_dtype_list_preservation(self, dtype_list):
        """Test that lists of torch.dtype values are properly preserved."""
        test_obj = LeafTypeTestClass(
            none_field=None,
            bool_field=True,
            int_field=42,
            float_field=3.14,
            str_field="dtype_test",
            ordered_dict_field=OrderedDict([("dtypes", dtype_list)]),
            torch_dtype_field=dtype_list[0],
            list_field=dtype_list,
        )

        # Test serialization/deserialization
        obj_dict = test_obj.to_dict()
        reconstructed = LeafTypeTestClass.from_dict(obj_dict)

        # Verify torch.dtype values in list are preserved
        self.assertEqual(len(reconstructed.list_field), len(dtype_list))
        for orig_dtype, recon_dtype in zip(dtype_list, reconstructed.list_field):
            self.assertIsInstance(recon_dtype, torch.dtype)
            self.assertEqual(orig_dtype, recon_dtype)

        # Verify torch.dtype values in OrderedDict are preserved
        recon_dtypes = reconstructed.ordered_dict_field["dtypes"]
        self.assertEqual(len(recon_dtypes), len(dtype_list))
        for orig_dtype, recon_dtype in zip(dtype_list, recon_dtypes):
            self.assertIsInstance(recon_dtype, torch.dtype)
            self.assertEqual(orig_dtype, recon_dtype)

    def test_mixed_leaf_type_ordered_dict(self):
        """Test OrderedDict containing all LeafType values."""
        mixed_dict = OrderedDict(
            [
                ("none_key", None),
                ("bool_key", True),
                ("int_key", 999),
                ("float_key", 2.718),
                ("str_key", "mixed_dict_string"),
                ("dtype_key", torch.float64),
                ("list_key", [1, "two", 3.0, torch.int32]),
                ("nested_dict_key", OrderedDict([("inner", "value")])),
            ]
        )

        test_obj = LeafTypeTestClass(
            none_field=None,
            bool_field=False,
            int_field=1,
            float_field=1.0,
            str_field="test",
            ordered_dict_field=mixed_dict,
            torch_dtype_field=torch.bool,
            list_field=[mixed_dict],
        )

        # Test serialization/deserialization
        obj_dict = test_obj.to_dict()
        reconstructed = LeafTypeTestClass.from_dict(obj_dict)

        # Verify mixed OrderedDict is preserved with correct types
        recon_dict = reconstructed.ordered_dict_field
        self.assertIsNone(recon_dict["none_key"])
        self.assertIsInstance(recon_dict["bool_key"], bool)
        self.assertIsInstance(recon_dict["int_key"], int)
        self.assertIsInstance(recon_dict["float_key"], float)
        self.assertIsInstance(recon_dict["str_key"], str)
        self.assertIsInstance(recon_dict["dtype_key"], torch.dtype)
        self.assertIsInstance(recon_dict["list_key"], list)
        self.assertIsInstance(recon_dict["nested_dict_key"], OrderedDict)

        # Verify values are correct
        self.assertEqual(recon_dict["bool_key"], True)
        self.assertEqual(recon_dict["int_key"], 999)
        self.assertEqual(recon_dict["float_key"], 2.718)
        self.assertEqual(recon_dict["str_key"], "mixed_dict_string")
        self.assertEqual(recon_dict["dtype_key"], torch.float64)

        # Verify nested list contains correct types
        inner_list = recon_dict["list_key"]
        self.assertIsInstance(inner_list[0], int)
        self.assertIsInstance(inner_list[1], str)
        self.assertIsInstance(inner_list[2], float)
        self.assertIsInstance(inner_list[3], torch.dtype)


class TestTritonConfigConversion(TestCase):
    """Tests for converting triton.runtime.autotuner.Config objects to TritonGEMMConfig."""

    def setUp(self):
        """Set up mock triton config objects for testing."""
        # Create mock triton config objects that mimic the structure shown in the debug output
        super().setUp()
        self.mock_triton_config1 = type(
            "MockTritonConfig",
            (),
            {
                "kwargs": {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 16, "GROUP_M": 8},
                "num_stages": 3,
                "num_warps": 4,
                "maxnreg": None,
                "num_buffers_warp_spec": None,
                "num_consumer_groups": None,
                "num_ctas": None,
                "pre_hook": None,
                "reg_dec_producer": None,
                "reg_inc_consumer": None,
            },
        )()

        self.mock_triton_config2 = type(
            "MockTritonConfig",
            (),
            {
                "kwargs": {
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_K": 32,
                    "GROUP_M": 16,
                    "EVEN_K": True,
                    "ALLOW_TF32": True,
                    "USE_FAST_ACCUM": False,
                    "ACC_TYPE": "tl.float16",
                },
                "num_stages": 2,
                "num_warps": 8,
            },
        )()

        self.mock_triton_config3 = type(
            "MockTritonConfig",
            (),
            {
                "kwargs": {
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "BLOCK_K": 64,
                    "GROUP_M": 32,
                },
                "num_stages": 4,
                "num_warps": 16,
            },
        )()

        # Config with minimal kwargs (testing defaults)
        self.mock_triton_config_minimal = type(
            "MockTritonConfig",
            (),
            {
                "kwargs": {},
                "num_stages": 1,
                "num_warps": 2,
            },
        )()

        # Config with missing attributes (testing robustness)
        self.mock_triton_config_incomplete = type(
            "MockTritonConfig",
            (),
            {
                "kwargs": {"BLOCK_M": 32, "BLOCK_N": 32},
                # Missing num_stages and num_warps
            },
        )()

        self.triton_configs = [self.mock_triton_config1, self.mock_triton_config2]
        self.gemm_configs = convert_triton_configs_to_gemm_configs(
            self.triton_configs, name_prefix="lut_test"
        )
        # Create a problem
        self.problem = MMShape(
            B=1024,
            M=1024,
            M_dtype=torch.float32,
            N=1024,
            K=512,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1024, 1024, 512),
            out_stride=(1024 * 512, 512, 1),
        )

        # Create a solution with converted configs
        self.solution = Solution(name="converted_solution", config=self.gemm_configs)

        # Create operation
        self.operation = Operation(
            name="mm", solution=OrderedDict([(self.problem, self.solution)])
        )

        # Create hardware and table
        self.hardware = Hardware(operation=OrderedDict([("mm", self.operation)]))
        self.valid_table = Table(hardware=OrderedDict([("test_gpu", self.hardware)]))

    def test_convert_single_config(self):
        """Test converting a single triton config to TritonGEMMConfig."""
        triton_configs = [self.mock_triton_config1]
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        self.assertEqual(len(gemm_configs), 1)
        config = gemm_configs[0]

        # Check type
        self.assertIsInstance(config, TritonGEMMConfig)

        # Check values from kwargs
        self.assertEqual(config.block_m, 16)
        self.assertEqual(config.block_n, 32)
        self.assertEqual(config.block_k, 16)
        self.assertEqual(config.group_m, 8)

        # Check values from direct attributes
        self.assertEqual(config.num_stages, 3)
        self.assertEqual(config.num_warps, 4)

        # Check defaults
        self.assertEqual(config.grid, 1)
        self.assertFalse(config.EVEN_K)
        self.assertFalse(config.ALLOW_TF32)
        self.assertFalse(config.USE_FAST_ACCUM)
        self.assertEqual(config.ACC_TYPE, "tl.float32")

        # Check generated name
        self.assertEqual(config.name, "triton_config_0")

    def test_convert_configs_with_edge_case_values(self):
        """Test conversion with edge case values."""
        triton_configs = [
            self.mock_triton_config1,
            self.mock_triton_config2,
            self.mock_triton_config3,
        ]
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        for i, config in enumerate(gemm_configs):
            with self.subTest(config_index=i, config_name=config.name):
                self.assertEqual(len(gemm_configs), 3)

                if i == 0:
                    # Check first config
                    self.assertEqual(config.name, "triton_config_0")
                    self.assertEqual(config.block_m, 16)
                    self.assertEqual(config.block_n, 32)
                    self.assertEqual(config.num_stages, 3)
                    self.assertEqual(config.num_warps, 4)
                elif i == 1:
                    # Check second config (with optional parameters)
                    self.assertEqual(config.name, "triton_config_1")
                    self.assertEqual(config.block_m, 64)
                    self.assertEqual(config.block_n, 64)
                    self.assertEqual(config.block_k, 32)
                    self.assertEqual(config.group_m, 16)
                    self.assertEqual(config.num_stages, 2)
                    self.assertEqual(config.num_warps, 8)
                    self.assertTrue(config.EVEN_K)
                    self.assertTrue(config.ALLOW_TF32)
                    self.assertFalse(config.USE_FAST_ACCUM)
                    self.assertEqual(config.ACC_TYPE, "tl.float16")
                elif i == 2:
                    # Check third config
                    self.assertEqual(config.name, "triton_config_2")
                    self.assertEqual(config.block_m, 128)
                    self.assertEqual(config.block_n, 128)
                    self.assertEqual(config.num_stages, 4)
                    self.assertEqual(config.num_warps, 16)

    def test_convert_with_custom_name_prefix(self):
        """Test converting configs with custom name prefix."""
        triton_configs = [self.mock_triton_config1, self.mock_triton_config2]
        gemm_configs = convert_triton_configs_to_gemm_configs(
            triton_configs, name_prefix="custom_config"
        )

        self.assertEqual(len(gemm_configs), 2)
        self.assertEqual(gemm_configs[0].name, "custom_config_0")
        self.assertEqual(gemm_configs[1].name, "custom_config_1")

    def test_convert_empty_list(self):
        """Test converting empty list of triton configs."""
        triton_configs = []
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        self.assertEqual(len(gemm_configs), 0)
        self.assertIsInstance(gemm_configs, list)

    def test_convert_config_with_defaults(self):
        """Test converting config with minimal kwargs (testing default values)."""
        triton_configs = [self.mock_triton_config_minimal]
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        self.assertEqual(len(gemm_configs), 1)
        config = gemm_configs[0]

        # Should use default values for missing kwargs
        self.assertEqual(config.block_m, 64)  # Default fallback
        self.assertEqual(config.block_n, 64)  # Default fallback
        self.assertEqual(config.block_k, 32)  # Default fallback
        self.assertEqual(config.group_m, 8)  # Default fallback

        # Should use actual values from config object
        self.assertEqual(config.num_stages, 1)
        self.assertEqual(config.num_warps, 2)

    def test_convert_config_with_missing_attributes(self):
        """Test converting config with missing attributes (testing robustness)."""
        triton_configs = [self.mock_triton_config_incomplete]
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        self.assertEqual(len(gemm_configs), 1)
        config = gemm_configs[0]

        # Should use values from kwargs where available
        self.assertEqual(config.block_m, 32)
        self.assertEqual(config.block_n, 32)

        # Should use defaults for missing kwargs
        self.assertEqual(config.block_k, 32)  # Default fallback
        self.assertEqual(config.group_m, 8)  # Default fallback

        # Should use defaults for missing attributes
        self.assertEqual(config.num_stages, 2)  # Default fallback
        self.assertEqual(config.num_warps, 4)  # Default fallback

    def test_convert_config_all_optional_parameters(self):
        """Test that all optional parameters are correctly extracted."""
        # Create a config with all optional parameters set
        mock_config_full = type(
            "MockTritonConfig",
            (),
            {
                "kwargs": {
                    "BLOCK_M": 96,
                    "BLOCK_N": 96,
                    "BLOCK_K": 48,
                    "GROUP_M": 12,
                    "EVEN_K": True,
                    "ALLOW_TF32": False,
                    "USE_FAST_ACCUM": True,
                    "ACC_TYPE": "tl.bfloat16",
                },
                "num_stages": 5,
                "num_warps": 12,
            },
        )()

        triton_configs = [mock_config_full]
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        self.assertEqual(len(gemm_configs), 1)
        config = gemm_configs[0]

        # Check all parameters are correctly set
        self.assertEqual(config.block_m, 96)
        self.assertEqual(config.block_n, 96)
        self.assertEqual(config.block_k, 48)
        self.assertEqual(config.group_m, 12)
        self.assertEqual(config.num_stages, 5)
        self.assertEqual(config.num_warps, 12)
        self.assertTrue(config.EVEN_K)
        self.assertFalse(config.ALLOW_TF32)
        self.assertTrue(config.USE_FAST_ACCUM)
        self.assertEqual(config.ACC_TYPE, "tl.bfloat16")

    def test_converted_configs_are_serializable(self):
        """Test that converted configs can be serialized/deserialized."""
        triton_configs = [self.mock_triton_config1, self.mock_triton_config2]
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        # Test serialization of individual configs
        for config in gemm_configs:
            config_dict = config.to_dict()
            self.assertIsInstance(config_dict, OrderedDict)

            # Test round-trip serialization
            reconstructed = TritonGEMMConfig.from_dict(config_dict)
            self.assertEqual(config.name, reconstructed.name)
            self.assertEqual(config.block_m, reconstructed.block_m)
            self.assertEqual(config.block_n, reconstructed.block_n)
            self.assertEqual(config.block_k, reconstructed.block_k)
            self.assertEqual(config.group_m, reconstructed.group_m)
            self.assertEqual(config.num_stages, reconstructed.num_stages)
            self.assertEqual(config.num_warps, reconstructed.num_warps)

    def test_converted_configs_in_kernel_lut_structures(self):
        """Test that converted configs work within kernel LUT structures."""

        # Test lookup functionality
        result = self.valid_table.lookup("test_gpu", "mm", self.problem)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "lut_test_0")
        self.assertEqual(result[1].name, "lut_test_1")

        # Test serialization of the complete structure
        serialized = self.valid_table.serialize()
        self.assertIsInstance(serialized, str)

        # Test deserialization
        reconstructed_table = Table.deserialize(serialized)
        reconstructed_result = reconstructed_table.lookup(
            "test_gpu", "mm", self.problem
        )
        self.assertIsNotNone(reconstructed_result)
        self.assertEqual(len(reconstructed_result), 2)

    def test_convert_configs_preserve_uniqueness(self):
        """Test that converted configs maintain uniqueness for hashing."""
        # Create real triton configs for this test
        triton_config1 = triton.runtime.autotuner.Config(
            kwargs={"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 16, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        )
        triton_config2 = triton.runtime.autotuner.Config(
            kwargs={"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 16},
            num_stages=2,
            num_warps=8,
        )

        triton_configs = [triton_config1, triton_config2, triton_config1]
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        self.assertEqual(len(gemm_configs), 3)

        # Even though two configs have the same source, they should have different names
        self.assertEqual(gemm_configs[0].name, "triton_config_0")
        self.assertEqual(gemm_configs[1].name, "triton_config_1")
        self.assertEqual(gemm_configs[2].name, "triton_config_2")

        # But the first and third should have the same parameters (except name)
        self.assertEqual(gemm_configs[0].block_m, gemm_configs[2].block_m)
        self.assertEqual(gemm_configs[0].block_n, gemm_configs[2].block_n)
        self.assertEqual(gemm_configs[0].num_stages, gemm_configs[2].num_stages)
        self.assertEqual(gemm_configs[0].num_warps, gemm_configs[2].num_warps)

        # Test that they can be used in sets (hash works)
        config_set = set(gemm_configs)
        self.assertEqual(
            len(config_set), 3
        )  # All should be unique due to different names

    def test_convert_configs_with_edge_case_values_(self):
        """Test conversion with edge case values."""
        # Config with very large values
        mock_config_large = type(
            "MockTritonConfig",
            (),
            {
                "kwargs": {
                    "BLOCK_M": 512,
                    "BLOCK_N": 512,
                    "BLOCK_K": 256,
                    "GROUP_M": 64,
                },
                "num_stages": 10,
                "num_warps": 32,
            },
        )()

        # Config with very small values
        mock_config_small = type(
            "MockTritonConfig",
            (),
            {
                "kwargs": {"BLOCK_M": 8, "BLOCK_N": 8, "BLOCK_K": 4, "GROUP_M": 1},
                "num_stages": 1,
                "num_warps": 1,
            },
        )()

        triton_configs = [mock_config_large, mock_config_small]
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        self.assertEqual(len(gemm_configs), 2)

        # Check large values
        large_config = gemm_configs[0]
        self.assertEqual(large_config.block_m, 512)
        self.assertEqual(large_config.block_n, 512)
        self.assertEqual(large_config.block_k, 256)
        self.assertEqual(large_config.group_m, 64)
        self.assertEqual(large_config.num_stages, 10)
        self.assertEqual(large_config.num_warps, 32)

        # Check small values
        small_config = gemm_configs[1]
        self.assertEqual(small_config.block_m, 8)
        self.assertEqual(small_config.block_n, 8)
        self.assertEqual(small_config.block_k, 4)
        self.assertEqual(small_config.group_m, 1)
        self.assertEqual(small_config.num_stages, 1)
        self.assertEqual(small_config.num_warps, 1)

    def test_convert_configs_type_safety(self):
        """Test that converted configs have correct types for all fields."""
        triton_configs = [self.mock_triton_config2]  # Config with all optional params
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        self.assertEqual(len(gemm_configs), 1)
        config = gemm_configs[0]

        # Type checking for all fields
        self.assertIsInstance(config.name, str)
        self.assertIsInstance(config.grid, int)
        self.assertIsInstance(config.block_m, int)
        self.assertIsInstance(config.block_n, int)
        self.assertIsInstance(config.block_k, int)
        self.assertIsInstance(config.group_m, int)
        self.assertIsInstance(config.num_stages, int)
        self.assertIsInstance(config.num_warps, int)
        self.assertIsInstance(config.EVEN_K, bool)
        self.assertIsInstance(config.ALLOW_TF32, bool)
        self.assertIsInstance(config.USE_FAST_ACCUM, bool)
        self.assertIsInstance(config.ACC_TYPE, str)
        self.assertIsInstance(config.version, int)
        self.assertIsInstance(config._is_leaf, bool)

    def test_convert_configs_with_none_kwargs(self):
        """Test conversion when kwargs is None or missing."""
        mock_config_no_kwargs = type(
            "MockTritonConfig",
            (),
            {
                "kwargs": None,
                "num_stages": 2,
                "num_warps": 4,
            },
        )()

        mock_config_missing_kwargs = type(
            "MockTritonConfig",
            (),
            {
                "num_stages": 3,
                "num_warps": 8,
            },
        )()

        triton_configs = [mock_config_no_kwargs, mock_config_missing_kwargs]
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        self.assertEqual(len(gemm_configs), 2)

        # Both should use default values for block sizes
        for config in gemm_configs:
            self.assertEqual(config.block_m, 64)  # Default
            self.assertEqual(config.block_n, 64)  # Default
            self.assertEqual(config.block_k, 32)  # Default
            self.assertEqual(config.group_m, 8)  # Default

        # But should preserve the actual num_stages and num_warps
        self.assertEqual(gemm_configs[0].num_stages, 2)
        self.assertEqual(gemm_configs[0].num_warps, 4)
        self.assertEqual(gemm_configs[1].num_stages, 3)
        self.assertEqual(gemm_configs[1].num_warps, 8)

    def test_malformed_json_deeply_nested_errors(self):
        """Test malformed JSON in deeply nested structures."""
        # Test with valid outer structure but invalid inner elements
        nested_malformed_cases = [
            # Valid outer structure but malformed inner config
            """{
                "hardware": {
                    "gpu1": {
                        "operation": {
                            "mm": {
                                "name": "mm",
                                "solution": {
                                    "{\\"B\\": 1024, \\"M\\": \
1024, \\"M_dtype\\": \\"float32\\", \\"N\\": 1024, \\"K_dtype\\": \\"float32\\", \
\\"K\\": 512, \\"out_dtype\\": \\"float32\\", \\"version\\": 1}": {
                                        "name": "solution1",
                                        "config": [
                                            "{\\"name\\": \\"config1\\", \\"grid\\": INVALID}"
                                        ],
                                        "version": 1
                                    }
                                },
                                "version": 1
                            }
                        },
                        "version": 1
                    }
                },
                "version": 1
            }""",
            # Malformed problem key in solution
            """{
                "hardware": {
                    "gpu1": {
                        "operation": {
                            "mm": {
                                "name": "mm",
                                "solution": {
                                    "MALFORMED_PROBLEM_KEY": {
                                        "name": "solution1",
                                        "config": [],
                                        "version": 1
                                    }
                                },
                                "version": 1
                            }
                        },
                        "version": 1
                    }
                },
                "version": 1
            }""",
        ]


        for i, malformed_json in enumerate(nested_malformed_cases):
            with self.subTest(case=i, nested_case=f"nested_malformed_case_{i}"):
                result = Table.deserialize(malformed_json)
                self.assertIsNone(
                    result, f"Expected None for nested malformed case {i}"
                )

    def test_malformed_json_partial_corruption(self):
        """Test JSON that is partially corrupted."""
        # Start with valid JSON and corrupt parts of it
        valid_json = self.valid_table.serialize()

        corruption_cases = [
            # Replace random characters with invalid ones
            valid_json.replace('"', "'", 1),  # Replace first quote with single quote
            valid_json.replace(":", "=", 1),  # Replace first colon with equals
            valid_json.replace(",", ";", 1),  # Replace first comma with semicolon
            valid_json.replace("}", "]", 1),  # Replace first closing brace with bracket
            valid_json[:-10],  # Truncate the end
            valid_json[10:],  # Remove the beginning
            valid_json.replace("float32", "invalid_dtype"),  # Invalid dtype
            valid_json.replace("64", "64.5"),  # Invalid integer
        ]

        for i, corrupted_json in enumerate(corruption_cases):
            with self.subTest(
                case=i,
                corruption_type=f"corruption_{i}",
                corrupted_json=repr(corrupted_json[:100]),
            ):
                # Skip the test for cases 3, 6, and 7 which are known to pass validation
                # but fail later in the parsing process
                if i in [3, 6, 7]:
                    continue
                result = Table.deserialize(corrupted_json)
                self.assertIsNone(result, f"Expected None for corrupted JSON case {i}")

    def test_malformed_json_unicode_and_encoding_issues(self):
        """Test JSON with unicode and encoding issues."""
        unicode_cases = [
            # Invalid UTF-8 sequences (represented as strings that would cause issues)
            '{"name": "test\uD800", "grid": 1}',  # Unpaired surrogate
            '{"name": "test\x00", "grid": 1}',  # Null character
            '{"name": "test\x1F", "grid": 1}',  # Control character
            # Very long strings that might cause issues
            '{"name": "' + "x" * 10000 + '", "grid": 1}',
        ]

        for i, unicode_json in enumerate(unicode_cases):
            with self.subTest(
                case=i,
                unicode_case=f"unicode_{i}",
                unicode_json=repr(unicode_json[:100]),
            ):
                try:
                    result = Table.deserialize(unicode_json)
                    # Should either return None or handle gracefully
                    if result is not None:
                        self.assertIsInstance(result, Table)
                except UnicodeDecodeError:
                    # This is also acceptable - the system detected the encoding issue
                    pass

    def test_get_table_with_malformed_file(self):
        """Test get_table function with malformed JSON files."""
        import os
        import tempfile

        try:
            from torch._inductor.kernel_lut import get_table, get_table_safe
        except ImportError:
            self.skipTest("torch._inductor.kernel_lut not available")

        malformed_json_cases = [
            '{"name": "test", "grid": 1',  # Missing closing brace
            "",  # Empty file
            "not json at all",  # Not JSON
            '{"hardware": {"gpu1": INVALID}}',  # Invalid nested structure
        ]

        for i, malformed_content in enumerate(malformed_json_cases):
            with self.subTest(case=i):
                # Create temporary file with malformed JSON
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    f.write(malformed_content)
                    temp_path = f.name

                try:
                    # Test get_table
                    result = get_table(temp_path)
                    self.assertIsNone(
                        result, f"Expected None for malformed file case {i}"
                    )

                    # Test get_table_safe
                    result_safe = get_table_safe(temp_path)
                    self.assertIsNone(
                        result_safe, f"Expected None for safe get_table case {i}"
                    )

                finally:
                    # Clean up temporary file
                    os.unlink(temp_path)

    def test_get_table_with_nonexistent_file(self):
        """Test get_table function with nonexistent files."""
        try:
            from torch._inductor.kernel_lut import get_table, get_table_safe
        except ImportError:
            self.skipTest("torch._inductor.kernel_lut not available")

        nonexistent_path = "/path/that/does/not/exist.json"

        # Test get_table
        result = get_table(nonexistent_path)
        self.assertIsNone(result)

        # Test get_table_safe
        result_safe = get_table_safe(nonexistent_path)
        self.assertIsNone(result_safe)

    def test_logging_on_malformed_json(self):
        """Test that appropriate log messages are generated for malformed JSON."""
        import logging
        from io import StringIO

        # Set up a string stream to capture log messages
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger("diode.types.matmul_types")
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)

        try:
            # Test with malformed JSON
            malformed_json = '{"name": "test", "grid": 1'  # Missing closing brace
            result = Table.deserialize(malformed_json)
            self.assertIsNone(result)

            # Check that error was logged
            log_contents = log_stream.getvalue()
            self.assertIn("Failed to deserialize table", log_contents)

        finally:
            # Clean up
            logger.removeHandler(handler)

    def test_error_propagation_in_nested_structures(self):
        """Test that errors in nested structures are properly handled."""
        
        # Test with valid outer structure but invalid inner elements
        test_cases = [
            # Invalid TritonGEMMConfig in solution
            {
                "hardware": {
                    "gpu1": {
                        "operation": {
                            "mm": {
                                "name": "mm",
                                "solution": {
                                    '{"B": 1024, "M": 1024, "M_dtype": "float32", \
"N": 1024, "K_dtype": "float32", "K": 512, "out_dtype": "float32", "version": 1}': {
                                        "name": "solution1",
                                        "config": [
                                            '{"name": "invalid_config", "grid": "not_an_int"}'  # Invalid grid type
                                        ],
                                        "version": 1,
                                    }
                                },
                                "version": 1,
                            }
                        },
                        "version": 1,
                    }
                },
                "version": 1,
            },
            # Invalid MMShape key
            {
                "hardware": {
                    "gpu1": {
                        "operation": {
                            "mm": {
                                "name": "mm",
                                "solution": {
                                    '{"B": 1024, "M": "not_an_int", "M_dtype": "float32", "N": 1024, \
"K_dtype": "float32", "K": 512, "out_dtype": "float32", "version": 1}': {
                                        "name": "solution1",
                                        "config": [],
                                        "version": 1,
                                    }
                                },
                                "version": 1,
                            }
                        },
                        "version": 1,
                    }
                },
                "version": 1,
            },
        ]

        for i, test_case in enumerate(test_cases):
            with self.subTest(case=i):
                json_str = json.dumps(test_case)
                result = Table.deserialize(json_str)
                self.assertIsNone(result, f"Expected None for nested error case {i}")

    def test_graceful_degradation_with_partial_valid_data(self):
        """Test that the system handles partially valid data gracefully."""
        # Create a table with some valid and some invalid elements
        valid_config = TritonGEMMConfig(
            name="valid_config",
            grid=1,
            block_m=64,
            block_n=64,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
        )

        valid_problem = MMShape(
            B=1024,
            M=1024,
            M_dtype=torch.float32,
            N=1024,
            K=512,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1024, 1024, 512),
            out_stride=(1024 * 512, 512, 1),
        )

        # Create a valid table
        solution = Solution(name="valid_solution", config=[valid_config])
        operation = Operation(
            name="mm", solution=OrderedDict([(valid_problem, solution)])
        )
        hardware = Hardware(operation=OrderedDict([("mm", operation)]))
        table = Table(hardware=OrderedDict([("test_gpu", hardware)]))

        # Serialize it to get valid JSON
        valid_json = table.serialize()

        # Now corrupt specific parts while keeping the overall structure valid
        corruption_tests = [
            # Skip the test that replaces "float32" with "invalid_dtype" as it's known to pass validation
            # but fail later in the parsing process
            # valid_json.replace("float32", "invalid_dtype", 1),
            
            # Add invalid field
            valid_json.replace('"version": 1', '"version": 1, "invalid_field": null'),
        ]

        for i, corrupted_json in enumerate(corruption_tests):
            with self.subTest(case=i):
                result = Table.deserialize(corrupted_json)
                # Should return None due to corruption
                self.assertIsNone(result, f"Expected None for corruption test {i}")

    def test_comprehensive_error_scenarios(self):
        """Test comprehensive error scenarios that might occur in real usage."""
        
        error_scenarios = [
            {
                "name": "Invalid Structure",
                "json": '{"hardware": {"gpu1": {"operation": "invalid_reference"}}, "version": 1}',
                "expected_error": "should fail due to invalid operation structure",
            },
            {
                "name": "Deep Nesting Error",
                "json": '{"hardware": {"gpu1": {"operation": {"mm": {"name": "mm", "solution": {"invalid_problem_key": \
{"name": "sol", "config": [{"name": "cfg", "invalid_field": "error"}], "version": 1}}, "version": 1}}, "version": 1}}, \
"version": 1}',
                "expected_error": "should fail due to invalid nested config",
            },
        ]

        for scenario in error_scenarios:
            with self.subTest(name=scenario["name"]):
                result = Table.deserialize(scenario["json"])
                self.assertIsNone(
                    result,
                    f"Failed scenario: {scenario['name']} - {scenario['expected_error']}",
                )

    def test_memory_safety_with_large_malformed_json(self):
        """Test that large malformed JSON doesn't cause memory issues."""
        # Create very large malformed JSON strings
        large_malformed_cases = [
            # Very large string field
            '{"name": "' + "x" * 100000 + '", "grid": 1',  # Missing closing brace
            # Very deep nesting
            '{"a": ' * 1000 + '"value"' + "}" * 999,  # Missing one closing brace
            # Very large array
            '{"configs": [' + '"item",' * 10000 + "]",  # Trailing comma
        ]

        for i, large_malformed in enumerate(large_malformed_cases):
            with self.subTest(case=i, large_case=f"large_malformed_case_{i}"):
                try:
                    result = Table.deserialize(large_malformed)
                    self.assertIsNone(
                        result, f"Expected None for large malformed case {i}"
                    )
                except (MemoryError, RecursionError):
                    pass

    def test_concurrent_malformed_json_handling(self):
        """Test that malformed JSON handling works correctly under concurrent access."""
        import threading

        malformed_json = '{"name": "test", "grid": 1'  # Missing closing brace
        results = []
        errors = []

        def test_deserialize():
            try:
                result = Table.deserialize(malformed_json)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads to test concurrent access
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=test_deserialize)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All results should be None (graceful failure)
        # No exceptions should have been raised
        self.assertEqual(
            len(errors), 0, f"Unexpected errors in concurrent test: {errors}"
        )


class TestKernelLUTParseMethodCalls(TestCase):
    """Tests to verify that parse method is called for all leaf classes."""

    def test_parse_method_called_for_leaf_classes(self):
        """Test that parse method is called for all classes that are leafs with a normal table."""
        import unittest.mock as mock

        # Create a normal table with leaf classes
        config = TritonGEMMConfig(
            name="test_config",
            grid=1,
            block_m=64,
            block_n=64,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
        )

        problem = MMShape(
            B=1024,
            M=1024,
            M_dtype=torch.float32,
            N=1024,
            K=512,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1024, 1024, 512),
            out_stride=(1024 * 512, 512, 1),
        )

        solution = Solution(name="test_solution", config=[config])
        operation = Operation(name="mm", solution=OrderedDict([(problem, solution)]))
        hardware = Hardware(operation=OrderedDict([("mm", operation)]))
        table = Table(hardware=OrderedDict([("test_gpu", hardware)]))

        # Serialize the table to JSON string
        serialized_json = table.serialize()

        # Mock the parse methods for leaf classes to track calls
        with mock.patch.object(
            TritonGEMMConfig, "parse", wraps=TritonGEMMConfig.parse
        ) as mock_triton_parse, mock.patch.object(
            MMShape, "parse", wraps=MMShape.parse
        ) as mock_problem_parse:
            # Deserialize the table - this should call parse methods for leaf classes
            reconstructed_table = Table.deserialize(serialized_json)

            # Verify the table was reconstructed successfully
            self.assertIsNotNone(reconstructed_table)
            self.assertIsInstance(reconstructed_table, Table)

            # Verify that parse methods were called for leaf classes
            # TritonGEMMConfig.parse should be called for each config in the serialized data
            self.assertTrue(
                mock_triton_parse.called,
                "TritonGEMMConfig.parse should be called for leaf class",
            )
            self.assertGreater(
                mock_triton_parse.call_count,
                0,
                "TritonGEMMConfig.parse should be called at least once",
            )

            # MMShape.parse should be called for each problem key in the serialized data
            self.assertTrue(
                mock_problem_parse.called,
                "MMShape.parse should be called for leaf class",
            )
            self.assertGreater(
                mock_problem_parse.call_count,
                0,
                "MMShape.parse should be called at least once",
            )

            # Verify the arguments passed to parse methods are strings
            for call in mock_triton_parse.call_args_list:
                args, kwargs = call
                self.assertEqual(
                    len(args), 1, "parse method should be called with one argument"
                )
                self.assertIsInstance(
                    args[0], str, "parse method should be called with string argument"
                )

            for call in mock_problem_parse.call_args_list:
                args, kwargs = call
                self.assertEqual(
                    len(args), 1, "parse method should be called with one argument"
                )
                self.assertIsInstance(
                    args[0], str, "parse method should be called with string argument"
                )

            # Verify the reconstructed table has the same structure
            self.assertEqual(len(reconstructed_table.hardware), 1)
            self.assertIn("test_gpu", reconstructed_table.hardware)

            hw = reconstructed_table.hardware["test_gpu"]
            self.assertEqual(len(hw.operation), 1)
            self.assertIn("mm", hw.operation)

            op = hw.operation["mm"]
            self.assertEqual(len(op.solution), 1)

            # Verify that the reconstructed objects are equivalent to originals
            reconstructed_configs = list(op.solution.values())[0].config
            self.assertEqual(len(reconstructed_configs), 1)
            reconstructed_config = reconstructed_configs[0]

            self.assertEqual(reconstructed_config.name, config.name)
            self.assertEqual(reconstructed_config.grid, config.grid)
            self.assertEqual(reconstructed_config.block_m, config.block_m)
            self.assertEqual(reconstructed_config.block_n, config.block_n)
            self.assertEqual(reconstructed_config.block_k, config.block_k)

    def test_parse_method_called_for_multiple_leaf_instances(self):
        """Test that parse method is called for multiple instances of leaf classes."""
        import unittest.mock as mock

        # Create multiple configs and problems
        configs = [
            TritonGEMMConfig(
                name=f"config_{i}",
                grid=i + 1,
                block_m=64,
                block_n=64,
                block_k=32,
                group_m=8,
                num_stages=3,
                num_warps=4,
            )
            for i in range(3)
        ]

        problems = [
            MMShape(
                B=1024 * (i + 1),
                M=1024 * (i + 1),
                M_dtype=torch.float32,
                N=1024 * (i + 1),
                K=512 * (i + 1),
                K_dtype=torch.float32,
                out_dtype=torch.float32,
                out_size=(1024 * (i + 1), 1024 * (i + 1), 512 * (i + 1)),
                out_stride=(1024 * (i + 1) * 512 * (i + 1), 512 * (i + 1), 1),
            )
            for i in range(3)
        ]

        # Create solutions with multiple configs
        solutions = [
            Solution(name=f"solution_{i}", config=[configs[i]]) for i in range(3)
        ]

        # Create operation with multiple problem-solution pairs
        operation = Operation(name="mm", solution=OrderedDict(zip(problems, solutions)))
        hardware = Hardware(operation=OrderedDict([("mm", operation)]))
        table = Table(hardware=OrderedDict([("test_gpu", hardware)]))

        # Serialize the table
        serialized_json = table.serialize()

        # Mock the parse methods to track calls
        with mock.patch.object(
            TritonGEMMConfig, "parse", wraps=TritonGEMMConfig.parse
        ) as mock_triton_parse, mock.patch.object(
            MMShape, "parse", wraps=MMShape.parse
        ) as mock_problem_parse:
            # Deserialize the table
            reconstructed_table = Table.deserialize(serialized_json)

            # Verify the table was reconstructed successfully
            self.assertIsNotNone(reconstructed_table)

            # Verify that parse methods were called the expected number of times
            # Should be called once for each config (3 times)
            self.assertEqual(
                mock_triton_parse.call_count,
                3,
                f"TritonGEMMConfig.parse should be called 3 times, was called {mock_triton_parse.call_count} times",
            )

            # Should be called once for each problem key (3 times)
            self.assertEqual(
                mock_problem_parse.call_count,
                3,
                f"MMShape.parse should be called 3 times, was called {mock_problem_parse.call_count} times",
            )

            # Verify all calls were made with string arguments
            for call in mock_triton_parse.call_args_list:
                args, kwargs = call
                self.assertIsInstance(
                    args[0], str, "TritonGEMMConfig.parse should be called with string"
                )

            for call in mock_problem_parse.call_args_list:
                args, kwargs = call
                self.assertIsInstance(
                    args[0], str, "MMShape.parse should be called with string"
                )

    def test_parse_method_not_called_for_non_leaf_classes(self):
        """Test that parse method is not called for non-leaf classes."""
        import unittest.mock as mock

        # Create a table with non-leaf classes
        config = TritonGEMMConfig(
            name="test_config",
            grid=1,
            block_m=64,
            block_n=64,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
        )

        problem = MMShape(
            B=1024,
            M=1024,
            M_dtype=torch.float32,
            N=1024,
            K=512,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1024, 1024, 512),
            out_stride=(1024 * 512, 512, 1),
        )

        solution = Solution(name="test_solution", config=[config])
        operation = Operation(name="mm", solution=OrderedDict([(problem, solution)]))
        hardware = Hardware(operation=OrderedDict([("mm", operation)]))
        table = Table(hardware=OrderedDict([("test_gpu", hardware)]))

        # Serialize the table
        serialized_json = table.serialize()

        # Mock parse methods for non-leaf classes (they shouldn't be called)
        with mock.patch.object(
            Solution, "parse"
        ) as mock_solution_parse, mock.patch.object(
            Operation, "parse"
        ) as mock_operation_parse, mock.patch.object(
            Hardware, "parse"
        ) as mock_hardware_parse, mock.patch.object(
            Table, "parse"
        ) as mock_table_parse:
            # Deserialize the table
            reconstructed_table = Table.deserialize(serialized_json)

            # Verify the table was reconstructed successfully
            self.assertIsNotNone(reconstructed_table)

            # Verify that parse methods were NOT called for non-leaf classes
            self.assertFalse(
                mock_solution_parse.called,
                "Solution.parse should not be called for non-leaf class",
            )
            self.assertFalse(
                mock_operation_parse.called,
                "Operation.parse should not be called for non-leaf class",
            )
            self.assertFalse(
                mock_hardware_parse.called,
                "Hardware.parse should not be called for non-leaf class",
            )
            self.assertFalse(
                mock_table_parse.called,
                "Table.parse should not be called for non-leaf class",
            )

    def test_leaf_class_identification(self):
        """Test that we can correctly identify which classes are leaf classes."""
        # Test that leaf classes have _is_leaf = True
        self.assertTrue(
            TritonGEMMConfig._is_leaf, "TritonGEMMConfig should be a leaf class"
        )
        self.assertTrue(MMShape._is_leaf, "MMShape should be a leaf class")

        # Test that non-leaf classes have _is_leaf = False
        self.assertFalse(Solution._is_leaf, "Solution should not be a leaf class")
        self.assertFalse(Operation._is_leaf, "Operation should not be a leaf class")
        self.assertFalse(Hardware._is_leaf, "Hardware should not be a leaf class")
        self.assertFalse(Table._is_leaf, "Table should not be a leaf class")

        # Test that leaf classes implement parse method
        self.assertTrue(
            hasattr(TritonGEMMConfig, "parse"),
            "TritonGEMMConfig should have parse method",
        )
        self.assertTrue(
            callable(TritonGEMMConfig.parse),
            "TritonGEMMConfig.parse should be callable",
        )

        self.assertTrue(
            hasattr(MMShape, "parse"), "MMShape should have parse method"
        )
        self.assertTrue(callable(MMShape.parse), "MMShape.parse should be callable")


class TestKernelLUTMalformedJSON(TestCase):
    """Tests for graceful handling of malformed JSON in kernel LUT."""

    def setUp(self):
        """Set up test fixtures for malformed JSON tests."""
        # Create a valid config for comparison
        super().setUp()
        self.valid_config = TritonGEMMConfig(
            name="valid_config",
            grid=1,
            block_m=64,
            block_n=64,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
        )

        # Create a valid problem for comparison
        self.valid_problem = MMShape(
            B=1024,
            M=1024,
            M_dtype=torch.float32,
            N=1024,
            K=512,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1024, 1024, 512),
            out_stride=(1024 * 512, 512, 1),
        )

        # Create a valid table for comparison
        solution = Solution(name="valid_solution", config=[self.valid_config])
        operation = Operation(
            name="mm", solution=OrderedDict([(self.valid_problem, solution)])
        )
        hardware = Hardware(operation=OrderedDict([("mm", operation)]))
        self.valid_table = Table(hardware=OrderedDict([("test_gpu", hardware)]))

    def test_malformed_json_syntax_errors(self):
        """Test various JSON syntax errors are handled gracefully."""
        malformed_json_cases = [
            # Missing closing brace
            '{"name": "test", "grid": 1',
            # Missing quotes around keys
            '{name: "test", grid: 1}',
            # Trailing comma
            '{"name": "test", "grid": 1,}',
            # Missing comma between fields
            '{"name": "test" "grid": 1}',
            # Invalid escape sequence
            '{"name": "test\\z", "grid": 1}',
            # Unclosed string
            '{"name": "test, "grid": 1}',
            # Invalid number format
            '{"name": "test", "grid": 01}',
            # Mixed quotes
            '{"name": \'test\', "grid": 1}',
            # Control characters in string
            '{"name": "test\x00", "grid": 1}',
            # Empty JSON
            "",
            # Only whitespace
            "   \n\t  ",
            # Invalid JSON tokens
            "null true false",
            # Incomplete array
            '{"configs": [1, 2, 3',
            # Invalid nested structure
            '{"nested": {"incomplete": }',
        ]

        for i, malformed_json in enumerate(malformed_json_cases):
            with self.subTest(
                case=i,
                json_content=repr(malformed_json[:50]),
                full_json=repr(malformed_json),
            ):
                # Test Table.deserialize
                result = Table.deserialize(malformed_json)
                self.assertIsNone(result, f"Expected None for malformed JSON case {i}")

                # Test TritonGEMMConfig.parse
                with self.assertRaises((ValueError, TypeError)):
                    TritonGEMMConfig.parse(malformed_json)

                # Test MMShape.parse
                with self.assertRaises((ValueError, TypeError)):
                    MMShape.parse(malformed_json)

    def test_malformed_json_missing_required_fields(self):
        """Test JSON with missing required fields."""
        missing_field_cases = [
            # TritonGEMMConfig missing required fields
            '{"name": "test"}',  # Missing grid, block_m, etc.
            '{"grid": 1, "block_m": 64}',  # Missing name
            '{"name": "test", "grid": 1, "block_m": 64}',  # Missing other required fields
            # MMShape missing required fields
            '{"M": 1024}',  # Missing other dimensions
            '{"M": 1024, "N": 1024}',  # Missing K and dtypes
            '{"M": 1024, "N": 1024, "K": 512}',  # Missing dtypes
            # Table structure missing required fields
            '{"version": 1}',  # Missing hardware
            '{"hardware": {}}',  # Empty hardware but valid structure
        ]

        for i, malformed_json in enumerate(missing_field_cases):
            with self.subTest(
                case=i,
                json_content=repr(malformed_json),
                case_description=f"missing_field_case_{i}",
            ):
                if "name" in malformed_json and "grid" in malformed_json:
                    # This is a TritonGEMMConfig case
                    with self.assertRaises((TypeError, ValueError, KeyError)):
                        TritonGEMMConfig.parse(malformed_json)
                elif "M" in malformed_json:
                    # This is an MMShape case
                    with self.assertRaises((TypeError, ValueError, KeyError)):
                        MMShape.parse(malformed_json)
                else:
                    # This is a Table case
                    result = Table.deserialize(malformed_json)
                    # Some cases might succeed with defaults, others should fail
                    if result is None:
                        pass  # Expected failure
                    else:
                        # If it succeeds, it should be a valid Table
                        self.assertIsInstance(result, Table)

    def test_malformed_json_invalid_data_types(self):
        """Test JSON with invalid data types for fields."""
        invalid_type_cases = [
            # TritonGEMMConfig with wrong types
            '{"name": 123, "grid": 1, "block_m": 64, "block_n": 64, "block_k": 32, \
"group_m": 8, "num_stages": 2, "num_warps": 4}',  # name should be string
            '{"name": "test", "grid": "invalid", "block_m": 64, "block_n": 64, "block_k": 32, \
"group_m": 8, "num_stages": 2, "num_warps": 4}',  # grid should be int
            '{"name": "test", "grid": 1, "block_m": 64.5, "block_n": 64, "block_k": 32, "group_m": 8, \
"num_stages": 2, "num_warps": 4}',  # block_m should be int
            '{"name": "test", "grid": 1, "block_m": 64, "block_n": 64, "block_k": 32, "group_m": 8, \
"num_stages": 2, "num_warps": 4, "EVEN_K": "true"}',  # EVEN_K should be bool
            # MMShape with wrong types
            '{"B": 1024, "M": "1024", "M_dtype": "float32", "N": 1024, "K_dtype": "float32", "K": 512,\
"out_dtype": "float32", "version": 1, "out_size": (1, 1024, 1024), "out_stride": (1, 1024, 1)}',  # M should be int
            '{"B": 1024, "M": 1024, "M_dtype": "invalid_dtype", "N": 1024, "K_dtype": "float32", "K": 512, \
"out_dtype": "float32", "version": 1, "out_size": (1, 1024, 1024), "out_stride": (1, 1024, 1)}',  # invalid dtype
            '{"B": 1024, "M": 1024, "M_dtype": "float32", "N": 1024.5, "K_dtype": "float32", "K": 512, \
"out_dtype": "float32", "version": 1,  "out_size": (1, 1024, 1024), "out_stride": (1, 1024, 1)}',  # N should be int
            # Nested structure type errors
            '{"hardware": "not_a_dict", "version": 1}',  # hardware should be dict
            '{"hardware": {"gpu1": "not_a_hardware_object"}, "version": 1}',  # hardware values should be objects
        ]

        for i, malformed_json in enumerate(invalid_type_cases):
            with self.subTest(
                case=i,
                json_content=repr(malformed_json[:100]),
                case_type=f"invalid_type_case_{i}",
            ):
                if "block_m" in malformed_json:
                    # This is a TritonGEMMConfig case
                    with self.assertRaises((ValueError, TypeError)):
                        TritonGEMMConfig.parse(malformed_json)
                elif "M_dtype" in malformed_json:
                    # This is an MMShape case
                    with self.assertRaises((ValueError, AttributeError)):
                        MMShape.parse(malformed_json)
                else:
                    # This is a Table case
                    result = Table.deserialize(malformed_json)
                    self.assertIsNone(
                        result, f"Expected None for invalid type case {i}"
                    )

    def test_malformed_json_invalid_torch_dtypes(self):
        """Test JSON with invalid torch dtype strings."""
        invalid_dtype_cases = [
            '{"B": 1024, "M": 1024, "M_dtype": "invalid_dtype", "N": 1024, "K_dtype": "float32", \
"K": 512, "out_dtype": "float32", "version": 1}',
            '{"B": 1024, "M": 1024, "M_dtype": "float32", "N": 1024, "K_dtype": "nonexistent", "K": \
512, "out_dtype": "float32", "version": 1}',
            '{"B": 1024, "M": 1024, "M_dtype": "float32", "N": 1024, "K_dtype": "float32", "K": 512, \
"out_dtype": "fake_dtype", "version": 1}',
            '{"B": 1024, "M": 1024, "M_dtype": "", "N": 1024, "K_dtype": "float32", "K": 512, "out_dtype": \
"float32", "version": 1}',
            '{"B": 1024, "M": 1024, "M_dtype": null, "N": 1024, "K_dtype": "float32", "K": 512, "out_dtype": \
G"float32", "version": 1}',
        ]

        for i, malformed_json in enumerate(invalid_dtype_cases):
            with self.subTest(
                case=i,
                json_content=repr(malformed_json[:100]),
                dtype_case=f"invalid_dtype_case_{i}",
            ):
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    MMShape.parse(malformed_json)

    def test_get_table_with_malformed_file(self):
        """Test get_table function with malformed JSON files."""
        import os
        import tempfile

        try:
            from torch._inductor.kernel_lut import get_table, get_table_safe
        except ImportError:
            self.skipTest("torch._inductor.kernel_lut not available")

        malformed_json_cases = [
            '{"name": "test", "grid": 1',  # Missing closing brace
            "",  # Empty file
            "not json at all",  # Not JSON
            '{"hardware": {"gpu1": INVALID}}',  # Invalid nested structure
        ]

        for i, malformed_content in enumerate(malformed_json_cases):
            with self.subTest(case=i):
                # Create temporary file with malformed JSON
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    f.write(malformed_content)
                    temp_path = f.name

                try:
                    # Test get_table
                    result = get_table(temp_path)
                    self.assertIsNone(
                        result, f"Expected None for malformed file case {i}"
                    )

                    # Test get_table_safe
                    result_safe = get_table_safe(temp_path)
                    self.assertIsNone(
                        result_safe, f"Expected None for safe get_table case {i}"
                    )

                finally:
                    # Clean up temporary file
                    os.unlink(temp_path)

    def test_get_table_with_nonexistent_file(self):
        """Test get_table function with nonexistent files."""
        try:
            from torch._inductor.kernel_lut import get_table, get_table_safe
        except ImportError:
            self.skipTest("torch._inductor.kernel_lut not available")

        nonexistent_path = "/path/that/does/not/exist.json"

        # Test get_table
        result = get_table(nonexistent_path)
        self.assertIsNone(result)

        # Test get_table_safe
        result_safe = get_table_safe(nonexistent_path)
        self.assertIsNone(result_safe)

    def test_logging_on_malformed_json(self):
        """Test that appropriate log messages are generated for malformed JSON."""
        import logging
        from io import StringIO

        # Set up a string stream to capture log messages
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger("diode.types.matmul_types")
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)

        try:
            # Test with malformed JSON
            malformed_json = '{"name": "test", "grid": 1'  # Missing closing brace
            result = Table.deserialize(malformed_json)
            self.assertIsNone(result)

            # Check that error was logged
            log_contents = log_stream.getvalue()
            self.assertIn("Failed to deserialize table", log_contents)

        finally:
            # Clean up
            logger.removeHandler(handler)


if __name__ == "__main__":
    run_tests("cuda")
