# Owner(s): ["module: diode"]
"""
Tests for kernel lookup table serialization and LeafType functionality.
"""

import json
import sys
import unittest
from collections import OrderedDict
from typing import Any
from unittest import TestCase

import msgpack
import torch

from diode.types.matmul_types import (
    Hardware,
    MMShape,
    Operation,
    Solution,
    Table,
    TritonGEMMConfig,
)

from hypothesis import given, strategies as st
from tests.types.test_kernel_lut_strategies import (
    ComplexLeafTypeTestClass,
    hardware_strategy,
    leaf_type_test_class_strategy,
    LeafTypeTestClass,
    mm_problem_strategy,
    nested_leaf_type_test_class_strategy,
    NestedLeafTypeTestClass,
    operation_strategy,
    table_strategy,
    torch_dtype_strategy,
    triton_gemm_config_strategy,
)


def run_tests():
    unittest.main()


class TestKernelLUTPropertyBased(TestCase):
    """Property-based tests for kernel LUT serialization."""

    @given(triton_gemm_config_strategy())
    def test_triton_gemm_config_roundtrip(self, config):
        """Test that TritonGEMMConfig to_dict/from_dict are inverses."""
        # Convert to dict and back
        config_dict = config.to_dict()

        # Verify the dict contains expected keys
        self.assertIn("name", config_dict)
        self.assertIn("grid", config_dict)

        # Test round-trip: from_dict should reconstruct the original object
        reconstructed = TritonGEMMConfig.from_dict(config_dict)

        # Type checking: verify reconstructed object is correct type
        self.assertIsInstance(reconstructed, TritonGEMMConfig)

        # Type checking: verify all fields have correct types
        self.assertIsInstance(reconstructed.name, str)
        self.assertIsInstance(reconstructed.grid, int)
        self.assertIsInstance(reconstructed.block_m, int)
        self.assertIsInstance(reconstructed.block_n, int)
        self.assertIsInstance(reconstructed.block_k, int)
        self.assertIsInstance(reconstructed.group_m, int)
        self.assertIsInstance(reconstructed.num_stages, int)
        self.assertIsInstance(reconstructed.num_warps, int)
        self.assertIsInstance(reconstructed.EVEN_K, bool)
        self.assertIsInstance(reconstructed.ALLOW_TF32, bool)
        self.assertIsInstance(reconstructed.USE_FAST_ACCUM, bool)
        self.assertIsInstance(reconstructed.ACC_TYPE, str)
        self.assertIsInstance(reconstructed.version, int)

        # Compare all fields
        self.assertEqual(config.name, reconstructed.name)
        self.assertEqual(config.grid, reconstructed.grid)
        self.assertEqual(config.block_m, reconstructed.block_m)
        self.assertEqual(config.block_n, reconstructed.block_n)
        self.assertEqual(config.block_k, reconstructed.block_k)
        self.assertEqual(config.group_m, reconstructed.group_m)
        self.assertEqual(config.num_stages, reconstructed.num_stages)
        self.assertEqual(config.num_warps, reconstructed.num_warps)
        self.assertEqual(config.EVEN_K, reconstructed.EVEN_K)
        self.assertEqual(config.ALLOW_TF32, reconstructed.ALLOW_TF32)
        self.assertEqual(config.USE_FAST_ACCUM, reconstructed.USE_FAST_ACCUM)
        self.assertEqual(config.ACC_TYPE, reconstructed.ACC_TYPE)
        self.assertEqual(config.version, reconstructed.version)

    @given(mm_problem_strategy())
    def test_mm_problem_roundtrip(self, problem):
        """Test that MMShape to_dict/from_dict are inverses."""
        # Convert to dict and back
        problem_dict = problem.to_dict()

        # Verify the dict contains expected keys
        self.assertIn("B", problem_dict)
        self.assertIn("M", problem_dict)
        self.assertIn("N", problem_dict)
        self.assertIn("K", problem_dict)

        # Test round-trip
        reconstructed = MMShape.from_dict(problem_dict)

        # Type checking: verify reconstructed object is correct type
        self.assertIsInstance(reconstructed, MMShape)

        # Type checking: verify all fields have correct types
        self.assertIsInstance(reconstructed.B, int)
        self.assertIsInstance(reconstructed.M, int)
        self.assertIsInstance(reconstructed.M_dtype, torch.dtype)
        self.assertIsInstance(reconstructed.N, int)
        self.assertIsInstance(reconstructed.K, int)
        self.assertIsInstance(reconstructed.out_dtype, torch.dtype)
        self.assertIsInstance(reconstructed.version, int)

        # Compare all fields
        self.assertEqual(problem.B, reconstructed.B)
        self.assertEqual(problem.M, reconstructed.M)
        self.assertEqual(problem.M_dtype, reconstructed.M_dtype)
        self.assertEqual(problem.N, reconstructed.N)
        self.assertEqual(problem.K, reconstructed.K)
        self.assertEqual(problem.out_dtype, reconstructed.out_dtype)
        self.assertEqual(problem.version, reconstructed.version)

    @given(operation_strategy())
    def test_operation_roundtrip(self, operation):
        """Test that Operation to_dict/from_dict are inverses."""
        # Convert to dict and back
        operation_dict = operation.to_dict()

        # Verify the dict contains expected keys
        self.assertIn("name", operation_dict)
        self.assertIn("solution", operation_dict)

        # Test round-trip
        reconstructed = Operation.from_dict(operation_dict)

        # Type checking: verify reconstructed object is correct type
        self.assertIsInstance(reconstructed, Operation)

        # Type checking: verify all fields have correct types
        self.assertIsInstance(reconstructed.name, str)
        self.assertIsInstance(reconstructed.solution, OrderedDict)
        self.assertIsInstance(reconstructed.version, int)

        # Compare fields
        self.assertEqual(operation.name, reconstructed.name)
        self.assertEqual(operation.version, reconstructed.version)
        self.assertEqual(len(operation.solution), len(reconstructed.solution))

        # Type check solution contents
        for key, value in reconstructed.solution.items():
            self.assertIsInstance(key, MMShape)  # Keys should be MMShape objects
            self.assertIsInstance(value, Solution)  # Values should be Solution objects

    @given(hardware_strategy())
    def test_hardware_roundtrip(self, hardware):
        """Test that Hardware to_dict/from_dict are inverses."""
        # Convert to dict and back
        hardware_dict = hardware.to_dict()

        # Verify the dict contains expected keys
        self.assertIn("operation", hardware_dict)

        # Test round-trip
        reconstructed = Hardware.from_dict(hardware_dict)

        # Type checking: verify reconstructed object is correct type
        self.assertIsInstance(reconstructed, Hardware)

        # Type checking: verify all fields have correct types
        self.assertIsInstance(reconstructed.operation, OrderedDict)
        self.assertIsInstance(reconstructed.version, int)

        # Compare fields
        self.assertEqual(len(hardware.operation), len(reconstructed.operation))
        self.assertEqual(hardware.version, reconstructed.version)

        # Type check operation contents
        for key, value in reconstructed.operation.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, Operation)

    @given(table_strategy())
    def test_table_roundtrip(self, table):
        """Test that Table to_dict/from_dict are inverses."""
        # Convert to dict and back
        table_dict = table.to_dict()

        # Verify the dict contains expected keys
        self.assertIn("hardware", table_dict)

        # Test round-trip
        reconstructed = Table.from_dict(table_dict)

        # Type checking: verify reconstructed object is correct type
        self.assertIsInstance(reconstructed, Table)

        # Type checking: verify all fields have correct types
        self.assertIsInstance(reconstructed.hardware, OrderedDict)
        self.assertIsInstance(reconstructed.version, int)
        self.assertIsInstance(reconstructed._set_cache, OrderedDict)

        # Compare fields
        self.assertEqual(len(table.hardware), len(reconstructed.hardware))
        self.assertEqual(table.version, reconstructed.version)

        # Type check hardware contents
        for key, value in reconstructed.hardware.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, Hardware)

    @given(table_strategy())
    def test_table_json_serialization_roundtrip(self, table):
        """Test that Table can be serialized to JSON and back."""
        # Serialize to JSON string
        json_str = table.serialize()

        # Verify it's valid JSON
        parsed_json = json.loads(json_str)
        self.assertIsInstance(parsed_json, dict)

        # Deserialize back to Table
        reconstructed = Table.deserialize(json_str)

        # Basic structure should be preserved
        self.assertEqual(len(table.hardware), len(reconstructed.hardware))
        self.assertEqual(table.version, reconstructed.version)

    @given(st.text(min_size=1, max_size=100))
    def test_json_serializable_version_field(self, name):
        """Test that version field is properly handled in serialization."""
        # Create a simple config with custom version
        config = TritonGEMMConfig(
            name=name,
            grid=1,
            block_m=32,
            block_n=32,
            block_k=32,
            group_m=8,
            num_stages=2,
            num_warps=4,
            version=42,
        )

        # Serialize and check version is preserved
        config_dict = config.to_dict()

        # Round-trip should preserve version
        reconstructed = TritonGEMMConfig.from_dict(config_dict)
        self.assertEqual(reconstructed.version, 42)

    def test_empty_structures_comprehensive(self):
        """Test that empty structures work correctly across serialization and lookup methods."""
        # Test empty table
        empty_table = Table(hardware=OrderedDict())

        # Test serialization
        table_dict = empty_table.to_dict()
        self.assertEqual(len(table_dict["hardware"]), 0)

        # Test round-trip
        reconstructed_table = Table.from_dict(table_dict)
        self.assertEqual(len(reconstructed_table.hardware), 0)
        self.assertEqual(empty_table.version, reconstructed_table.version)

        # Test JSON serialization
        serialized = empty_table.serialize()
        deserialized_table = Table.deserialize(serialized)
        self.assertNotEqual(deserialized_table, None)
        self.assertEqual(len(deserialized_table.hardware), 0)

        # Test all lookup methods return None for empty table
        dummy_problem = MMShape(
            B=100,
            M=100,
            M_dtype=torch.float32,
            N=100,
            K=100,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(100, 100, 100),
            out_stride=(100 * 100, 100, 1),
        )
        dummy_config = TritonGEMMConfig(
            name="dummy",
            grid=1,
            block_m=32,
            block_n=32,
            block_k=32,
            group_m=8,
            num_stages=2,
            num_warps=4,
        )

        self.assertIsNone(deserialized_table.lookup("any_hw", "any_op", dummy_problem))
        self.assertIsNone(
            deserialized_table.lookup_set("any_hw", "any_op", dummy_problem)
        )
        self.assertIsNone(
            deserialized_table.filter("any_hw", "any_op", dummy_problem, [dummy_config])
        )

        # Test empty hardware
        empty_hardware = Hardware(operation=OrderedDict([]))

        # Test serialization
        hardware_dict = empty_hardware.to_dict()
        self.assertEqual(len(hardware_dict["operation"]), 0)

        # Test round-trip
        reconstructed_hardware = Hardware.from_dict(hardware_dict)
        self.assertEqual(len(reconstructed_hardware.operation), 0)
        self.assertEqual(empty_hardware.version, reconstructed_hardware.version)

        # Test table with empty hardware
        table_with_empty_hw = Table(
            hardware=OrderedDict([("empty_gpu", empty_hardware)])
        )
        self.assertIsNone(
            table_with_empty_hw.lookup("empty_gpu", "any_op", dummy_problem)
        )

        # Test empty operation
        empty_operation = Operation(name="empty_op", solution=OrderedDict())
        hardware_with_empty_op = Hardware(
            operation=OrderedDict([("empty_op", empty_operation)])
        )
        table_with_empty_op = Table(
            hardware=OrderedDict([("gpu_empty_op", hardware_with_empty_op)])
        )
        self.assertIsNone(
            table_with_empty_op.lookup("gpu_empty_op", "empty_op", dummy_problem)
        )

    @given(st.integers(min_value=1, max_value=1000))
    def test_version_field_consistency(self, version):
        """Test that version field is consistently handled across all classes."""
        config = TritonGEMMConfig(
            name="test",
            grid=1,
            block_m=32,
            block_n=32,
            block_k=32,
            group_m=8,
            num_stages=2,
            num_warps=4,
            version=version,
        )
        problem = MMShape(
            B=1,
            M=1,
            M_dtype=torch.float32,
            N=1,
            K=1,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1, 1, 1),
            out_stride=(1, 1, 1),
            version=version,
        )

        # Both should have the same version after serialization
        config_dict = config.to_dict()
        problem_dict = problem.to_dict()



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
        expected_keys = {"complex_dict", "dict_list"}
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
        reconstructed_dict = reconstructed.ordered_dict_field

        # Check each key-value pair has correct type
        self.assertIsNone(reconstructed_dict["none_key"])
        self.assertIsInstance(reconstructed_dict["bool_key"], bool)
        self.assertIsInstance(reconstructed_dict["int_key"], int)
        self.assertIsInstance(reconstructed_dict["float_key"], float)
        self.assertIsInstance(reconstructed_dict["str_key"], str)
        self.assertIsInstance(reconstructed_dict["dtype_key"], torch.dtype)
        self.assertIsInstance(reconstructed_dict["list_key"], list)
        self.assertIsInstance(reconstructed_dict["nested_dict_key"], OrderedDict)

        # Check values are preserved
        self.assertEqual(reconstructed_dict["bool_key"], True)
        self.assertEqual(reconstructed_dict["int_key"], 999)
        self.assertEqual(reconstructed_dict["float_key"], 2.718)
        self.assertEqual(reconstructed_dict["str_key"], "mixed_dict_string")
        self.assertEqual(reconstructed_dict["dtype_key"], torch.float64)

        # Check nested list types
        list_val = reconstructed_dict["list_key"]
        self.assertIsInstance(list_val[0], int)
        self.assertIsInstance(list_val[1], str)
        self.assertIsInstance(list_val[2], float)
        self.assertIsInstance(list_val[3], torch.dtype)

        # Check nested OrderedDict
        nested_dict = reconstructed_dict["nested_dict_key"]
        self.assertEqual(nested_dict["inner"], "value")


class TestMessagePackSerialization(TestCase):
    """Tests for MessagePack serialization functionality."""

    @given(triton_gemm_config_strategy())
    def test_triton_gemm_config_msgpack_roundtrip(self, config):
        """Test that TritonGEMMConfig MessagePack serialization/deserialization works correctly."""
        # Serialize to MessagePack
        msgpack_data = config.to_msgpack()
        self.assertIsInstance(msgpack_data, bytes)
        self.assertGreater(len(msgpack_data), 0)

        # Deserialize from MessagePack
        reconstructed = TritonGEMMConfig.from_msgpack(msgpack_data)
        self.assertIsInstance(reconstructed, TritonGEMMConfig)

        # Verify all fields are preserved
        self.assertEqual(config.name, reconstructed.name)
        self.assertEqual(config.grid, reconstructed.grid)
        self.assertEqual(config.block_m, reconstructed.block_m)
        self.assertEqual(config.block_n, reconstructed.block_n)
        self.assertEqual(config.block_k, reconstructed.block_k)
        self.assertEqual(config.group_m, reconstructed.group_m)
        self.assertEqual(config.num_stages, reconstructed.num_stages)
        self.assertEqual(config.num_warps, reconstructed.num_warps)
        self.assertEqual(config.EVEN_K, reconstructed.EVEN_K)
        self.assertEqual(config.ALLOW_TF32, reconstructed.ALLOW_TF32)
        self.assertEqual(config.USE_FAST_ACCUM, reconstructed.USE_FAST_ACCUM)
        self.assertEqual(config.ACC_TYPE, reconstructed.ACC_TYPE)
        self.assertEqual(config.version, reconstructed.version)

    @given(mm_problem_strategy())
    def test_mm_problem_msgpack_roundtrip(self, problem):
        """Test that MMShape MessagePack serialization/deserialization works correctly."""
        # Serialize to MessagePack
        msgpack_data = problem.to_msgpack()
        self.assertIsInstance(msgpack_data, bytes)
        self.assertGreater(len(msgpack_data), 0)

        # Deserialize from MessagePack
        reconstructed = MMShape.from_msgpack(msgpack_data)
        self.assertIsInstance(reconstructed, MMShape)

        # Verify all fields are preserved
        self.assertEqual(problem.B, reconstructed.B)
        self.assertEqual(problem.M, reconstructed.M)
        self.assertEqual(problem.M_dtype, reconstructed.M_dtype)
        self.assertEqual(problem.N, reconstructed.N)
        self.assertEqual(problem.K, reconstructed.K)
        self.assertEqual(problem.out_dtype, reconstructed.out_dtype)
        self.assertEqual(problem.version, reconstructed.version)

    @given(table_strategy())
    def test_table_msgpack_roundtrip(self, table):
        """Test that Table MessagePack serialization/deserialization works correctly."""
        # Serialize to MessagePack
        msgpack_data = table.to_msgpack()
        self.assertIsInstance(msgpack_data, bytes)
        self.assertGreater(len(msgpack_data), 0)

        # Deserialize from MessagePack
        reconstructed = Table.from_msgpack(msgpack_data)
        self.assertIsInstance(reconstructed, Table)

        # Verify structure is preserved
        self.assertEqual(len(table.hardware), len(reconstructed.hardware))
        self.assertEqual(table.version, reconstructed.version)

        # Verify hardware contents
        for key, value in reconstructed.hardware.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, Hardware)

    @given(leaf_type_test_class_strategy())
    def test_leaf_type_msgpack_roundtrip(self, test_obj):
        """Test MessagePack serialization/deserialization for LeafTypeTestClass."""
        # Serialize to MessagePack
        msgpack_data = test_obj.to_msgpack()
        self.assertIsInstance(msgpack_data, bytes)
        self.assertGreater(len(msgpack_data), 0)

        # Deserialize from MessagePack
        reconstructed = LeafTypeTestClass.from_msgpack(msgpack_data)
        self.assertIsInstance(reconstructed, LeafTypeTestClass)

        # Verify all LeafType fields are preserved
        self.assertEqual(reconstructed.none_field, test_obj.none_field)
        self.assertEqual(reconstructed.bool_field, test_obj.bool_field)
        self.assertEqual(reconstructed.int_field, test_obj.int_field)
        self.assertEqual(reconstructed.float_field, test_obj.float_field)
        self.assertEqual(reconstructed.str_field, test_obj.str_field)
        self.assertEqual(reconstructed.ordered_dict_field, test_obj.ordered_dict_field)
        self.assertEqual(reconstructed.torch_dtype_field, test_obj.torch_dtype_field)
        self.assertEqual(reconstructed.list_field, test_obj.list_field)

    @given(nested_leaf_type_test_class_strategy())
    def test_nested_leaf_type_msgpack_roundtrip(self, test_obj):
        """Test MessagePack serialization/deserialization for NestedLeafTypeTestClass."""
        # Serialize to MessagePack
        msgpack_data = test_obj.to_msgpack()
        self.assertIsInstance(msgpack_data, bytes)
        self.assertGreater(len(msgpack_data), 0)

        # Deserialize from MessagePack
        reconstructed = NestedLeafTypeTestClass.from_msgpack(msgpack_data)
        self.assertIsInstance(reconstructed, NestedLeafTypeTestClass)

        # Verify nested structures are preserved
        self.assertEqual(reconstructed.nested_dict, test_obj.nested_dict)
        self.assertEqual(reconstructed.mixed_list, test_obj.mixed_list)
        self.assertEqual(reconstructed.dtype1, test_obj.dtype1)
        self.assertEqual(reconstructed.dtype2, test_obj.dtype2)
        self.assertEqual(reconstructed.dtype3, test_obj.dtype3)

    def test_msgpack_alias_methods(self):
        """Test that serialize_msgpack and deserialize_msgpack alias methods work correctly."""
        config = TritonGEMMConfig(
            name="alias_test",
            grid=1,
            block_m=32,
            block_n=32,
            block_k=32,
            group_m=8,
            num_stages=2,
            num_warps=4,
        )

        # Test serialize_msgpack alias
        msgpack_data1 = config.serialize_msgpack()
        msgpack_data2 = config.to_msgpack()
        self.assertEqual(msgpack_data1, msgpack_data2)

        # Test deserialize_msgpack alias
        reconstructed1 = TritonGEMMConfig.deserialize_msgpack(msgpack_data1)
        reconstructed2 = TritonGEMMConfig.from_msgpack(msgpack_data1)

        self.assertEqual(reconstructed1.name, reconstructed2.name)
        self.assertEqual(reconstructed1.grid, reconstructed2.grid)
        self.assertEqual(reconstructed1.block_m, reconstructed2.block_m)

    def test_msgpack_vs_json_size_comparison(self):
        """Test that MessagePack serialization produces different (typically smaller) output than JSON."""
        # Create a complex table with multiple entries
        configs = [
            TritonGEMMConfig(
                name=f"config_{i}",
                grid=i + 1,
                block_m=32 * (i + 1),
                block_n=32 * (i + 1),
                block_k=32 * (i + 1),
                group_m=8,
                num_stages=2,
                num_warps=4,
            )
            for i in range(10)
        ]

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
            for i in range(10)
        ]

        solutions = [
            Solution(name=f"solution_{i}", config=[configs[i]]) for i in range(10)
        ]

        operation = Operation(
            name="size_test_mm", solution=OrderedDict(zip(problems, solutions))
        )
        hardware = Hardware(operation=OrderedDict([("size_test_mm", operation)]))
        table = Table(hardware=OrderedDict([("size_test_gpu", hardware)]))

        # Get JSON and MessagePack representations
        json_str = table.serialize()
        msgpack_data = table.to_msgpack()

        # Both should be non-empty
        self.assertGreater(len(json_str), 0)
        self.assertGreater(len(msgpack_data), 0)

        # MessagePack and JSON should produce different outputs
        self.assertNotEqual(json_str.encode("utf-8"), msgpack_data)

        # Both should deserialize to equivalent objects
        json_reconstructed = Table.deserialize(json_str)
        msgpack_reconstructed = Table.from_msgpack(msgpack_data)

        self.assertEqual(
            len(json_reconstructed.hardware), len(msgpack_reconstructed.hardware)
        )

    def test_msgpack_error_handling(self):
        """Test error handling for MessagePack serialization/deserialization."""
        # Test deserialization with invalid MessagePack data
        with self.assertRaises(ValueError) as context:
            TritonGEMMConfig.from_msgpack(b"invalid msgpack data")
        self.assertIn("Failed to deserialize", str(context.exception))

        # Test deserialization with valid MessagePack but wrong structure
        valid_msgpack_wrong_structure = msgpack.packb({"wrong": "structure"})
        with self.assertRaises(ValueError) as context:
            TritonGEMMConfig.from_msgpack(valid_msgpack_wrong_structure)
        self.assertIn("Malformed data", str(context.exception))

    def test_msgpack_empty_structures(self):
        """Test MessagePack serialization with empty structures."""
        # Test empty table
        empty_table = Table(hardware=OrderedDict())
        msgpack_data = empty_table.to_msgpack()
        reconstructed = Table.from_msgpack(msgpack_data)

        self.assertEqual(len(reconstructed.hardware), 0)
        self.assertEqual(empty_table.version, reconstructed.version)

        # Test empty LeafTypeTestClass
        empty_leaf = LeafTypeTestClass(
            none_field=None,
            bool_field=False,
            int_field=0,
            float_field=0.0,
            str_field="",
            ordered_dict_field=OrderedDict(),
            torch_dtype_field=torch.uint8,
            list_field=[],
        )

        msgpack_data = empty_leaf.to_msgpack()
        reconstructed = LeafTypeTestClass.from_msgpack(msgpack_data)

        self.assertEqual(len(reconstructed.ordered_dict_field), 0)
        self.assertEqual(len(reconstructed.list_field), 0)
        self.assertEqual(reconstructed.str_field, "")

    @given(st.lists(torch_dtype_strategy(), min_size=1, max_size=5))
    def test_msgpack_torch_dtype_preservation(self, dtype_list):
        """Test that torch.dtype values are properly preserved in MessagePack."""
        test_obj = LeafTypeTestClass(
            none_field=None,
            bool_field=True,
            int_field=42,
            float_field=3.14,
            str_field="dtype_msgpack_test",
            ordered_dict_field=OrderedDict([("dtypes", dtype_list)]),
            torch_dtype_field=dtype_list[0],
            list_field=dtype_list,
        )

        # Serialize to MessagePack and back
        msgpack_data = test_obj.to_msgpack()
        reconstructed = LeafTypeTestClass.from_msgpack(msgpack_data)

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

    def test_msgpack_binary_data_handling(self):
        """Test that MessagePack handles binary data correctly."""
        config = TritonGEMMConfig(
            name="binary_test",
            grid=1,
            block_m=32,
            block_n=32,
            block_k=32,
            group_m=8,
            num_stages=2,
            num_warps=4,
        )

        # Serialize to MessagePack
        msgpack_data = config.to_msgpack()

        # Verify it's actually binary data
        self.assertIsInstance(msgpack_data, bytes)

        # Verify it's not valid UTF-8 text (should be binary)
        try:
            msgpack_data.decode("utf-8")
            # If we get here, it might be coincidentally valid UTF-8, but that's unlikely
            # for MessagePack data, so we'll just verify it's different from JSON
            json_str = str(config)  # Use __str__ method instead of serialize
            self.assertNotEqual(msgpack_data, json_str.encode("utf-8"))
        except UnicodeDecodeError:
            # This is expected - MessagePack data should not be valid UTF-8
            pass

        # Verify we can still deserialize correctly
        reconstructed = TritonGEMMConfig.from_msgpack(msgpack_data)
        self.assertEqual(config.name, reconstructed.name)

    def test_msgpack_complex_nested_structures(self):
        """Test MessagePack with complex nested structures."""
        complex_obj = ComplexLeafTypeTestClass()

        # Serialize to MessagePack
        msgpack_data = complex_obj.to_msgpack()
        self.assertIsInstance(msgpack_data, bytes)
        self.assertGreater(len(msgpack_data), 0)

        # Deserialize from MessagePack
        reconstructed = ComplexLeafTypeTestClass.from_msgpack(msgpack_data)
        self.assertIsInstance(reconstructed, ComplexLeafTypeTestClass)

        # Verify complex nested structures are preserved
        self.assertEqual(reconstructed.complex_dict, complex_obj.complex_dict)
        self.assertEqual(reconstructed.dict_list, complex_obj.dict_list)

        # Deep verification of nested torch.dtype values
        level1 = reconstructed.complex_dict["level1"]
        for item in level1["level2_list"]:
            if isinstance(item, torch.dtype):
                self.assertIsInstance(item, torch.dtype)

        # Verify dict_list torch.dtype values
        for dict_item in reconstructed.dict_list:
            for key, value in dict_item.items():
                if key.endswith("_dtype"):
                    self.assertIsInstance(value, torch.dtype)

    @given(st.integers(min_value=1, max_value=1000))
    def test_msgpack_version_consistency(self, version):
        """Test that version field is consistently handled in MessagePack serialization."""
        config = TritonGEMMConfig(
            name="version_test",
            grid=1,
            block_m=32,
            block_n=32,
            block_k=32,
            group_m=8,
            num_stages=2,
            num_warps=4,
            version=version,
        )

        # Serialize to MessagePack and back
        msgpack_data = config.to_msgpack()
        reconstructed = TritonGEMMConfig.from_msgpack(msgpack_data)

        # Version should be preserved
        self.assertEqual(reconstructed.version, version)

    def test_msgpack_interoperability_with_json(self):
        """Test that objects serialized with MessagePack and JSON produce equivalent results."""
        config = TritonGEMMConfig(
            name="interop_test",
            grid=2,
            block_m=64,
            block_n=64,
            block_k=64,
            group_m=8,
            num_stages=3,
            num_warps=8,
        )

        # Create a table to test serialization methods
        problem = MMShape(
            B=1,
            M=1,
            M_dtype=torch.float32,
            N=1,
            K=1,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1, 1, 1),
            out_stride=(1, 1, 1),
        )
        solution = Solution(name="interop_solution", config=[config])
        operation = Operation(name="interop_op", solution=OrderedDict([(problem, solution)]))
        hardware = Hardware(operation=OrderedDict([("interop_op", operation)]))
        table = Table(hardware=OrderedDict([("interop_hw", hardware)]))

        # Serialize with both methods
        json_str = table.serialize()
        msgpack_data = table.to_msgpack()

        # Deserialize with both methods
        json_reconstructed = Table.deserialize(json_str)
        msgpack_reconstructed = Table.from_msgpack(msgpack_data)

        # Extract configs from both reconstructed tables
        json_config = list(json_reconstructed.hardware["interop_hw"].operation["interop_op"].solution.values())[0].config[0]
        msgpack_config = list(msgpack_reconstructed.hardware["interop_hw"].operation["interop_op"].solution.values())[0].config[0]

        # Both should produce equivalent objects
        self.assertEqual(json_config.name, msgpack_config.name)
        self.assertEqual(json_config.grid, msgpack_config.grid)
        self.assertEqual(json_config.block_m, msgpack_config.block_m)
        self.assertEqual(json_config.block_n, msgpack_config.block_n)
        self.assertEqual(json_config.block_k, msgpack_config.block_k)
        self.assertEqual(json_config.group_m, msgpack_config.group_m)
        self.assertEqual(json_config.num_stages, msgpack_config.num_stages)
        self.assertEqual(json_config.num_warps, msgpack_config.num_warps)
        self.assertEqual(json_config.EVEN_K, msgpack_config.EVEN_K)
        self.assertEqual(json_config.ALLOW_TF32, msgpack_config.ALLOW_TF32)
        self.assertEqual(json_config.USE_FAST_ACCUM, msgpack_config.USE_FAST_ACCUM)
        self.assertEqual(json_config.ACC_TYPE, msgpack_config.ACC_TYPE)
        self.assertEqual(json_config.version, msgpack_config.version)


if __name__ == "__main__":
    run_tests()
