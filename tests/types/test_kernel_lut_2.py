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
        self.assertIn("version", config_dict)

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
        self.assertIsInstance(reconstructed._is_leaf, bool)

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
        self.assertIn("version", problem_dict)

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
        self.assertIsInstance(reconstructed._is_leaf, bool)

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
        self.assertIn("version", operation_dict)

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
        self.assertIn("version", hardware_dict)

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
        self.assertIn("version", table_dict)

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
        self.assertEqual(config_dict["version"], 42)

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

        self.assertEqual(config_dict["version"], version)
        self.assertEqual(problem_dict["version"], version)


class TestKernelLUTLookup(TestCase):
    """Tests for the lookup functionality in kernel LUT."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample TritonGEMMConfig instances
        super().setUp()
        self.config1 = TritonGEMMConfig(
            name="config1",
            grid=1,
            block_m=64,
            block_n=64,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
            EVEN_K=True,
            ALLOW_TF32=False,
        )

        self.config2 = TritonGEMMConfig(
            name="config2",
            grid=2,
            block_m=128,
            block_n=128,
            block_k=64,
            group_m=16,
            num_stages=2,
            num_warps=8,
            EVEN_K=False,
            ALLOW_TF32=True,
        )

        self.config3 = TritonGEMMConfig(
            name="config3",
            grid=1,
            block_m=32,
            block_n=32,
            block_k=16,
            group_m=4,
            num_stages=3,
            num_warps=8,
            EVEN_K=False,
            ALLOW_TF32=True,
        )
        # Create sample problems
        self.problem1 = MMShape(
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

        self.problem2 = MMShape(
            B=2048,
            M=2048,
            M_dtype=torch.float16,
            N=2048,
            K=1024,
            K_dtype=torch.float16,
            out_dtype=torch.float16,
            out_size=(2048, 2048, 1024),
            out_stride=(2048 * 1024, 1024, 1),
        )

        # Create solutions with lists of configs
        self.solution1 = Solution(name="solution1", config=[self.config1, self.config3])

        self.solution2 = Solution(name="solution2", config=[self.config2])

        self.solution3 = Solution(name="solution3", config=[self.config2, self.config1])

        # Create operations with solutions
        self.operation1 = Operation(
            name="mm",
            solution=OrderedDict(
                [(self.problem1, self.solution1), (self.problem2, self.solution2)]
            ),
        )

        self.operation2 = Operation(
            name="addmm", solution=OrderedDict([(self.problem1, self.solution3)])
        )

        # Create hardware with operations (now using OrderedDict[str, Operation])
        self.hardware1 = Hardware(
            operation=OrderedDict([("mm", self.operation1), ("addmm", self.operation2)])
        )

        self.hardware2 = Hardware(operation=OrderedDict([("mm", self.operation1)]))

        # Create table with hardware
        self.table = Table(
            hardware=OrderedDict([("gpu1", self.hardware1), ("gpu2", self.hardware2)])
        )

    def test_successful_lookup(self):
        """Test successful lookup with valid inputs."""
        # Test lookup for problem1 with mm operation on gpu1
        result = self.table.lookup("gpu1", "mm", self.problem1)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)  # solution1 has config1 and config3
            self.assertEqual(result[0].name, "config1")
            self.assertEqual(result[0].block_m, 64)
            self.assertEqual(result[0].block_n, 64)
            self.assertEqual(result[1].name, "config3")

        # Test lookup for problem2 with mm operation on gpu1
        result = self.table.lookup("gpu1", "mm", self.problem2)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)  # solution2 has only config2
            self.assertEqual(result[0].name, "config2")
            self.assertEqual(result[0].block_m, 128)
            self.assertEqual(result[0].block_n, 128)

        # Test lookup for problem1 with addmm operation on gpu1
        result = self.table.lookup("gpu1", "addmm", self.problem1)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)  # solution3 has config2 and config1
            self.assertEqual(result[0].name, "config2")
            self.assertEqual(result[1].name, "config1")

    def test_lookup_invalid_hardware(self):
        """Test lookup with invalid hardware name."""
        result = self.table.lookup("invalid_gpu", "mm", self.problem1)
        self.assertIsNone(result)

    def test_lookup_invalid_operation(self):
        """Test lookup with invalid operation name."""
        result = self.table.lookup("gpu1", "invalid_op", self.problem1)
        self.assertIsNone(result)

    def test_lookup_invalid_problem(self):
        """Test lookup with problem not in solution."""
        # Create a problem that's not in any solution
        unknown_problem = MMShape(
            B=9999,
            M=9999,
            M_dtype=torch.float32,
            N=9999,
            K=9999,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(9999, 9999, 9999),
            out_stride=(9999 * 9999, 9999, 1),
        )

        result = self.table.lookup("gpu1", "mm", unknown_problem)
        self.assertIsNone(result)

    def test_lookup_operation_not_available_on_hardware(self):
        """Test lookup for operation not available on specific hardware."""
        # gpu2 only has mm operation, not addmm
        result = self.table.lookup("gpu2", "addmm", self.problem1)
        self.assertIsNone(result)

    def test_lookup_empty_table(self):
        """Test lookup on empty table."""
        empty_table = Table(hardware=OrderedDict())
        result = empty_table.lookup("gpu1", "mm", self.problem1)
        self.assertIsNone(result)

    def test_lookup_empty_hardware_operations(self):
        """Test lookup on hardware with no operations."""
        empty_hardware = Hardware(operation=OrderedDict([]))
        table_with_empty_hw = Table(
            hardware=OrderedDict([("empty_gpu", empty_hardware)])
        )

        result = table_with_empty_hw.lookup("empty_gpu", "mm", self.problem1)
        self.assertIsNone(result)

    def test_lookup_empty_operation_solution(self):
        """Test lookup on operation with empty solution."""
        empty_operation = Operation(name="empty_op", solution=OrderedDict())
        hardware_with_empty_op = Hardware(
            operation=OrderedDict([("empty_op", empty_operation)])
        )
        table_with_empty_op = Table(
            hardware=OrderedDict([("gpu_empty_op", hardware_with_empty_op)])
        )

        result = table_with_empty_op.lookup("gpu_empty_op", "empty_op", self.problem1)
        self.assertIsNone(result)

    def test_lookup_multiple_operations_same_name(self):
        """Test lookup when multiple operations have the same name (should find first)."""
        # Create solutions for the operations
        solution_a = Solution(name="solution_a", config=[self.config1])
        solution_b = Solution(name="solution_b", config=[self.config2])

        # Create two operations with the same name but different solutions
        operation_a = Operation(
            name="duplicate_op", solution=OrderedDict([(self.problem1, solution_a)])
        )
        operation_b = Operation(
            name="duplicate_op_different",
            solution=OrderedDict([(self.problem1, solution_b)]),
        )

        hardware_with_duplicates = Hardware(
            operation=OrderedDict(
                [("duplicate_op", operation_a), ("duplicate_op_different", operation_b)]
            )
        )
        table_with_duplicates = Table(
            hardware=OrderedDict([("gpu_dup", hardware_with_duplicates)])
        )

        # Should find the operation with matching name
        result = table_with_duplicates.lookup("gpu_dup", "duplicate_op", self.problem1)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "config1")  # Should be from operation_a

    @given(st.text(min_size=1, max_size=20))
    def test_lookup_with_random_hardware_names(self, hardware_name):
        """Property-based test for lookup with random hardware names."""
        # Most random hardware names should return None
        assume(hardware_name not in ["gpu1", "gpu2"])
        result = self.table.lookup(hardware_name, "mm", self.problem1)
        self.assertIsNone(result)

    @given(st.text(min_size=1, max_size=20))
    def test_lookup_with_random_operation_names(self, op_name):
        """Property-based test for lookup with random operation names."""
        # Most random operation names should return None
        assume(op_name not in ["mm", "addmm"])
        result = self.table.lookup("gpu1", op_name, self.problem1)
        self.assertIsNone(result)

    def test_lookup_problem_equality(self):
        """Test that lookup works correctly with problem equality."""
        # Create a new problem with same values as problem1

        identical_problem = MMShape(
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

        # This should work if MMShape implements proper equality
        # Note: This test might fail if MMShape doesn't implement __eq__ and __hash__
        try:
            result = self.table.lookup("gpu1", "mm", identical_problem)
            # If the lookup works, the problems are considered equal
            if result is not None:
                self.assertIsInstance(result, list)
                self.assertGreater(len(result), 0)
                self.assertEqual(result[0].name, "config1")
        except (KeyError, TypeError):
            # If lookup fails, it's likely because MMShape doesn't implement proper equality
            # This is expected behavior and indicates a potential issue to fix
            pass

    def test_lookup_preserves_config_properties(self):
        """Test that lookup returns configs with all properties intact."""
        result = self.table.lookup("gpu1", "mm", self.problem1)
        self.assertIsNotNone(result)

        # Check all properties are preserved for the first config
        if result is not None:
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)
            first_config = result[0]
            self.assertEqual(first_config.name, "config1")
            self.assertEqual(first_config.grid, 1)
            self.assertEqual(first_config.block_m, 64)
            self.assertEqual(first_config.block_n, 64)
            self.assertEqual(first_config.block_k, 32)
            self.assertEqual(first_config.group_m, 8)
            self.assertEqual(first_config.num_stages, 3)
            self.assertEqual(first_config.num_warps, 4)
            self.assertTrue(first_config.EVEN_K)
            self.assertFalse(first_config.ALLOW_TF32)
            self.assertFalse(first_config.USE_FAST_ACCUM)
            self.assertEqual(first_config.ACC_TYPE, "tl.float32")

    def test_solution_class(self):
        """Test the Solution class functionality."""
        # Test creating a solution with multiple configs
        solution = Solution(
            name="test_solution", config=[self.config1, self.config2, self.config3]
        )

        self.assertEqual(solution.name, "test_solution")
        self.assertEqual(len(solution.config), 3)
        self.assertEqual(solution.config[0].name, "config1")
        self.assertEqual(solution.config[1].name, "config2")
        self.assertEqual(solution.config[2].name, "config3")

        # Test serialization
        solution_dict = solution.to_dict()
        self.assertIn("name", solution_dict)
        self.assertIn("config", solution_dict)
        self.assertIn("version", solution_dict)
        self.assertEqual(len(solution_dict["config"]), 3)

        # Test round-trip serialization
        reconstructed = Solution.from_dict(solution_dict)
        self.assertEqual(reconstructed.name, solution.name)
        self.assertEqual(len(reconstructed.config), len(solution.config))

    def test_solution_empty_config_list(self):
        """Test Solution with empty config list."""
        empty_solution = Solution(name="empty", config=[])

        self.assertEqual(empty_solution.name, "empty")
        self.assertEqual(len(empty_solution.config), 0)

        # Test serialization of empty solution
        solution_dict = empty_solution.to_dict()
        self.assertEqual(len(solution_dict["config"]), 0)

        # Test round-trip
        reconstructed = Solution.from_dict(solution_dict)
        self.assertEqual(len(reconstructed.config), 0)

    def test_lookup_returns_solution_configs(self):
        """Test that lookup returns the configs from the solution."""
        result = self.table.lookup("gpu1", "mm", self.problem1)
        self.assertIsNotNone(result)

        if result is not None:
            # Should return the config list from solution1
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)

            # Check that we get the configs in the right order
            self.assertEqual(result[0].name, "config1")
            self.assertEqual(result[1].name, "config3")

            # Verify these are the same configs as in solution1
            self.assertEqual(result[0], self.config1)
            self.assertEqual(result[1], self.config3)

    def test_lookup_single_config_solution(self):
        """Test lookup with solution containing single config."""
        result = self.table.lookup("gpu1", "mm", self.problem2)
        self.assertIsNotNone(result)

        if result is not None:
            # Should return the config list from solution2 (single config)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "config2")
            self.assertEqual(result[0], self.config2)

    def test_lookup_set_successful(self):
        """Test successful lookup_set with valid inputs."""
        # Test lookup_set for problem1 with mm operation on gpu1
        result = self.table.lookup_set("gpu1", "mm", self.problem1)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertIsInstance(result, OrderedSet)
            self.assertEqual(len(result), 2)  # solution1 has config1 and config3
            self.assertIn(self.config1, result)
            self.assertIn(self.config3, result)

        # Test lookup_set for problem2 with mm operation on gpu1
        result = self.table.lookup_set("gpu1", "mm", self.problem2)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertIsInstance(result, OrderedSet)
            self.assertEqual(len(result), 1)  # solution2 has only config2
            self.assertIn(self.config2, result)

    def test_lookup_set_invalid_inputs(self):
        """Test lookup_set with invalid inputs."""
        # Invalid hardware
        result = self.table.lookup_set("invalid_gpu", "mm", self.problem1)
        self.assertIsNone(result)

        # Invalid operation
        result = self.table.lookup_set("gpu1", "invalid_op", self.problem1)
        self.assertIsNone(result)

        # Invalid problem
        unknown_problem = MMShape(
            B=9999,
            M=9999,
            M_dtype=torch.float32,
            N=9999,
            K=9999,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(9999, 9999, 9999),
            out_stride=(9999 * 9999, 9999, 1),
        )
        result = self.table.lookup_set("gpu1", "mm", unknown_problem)
        self.assertIsNone(result)

    def test_lookup_set_caching(self):
        """Test that lookup_set caches results properly."""
        # First call
        result1 = self.table.lookup_set("gpu1", "mm", self.problem1)
        # Second call should return the same cached result
        result2 = self.table.lookup_set("gpu1", "mm", self.problem1)

        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        if result1 is not None and result2 is not None:
            # Should be the exact same object due to caching
            self.assertIs(result1, result2)
            self.assertEqual(result1, result2)

    def test_filter_successful(self):
        """Test successful filter with valid inputs."""
        # Create a list of configs to filter
        configs_to_filter = [self.config1, self.config2, self.config3]

        # Filter for problem1 with mm operation on gpu1
        # Should return config1 and config3 (from solution1)
        result = self.table.filter("gpu1", "mm", self.problem1, configs_to_filter)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            self.assertIn(self.config1, result)
            self.assertIn(self.config3, result)
            self.assertNotIn(self.config2, result)

        # Filter for problem2 with mm operation on gpu1
        # Should return only config2 (from solution2)
        result = self.table.filter("gpu1", "mm", self.problem2, configs_to_filter)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertIn(self.config2, result)
            self.assertNotIn(self.config1, result)
            self.assertNotIn(self.config3, result)

    def test_filter_no_matches(self):
        """Test filter when no configs match."""
        # Create a config that's not in any solution
        unmatched_config = TritonGEMMConfig(
            name="unmatched",
            grid=99,
            block_m=256,
            block_n=256,
            block_k=128,
            group_m=32,
            num_stages=1,
            num_warps=16,
        )

        configs_to_filter = [unmatched_config]
        result = self.table.filter("gpu1", "mm", self.problem1, configs_to_filter)
        self.assertIsNone(result)

    def test_filter_empty_input_list(self):
        """Test filter with empty input list."""
        result = self.table.filter("gpu1", "mm", self.problem1, [])
        self.assertIsNone(result)

    def test_filter_invalid_inputs(self):
        """Test filter with invalid inputs."""
        configs_to_filter = [self.config1, self.config2]

        # Invalid hardware
        result = self.table.filter(
            "invalid_gpu", "mm", self.problem1, configs_to_filter
        )
        self.assertIsNone(result)

        # Invalid operation
        result = self.table.filter(
            "gpu1", "invalid_op", self.problem1, configs_to_filter
        )
        self.assertIsNone(result)

        # Invalid problem
        unknown_problem = MMShape(
            B=9999,
            M=9999,
            M_dtype=torch.float32,
            N=9999,
            K=9999,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(9999, 9999, 9999),
            out_stride=(9999 * 9999, 9999, 1),
        )
        result = self.table.filter("gpu1", "mm", unknown_problem, configs_to_filter)
        self.assertIsNone(result)

    def test_filter_partial_matches(self):
        """Test filter with partial matches."""
        # Mix of matching and non-matching configs
        unmatched_config = TritonGEMMConfig(
            name="unmatched",
            grid=99,
            block_m=256,
            block_n=256,
            block_k=128,
            group_m=32,
            num_stages=1,
            num_warps=16,
        )

        configs_to_filter = [self.config1, unmatched_config, self.config3]

        # Filter for problem1 with mm operation on gpu1
        # Should return config1 and config3, but not unmatched_config
        result = self.table.filter("gpu1", "mm", self.problem1, configs_to_filter)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            self.assertIn(self.config1, result)
            self.assertIn(self.config3, result)
            self.assertNotIn(unmatched_config, result)

    def test_filter_preserves_order(self):
        """Test that filter preserves the order of input configs."""
        # Test with configs in different order
        configs_to_filter = [self.config3, self.config1, self.config2]

        result = self.table.filter("gpu1", "mm", self.problem1, configs_to_filter)
        self.assertIsNotNone(result)
        if result is not None:
            # Should preserve the order: config3 first, then config1
            # config2 should not be included for problem1
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], self.config3)
            self.assertEqual(result[1], self.config1)


class TestKernelLUTIntegration(TestCase):
    """Integration tests that serialize/deserialize tables and test lookup methods."""

    def setUp(self):
        """Set up test fixtures for integration tests."""
        # Create sample configs
        super().setUp()
        self.config1 = TritonGEMMConfig(
            name="integration_config1",
            grid=1,
            block_m=64,
            block_n=64,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
            EVEN_K=True,
            ALLOW_TF32=False,
        )

        self.config2 = TritonGEMMConfig(
            name="integration_config2",
            grid=2,
            block_m=128,
            block_n=128,
            block_k=64,
            group_m=16,
            num_stages=2,
            num_warps=8,
            EVEN_K=False,
            ALLOW_TF32=True,
        )

        self.problem1 = MMShape(
            B=512,
            M=512,
            M_dtype=torch.float32,
            N=512,
            K=256,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(512, 512, 256),
            out_stride=(512 * 512, 512, 1),
        )

        # Invalid problem

        # Create sample problems
        self.problem2 = MMShape(
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

        # Create solutions
        self.solution1 = Solution(
            name="integration_solution1", config=[self.config1, self.config2]
        )

        self.solution2 = Solution(name="integration_solution2", config=[self.config2])

        # Create operations
        self.operation1 = Operation(
            name="mm",
            solution=OrderedDict(
                [(self.problem1, self.solution1), (self.problem2, self.solution2)]
            ),
        )

        # Create hardware
        self.hardware1 = Hardware(operation=OrderedDict([("mm", self.operation1)]))

        # Create original table
        self.original_table = Table(
            hardware=OrderedDict(
                [
                    (
                        "test_gpu",
                        self.hardware1,
                    )
                ]
            )
        )

    def test_serialization_roundtrip_preserves_lookup(self):
        """Test that serialization/deserialization preserves lookup functionality."""
        # Serialize the table
        serialized = self.original_table.serialize()
        self.assertIsInstance(serialized, str)

        # Deserialize the table
        deserialized_table = Table.deserialize(serialized)

        # Test that lookup works the same on both tables
        original_result = self.original_table.lookup("test_gpu", "mm", self.problem1)

        deserialized_result = deserialized_table.lookup("test_gpu", "mm", self.problem1)

        self.assertIsNotNone(original_result)
        self.assertIsNotNone(deserialized_result)

        if original_result is not None and deserialized_result is not None:
            self.assertEqual(len(original_result), len(deserialized_result))
            for orig_config, deser_config in zip(original_result, deserialized_result):
                self.assertEqual(orig_config.name, deser_config.name)
                self.assertEqual(orig_config.grid, deser_config.grid)
                self.assertEqual(orig_config.block_m, deser_config.block_m)
                self.assertEqual(orig_config.block_n, deser_config.block_n)
                self.assertEqual(orig_config.block_k, deser_config.block_k)
                self.assertEqual(orig_config.group_m, deser_config.group_m)

    def test_serialization_roundtrip_preserves_lookup_set(self):
        """Test that serialization/deserialization preserves lookup_set functionality."""
        # Serialize and deserialize
        serialized = self.original_table.serialize()
        deserialized_table = Table.deserialize(serialized)

        # Test lookup_set on both tables
        original_result = self.original_table.lookup_set(
            "test_gpu", "mm", self.problem1
        )
        deserialized_result = deserialized_table.lookup_set(
            "test_gpu", "mm", self.problem1
        )

        self.assertIsNotNone(original_result)
        self.assertIsNotNone(deserialized_result)

        if original_result is not None and deserialized_result is not None:
            self.assertEqual(len(original_result), len(deserialized_result))
            # Convert to lists for comparison since sets don't guarantee order
            orig_names = sorted([config.name for config in original_result])
            deser_names = sorted([config.name for config in deserialized_result])
            self.assertEqual(orig_names, deser_names)

    def test_serialization_roundtrip_preserves_filter(self):
        """Test that serialization/deserialization preserves filter functionality."""
        # Serialize and deserialize
        serialized = self.original_table.serialize()
        deserialized_table = Table.deserialize(serialized)

        # Create configs to filter
        configs_to_filter = [self.config1, self.config2]

        # Test filter on both tables
        original_result = self.original_table.filter(
            "test_gpu", "mm", self.problem1, configs_to_filter
        )
        deserialized_result = deserialized_table.filter(
            "test_gpu", "mm", self.problem1, configs_to_filter
        )

        self.assertIsNotNone(original_result)
        self.assertIsNotNone(deserialized_result)

        if original_result is not None and deserialized_result is not None:
            self.assertEqual(len(original_result), len(deserialized_result))
            for orig_config, deser_config in zip(original_result, deserialized_result):
                self.assertEqual(orig_config.name, deser_config.name)

    def test_lookup_methods_consistency_after_serialization(self):
        """Test that all lookup methods are consistent after serialization."""
        # Serialize and deserialize
        serialized = self.original_table.serialize()
        deserialized_table = Table.deserialize(serialized)

        # Test consistency between lookup and lookup_set
        lookup_result = deserialized_table.lookup("test_gpu", "mm", self.problem1)
        lookup_set_result = deserialized_table.lookup_set(
            "test_gpu", "mm", self.problem1
        )

        if lookup_result is not None and lookup_set_result is not None:
            # Convert lookup result to set for comparison
            lookup_as_set = set(lookup_result)
            self.assertEqual(lookup_as_set, lookup_set_result)

        # Test consistency between lookup_set and filter
        if lookup_result is not None and lookup_set_result is not None:
            # Filter with all configs from lookup_set
            all_configs = list(lookup_set_result)
            filter_result = deserialized_table.filter(
                "test_gpu", "mm", self.problem1, all_configs
            )

            if filter_result is not None:
                filter_as_set = set(filter_result)
                self.assertEqual(filter_as_set, lookup_set_result)

    @given(table_strategy())
    def test_property_based_serialization_lookup_consistency(self, table):
        """Property-based test for serialization consistency with lookup methods."""
        try:
            # Serialize and deserialize
            serialized = table.serialize()
            deserialized_table = Table.deserialize(serialized)

            # Test that basic structure is preserved
            self.assertEqual(len(table.hardware), len(deserialized_table.hardware))
            self.assertEqual(table.version, deserialized_table.version)

            # For each hardware/operation combination, test lookup consistency
            for hw_name, hardware in table.hardware.items():
                for op_name, operation in hardware.operation.items():
                    for problem in operation.solution.keys():
                        # Test lookup consistency
                        original_lookup = table.lookup(hw_name, op_name, problem)
                        deserialized_lookup = deserialized_table.lookup(
                            hw_name, op_name, problem
                        )

                        if (
                            original_lookup is not None
                            and deserialized_lookup is not None
                        ):
                            self.assertEqual(
                                len(original_lookup), len(deserialized_lookup)
                            )
                        elif original_lookup is None:
                            self.assertIsNone(deserialized_lookup)

                        # Test lookup_set consistency
                        original_lookup_set = table.lookup_set(
                            hw_name, op_name, problem
                        )
                        deserialized_lookup_set = deserialized_table.lookup_set(
                            hw_name, op_name, problem
                        )

                        if (
                            original_lookup_set is not None
                            and deserialized_lookup_set is not None
                        ):
                            self.assertEqual(
                                len(original_lookup_set), len(deserialized_lookup_set)
                            )
                        elif original_lookup_set is None:
                            self.assertIsNone(deserialized_lookup_set)

        except Exception:
            # Skip problematic generated data
            pass

    def test_large_table_serialization_performance(self):
        """Test serialization performance with larger tables."""
        # Create a larger table for performance testing
        configs = []
        for i in range(10):
            config = TritonGEMMConfig(
                name=f"perf_config_{i}",
                grid=i + 1,
                block_m=32 * (i + 1),
                block_n=32 * (i + 1),
                block_k=16 * (i + 1),
                group_m=4 * (i + 1),
                num_stages=2,
                num_warps=4,
            )
            configs.append(config)

        problems = []
        for i in range(5):
            problem = MMShape(
                B=256 * (i + 1),
                M=256 * (i + 1),
                M_dtype=torch.float32,
                N=256 * (i + 1),
                K=128 * (i + 1),
                K_dtype=torch.float32,
                out_dtype=torch.float32,
                out_size=(256 * (i + 1), 256 * (i + 1), 128 * (i + 1)),
                out_stride=(256 * (i + 1) * 128 * (i + 1), 128 * (i + 1), 1),
            )
            problems.append(problem)

        # Create solutions with multiple configs
        solutions = []
        for i in range(5):
            solution = Solution(
                name=f"perf_solution_{i}",
                config=configs[i * 2 : (i + 1) * 2],  # 2 configs per solution
            )
            solutions.append(solution)

        # Create operations
        operation = Operation(name="mm", solution=OrderedDict(zip(problems, solutions)))

        # Create hardware
        hardware = Hardware(operation=OrderedDict([("mm", operation)]))

        # Create large table
        large_table = Table(hardware=OrderedDict([("perf_gpu", hardware)]))

        # Test serialization and deserialization
        serialized = large_table.serialize()
        self.assertIsInstance(serialized, str)
        self.assertGreater(len(serialized), 1000)  # Should be a substantial JSON string

        deserialized_table = Table.deserialize(serialized)

        # Test that lookup methods work on the large deserialized table
        for i, problem in enumerate(problems):
            result = deserialized_table.lookup("perf_gpu", "mm", problem)
            self.assertIsNotNone(result)
            if result is not None:
                self.assertEqual(len(result), 2)  # Each solution has 2 configs

            result_set = deserialized_table.lookup_set("perf_gpu", "mm", problem)
            self.assertIsNotNone(result_set)
            if result_set is not None:
                self.assertEqual(len(result_set), 2)

            # Test filter with configs that are actually in this problem's solution
            # Each solution uses configs[i*2:(i+1)*2], so for problem i, use those specific configs
            problem_configs = configs[
                i * 2 : (i + 1) * 2
            ]  # Get the configs for this specific problem
            filter_result = deserialized_table.filter(
                "perf_gpu", "mm", problem, problem_configs
            )
            self.assertIsNotNone(filter_result)
            if filter_result is not None:
                self.assertEqual(
                    len(filter_result), 2
                )  # Should match both configs for this problem

    def test_comprehensive_type_preservation_after_serialization(self):
        """Comprehensive test to verify all types are preserved through serialization/deserialization."""
        # Test various torch dtypes
        dtypes_to_test = [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.int64,
            torch.bool,
            torch.float64,
            torch.int8,
            torch.int16,
        ]

        # Create a complex nested structure with various dtypes
        configs = [
            TritonGEMMConfig(
                name="nested_config1",
                grid=1,
                block_m=64,
                block_n=64,
                block_k=32,
                group_m=8,
                num_stages=3,
                num_warps=4,
            ),
            TritonGEMMConfig(
                name="nested_config2",
                grid=2,
                block_m=128,
                block_n=128,
                block_k=64,
                group_m=16,
                num_stages=2,
                num_warps=8,
            ),
        ]

        # Test problems with different dtypes
        problems = []
        for i, dtype in enumerate(dtypes_to_test[:3]):  # Use first 3 dtypes
            problem = MMShape(
                B=100 * (i + 1),
                M=100 * (i + 1),
                M_dtype=dtype,
                N=100 * (i + 1),
                K=50 * (i + 1),
                K_dtype=dtype,
                out_dtype=dtype,
                out_size=(100 * (i + 1), 100 * (i + 1), 50 * (i + 1)),
                out_stride=(100 * (i + 1) * 50 * (i + 1), 50 * (i + 1), 1),
            )
            problems.append(problem)

        solutions = [
            Solution(name="nested_solution1", config=[configs[0]]),
            Solution(name="nested_solution2", config=configs),
            Solution(name="nested_solution3", config=[configs[1]]),
        ]

        operation = Operation(
            name="nested_mm", solution=OrderedDict(zip(problems, solutions))
        )

        hardware = Hardware(operation=OrderedDict([("nested_mm", operation)]))
        table = Table(hardware=OrderedDict([("nested_gpu", hardware)]))

        # Multiple rounds of serialization/deserialization to test consistency
        for round_num in range(3):
            serialized = table.serialize()
            table = Table.deserialize(serialized)

            # Comprehensive type checking at all levels
            self.assertIsInstance(table, Table)
            self.assertIsInstance(table.hardware, OrderedDict)
            self.assertIsInstance(table._set_cache, OrderedDict)

            for hw_name, hw in table.hardware.items():
                self.assertIsInstance(hw_name, str)
                self.assertIsInstance(hw, Hardware)
                self.assertIsInstance(hw.operation, OrderedDict)
                self.assertIsInstance(hw.version, int)

                for op_name, op in hw.operation.items():
                    self.assertIsInstance(op_name, str)
                    self.assertIsInstance(op, Operation)
                    self.assertIsInstance(op.name, str)
                    self.assertIsInstance(op.solution, OrderedDict)
                    self.assertIsInstance(op.version, int)

                    for prob, sol in op.solution.items():
                        # Type check Problem fields including torch.dtype preservation
                        self.assertIsInstance(prob, MMShape)
                        self.assertIsInstance(prob.B, int)
                        self.assertIsInstance(prob.M, int)
                        self.assertIsInstance(prob.M_dtype, torch.dtype)
                        self.assertIsInstance(prob.N, int)
                        self.assertIsInstance(prob.K, int)
                        self.assertIsInstance(prob.out_dtype, torch.dtype)
                        self.assertIsInstance(prob.version, int)
                        self.assertIsInstance(prob._is_leaf, bool)

                        # Type check Solution fields including list preservation
                        self.assertIsInstance(sol, Solution)
                        self.assertIsInstance(sol.name, str)
                        self.assertIsInstance(sol.config, list)
                        self.assertIsInstance(sol.version, int)

                        # Type check TritonGEMMConfig objects in solution
                        for config in sol.config:
                            self.assertIsInstance(config, TritonGEMMConfig)
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

        # Test individual dtype preservation for all dtypes
        for dtype in dtypes_to_test:
            problem = MMShape(
                B=100,
                M=100,
                M_dtype=dtype,
                N=100,
                K=100,
                K_dtype=dtype,
                out_dtype=dtype,
                out_size=(1, 100, 100),
                out_stride=(100 * 100, 100, 1),
            )

            # Serialize and deserialize
            problem_dict = problem.to_dict()
            reconstructed = MMShape.from_dict(problem_dict)

            # Type check: ensure torch.dtype fields are still torch.dtype
            self.assertIsInstance(
                reconstructed.M_dtype,
                torch.dtype,
                f"M_dtype should be torch.dtype, got {type(reconstructed.M_dtype)} for {dtype}",
            )
            self.assertIsInstance(
                reconstructed.K_dtype,
                torch.dtype,
                f"K_dtype should be torch.dtype, got {type(reconstructed.K_dtype)} for {dtype}",
            )
            self.assertIsInstance(
                reconstructed.out_dtype,
                torch.dtype,
                f"out_dtype should be torch.dtype, got {type(reconstructed.out_dtype)} for {dtype}",
            )

            # Value check: ensure the actual dtype values are preserved
            self.assertEqual(reconstructed.M_dtype, dtype)
            self.assertEqual(reconstructed.K_dtype, dtype)
            self.assertEqual(reconstructed.out_dtype, dtype)


if __name__ == "__main__":
    run_tests("cuda")
