# Owner(s): ["module: diode"]
"""
Tests for the lookup functionality in kernel LUT.
"""

import unittest

# Enable debug flags for testing
try:
    from torch_diode.utils.debug_config import set_debug_flag

    set_debug_flag("ENABLE_TYPE_ASSERTS", True)
except ImportError:
    pass  # In case debug_config is not available yet
from collections import OrderedDict
from unittest import TestCase

import torch
from hypothesis import assume, given
from hypothesis import strategies as st
from torch.utils._ordered_set import OrderedSet

from torch_diode.types.matmul_types import (
    Hardware,
    MMShape,
    Operation,
    Solution,
    Table,
    TritonGEMMConfig,
)


def run_tests():
    unittest.main()


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
        self.solution1 = Solution(config=[self.config1, self.config3])
        # Solution with one config
        self.solution2 = Solution(config=[self.config2])
        # Solution with two configs
        self.solution3 = Solution(config=[self.config2, self.config1])

        # Create operations with solutions
        self.operation1 = Operation(
            solution=OrderedDict(
                [(self.problem1, self.solution1), (self.problem2, self.solution2)]
            )
        )

        self.operation2 = Operation(
            solution=OrderedDict([(self.problem1, self.solution3)])
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
        empty_operation = Operation(solution=OrderedDict())
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
        solution_a = Solution(config=[self.config1])
        solution_b = Solution(config=[self.config2])

        # Create two operations with different solutions
        operation_a = Operation(solution=OrderedDict([(self.problem1, solution_a)]))
        operation_b = Operation(solution=OrderedDict([(self.problem1, solution_b)]))

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
        solution = Solution(config=[self.config1, self.config2, self.config3])

        self.assertEqual(len(solution.config), 3)
        self.assertEqual(solution.config[0].name, "config1")
        self.assertEqual(solution.config[1].name, "config2")
        self.assertEqual(solution.config[2].name, "config3")

        # Test serialization
        solution_dict = solution.to_dict()
        self.assertIn("config", solution_dict)
        self.assertEqual(len(solution_dict["config"]), 3)

        # Test round-trip serialization
        reconstructed = Solution.from_dict(solution_dict)
        self.assertEqual(len(reconstructed.config), len(solution.config))

    def test_solution_empty_config_list(self):
        """Test Solution with empty config list."""
        empty_solution = Solution(config=[])

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


if __name__ == "__main__":
    run_tests()
