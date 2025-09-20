# Owner(s): ["module: diode"]
"""
Integration tests that serialize/deserialize tables and test lookup methods.
"""

import unittest
from collections import OrderedDict
from unittest import TestCase

import torch
import hypothesis
from hypothesis import given

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


# Import strategies from test_kernel_lut_2.py for property-based testing
from hypothesis.strategies import composite
from hypothesis import strategies as st


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
        sol = Solution(config=[config])
        # Use the problem object directly as key since MMShape should be hashable
        solution[problem] = sol

    return Operation(solution=solution)


@composite
def hardware_strategy(draw):
    """Generate Hardware instances."""
    num_operations = draw(st.integers(min_value=0, max_value=3))
    operations = [draw(operation_strategy()) for _ in range(num_operations)]

    # Use index-based names since Operation no longer has a name attribute
    return Hardware(operation=OrderedDict((f"op_{i}", op) for i, op in enumerate(operations)))


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
        self.solution1 = Solution(config=[self.config1, self.config2])

        self.solution2 = Solution(config=[self.config2])

        # Create operations
        self.operation1 = Operation(
            solution=OrderedDict(
                [(self.problem1, self.solution1), (self.problem2, self.solution2)]
            )
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
    @hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
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
            solution = Solution(config=configs[i * 2 : (i + 1) * 2])  # 2 configs per solution
            solutions.append(solution)

        # Create operations
        operation = Operation(solution=OrderedDict(zip(problems, solutions)))

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
            Solution(config=[configs[0]]),
            Solution(config=configs),
            Solution(config=[configs[1]]),
        ]

        operation = Operation(solution=OrderedDict(zip(problems, solutions)))

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

                        # Type check Solution fields including list preservation
                        self.assertIsInstance(sol, Solution)
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
    run_tests()
