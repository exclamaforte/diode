"""
Simple tests for diode.types.matmul_types module to improve coverage.
"""

import json
# Enable debug flags for testing
try:
    from torch_diode.utils.debug_config import set_debug_flag
    set_debug_flag("ENABLE_TYPE_ASSERTS", True)
except ImportError:
    pass  # In case debug_config is not available yet
import tempfile
from unittest.mock import Mock, patch

import pytest
import torch

from torch_diode.types.matmul_types import (
    Hardware,
    MMShape,
    Operation,
    Solution,
    Table,
    TritonGEMMConfig,
)


class TestMatmulTypesSimple:
    """Simple test class for matmul types."""

    def test_triton_gemm_config_basic(self):
        """Test TritonGEMMConfig basic functionality."""
        config = TritonGEMMConfig(
            name="test_config",
            grid=4,
            block_m=64,
            block_n=128,
            block_k=32,
            group_m=8,
            num_stages=4,
            num_warps=8,
        )

        assert config.name == "test_config"
        assert config.grid == 4
        assert config.block_m == 64
        assert config.block_n == 128
        assert config.block_k == 32
        assert config.group_m == 8
        assert config.num_stages == 4
        assert config.num_warps == 8

    def test_triton_gemm_config_with_optional_params(self):
        """Test TritonGEMMConfig with optional parameters."""
        config = TritonGEMMConfig(
            name="test_config_opt",
            grid=2,
            block_m=32,
            block_n=64,
            block_k=16,
            group_m=4,
            num_stages=2,
            num_warps=4,
            EVEN_K=True,
            ALLOW_TF32=False,
            USE_FAST_ACCUM=True,
            ACC_TYPE="tl.float16",
        )

        assert config.EVEN_K is True
        assert config.ALLOW_TF32 is False
        assert config.USE_FAST_ACCUM is True
        assert config.ACC_TYPE == "tl.float16"

    def test_mmshape_basic(self):
        """Test MMShape basic functionality."""
        shape = MMShape(
            B=8,
            M=64,
            N=128,
            K=256,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(8, 64, 128),
            out_stride=(8192, 128, 1),
        )

        assert shape.B == 8
        assert shape.M == 64
        assert shape.N == 128
        assert shape.K == 256

    def test_mmshape_equality(self):
        """Test MMShape equality."""
        shape1 = MMShape(
            B=4,
            M=32,
            N=64,
            K=128,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(4, 32, 64),
            out_stride=(2048, 64, 1),
        )
        shape2 = MMShape(
            B=4,
            M=32,
            N=64,
            K=128,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(4, 32, 64),
            out_stride=(2048, 64, 1),
        )
        shape3 = MMShape(
            B=4,
            M=64,
            N=64,
            K=128,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(4, 64, 64),
            out_stride=(4096, 64, 1),
        )

        assert shape1 == shape2
        assert shape1 != shape3

    def test_mmshape_hash(self):
        """Test MMShape hash function."""
        shape1 = MMShape(
            B=4,
            M=32,
            N=64,
            K=128,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(4, 32, 64),
            out_stride=(2048, 64, 1),
        )
        shape2 = MMShape(
            B=4,
            M=32,
            N=64,
            K=128,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(4, 32, 64),
            out_stride=(2048, 64, 1),
        )
        shape3 = MMShape(
            B=4,
            M=64,
            N=64,
            K=128,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(4, 64, 64),
            out_stride=(4096, 64, 1),
        )

        assert hash(shape1) == hash(shape2)
        assert hash(shape1) != hash(shape3)

    def test_solution_basic(self):
        """Test Solution basic functionality."""
        configs = [
            TritonGEMMConfig(
                name="config1",
                grid=2,
                block_m=64,
                block_n=128,
                block_k=32,
                group_m=8,
                num_stages=4,
                num_warps=8,
            ),
            TritonGEMMConfig(
                name="config2",
                grid=1,
                block_m=32,
                block_n=64,
                block_k=16,
                group_m=4,
                num_stages=2,
                num_warps=4,
            ),
        ]

        solution = Solution(config=configs)

        assert len(solution.config) == 2
        assert solution.config[0].name == "config1"
        assert solution.config[1].name == "config2"

    def test_operation_basic(self):
        """Test Operation basic functionality."""
        from collections import OrderedDict

        shape = MMShape(
            B=1,
            M=64,
            N=128,
            K=256,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1, 64, 128),
            out_stride=(8192, 128, 1),
        )
        config = TritonGEMMConfig(
            name="op_config",
            grid=2,
            block_m=64,
            block_n=128,
            block_k=32,
            group_m=8,
            num_stages=4,
            num_warps=8,
        )
        solution = Solution(config=[config])

        solutions = OrderedDict()
        solutions[shape] = solution

        operation = Operation(solution=solutions)

        assert len(operation.solution) == 1
        assert shape in operation.solution
        assert operation.solution[shape] == solution

    def test_hardware_basic(self):
        """Test Hardware basic functionality."""
        from collections import OrderedDict

        # Create a simple operation
        shape = MMShape(
            B=1,
            M=32,
            N=64,
            K=128,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1, 32, 64),
            out_stride=(2048, 64, 1),
        )
        config = TritonGEMMConfig(
            name="hw_config",
            grid=1,
            block_m=32,
            block_n=64,
            block_k=16,
            group_m=4,
            num_stages=2,
            num_warps=4,
        )
        solution = Solution(config=[config])

        solutions = OrderedDict()
        solutions[shape] = solution

        operation = Operation(solution=solutions)

        operations = OrderedDict()
        operations["addmm"] = operation

        hardware = Hardware(operation=operations)

        assert len(hardware.operation) == 1
        assert "addmm" in hardware.operation
        assert hardware.operation["addmm"] == operation

    def test_table_basic(self):
        """Test Table basic functionality."""
        from collections import OrderedDict

        # Create a complete table structure
        shape = MMShape(
            B=1,
            M=64,
            N=128,
            K=256,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1, 64, 128),
            out_stride=(8192, 128, 1),
        )
        config = TritonGEMMConfig(
            name="table_config",
            grid=2,
            block_m=64,
            block_n=128,
            block_k=32,
            group_m=8,
            num_stages=4,
            num_warps=8,
        )
        solution = Solution(config=[config])

        solutions = OrderedDict()
        solutions[shape] = solution

        operation = Operation(solution=solutions)

        operations = OrderedDict()
        operations["bmm"] = operation

        hardware = Hardware(operation=operations)

        hardware_dict = OrderedDict()
        hardware_dict["gpu1"] = hardware

        table = Table(hardware=hardware_dict)

        assert len(table.hardware) == 1
        assert "gpu1" in table.hardware
        assert table.hardware["gpu1"] == hardware

    def test_table_lookup_success(self):
        """Test Table.lookup with successful lookup."""
        from collections import OrderedDict

        # Create table with known data
        shape = MMShape(
            B=1,
            M=32,
            N=64,
            K=128,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1, 32, 64),
            out_stride=(2048, 64, 1),
        )

        config1 = TritonGEMMConfig(
            name="lookup_config1",
            grid=1,
            block_m=32,
            block_n=64,
            block_k=16,
            group_m=4,
            num_stages=2,
            num_warps=4,
        )
        config2 = TritonGEMMConfig(
            name="lookup_config2",
            grid=2,
            block_m=64,
            block_n=128,
            block_k=32,
            group_m=8,
            num_stages=4,
            num_warps=8,
        )

        solution = Solution(config=[config1, config2])

        solutions = OrderedDict()
        solutions[shape] = solution

        operation = Operation(solution=solutions)

        operations = OrderedDict()
        operations["mm"] = operation

        hardware = Hardware(operation=operations)

        hardware_dict = OrderedDict()
        hardware_dict["test_gpu"] = hardware

        table = Table(hardware=hardware_dict)

        # Test successful lookup
        result = table.lookup("test_gpu", "mm", shape)

        assert result is not None
        assert len(result) == 2
        assert result[0].name == "lookup_config1"
        assert result[1].name == "lookup_config2"

    def test_table_lookup_missing_hardware(self):
        """Test Table.lookup with missing hardware."""
        from collections import OrderedDict

        table = Table(hardware=OrderedDict())

        shape = MMShape(
            B=1,
            M=32,
            N=64,
            K=128,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1, 32, 64),
            out_stride=(2048, 64, 1),
        )

        result = table.lookup("nonexistent_gpu", "mm", shape)
        assert result is None

    def test_table_lookup_missing_operation(self):
        """Test Table.lookup with missing operation."""
        from collections import OrderedDict

        # Create table with hardware but no operations
        hardware = Hardware(operation=OrderedDict())
        hardware_dict = OrderedDict()
        hardware_dict["test_gpu"] = hardware
        table = Table(hardware=hardware_dict)

        shape = MMShape(
            B=1,
            M=32,
            N=64,
            K=128,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1, 32, 64),
            out_stride=(2048, 64, 1),
        )

        result = table.lookup("test_gpu", "nonexistent_op", shape)
        assert result is None

    def test_table_serialization(self):
        """Test Table serialization."""
        from collections import OrderedDict

        # Create a simple table
        shape = MMShape(
            B=1,
            M=32,
            N=64,
            K=128,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1, 32, 64),
            out_stride=(2048, 64, 1),
        )
        config = TritonGEMMConfig(
            name="serial_config",
            grid=1,
            block_m=32,
            block_n=64,
            block_k=16,
            group_m=4,
            num_stages=2,
            num_warps=4,
        )
        solution = Solution(config=[config])

        solutions = OrderedDict()
        solutions[shape] = solution

        operation = Operation(solution=solutions)
        operations = OrderedDict()
        operations["mm"] = operation

        hardware = Hardware(operation=operations)
        hardware_dict = OrderedDict()
        hardware_dict["serial_gpu"] = hardware

        table = Table(hardware=hardware_dict)

        # Serialize
        json_str = table.serialize()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_table_deserialization_invalid_json(self):
        """Test Table.deserialize with invalid JSON."""
        invalid_json = "{ invalid json structure"

        result = Table.deserialize(invalid_json)
        assert result is None

    def test_triton_gemm_config_parse_valid(self):
        """Test TritonGEMMConfig.parse with valid JSON."""
        config_dict = {
            "name": "parsed_config",
            "grid": 3,
            "block_m": 128,
            "block_n": 256,
            "block_k": 64,
            "group_m": 16,
            "num_stages": 5,
            "num_warps": 16,
            "EVEN_K": True,
            "ALLOW_TF32": True,
            "USE_FAST_ACCUM": False,
            "ACC_TYPE": "tl.float32",
        }

        json_str = json.dumps(config_dict)
        config = TritonGEMMConfig.parse(json_str)

        assert config.name == "parsed_config"
        assert config.grid == 3
        assert config.block_m == 128
        assert config.EVEN_K is True
        assert config.ACC_TYPE == "tl.float32"

    def test_triton_gemm_config_parse_missing_field(self):
        """Test TritonGEMMConfig.parse with missing required field."""
        config_dict = {
            "name": "incomplete_config",
            # Missing grid field
            "block_m": 64,
            "block_n": 128,
            "block_k": 32,
            "group_m": 8,
            "num_stages": 4,
            "num_warps": 8,
        }

        json_str = json.dumps(config_dict)

        with pytest.raises(KeyError, match="Missing required field: grid"):
            TritonGEMMConfig.parse(json_str)

    def test_triton_gemm_config_parse_wrong_type(self):
        """Test TritonGEMMConfig.parse with wrong field type."""
        config_dict = {
            "name": "bad_config",
            "grid": "not_an_int",  # Should be int
            "block_m": 64,
            "block_n": 128,
            "block_k": 32,
            "group_m": 8,
            "num_stages": 4,
            "num_warps": 8,
        }

        json_str = json.dumps(config_dict)

        with pytest.raises(TypeError, match="grid must be an int"):
            TritonGEMMConfig.parse(json_str)

    def test_mmshape_str(self):
        """Test MMShape string representation."""
        shape = MMShape(
            B=2,
            M=32,
            N=64,
            K=128,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(2, 32, 64),
            out_stride=(2048, 64, 1),
        )

        str_repr = str(shape)
        assert isinstance(str_repr, str)

        # Parse back to check it's valid JSON
        parsed = json.loads(str_repr)
        assert parsed["B"] == 2
        assert parsed["M"] == 32
        assert parsed["N"] == 64
        assert parsed["K"] == 128

    def test_solution_empty_config(self):
        """Test Solution with empty config list."""
        solution = Solution(config=[])
        assert len(solution.config) == 0

    def test_table_empty_hardware(self):
        """Test Table with empty hardware."""
        from collections import OrderedDict

        table = Table(hardware=OrderedDict())
        assert len(table.hardware) == 0

    def test_table_lookup_set_caching(self):
        """Test Table.lookup_set caching functionality."""
        from collections import OrderedDict

        shape = MMShape(
            B=1,
            M=32,
            N=64,
            K=128,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1, 32, 64),
            out_stride=(2048, 64, 1),
        )
        config1 = TritonGEMMConfig(
            name="cache_config1",
            grid=1,
            block_m=32,
            block_n=64,
            block_k=16,
            group_m=4,
            num_stages=2,
            num_warps=4,
        )
        config2 = TritonGEMMConfig(
            name="cache_config2",
            grid=2,
            block_m=64,
            block_n=128,
            block_k=32,
            group_m=8,
            num_stages=4,
            num_warps=8,
        )

        solution = Solution(config=[config1, config2])

        solutions = OrderedDict()
        solutions[shape] = solution

        operation = Operation(solution=solutions)
        operations = OrderedDict()
        operations["mm"] = operation

        hardware = Hardware(operation=operations)
        hardware_dict = OrderedDict()
        hardware_dict["cache_gpu"] = hardware

        table = Table(hardware=hardware_dict)

        # First call should create cache
        result1 = table.lookup_set("cache_gpu", "mm", shape)
        assert result1 is not None
        assert len(result1) == 2
        assert config1 in result1
        assert config2 in result1

        # Second call should use cache
        result2 = table.lookup_set("cache_gpu", "mm", shape)
        assert result2 is result1  # Should be the same object (cached)

    def test_table_filter_success(self):
        """Test Table.filter with successful filtering."""
        from collections import OrderedDict

        shape = MMShape(
            B=1,
            M=32,
            N=64,
            K=128,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1, 32, 64),
            out_stride=(2048, 64, 1),
        )

        # Configs that are in the table
        config1 = TritonGEMMConfig(
            name="filter_config1",
            grid=1,
            block_m=32,
            block_n=64,
            block_k=16,
            group_m=4,
            num_stages=2,
            num_warps=4,
        )
        config2 = TritonGEMMConfig(
            name="filter_config2",
            grid=2,
            block_m=64,
            block_n=128,
            block_k=32,
            group_m=8,
            num_stages=4,
            num_warps=8,
        )

        # Config that is NOT in the table
        config3 = TritonGEMMConfig(
            name="filter_config3",
            grid=3,
            block_m=128,
            block_n=256,
            block_k=64,
            group_m=16,
            num_stages=5,
            num_warps=16,
        )

        solution = Solution(
            config=[config1, config2]
        )  # Only config1 and config2

        solutions = OrderedDict()
        solutions[shape] = solution

        operation = Operation(solution=solutions)
        operations = OrderedDict()
        operations["mm"] = operation

        hardware = Hardware(operation=operations)
        hardware_dict = OrderedDict()
        hardware_dict["filter_gpu"] = hardware

        table = Table(hardware=hardware_dict)

        # Filter a list that includes all configs
        to_filter = [config1, config2, config3]
        result = table.filter("filter_gpu", "mm", shape, to_filter)

        # Should only return configs that are in the table
        assert result is not None
        assert len(result) == 2
        assert config1 in result
        assert config2 in result
        assert config3 not in result
