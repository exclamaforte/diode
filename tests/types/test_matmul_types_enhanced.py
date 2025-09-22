"""
Enhanced tests for diode.types.matmul_types module to improve coverage.
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
from typing import OrderedDict
import pytest
import torch
from torch.utils._ordered_set import OrderedSet

from torch_diode.types.matmul_types import (
    TritonGEMMConfig,
    MMShape,
    Solution,
    Hardware,
    Operation, 
    Table,
)


class TestMatmulTypesEnhanced:
    """Enhanced test class for matmul types."""

    def test_triton_gemm_config_initialization(self):
        """Test TritonGEMMConfig initialization with all attributes."""
        config = TritonGEMMConfig(
            name="test_config",
            grid=4,
            block_m=64,
            block_n=128,
            block_k=32,
            group_m=8,
            num_stages=4,
            num_warps=8,
            EVEN_K=True,
            ALLOW_TF32=False,
            USE_FAST_ACCUM=True,
            ACC_TYPE="tl.float16"
        )

        assert config.name == "test_config"
        assert config.grid == 4
        assert config.block_m == 64
        assert config.block_n == 128
        assert config.block_k == 32
        assert config.group_m == 8
        assert config.num_stages == 4
        assert config.num_warps == 8
        assert config.EVEN_K is True
        assert config.ALLOW_TF32 is False
        assert config.USE_FAST_ACCUM is True
        assert config.ACC_TYPE == "tl.float16"

    def test_triton_gemm_config_hash(self):
        """Test TritonGEMMConfig hash function."""
        config1 = TritonGEMMConfig(
            name="config1", grid=2, block_m=32, block_n=64, block_k=16,
            group_m=4, num_stages=2, num_warps=4
        )
        
        config2 = TritonGEMMConfig(
            name="config1", grid=2, block_m=32, block_n=64, block_k=16,
            group_m=4, num_stages=2, num_warps=4
        )
        
        config3 = TritonGEMMConfig(
            name="config3", grid=2, block_m=32, block_n=64, block_k=16,
            group_m=4, num_stages=2, num_warps=4
        )
        
        # Same configurations should have same hash
        assert hash(config1) == hash(config2)
        
        # Different configurations should have different hash (usually)
        assert hash(config1) != hash(config3)

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
            "ACC_TYPE": "tl.float32"
        }
        
        json_str = json.dumps(config_dict)
        config = TritonGEMMConfig.parse(json_str)
        
        assert config.name == "parsed_config"
        assert config.grid == 3
        assert config.block_m == 128
        assert config.EVEN_K is True
        assert config.ACC_TYPE == "tl.float32"

    def test_triton_gemm_config_parse_missing_required_field(self):
        """Test TritonGEMMConfig.parse with missing required field."""
        config_dict = {
            "name": "incomplete_config",
            # Missing grid field
            "block_m": 64,
            "block_n": 128,
            "block_k": 32,
            "group_m": 8,
            "num_stages": 4,
            "num_warps": 8
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
            "num_warps": 8
        }
        
        json_str = json.dumps(config_dict)
        
        with pytest.raises(TypeError, match="grid must be an int"):
            TritonGEMMConfig.parse(json_str)

    def test_triton_gemm_config_parse_optional_boolean_wrong_type(self):
        """Test TritonGEMMConfig.parse with wrong type for optional boolean."""
        config_dict = {
            "name": "bad_bool_config",
            "grid": 2,
            "block_m": 64,
            "block_n": 128,
            "block_k": 32,
            "group_m": 8,
            "num_stages": 4,
            "num_warps": 8,
            "EVEN_K": "not_a_bool"  # Should be bool
        }
        
        json_str = json.dumps(config_dict)
        
        with pytest.raises(TypeError, match="EVEN_K must be a bool"):
            TritonGEMMConfig.parse(json_str)

    def test_mm_shape_initialization(self):
        """Test MMShape initialization."""
        problem = MMShape(
            B=8,
            M=64,
            M_dtype=torch.float32,
            N=128,
            K=256,
            K_dtype=torch.float16,
            out_dtype=torch.float32,
            out_size=(8, 64, 128),
            out_stride=(8192, 128, 1)
        )
        
        assert problem.B == 8
        assert problem.M == 64
        assert problem.M_dtype == torch.float32
        assert problem.N == 128
        assert problem.K == 256
        assert problem.K_dtype == torch.float16
        assert problem.out_dtype == torch.float32
        assert problem.out_size == (8, 64, 128)
        assert problem.out_stride == (8192, 128, 1)

    def test_mm_shape_hash(self):
        """Test MMShape hash function."""
        problem1 = MMShape(
            B=4, M=32, M_dtype=torch.float32, N=64, K=128,
            K_dtype=torch.float32, out_dtype=torch.float32,
            out_size=(4, 32, 64), out_stride=(2048, 64, 1)
        )
        
        problem2 = MMShape(
            B=4, M=32, M_dtype=torch.float32, N=64, K=128,
            K_dtype=torch.float32, out_dtype=torch.float32,
            out_size=(4, 32, 64), out_stride=(2048, 64, 1)
        )
        
        problem3 = MMShape(
            B=4, M=64, M_dtype=torch.float32, N=64, K=128,  # Different M
            K_dtype=torch.float32, out_dtype=torch.float32,
            out_size=(4, 32, 64), out_stride=(2048, 64, 1)
        )
        
        # Same problems should have same hash
        assert hash(problem1) == hash(problem2)
        
        # Different problems should have different hash
        assert hash(problem1) != hash(problem3)

    def test_mm_shape_str(self):
        """Test MMShape string representation."""
        problem = MMShape(
            B=2, M=32, M_dtype=torch.float32, N=64, K=128,
            K_dtype=torch.float16, out_dtype=torch.float32,
            out_size=(2, 32, 64), out_stride=(2048, 64, 1)
        )
        
        str_repr = str(problem)
        assert isinstance(str_repr, str)
        
        # Parse back to check it's valid JSON
        parsed = json.loads(str_repr)
        assert parsed["B"] == 2
        assert parsed["M"] == 32
        assert parsed["M_dtype"] == "float32"
        assert parsed["K_dtype"] == "float16"
        assert parsed["out_size"] == [2, 32, 64]
        assert parsed["out_stride"] == [2048, 64, 1]

    def test_mm_shape_parse_valid(self):
        """Test MMShape.parse with valid JSON."""
        problem_dict = {
            "B": 4,
            "M": 64,
            "M_dtype": "float32",
            "N": 128,
            "K": 256,
            "K_dtype": "float16",
            "out_dtype": "float32",
            "out_size": [4, 64, 128],
            "out_stride": [8192, 128, 1]
        }
        
        json_str = json.dumps(problem_dict)
        problem = MMShape.parse(json_str)
        
        assert problem.B == 4
        assert problem.M == 64
        assert problem.M_dtype == torch.float32
        assert problem.K_dtype == torch.float16
        assert problem.out_size == (4, 64, 128)
        assert problem.out_stride == (8192, 128, 1)

    def test_mm_shape_parse_invalid_dtype(self):
        """Test MMShape.parse with invalid dtype."""
        problem_dict = {
            "B": 4,
            "M": 64,
            "M_dtype": "invalid_dtype",  # Invalid dtype
            "N": 128,
            "K": 256,
            "K_dtype": "float16",
            "out_dtype": "float32",
            "out_size": [4, 64, 128],
            "out_stride": [8192, 128, 1]
        }
        
        json_str = json.dumps(problem_dict)
        
        with pytest.raises(ValueError, match="Invalid torch dtype: invalid_dtype"):
            MMShape.parse(json_str)

    def test_mm_shape_parse_missing_field(self):
        """Test MMShape.parse with missing required field."""
        problem_dict = {
            "B": 4,
            "M": 64,
            # Missing M_dtype
            "N": 128,
            "K": 256,
            "K_dtype": "float16",
            "out_dtype": "float32",
            "out_size": [4, 64, 128],
            "out_stride": [8192, 128, 1]
        }
        
        json_str = json.dumps(problem_dict)
        
        with pytest.raises(KeyError, match="Missing required field: M_dtype"):
            MMShape.parse(json_str)

    def test_solution_initialization(self):
        """Test Solution initialization."""
        configs = [
            TritonGEMMConfig(name="config1", grid=2, block_m=64, block_n=128, block_k=32,
                           group_m=8, num_stages=4, num_warps=8),
            TritonGEMMConfig(name="config2", grid=1, block_m=32, block_n=64, block_k=16,
                           group_m=4, num_stages=2, num_warps=4)
        ]
        
        solution = Solution(config=configs)
        
        assert len(solution.config) == 2
        assert solution.config[0].name == "config1"
        assert solution.config[1].name == "config2"

    def test_operation_initialization(self):
        """Test Operation initialization."""
        problem1 = MMShape(
            B=1, M=64, M_dtype=torch.float32, N=128, K=256,
            K_dtype=torch.float32, out_dtype=torch.float32,
            out_size=(1, 64, 128), out_stride=(8192, 128, 1)
        )
        
        config = TritonGEMMConfig(name="op_config", grid=2, block_m=64, block_n=128, block_k=32,
                                group_m=8, num_stages=4, num_warps=8)
        solution = Solution(config=[config])
        
        solutions = OrderedDict()
        solutions[problem1] = solution
        
        operation = Operation(solution=solutions)
        
        assert len(operation.solution) == 1
        assert problem1 in operation.solution
        assert operation.solution[problem1] == solution

    def test_hardware_initialization(self):
        """Test Hardware initialization."""
        # Create a simple operation
        problem = MMShape(
            B=1, M=32, M_dtype=torch.float32, N=64, K=128,
            K_dtype=torch.float32, out_dtype=torch.float32,
            out_size=(1, 32, 64), out_stride=(2048, 64, 1)
        )
        
        config = TritonGEMMConfig(name="hw_config", grid=1, block_m=32, block_n=64, block_k=16,
                                group_m=4, num_stages=2, num_warps=4)
        solution = Solution(config=[config])
        
        solutions = OrderedDict()
        solutions[problem] = solution
        
        operation = Operation(solution=solutions)
        
        operations = OrderedDict()
        operations["addmm"] = operation
        
        hardware = Hardware(operation=operations)
        
        assert len(hardware.operation) == 1
        assert "addmm" in hardware.operation
        assert hardware.operation["addmm"] == operation

    def test_table_initialization(self):
        """Test Table initialization."""
        # Create a complete table structure
        problem = MMShape(
            B=1, M=64, M_dtype=torch.float32, N=128, K=256,
            K_dtype=torch.float32, out_dtype=torch.float32,
            out_size=(1, 64, 128), out_stride=(8192, 128, 1)
        )
        
        config = TritonGEMMConfig(name="table_config", grid=2, block_m=64, block_n=128, block_k=32,
                                group_m=8, num_stages=4, num_warps=8)
        solution = Solution(config=[config])
        
        solutions = OrderedDict()
        solutions[problem] = solution
        
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
        # Create table with known data
        problem = MMShape(
            B=1, M=32, M_dtype=torch.float32, N=64, K=128,
            K_dtype=torch.float32, out_dtype=torch.float32,
            out_size=(1, 32, 64), out_stride=(2048, 64, 1)
        )
        
        config1 = TritonGEMMConfig(name="lookup_config1", grid=1, block_m=32, block_n=64, block_k=16,
                                 group_m=4, num_stages=2, num_warps=4)
        config2 = TritonGEMMConfig(name="lookup_config2", grid=2, block_m=64, block_n=128, block_k=32,
                                 group_m=8, num_stages=4, num_warps=8)
        
        solution = Solution(config=[config1, config2])
        
        solutions = OrderedDict()
        solutions[problem] = solution
        
        operation = Operation(solution=solutions)
        
        operations = OrderedDict()
        operations["mm"] = operation
        
        hardware = Hardware(operation=operations)
        
        hardware_dict = OrderedDict()
        hardware_dict["test_gpu"] = hardware
        
        table = Table(hardware=hardware_dict)
        
        # Test successful lookup
        result = table.lookup("test_gpu", "mm", problem)
        
        assert result is not None
        assert len(result) == 2
        assert result[0].name == "lookup_config1"
        assert result[1].name == "lookup_config2"

    def test_table_lookup_missing_hardware(self):
        """Test Table.lookup with missing hardware."""
        table = Table(hardware=OrderedDict())
        
        problem = MMShape(
            B=1, M=32, M_dtype=torch.float32, N=64, K=128,
            K_dtype=torch.float32, out_dtype=torch.float32,
            out_size=(1, 32, 64), out_stride=(2048, 64, 1)
        )
        
        result = table.lookup("nonexistent_gpu", "mm", problem)
        assert result is None

    def test_table_lookup_missing_operation(self):
        """Test Table.lookup with missing operation."""
        # Create table with hardware but no operations
        hardware = Hardware(operation=OrderedDict())
        hardware_dict = OrderedDict()
        hardware_dict["test_gpu"] = hardware
        table = Table(hardware=hardware_dict)
        
        problem = MMShape(
            B=1, M=32, M_dtype=torch.float32, N=64, K=128,
            K_dtype=torch.float32, out_dtype=torch.float32,
            out_size=(1, 32, 64), out_stride=(2048, 64, 1)
        )
        
        result = table.lookup("test_gpu", "nonexistent_op", problem)
        assert result is None

    def test_table_lookup_missing_problem(self):
        """Test Table.lookup with missing problem."""
        # Create table with operation but different problem
        existing_problem = MMShape(
            B=1, M=32, M_dtype=torch.float32, N=64, K=128,
            K_dtype=torch.float32, out_dtype=torch.float32,
            out_size=(1, 32, 64), out_stride=(2048, 64, 1)
        )
        
        different_problem = MMShape(
            B=1, M=64, M_dtype=torch.float32, N=128, K=256,  # Different dimensions
            K_dtype=torch.float32, out_dtype=torch.float32,
            out_size=(1, 64, 128), out_stride=(8192, 128, 1)
        )
        
        config = TritonGEMMConfig(name="existing_config", grid=1, block_m=32, block_n=64, block_k=16,
                                group_m=4, num_stages=2, num_warps=4)
        solution = Solution(config=[config])
        
        solutions = OrderedDict()
        solutions[existing_problem] = solution  # Only has existing_problem
        
        operation = Operation(solution=solutions)
        operations = OrderedDict()
        operations["mm"] = operation
        
        hardware = Hardware(operation=operations)
        hardware_dict = OrderedDict()
        hardware_dict["test_gpu"] = hardware
        
        table = Table(hardware=hardware_dict)
        
        # Lookup with different problem should return None
        result = table.lookup("test_gpu", "mm", different_problem)
        assert result is None

    def test_table_lookup_set_caching(self):
        """Test Table.lookup_set caching functionality."""
        problem = MMShape(
            B=1, M=32, M_dtype=torch.float32, N=64, K=128,
            K_dtype=torch.float32, out_dtype=torch.float32,
            out_size=(1, 32, 64), out_stride=(2048, 64, 1)
        )
        
        config1 = TritonGEMMConfig(name="cache_config1", grid=1, block_m=32, block_n=64, block_k=16,
                                 group_m=4, num_stages=2, num_warps=4)
        config2 = TritonGEMMConfig(name="cache_config2", grid=2, block_m=64, block_n=128, block_k=32,
                                 group_m=8, num_stages=4, num_warps=8)
        
        solution = Solution(config=[config1, config2])
        
        solutions = OrderedDict()
        solutions[problem] = solution
        
        operation = Operation(solution=solutions)
        operations = OrderedDict()
        operations["mm"] = operation
        
        hardware = Hardware(operation=operations)
        hardware_dict = OrderedDict()
        hardware_dict["cache_gpu"] = hardware
        
        table = Table(hardware=hardware_dict)
        
        # First call should create cache
        result1 = table.lookup_set("cache_gpu", "mm", problem)
        assert result1 is not None
        assert len(result1) == 2
        assert config1 in result1
        assert config2 in result1
        
        # Second call should use cache
        result2 = table.lookup_set("cache_gpu", "mm", problem)
        assert result2 is result1  # Should be the same object (cached)
        
        # Check cache was populated
        cache_key = ("cache_gpu", "mm", problem)
        assert cache_key in table._set_cache
        assert table._set_cache[cache_key] is result1

    def test_table_filter_success(self):
        """Test Table.filter with successful filtering."""
        problem = MMShape(
            B=1, M=32, M_dtype=torch.float32, N=64, K=128,
            K_dtype=torch.float32, out_dtype=torch.float32,
            out_size=(1, 32, 64), out_stride=(2048, 64, 1)
        )
        
        # Configs that are in the table
        config1 = TritonGEMMConfig(name="filter_config1", grid=1, block_m=32, block_n=64, block_k=16,
                                 group_m=4, num_stages=2, num_warps=4)
        config2 = TritonGEMMConfig(name="filter_config2", grid=2, block_m=64, block_n=128, block_k=32,
                                 group_m=8, num_stages=4, num_warps=8)
        
        # Config that is NOT in the table
        config3 = TritonGEMMConfig(name="filter_config3", grid=3, block_m=128, block_n=256, block_k=64,
                                 group_m=16, num_stages=5, num_warps=16)
        
        solution = Solution(config=[config1, config2])  # Only config1 and config2
        
        solutions = OrderedDict()
        solutions[problem] = solution
        
        operation = Operation(solution=solutions)
        operations = OrderedDict()
        operations["mm"] = operation
        
        hardware = Hardware(operation=operations)
        hardware_dict = OrderedDict()
        hardware_dict["filter_gpu"] = hardware
        
        table = Table(hardware=hardware_dict)
        
        # Filter a list that includes all configs
        to_filter = [config1, config2, config3]
        result = table.filter("filter_gpu", "mm", problem, to_filter)
        
        # Should only return configs that are in the table
        assert result is not None
        assert len(result) == 2
        assert config1 in result
        assert config2 in result
        assert config3 not in result

    def test_table_filter_no_matches(self):
        """Test Table.filter with no matching configs."""
        problem = MMShape(
            B=1, M=32, M_dtype=torch.float32, N=64, K=128,
            K_dtype=torch.float32, out_dtype=torch.float32,
            out_size=(1, 32, 64), out_stride=(2048, 64, 1)
        )
        
        # Config in the table
        table_config = TritonGEMMConfig(name="table_config", grid=1, block_m=32, block_n=64, block_k=16,
                                      group_m=4, num_stages=2, num_warps=4)
        
        # Config NOT in the table 
        external_config = TritonGEMMConfig(name="external_config", grid=2, block_m=64, block_n=128, block_k=32,
                                         group_m=8, num_stages=4, num_warps=8)
        
        solution = Solution(config=[table_config])
        
        solutions = OrderedDict()
        solutions[problem] = solution
        
        operation = Operation(solution=solutions)
        operations = OrderedDict()
        operations["mm"] = operation
        
        hardware = Hardware(operation=operations)
        hardware_dict = OrderedDict()
        hardware_dict["no_match_gpu"] = hardware
        
        table = Table(hardware=hardware_dict)
        
        # Filter with only external config
        to_filter = [external_config]
        result = table.filter("no_match_gpu", "mm", problem, to_filter)
        
        # Should return None since no configs match
        assert result is None

    def test_table_serialization_deserialization(self):
        """Test Table serialization and deserialization."""
        # Create a simple table
        problem = MMShape(
            B=1, M=32, M_dtype=torch.float32, N=64, K=128,
            K_dtype=torch.float32, out_dtype=torch.float32,
            out_size=(1, 32, 64), out_stride=(2048, 64, 1)
        )
        
        config = TritonGEMMConfig(name="serial_config", grid=1, block_m=32, block_n=64, block_k=16,
                                group_m=4, num_stages=2, num_warps=4)
        solution = Solution(config=[config])
        
        solutions = OrderedDict()
        solutions[problem] = solution
        
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
        
        # Deserialize
        deserialized = Table.deserialize(json_str)
        assert deserialized is not None
        assert len(deserialized.hardware) == 1
        assert "serial_gpu" in deserialized.hardware

    def test_table_deserialize_invalid_json(self):
        """Test Table.deserialize with invalid JSON."""
        invalid_json = "{ invalid json structure"
        
        with patch("torch_diode.types.matmul_types.logger") as mock_logger:
            result = Table.deserialize(invalid_json)
            
            assert result is None
            mock_logger.error.assert_called_once()

    def test_table_empty_initialization(self):
        """Test Table with empty hardware dictionary."""
        table = Table(hardware=OrderedDict())
        
        assert len(table.hardware) == 0
        assert len(table._set_cache) == 0

    def test_triton_gemm_config_default_values(self):
        """Test TritonGEMMConfig with default boolean values."""
        config = TritonGEMMConfig(
            name="default_config",
            grid=1,
            block_m=64,
            block_n=128,
            block_k=32,
            group_m=8,
            num_stages=4,
            num_warps=8
            # Not specifying optional boolean fields - should use defaults
        )
        
        assert config.EVEN_K is False  # Default
        assert config.ALLOW_TF32 is False  # Default
        assert config.USE_FAST_ACCUM is False  # Default
        assert config.ACC_TYPE == "tl.float32"  # Default

    def test_multiple_torch_dtypes(self):
        """Test MMShape with various torch dtypes."""
        dtypes = [torch.float16, torch.float32, torch.float64, torch.bfloat16]
        
        for dtype in dtypes:
            problem = MMShape(
                B=1, M=32, M_dtype=dtype, N=64, K=128,
                K_dtype=dtype, out_dtype=dtype,
                out_size=(1, 32, 64), out_stride=(2048, 64, 1)
            )
            
            assert problem.M_dtype == dtype
            assert problem.K_dtype == dtype
            assert problem.out_dtype == dtype
