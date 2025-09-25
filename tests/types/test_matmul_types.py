"""
Tests for diode.types.matmul_types module.
"""

import json

# Enable debug flags for testing
try:
    from torch_diode.utils.debug_config import set_debug_flag

    set_debug_flag("ENABLE_TYPE_ASSERTS", True)
except ImportError:
    pass  # In case debug_config is not available yet

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


class TestTritonGEMMConfig:
    """Test TritonGEMMConfig dataclass."""

    def test_init_basic(self):
        """Test basic initialization."""
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
        assert config.name == "test_config"
        assert config.grid == 1
        assert config.block_m == 64
        assert config.block_n == 64
        assert config.block_k == 32
        assert config.group_m == 8
        assert config.num_stages == 3
        assert config.num_warps == 4
        assert config.EVEN_K is False
        assert config.ALLOW_TF32 is False
        assert config.USE_FAST_ACCUM is False
        assert config.ACC_TYPE == "tl.float32"

    def test_init_with_optional_flags(self):
        """Test initialization with optional flags."""
        config = TritonGEMMConfig(
            name="test_config",
            grid=1,
            block_m=64,
            block_n=64,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
            EVEN_K=True,
            ALLOW_TF32=True,
            USE_FAST_ACCUM=True,
            ACC_TYPE="tl.float16",
        )
        assert config.EVEN_K is True
        assert config.ALLOW_TF32 is True
        assert config.USE_FAST_ACCUM is True
        assert config.ACC_TYPE == "tl.float16"

    def test_hash(self):
        """Test hash method."""
        config1 = TritonGEMMConfig(
            name="test_config",
            grid=1,
            block_m=64,
            block_n=64,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
        )
        config2 = TritonGEMMConfig(
            name="test_config",
            grid=1,
            block_m=64,
            block_n=64,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
        )
        config3 = TritonGEMMConfig(
            name="different_config",
            grid=1,
            block_m=64,
            block_n=64,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
        )

        # Same configs should have same hash
        assert hash(config1) == hash(config2)
        # Different configs should have different hash
        assert hash(config1) != hash(config3)

    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        json_str = json.dumps(
            {
                "name": "test_config",
                "grid": 1,
                "block_m": 64,
                "block_n": 64,
                "block_k": 32,
                "group_m": 8,
                "num_stages": 3,
                "num_warps": 4,
                "EVEN_K": True,
                "ALLOW_TF32": False,
                "USE_FAST_ACCUM": True,
                "ACC_TYPE": "tl.float16",
            }
        )

        config = TritonGEMMConfig.parse(json_str)
        assert config.name == "test_config"
        assert config.EVEN_K is True
        assert config.ALLOW_TF32 is False
        assert config.USE_FAST_ACCUM is True
        assert config.ACC_TYPE == "tl.float16"

    def test_parse_missing_required_field(self):
        """Test parsing JSON with missing required field."""
        json_str = json.dumps(
            {
                "grid": 1,
                "block_m": 64,
                "block_n": 64,
                "block_k": 32,
                "group_m": 8,
                "num_stages": 3,
                "num_warps": 4,
                # Missing "name"
            }
        )

        with pytest.raises(KeyError, match="Missing required field: name"):
            TritonGEMMConfig.parse(json_str)

    def test_parse_invalid_type(self):
        """Test parsing JSON with invalid type."""
        json_str = json.dumps(
            {
                "name": 123,  # Should be string
                "grid": 1,
                "block_m": 64,
                "block_n": 64,
                "block_k": 32,
                "group_m": 8,
                "num_stages": 3,
                "num_warps": 4,
            }
        )

        with pytest.raises(TypeError, match="name must be a string"):
            TritonGEMMConfig.parse(json_str)


class TestMMShape:
    """Test MMShape dataclass."""

    def test_init_basic(self):
        """Test basic initialization."""
        problem = MMShape(
            B=1,
            M=64,
            M_dtype=torch.float32,
            N=128,
            K=256,
            K_dtype=torch.float16,
            out_dtype=torch.float32,
            out_size=(1, 64, 128),
            out_stride=(8192, 128, 1),
        )
        assert problem.B == 1
        assert problem.M == 64
        assert problem.M_dtype == torch.float32
        assert problem.N == 128
        assert problem.K == 256
        assert problem.K_dtype == torch.float16
        assert problem.out_dtype == torch.float32
        assert problem.out_size == (1, 64, 128)
        assert problem.out_stride == (8192, 128, 1)

    def test_hash(self):
        """Test hash method."""
        problem1 = MMShape(
            B=1,
            M=64,
            M_dtype=torch.float32,
            N=128,
            K=256,
            K_dtype=torch.float16,
            out_dtype=torch.float32,
            out_size=(1, 64, 128),
            out_stride=(8192, 128, 1),
        )
        problem2 = MMShape(
            B=1,
            M=64,
            M_dtype=torch.float32,
            N=128,
            K=256,
            K_dtype=torch.float16,
            out_dtype=torch.float32,
            out_size=(1, 64, 128),
            out_stride=(8192, 128, 1),
        )
        problem3 = MMShape(
            B=2,
            M=64,
            M_dtype=torch.float32,
            N=128,
            K=256,
            K_dtype=torch.float16,
            out_dtype=torch.float32,
            out_size=(2, 64, 128),
            out_stride=(8192, 128, 1),
        )

        # Same problems should have same hash
        assert hash(problem1) == hash(problem2)
        # Different problems should have different hash
        assert hash(problem1) != hash(problem3)

    def test_str_representation(self):
        """Test string representation."""
        problem = MMShape(
            B=1,
            M=64,
            M_dtype=torch.float32,
            N=128,
            K=256,
            K_dtype=torch.float16,
            out_dtype=torch.float32,
            out_size=(1, 64, 128),
            out_stride=(8192, 128, 1),
        )

        str_repr = str(problem)
        assert isinstance(str_repr, str)
        assert "B" in str_repr
        assert "64" in str_repr
        assert "float32" in str_repr

    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        json_str = json.dumps(
            {
                "B": 1,
                "M": 64,
                "M_dtype": "float32",
                "N": 128,
                "K": 256,
                "K_dtype": "float16",
                "out_dtype": "float32",
                "out_size": [1, 64, 128],
                "out_stride": [8192, 128, 1],
            }
        )

        problem = MMShape.parse(json_str)
        assert problem.B == 1
        assert problem.M == 64
        assert problem.M_dtype == torch.float32
        assert problem.K_dtype == torch.float16
        assert problem.out_size == (1, 64, 128)

    def test_parse_invalid_dtype(self):
        """Test parsing with invalid dtype."""
        json_str = json.dumps(
            {
                "B": 1,
                "M": 64,
                "M_dtype": "invalid_dtype",
                "N": 128,
                "K": 256,
                "K_dtype": "float16",
                "out_dtype": "float32",
                "out_size": [1, 64, 128],
                "out_stride": [8192, 128, 1],
            }
        )

        with pytest.raises(ValueError, match="Invalid torch dtype"):
            MMShape.parse(json_str)


class TestSolution:
    """Test Solution dataclass."""

    def test_init_basic(self):
        """Test basic initialization."""
        config1 = TritonGEMMConfig(
            name="config1",
            grid=1,
            block_m=64,
            block_n=64,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
        )
        config2 = TritonGEMMConfig(
            name="config2",
            grid=1,
            block_m=128,
            block_n=128,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
        )

        solution = Solution(config=[config1, config2])
        assert len(solution.config) == 2
        assert solution.config[0] == config1
        assert solution.config[1] == config2


class TestOperation:
    """Test Operation dataclass."""

    def test_init_basic(self):
        """Test basic initialization."""
        from collections import OrderedDict

        problem = MMShape(
            B=1,
            M=64,
            M_dtype=torch.float32,
            N=128,
            K=256,
            K_dtype=torch.float16,
            out_dtype=torch.float32,
            out_size=(1, 64, 128),
            out_stride=(8192, 128, 1),
        )
        config = TritonGEMMConfig(
            name="config1",
            grid=1,
            block_m=64,
            block_n=64,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
        )
        solution = Solution(config=[config])

        operation = Operation(solution=OrderedDict([(problem, solution)]))
        assert len(operation.solution) == 1
        assert operation.solution[problem] == solution


class TestHardware:
    """Test Hardware dataclass."""

    def test_init_basic(self):
        """Test basic initialization."""
        from collections import OrderedDict

        problem = MMShape(
            B=1,
            M=64,
            M_dtype=torch.float32,
            N=128,
            K=256,
            K_dtype=torch.float16,
            out_dtype=torch.float32,
            out_size=(1, 64, 128),
            out_stride=(8192, 128, 1),
        )
        config = TritonGEMMConfig(
            name="config1",
            grid=1,
            block_m=64,
            block_n=64,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
        )
        solution = Solution(config=[config])
        operation = Operation(solution=OrderedDict([(problem, solution)]))

        hardware = Hardware(operation=OrderedDict([("mm", operation)]))
        assert len(hardware.operation) == 1
        assert hardware.operation["mm"] == operation


class TestTable:
    """Test Table dataclass."""

    def setup_method(self):
        """Set up test fixtures."""
        from collections import OrderedDict

        self.problem = MMShape(
            B=1,
            M=64,
            M_dtype=torch.float32,
            N=128,
            K=256,
            K_dtype=torch.float16,
            out_dtype=torch.float32,
            out_size=(1, 64, 128),
            out_stride=(8192, 128, 1),
        )
        self.config = TritonGEMMConfig(
            name="config1",
            grid=1,
            block_m=64,
            block_n=64,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
        )
        self.solution = Solution(config=[self.config])
        self.operation = Operation(
            solution=OrderedDict([(self.problem, self.solution)])
        )
        self.hardware = Hardware(operation=OrderedDict([("mm", self.operation)]))

    def test_init_basic(self):
        """Test basic initialization."""
        from collections import OrderedDict

        table = Table(hardware=OrderedDict([("gpu1", self.hardware)]))
        assert len(table.hardware) == 1
        assert table.hardware["gpu1"] == self.hardware

    def test_serialize_deserialize(self):
        """Test serialization and deserialization."""
        from collections import OrderedDict

        table = Table(hardware=OrderedDict([("gpu1", self.hardware)]))

        # Serialize
        serialized = table.serialize()
        assert isinstance(serialized, str)

        # Deserialize
        deserialized = Table.deserialize(serialized)
        assert deserialized is not None
        assert len(deserialized.hardware) == 1

    def test_deserialize_invalid_json(self):
        """Test deserialization with invalid JSON."""
        result = Table.deserialize("invalid json")
        assert result is None

    def test_lookup_success(self):
        """Test successful lookup."""
        from collections import OrderedDict

        table = Table(hardware=OrderedDict([("gpu1", self.hardware)]))

        result = table.lookup("gpu1", "mm", self.problem)
        assert result is not None
        assert result == [self.config]

    def test_lookup_missing_hardware(self):
        """Test lookup with missing hardware."""
        from collections import OrderedDict

        table = Table(hardware=OrderedDict([("gpu1", self.hardware)]))

        result = table.lookup("gpu2", "mm", self.problem)
        assert result is None

    def test_lookup_missing_operation(self):
        """Test lookup with missing operation."""
        from collections import OrderedDict

        table = Table(hardware=OrderedDict([("gpu1", self.hardware)]))

        result = table.lookup("gpu1", "addmm", self.problem)
        assert result is None

    def test_lookup_missing_problem(self):
        """Test lookup with missing problem."""
        from collections import OrderedDict

        table = Table(hardware=OrderedDict([("gpu1", self.hardware)]))

        different_problem = MMShape(
            B=2,
            M=128,
            M_dtype=torch.float32,
            N=256,
            K=512,
            K_dtype=torch.float16,
            out_dtype=torch.float32,
            out_size=(2, 128, 256),
            out_stride=(65536, 256, 1),
        )

        result = table.lookup("gpu1", "mm", different_problem)
        assert result is None

    def test_lookup_set(self):
        """Test lookup_set method."""
        from collections import OrderedDict

        from torch.utils._ordered_set import OrderedSet

        table = Table(hardware=OrderedDict([("gpu1", self.hardware)]))

        result = table.lookup_set("gpu1", "mm", self.problem)
        assert result is not None
        assert isinstance(result, OrderedSet)
        assert self.config in result

    def test_lookup_set_caching(self):
        """Test that lookup_set caches results."""
        from collections import OrderedDict

        table = Table(hardware=OrderedDict([("gpu1", self.hardware)]))

        # First call
        result1 = table.lookup_set("gpu1", "mm", self.problem)
        # Second call should use cache
        result2 = table.lookup_set("gpu1", "mm", self.problem)

        assert result1 is result2  # Same object due to caching

    def test_filter_configs(self):
        """Test filter method."""
        from collections import OrderedDict

        config2 = TritonGEMMConfig(
            name="config2",
            grid=1,
            block_m=128,
            block_n=128,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
        )

        table = Table(hardware=OrderedDict([("gpu1", self.hardware)]))

        # Filter with configs that include the one in our table
        to_filter = [self.config, config2]
        result = table.filter("gpu1", "mm", self.problem, to_filter)

        assert result is not None
        assert len(result) == 1
        assert result[0] == self.config

    def test_filter_no_matching_configs(self):
        """Test filter with no matching configs."""
        from collections import OrderedDict

        config2 = TritonGEMMConfig(
            name="config2",
            grid=1,
            block_m=128,
            block_n=128,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
        )

        table = Table(hardware=OrderedDict([("gpu1", self.hardware)]))

        # Filter with configs that don't include the one in our table
        to_filter = [config2]
        result = table.filter("gpu1", "mm", self.problem, to_filter)

        assert result is None

    def test_filter_missing_problem(self):
        """Test filter with missing problem."""
        from collections import OrderedDict

        different_problem = MMShape(
            B=2,
            M=128,
            M_dtype=torch.float32,
            N=256,
            K=512,
            K_dtype=torch.float16,
            out_dtype=torch.float32,
            out_size=(2, 128, 256),
            out_stride=(65536, 256, 1),
        )

        table = Table(hardware=OrderedDict([("gpu1", self.hardware)]))

        result = table.filter("gpu1", "mm", different_problem, [self.config])
        assert result is None


class TestIntegrationMatmulTypes:
    """Integration tests for the matmul types module."""

    def test_complete_workflow(self):
        """Test a complete workflow using all types."""
        from collections import OrderedDict

        # Create multiple configs
        config1 = TritonGEMMConfig(
            name="fast_config",
            grid=1,
            block_m=64,
            block_n=64,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
        )
        config2 = TritonGEMMConfig(
            name="large_config",
            grid=1,
            block_m=128,
            block_n=128,
            block_k=64,
            group_m=8,
            num_stages=4,
            num_warps=8,
        )

        # Create problems
        problem1 = MMShape(
            B=1,
            M=64,
            M_dtype=torch.float32,
            N=128,
            K=256,
            K_dtype=torch.float16,
            out_dtype=torch.float32,
            out_size=(1, 64, 128),
            out_stride=(8192, 128, 1),
        )
        problem2 = MMShape(
            B=1,
            M=128,
            M_dtype=torch.float32,
            N=256,
            K=512,
            K_dtype=torch.float16,
            out_dtype=torch.float32,
            out_size=(1, 128, 256),
            out_stride=(32768, 256, 1),
        )

        # Create solutions
        solution1 = Solution(config=[config1])
        solution2 = Solution(config=[config2])

        # Create operations
        mm_operation = Operation(
            solution=OrderedDict([(problem1, solution1), (problem2, solution2)]),
        )

        # Create hardware
        gpu_hardware = Hardware(operation=OrderedDict([("mm", mm_operation)]))

        # Create table
        table = Table(hardware=OrderedDict([("A100", gpu_hardware)]))

        # Test lookups
        result1 = table.lookup("A100", "mm", problem1)
        assert result1 == [config1]

        result2 = table.lookup("A100", "mm", problem2)
        assert result2 == [config2]

        # Test filtering
        all_configs = [config1, config2]
        filtered = table.filter("A100", "mm", problem1, all_configs)
        assert filtered == [config1]

        # Test serialization round-trip
        serialized = table.serialize()
        deserialized = Table.deserialize(serialized)
        assert deserialized is not None

        # Verify deserialized table works
        result3 = deserialized.lookup("A100", "mm", problem1)
        assert result3 is not None
        assert len(result3) == 1
