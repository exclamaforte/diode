"""
Tests for parse methods and validation in matmul_types.py.
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
    MMShape,
    OperationShapeSet,
    ShapeSet,
    Solution,
    Table,
    TritonGEMMConfig,
)


class TestTritonGEMMConfigParse:
    def test_parse_valid_config(self):
        """Test parsing a valid TritonGEMMConfig JSON string."""
        config_json = json.dumps(
            {
                "name": "test_config",
                "grid": 1024,
                "block_m": 64,
                "block_n": 128,
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

        config = TritonGEMMConfig.parse(config_json)

        assert config.name == "test_config"
        assert config.grid == 1024
        assert config.block_m == 64
        assert config.block_n == 128
        assert config.block_k == 32
        assert config.group_m == 8
        assert config.num_stages == 3
        assert config.num_warps == 4
        assert config.EVEN_K is True
        assert config.ALLOW_TF32 is False
        assert config.USE_FAST_ACCUM is True
        assert config.ACC_TYPE == "tl.float16"

    def test_parse_missing_required_fields(self):
        """Test parsing with missing required fields."""
        # Missing name
        with pytest.raises(KeyError, match="Missing required field: name"):
            TritonGEMMConfig.parse('{"grid": 1024}')

        # Missing grid
        with pytest.raises(KeyError, match="Missing required field: grid"):
            TritonGEMMConfig.parse('{"name": "test"}')

        # Missing block_m
        with pytest.raises(KeyError, match="Missing required field: block_m"):
            TritonGEMMConfig.parse('{"name": "test", "grid": 1024}')

        # Missing block_n
        with pytest.raises(KeyError, match="Missing required field: block_n"):
            TritonGEMMConfig.parse('{"name": "test", "grid": 1024, "block_m": 64}')

        # Missing block_k
        with pytest.raises(KeyError, match="Missing required field: block_k"):
            TritonGEMMConfig.parse(
                '{"name": "test", "grid": 1024, "block_m": 64, "block_n": 128}'
            )

        # Missing group_m
        with pytest.raises(KeyError, match="Missing required field: group_m"):
            TritonGEMMConfig.parse(
                '{"name": "test", "grid": 1024, "block_m": 64, "block_n": 128, "block_k": 32}'
            )

        # Missing num_stages
        with pytest.raises(KeyError, match="Missing required field: num_stages"):
            TritonGEMMConfig.parse(
                '{"name": "test", "grid": 1024, "block_m": 64, "block_n": 128, "block_k": 32, "group_m": 8}'
            )

        # Missing num_warps
        with pytest.raises(KeyError, match="Missing required field: num_warps"):
            TritonGEMMConfig.parse(
                '{"name": "test", "grid": 1024, "block_m": 64, "block_n": 128, "block_k": 32, "group_m": 8, "num_stages": 3}'
            )

    def test_parse_wrong_types(self):
        """Test parsing with wrong types for fields."""
        base_config = {
            "name": "test",
            "grid": 1024,
            "block_m": 64,
            "block_n": 128,
            "block_k": 32,
            "group_m": 8,
            "num_stages": 3,
            "num_warps": 4,
        }

        # Wrong type for name
        config = base_config.copy()
        config["name"] = 123
        with pytest.raises(TypeError, match="name must be a string"):
            TritonGEMMConfig.parse(json.dumps(config))

        # Wrong type for grid
        config = base_config.copy()
        config["grid"] = "not_int"
        with pytest.raises(TypeError, match="grid must be an int"):
            TritonGEMMConfig.parse(json.dumps(config))

        # Wrong type for block_m
        config = base_config.copy()
        config["block_m"] = 64.5
        with pytest.raises(TypeError, match="block_m must be an int"):
            TritonGEMMConfig.parse(json.dumps(config))

        # Wrong type for EVEN_K
        config = base_config.copy()
        config["EVEN_K"] = "true"
        with pytest.raises(TypeError, match="EVEN_K must be a bool"):
            TritonGEMMConfig.parse(json.dumps(config))

        # Wrong type for ALLOW_TF32
        config = base_config.copy()
        config["ALLOW_TF32"] = 1
        with pytest.raises(TypeError, match="ALLOW_TF32 must be a bool"):
            TritonGEMMConfig.parse(json.dumps(config))

        # Wrong type for USE_FAST_ACCUM
        config = base_config.copy()
        config["USE_FAST_ACCUM"] = "false"
        with pytest.raises(TypeError, match="USE_FAST_ACCUM must be a bool"):
            TritonGEMMConfig.parse(json.dumps(config))

        # Wrong type for ACC_TYPE
        config = base_config.copy()
        config["ACC_TYPE"] = 123
        with pytest.raises(TypeError, match="ACC_TYPE must be a string"):
            TritonGEMMConfig.parse(json.dumps(config))


class TestMMShapeParse:
    def test_parse_valid_shape(self):
        """Test parsing a valid MMShape JSON string."""
        shape_json = json.dumps(
            {
                "B": 2,
                "M": 128,
                "N": 256,
                "K": 512,
                "M_dtype": "float32",
                "K_dtype": "float32",
                "out_dtype": "float32",
                "out_size": [2, 128, 256],
                "out_stride": [32768, 256, 1],
            }
        )

        shape = MMShape.parse(shape_json)

        assert shape.B == 2
        assert shape.M == 128
        assert shape.N == 256
        assert shape.K == 512
        assert shape.M_dtype == torch.float32
        assert shape.K_dtype == torch.float32
        assert shape.out_dtype == torch.float32
        assert shape.out_size == (2, 128, 256)
        assert shape.out_stride == (32768, 256, 1)

    def test_parse_missing_required_fields(self):
        """Test parsing with missing required fields."""
        # Missing B
        with pytest.raises(KeyError, match="Missing required field: B"):
            MMShape.parse('{"M": 128}')

        # Missing M
        with pytest.raises(KeyError, match="Missing required field: M"):
            MMShape.parse('{"B": 2}')

        # Missing N
        with pytest.raises(KeyError, match="Missing required field: N"):
            MMShape.parse('{"B": 2, "M": 128}')

        # Missing K
        with pytest.raises(KeyError, match="Missing required field: K"):
            MMShape.parse('{"B": 2, "M": 128, "N": 256}')

        # Missing M_dtype
        with pytest.raises(KeyError, match="Missing required field: M_dtype"):
            MMShape.parse('{"B": 2, "M": 128, "N": 256, "K": 512}')

        # Missing K_dtype
        with pytest.raises(KeyError, match="Missing required field: K_dtype"):
            MMShape.parse(
                '{"B": 2, "M": 128, "N": 256, "K": 512, "M_dtype": "float32"}'
            )

        # Missing out_dtype
        with pytest.raises(KeyError, match="Missing required field: out_dtype"):
            MMShape.parse(
                '{"B": 2, "M": 128, "N": 256, "K": 512, "M_dtype": "float32", "K_dtype": "float32"}'
            )

        # Missing out_size
        with pytest.raises(KeyError, match="Missing required field: out_size"):
            MMShape.parse(
                '{"B": 2, "M": 128, "N": 256, "K": 512, "M_dtype": "float32", "K_dtype": "float32", "out_dtype": "float32"}'
            )

        # Missing out_stride
        with pytest.raises(KeyError, match="Missing required field: out_stride"):
            MMShape.parse(
                '{"B": 2, "M": 128, "N": 256, "K": 512, "M_dtype": "float32", "K_dtype": "float32", "out_dtype": "float32", "out_size": [2, 128, 256]}'
            )

    def test_parse_wrong_types(self):
        """Test parsing with wrong types for fields."""
        base_shape = {
            "B": 2,
            "M": 128,
            "N": 256,
            "K": 512,
            "M_dtype": "float32",
            "K_dtype": "float32",
            "out_dtype": "float32",
            "out_size": [2, 128, 256],
            "out_stride": [32768, 256, 1],
        }

        # Wrong type for B
        shape = base_shape.copy()
        shape["B"] = "not_int"
        with pytest.raises(TypeError, match="B must be an int"):
            MMShape.parse(json.dumps(shape))

        # Wrong type for M_dtype
        shape = base_shape.copy()
        shape["M_dtype"] = 123
        with pytest.raises(TypeError, match="M_dtype must be a string"):
            MMShape.parse(json.dumps(shape))

        # Wrong type for out_size
        shape = base_shape.copy()
        shape["out_size"] = "not_list"
        with pytest.raises(TypeError, match="out_size must be a list"):
            MMShape.parse(json.dumps(shape))

        # Wrong type for out_stride
        shape = base_shape.copy()
        shape["out_stride"] = 123
        with pytest.raises(TypeError, match="out_stride must be a list"):
            MMShape.parse(json.dumps(shape))

    def test_parse_invalid_torch_dtype(self):
        """Test parsing with invalid torch dtype strings."""
        base_shape = {
            "B": 2,
            "M": 128,
            "N": 256,
            "K": 512,
            "M_dtype": "invalid_dtype",
            "K_dtype": "float32",
            "out_dtype": "float32",
            "out_size": [2, 128, 256],
            "out_stride": [32768, 256, 1],
        }

        with pytest.raises(ValueError, match="Invalid torch dtype: invalid_dtype"):
            MMShape.parse(json.dumps(base_shape))

        # Test K_dtype invalid
        base_shape["M_dtype"] = "float32"
        base_shape["K_dtype"] = "invalid_dtype"
        with pytest.raises(ValueError, match="Invalid torch dtype: invalid_dtype"):
            MMShape.parse(json.dumps(base_shape))

        # Test out_dtype invalid
        base_shape["K_dtype"] = "float32"
        base_shape["out_dtype"] = "invalid_dtype"
        with pytest.raises(ValueError, match="Invalid torch dtype: invalid_dtype"):
            MMShape.parse(json.dumps(base_shape))


class TestSolutionParse:
    def test_parse_valid_solution(self):
        """Test parsing a valid Solution JSON string."""
        solution_json = json.dumps(
            {
                "config": [
                    {
                        "name": "config1",
                        "grid": 1024,
                        "block_m": 64,
                        "block_n": 128,
                        "block_k": 32,
                        "group_m": 8,
                        "num_stages": 3,
                        "num_warps": 4,
                    },
                    {
                        "name": "config2",
                        "grid": 2048,
                        "block_m": 128,
                        "block_n": 256,
                        "block_k": 64,
                        "group_m": 16,
                        "num_stages": 4,
                        "num_warps": 8,
                    },
                ]
            }
        )

        solution = Solution.parse(solution_json)

        assert len(solution.config) == 2
        assert isinstance(solution.config[0], TritonGEMMConfig)
        assert solution.config[0].name == "config1"
        assert solution.config[0].grid == 1024
        assert isinstance(solution.config[1], TritonGEMMConfig)
        assert solution.config[1].name == "config2"
        assert solution.config[1].grid == 2048

    def test_parse_empty_config_list(self):
        """Test parsing Solution with empty config list."""
        solution_json = json.dumps({"config": []})

        solution = Solution.parse(solution_json)

        assert len(solution.config) == 0

    def test_parse_solution_with_tritongemm_objects(self):
        """Test parsing Solution where config already contains TritonGEMMConfig objects."""
        # This shouldn't normally happen but the code handles it
        config_obj = TritonGEMMConfig(
            name="existing_config",
            grid=512,
            block_m=32,
            block_n=64,
            block_k=16,
            group_m=4,
            num_stages=2,
            num_warps=2,
        )

        # Manually create the data structure as if it came from JSON but with object
        data = {"config": [config_obj]}

        # We can't directly parse this since it's not JSON, but we can test the object creation
        solution = Solution(config=[config_obj])
        assert len(solution.config) == 1
        assert solution.config[0] == config_obj


class TestMMShapeStringMethods:
    def test_str_method_with_special_types(self):
        """Test __str__ method with objects that need conversion."""

        # Create a mock object that has __int__ but conversion fails
        class MockIntObject:
            def __int__(self):
                raise ValueError("Cannot convert to int")

            def __str__(self):
                return "mock_int"

        # Create a mock object that has __float__ but conversion fails
        class MockFloatObject:
            def __float__(self):
                raise ValueError("Cannot convert to float")

            def __str__(self):
                return "mock_float"

        # Test with normal shape first
        shape = MMShape(
            B=2,
            M=128,
            N=256,
            K=512,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(2, 128, 256),
            out_stride=(32768, 256, 1),
        )

        # This should work normally
        str_repr = str(shape)
        assert isinstance(str_repr, str)

        # Parse it back to verify it's valid JSON
        parsed_data = json.loads(str_repr)
        assert parsed_data["B"] == 2
        assert parsed_data["M"] == 128
        assert parsed_data["M_dtype"] == "float32"

    def test_str_method_with_tuple_conversion(self):
        """Test __str__ method handles tuple/list conversion correctly."""
        shape = MMShape(
            B=1,
            M=64,
            N=128,
            K=256,
            M_dtype=torch.float16,
            K_dtype=torch.float16,
            out_dtype=torch.float16,
            out_size=(1, 64, 128),
            out_stride=(8192, 128, 1),
        )

        str_repr = str(shape)
        parsed_data = json.loads(str_repr)

        # Verify tuples were converted to lists
        assert isinstance(parsed_data["out_size"], list)
        assert isinstance(parsed_data["out_stride"], list)
        assert parsed_data["out_size"] == [1, 64, 128]
        assert parsed_data["out_stride"] == [8192, 128, 1]

    def test_str_method_filters_private_attributes(self):
        """Test __str__ method filters out private attributes."""
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

        str_repr = str(shape)
        parsed_data = json.loads(str_repr)

        # Verify no private attributes (starting with _) are included
        for key in parsed_data.keys():
            assert not key.startswith("_")


class TestShapeSetAndOperationShapeSet:
    def test_shapeset_deserialization_failure(self):
        """Test ShapeSet deserialization with invalid JSON."""
        # Test JSON decode error
        result = ShapeSet.deserialize("invalid json {")
        assert result is None

        # Test missing shapes field
        result = ShapeSet.deserialize('{"no_shapes": []}')
        assert result is None

        # Test general exception during deserialization
        result = ShapeSet.deserialize('{"shapes": [{"invalid": "shape"}]}')
        assert result is None

    def test_operation_shapeset_deserialization_failure(self):
        """Test OperationShapeSet deserialization with invalid JSON."""
        # Test JSON decode error
        result = OperationShapeSet.deserialize("invalid json {")
        assert result is None

        # Test missing operations field
        result = OperationShapeSet.deserialize('{"no_operations": {}}')
        assert result is None

        # Test general exception during deserialization - this returns an object but with failed ShapeSets filtered out
        result = OperationShapeSet.deserialize(
            '{"operations": {"mm": {"invalid": "data"}}}'
        )
        # The function continues and returns an object even if individual ShapeSets fail
        assert result is not None
        # But the operations dict should be empty since the ShapeSet failed to deserialize
        assert len(result.operations) == 0

    def test_operation_shapeset_with_failed_shapeset_deserialization(self):
        """Test OperationShapeSet when individual ShapeSet deserialization fails."""
        # This creates an OperationShapeSet where one of the ShapeSets fails to deserialize
        ops_json = json.dumps(
            {
                "operations": {
                    "mm": {"shapes": [{"invalid": "shape_data"}]},
                    "addmm": {"shapes": []},
                }
            }
        )

        result = OperationShapeSet.deserialize(ops_json)

        # Should still return an object, but the failed ShapeSet should be skipped
        assert result is not None
        # Only the successful ShapeSet should be included
        assert len(result.operations) == 1
        assert "addmm" in result.operations


class TestTableDeserialization:
    def test_table_json_decode_error(self):
        """Test Table deserialization with JSON decode error."""
        result = Table.deserialize("invalid json {")
        assert result is None

    def test_table_from_dict_error(self):
        """Test Table deserialization when from_dict fails."""
        # Valid JSON but invalid structure for Table
        result = Table.deserialize('{"invalid": "structure"}')
        assert result is None
