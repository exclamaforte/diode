# Owner(s): ["module: diode"]
"""
Tests for graceful handling of malformed JSON in kernel LUT.
"""

import json
import logging
import os
import tempfile
import threading
import unittest
from collections import OrderedDict
from io import StringIO
from unittest import TestCase

import torch

from diode.types.matmul_types import (
    Hardware,
    MMShape,
    Operation,
    Solution,
    Table,
    TritonGEMMConfig,
)


def run_tests():
    unittest.main()


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

            # Verify that parse methods were called for objects used as keys
            # MMShape.parse should be called for each problem key in the serialized data
            # (MMShape objects are used as keys in OrderedDict, so they get serialized as strings)
            self.assertTrue(
                mock_problem_parse.called,
                "MMShape.parse should be called when MMShape is used as OrderedDict key",
            )
            self.assertGreater(
                mock_problem_parse.call_count,
                0,
                "MMShape.parse should be called at least once",
            )

            # TritonGEMMConfig.parse should NOT be called because TritonGEMMConfig objects
            # are stored in lists, not used as keys, so they use from_dict instead of parse
            self.assertFalse(
                mock_triton_parse.called,
                "TritonGEMMConfig.parse should NOT be called when stored in lists",
            )


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

    def test_get_table_with_malformed_file(self):
        """Test get_table function with malformed JSON files."""
        from diode.types.kernel_lut import get_table, get_table_safe

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
        from diode.types.kernel_lut import get_table, get_table_safe

        nonexistent_path = "/path/that/does/not/exist.json"

        # Test get_table
        result = get_table(nonexistent_path)
        self.assertIsNone(result)

        # Test get_table_safe
        result_safe = get_table_safe(nonexistent_path)
        self.assertIsNone(result_safe)

    def test_logging_on_malformed_json(self):
        """Test that appropriate log messages are generated for malformed JSON."""
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


if __name__ == "__main__":
    run_tests()
