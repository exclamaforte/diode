"""
Tests for diode.collection.generic_data_utils module.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

from diode.collection.generic_data_utils import convert_json_to_msgpack


class TestConvertJsonToMsgpack:
    """Test the convert_json_to_msgpack function."""

    def test_convert_single_file_success(self):
        """Test successful conversion of a single JSON file."""
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input JSON file
            input_file = os.path.join(temp_dir, "test_input.json")
            test_data = {"key": "value", "number": 42, "array": [1, 2, 3]}

            with open(input_file, "w") as f:
                json.dump(test_data, f)

            # Convert the file
            convert_json_to_msgpack([input_file])

            # Check that output file was created
            expected_output = os.path.join(temp_dir, "test_input.msgpack")
            assert os.path.exists(expected_output)

            # Verify the content by reading back
            import msgpack

            with open(expected_output, "rb") as f:
                loaded_data = msgpack.unpack(f)

            assert loaded_data == test_data

    def test_convert_multiple_files_success(self):
        """Test successful conversion of multiple JSON files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple input files
            files_data = [
                ("file1.json", {"data": "file1"}),
                ("file2.json", {"data": "file2"}),
                ("file3.json", {"data": "file3"}),
            ]

            input_files = []
            for filename, data in files_data:
                input_file = os.path.join(temp_dir, filename)
                with open(input_file, "w") as f:
                    json.dump(data, f)
                input_files.append(input_file)

            # Convert all files
            convert_json_to_msgpack(input_files)

            # Verify all output files exist
            for filename, expected_data in files_data:
                base_name = os.path.splitext(filename)[0]
                output_file = os.path.join(temp_dir, f"{base_name}.msgpack")
                assert os.path.exists(output_file)

                # Verify content
                import msgpack

                with open(output_file, "rb") as f:
                    loaded_data = msgpack.unpack(f)
                assert loaded_data == expected_data

    def test_convert_with_output_directory(self):
        """Test conversion with specified output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input file
            input_file = os.path.join(temp_dir, "input.json")
            test_data = {"test": "data"}

            with open(input_file, "w") as f:
                json.dump(test_data, f)

            # Create output directory
            output_dir = os.path.join(temp_dir, "output")

            # Convert with output directory
            convert_json_to_msgpack([input_file], output_dir=output_dir)

            # Check output file in specified directory
            expected_output = os.path.join(output_dir, "input.msgpack")
            assert os.path.exists(expected_output)
            assert os.path.exists(output_dir)

    def test_convert_nonexistent_file(self):
        """Test handling of nonexistent input file."""
        nonexistent_file = "/path/that/does/not/exist.json"

        # Should not raise exception, but log error
        with patch("diode.collection.generic_data_utils.logger") as mock_logger:
            convert_json_to_msgpack([nonexistent_file])

            # Verify error was logged
            mock_logger.error.assert_called()
            error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
            assert any("Input file does not exist" in call for call in error_calls)

    def test_convert_invalid_json(self):
        """Test handling of invalid JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid JSON file
            input_file = os.path.join(temp_dir, "invalid.json")
            with open(input_file, "w") as f:
                f.write("{ invalid json content")

            with patch("diode.collection.generic_data_utils.logger") as mock_logger:
                convert_json_to_msgpack([input_file])

                # Verify JSON decode error was logged
                mock_logger.error.assert_called()
                error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
                assert any("Failed to parse JSON file" in call for call in error_calls)

    def test_convert_with_msgpack_error(self):
        """Test handling of msgpack serialization error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create valid JSON file
            input_file = os.path.join(temp_dir, "test.json")
            with open(input_file, "w") as f:
                json.dump({"test": "data"}, f)

            # Mock msgpack.pack to raise an exception
            with patch(
                "diode.collection.generic_data_utils.msgpack.pack",
                side_effect=Exception("Mock error"),
            ):
                with patch("diode.collection.generic_data_utils.logger") as mock_logger:
                    convert_json_to_msgpack([input_file])

                    # Verify error was logged
                    mock_logger.error.assert_called()
                    error_calls = [
                        call[0][0] for call in mock_logger.error.call_args_list
                    ]
                    assert any("Error converting" in call for call in error_calls)

    def test_convert_empty_file_list(self):
        """Test conversion with empty file list."""
        with patch("diode.collection.generic_data_utils.logger") as mock_logger:
            convert_json_to_msgpack([])

            # Should log completion with 0 files
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Converted: 0 files" in call for call in info_calls)

    def test_convert_creates_output_directory(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input file
            input_file = os.path.join(temp_dir, "test.json")
            with open(input_file, "w") as f:
                json.dump({"test": "data"}, f)

            # Specify non-existent output directory
            output_dir = os.path.join(temp_dir, "new_output_dir")
            assert not os.path.exists(output_dir)

            convert_json_to_msgpack([input_file], output_dir=output_dir)

            # Directory should be created
            assert os.path.exists(output_dir)
            expected_output = os.path.join(output_dir, "test.msgpack")
            assert os.path.exists(expected_output)

    def test_convert_logs_summary(self):
        """Test that conversion logs summary information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            input_files = []
            for i in range(3):
                input_file = os.path.join(temp_dir, f"test{i}.json")
                with open(input_file, "w") as f:
                    json.dump({"data": i}, f)
                input_files.append(input_file)

            with patch("diode.collection.generic_data_utils.logger") as mock_logger:
                convert_json_to_msgpack(input_files)

                # Check summary logs
                info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
                assert any("Conversion completed:" in call for call in info_calls)
                assert any("Converted: 3 files" in call for call in info_calls)
                assert any("Errors: 0 files" in call for call in info_calls)

    def test_convert_mixed_success_and_failure(self):
        """Test conversion with some successful and some failed files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create one valid JSON file
            valid_file = os.path.join(temp_dir, "valid.json")
            with open(valid_file, "w") as f:
                json.dump({"valid": "data"}, f)

            # Create one invalid JSON file
            invalid_file = os.path.join(temp_dir, "invalid.json")
            with open(invalid_file, "w") as f:
                f.write("{ invalid json")

            # Include one nonexistent file
            nonexistent_file = os.path.join(temp_dir, "nonexistent.json")

            with patch("diode.collection.generic_data_utils.logger") as mock_logger:
                convert_json_to_msgpack([valid_file, invalid_file, nonexistent_file])

                # Check summary
                info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
                assert any("Converted: 1 files" in call for call in info_calls)
                assert any("Errors: 2 files" in call for call in info_calls)

    def test_convert_preserves_directory_structure_with_output_dir(self):
        """Test that file names are preserved when using output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input file with complex name
            input_file = os.path.join(temp_dir, "complex_name.with.dots.json")
            with open(input_file, "w") as f:
                json.dump({"test": "data"}, f)

            output_dir = os.path.join(temp_dir, "output")
            convert_json_to_msgpack([input_file], output_dir=output_dir)

            # Check that output file has correct name
            expected_output = os.path.join(output_dir, "complex_name.with.dots.msgpack")
            assert os.path.exists(expected_output)
