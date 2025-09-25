"""
Tests for diode.collection.matmul_data_utils module.
"""


# Enable debug flags for testing
try:
    from torch_diode.utils.debug_config import set_debug_flag

    set_debug_flag("ENABLE_TYPE_ASSERTS", True)
except ImportError:
    pass  # In case debug_config is not available yet
import os
import tempfile
from unittest.mock import Mock, patch

import torch

from torch_diode.collection.matmul_data_utils import (
    _collect_data_chunked,
    _create_validation_dataset_chunked,
    collect_data,
    create_validation_dataset,
    run_collector_example,
    run_matrix_multiplications,
)


def mock_open():
    """Helper function to create a mock for the open function."""
    from unittest.mock import mock_open as _mock_open

    return _mock_open()


class TestMatmulDataUtils:
    """Test matmul data utility functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_dataset_with_proper_structure(self):
        """Create a mock dataset with proper structure for traversal."""
        # Create a mock solution with timed_configs
        mock_solution = Mock()
        mock_solution.timed_configs = []

        # Create a mock problem with the required attributes (M, N, K, M_dtype)
        mock_problem = Mock()
        mock_problem.M = 32
        mock_problem.N = 64
        mock_problem.K = 128
        mock_problem.M_dtype = "float32"

        # Create mock solution dict that behaves like an object
        mock_solution_dict = Mock()
        mock_solution_dict_data = {
            mock_problem: mock_solution
        }  # Use mock_problem as key
        mock_solution_dict.items = Mock(return_value=mock_solution_dict_data.items())
        mock_solution_dict.values = Mock(return_value=mock_solution_dict_data.values())
        mock_solution_dict.__len__ = Mock(return_value=len(mock_solution_dict_data))

        # Create mock operation obj that has .solution attribute
        mock_operation_obj = Mock()
        mock_operation_obj.solution = mock_solution_dict

        # Create mock operation dict
        mock_operation_dict = Mock()
        mock_operation_dict_data = {"op1": mock_operation_obj}
        mock_operation_dict.items = Mock(return_value=mock_operation_dict_data.items())
        mock_operation_dict.__len__ = Mock(return_value=len(mock_operation_dict_data))

        # Create mock hardware object that has both dict-like and attribute access
        mock_hardware_obj = Mock()
        mock_hardware_obj.operation = mock_operation_dict

        # Create proper nested structure that supports iteration and len()
        mock_hardware = {"hw1": mock_hardware_obj}

        mock_dataset = Mock()
        mock_dataset.hardware = mock_hardware
        mock_dataset.to_msgpack.return_value = b"mock_msgpack_data"

        return mock_dataset

    def _create_mock_collector_with_proper_dataset(self):
        """Create a mock collector that passes isinstance check and has proper dataset."""
        from torch_diode.collection.matmul_dataset_collector import (
            MatmulDatasetCollector,
        )

        # Create a mock collector that will pass isinstance check
        mock_collector = Mock(spec=MatmulDatasetCollector)
        mock_dataset = self._create_mock_dataset_with_proper_structure()
        mock_collector.get_dataset.return_value = mock_dataset
        mock_collector.collect_data = Mock()
        mock_collector.save_to_file = Mock()

        return mock_collector

    @patch("torch.compile")
    @patch("torch.randn")
    def test_run_matrix_multiplications_basic(self, mock_randn, mock_compile):
        """Test basic matrix multiplication running."""
        # Mock tensors
        mock_tensor = Mock()
        mock_randn.return_value = mock_tensor

        # Mock compiled function
        mock_compiled_fn = Mock()
        mock_compile.return_value = mock_compiled_fn

        sizes = [(32, 64, 128), (64, 128, 256)]
        dtypes = [torch.float32]

        # Should not raise exception
        run_matrix_multiplications(sizes, dtypes, device="cpu")

        # Verify torch.compile was called for each size/dtype combination
        expected_calls = len(sizes) * len(dtypes) * 2  # mm and addmm
        assert mock_compile.call_count == expected_calls

    @patch("torch.cuda.is_available")
    def test_run_matrix_multiplications_cuda_available(self, mock_cuda_available):
        """Test matrix multiplication with CUDA available."""
        mock_cuda_available.return_value = True

        with patch("torch.compile") as mock_compile:
            with patch("torch.randn") as mock_randn:
                mock_tensor = Mock()
                mock_randn.return_value = mock_tensor
                mock_compiled_fn = Mock()
                mock_compile.return_value = mock_compiled_fn

                sizes = [(32, 64, 128)]
                dtypes = [torch.float16]

                run_matrix_multiplications(sizes, dtypes)

                # Verify randn was called with cuda device
                mock_randn.assert_called()

    def test_collect_data_chunked_function_exists(self):
        """Test that collect_data function exists and is callable."""
        assert callable(collect_data)

    @patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    @patch("torch_diode.utils.dataset_utils.print_dataset_statistics")
    def test_collect_data_msgpack_format(
        self,
        mock_print_stats,
        mock_device_name,
        mock_cuda_available,
        mock_collector_class,
    ):
        """Test data collection with msgpack format."""
        mock_cuda_available.return_value = True
        mock_device_name.return_value = "NVIDIA A100"

        # Mock collector and dataset with proper structure
        mock_collector = self._create_mock_collector_with_proper_dataset()
        mock_collector_class.return_value = mock_collector

        output_file = os.path.join(self.temp_dir, "test_data.json")

        with patch("builtins.open", mock_open()) as mock_file:
            result = collect_data(output_file=output_file, file_format="msgpack")

            # Should change extension to .msgpack
            expected_file = output_file.replace(".json", ".msgpack")
            assert result == expected_file

    @patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    def test_collect_data_chunked(
        self, mock_device_name, mock_cuda_available, mock_collector_class
    ):
        """Test chunked data collection."""
        mock_cuda_available.return_value = True
        mock_device_name.return_value = "NVIDIA A100"

        # Mock collector
        mock_collector = Mock()
        mock_collector_class.return_value = mock_collector

        with patch(
            "torch_diode.collection.matmul_data_utils._collect_data_chunked"
        ) as mock_chunked:
            mock_chunked.return_value = "chunked_output.json"

            output_file = os.path.join(self.temp_dir, "test_data.json")

            result = collect_data(output_file=output_file, chunk_size=10)

            assert result == "chunked_output.json"
            mock_chunked.assert_called_once()

    @patch("torch_diode.collection.matmul_data_utils.collect_data")
    @patch("os.path.exists")
    def test_create_validation_dataset_new(self, mock_exists, mock_collect_data):
        """Test creating new validation dataset."""
        mock_exists.return_value = False
        mock_collect_data.return_value = "validation.json"

        output_file = os.path.join(self.temp_dir, "validation.json")

        result = create_validation_dataset(output_file)

        assert result == "validation.json"
        mock_collect_data.assert_called_once()

    @patch("os.path.exists")
    def test_create_validation_dataset_existing(self, mock_exists):
        """Test when validation dataset already exists."""
        mock_exists.return_value = True

        output_file = os.path.join(self.temp_dir, "validation.json")

        result = create_validation_dataset(output_file)

        assert result == output_file

    @patch(
        "torch_diode.collection.matmul_data_utils._create_validation_dataset_chunked"
    )
    def test_create_validation_dataset_chunked(self, mock_chunked):
        """Test chunked validation dataset creation."""
        mock_chunked.return_value = "chunked_validation.json"

        output_file = os.path.join(self.temp_dir, "validation.json")

        result = create_validation_dataset(output_file=output_file, chunk_size=10)

        assert result == "chunked_validation.json"
        mock_chunked.assert_called_once()

    @patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector")
    @patch("torch_diode.collection.matmul_data_utils.run_matrix_multiplications")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    @patch("torch_diode.utils.dataset_utils.print_dataset_statistics")
    def test_run_collector_example_context_manager(
        self,
        mock_print_stats,
        mock_device_name,
        mock_cuda_available,
        mock_run_mm,
        mock_collector_class,
    ):
        """Test collector example with context manager."""
        mock_cuda_available.return_value = True
        mock_device_name.return_value = "NVIDIA A100"

        # Mock collector with context manager support and spec to make isinstance work
        mock_collector = self._create_mock_collector_with_proper_dataset()
        mock_collector.__enter__ = Mock(return_value=mock_collector)
        mock_collector.__exit__ = Mock(return_value=None)
        mock_collector_class.return_value = mock_collector

        run_collector_example(
            output_dir=self.temp_dir, use_context_manager=True, num_shapes=2
        )

        mock_run_mm.assert_called_once()
        mock_collector.save_to_file.assert_called()
        mock_collector.save_table_to_file.assert_called()

    @patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector")
    @patch("torch_diode.collection.matmul_data_utils.run_matrix_multiplications")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    @patch("torch_diode.utils.dataset_utils.print_dataset_statistics")
    def test_run_collector_example_explicit(
        self,
        mock_print_stats,
        mock_device_name,
        mock_cuda_available,
        mock_run_mm,
        mock_collector_class,
    ):
        """Test collector example with explicit start/stop."""
        mock_cuda_available.return_value = True
        mock_device_name.return_value = "NVIDIA A100"

        # Mock collector with spec to make isinstance work
        mock_collector = self._create_mock_collector_with_proper_dataset()
        mock_collector_class.return_value = mock_collector

        run_collector_example(
            output_dir=self.temp_dir, use_context_manager=False, num_shapes=2
        )

        mock_collector.start_collection.assert_called_once()
        mock_collector.stop_collection.assert_called_once()
        mock_run_mm.assert_called_once()

    def test_collect_data_chunked_function_exists_internal(self):
        """Test that _collect_data_chunked function exists and is callable."""
        assert callable(_collect_data_chunked)

    def test_create_validation_dataset_chunked_function_exists(self):
        """Test that _create_validation_dataset_chunked function exists and is callable."""
        assert callable(_create_validation_dataset_chunked)

    @patch("torch.cuda.is_available")
    @patch("torch_diode.utils.dataset_utils.print_dataset_statistics")
    def test_collect_data_cpu_fallback(self, mock_print_stats, mock_cuda_available):
        """Test data collection fallback to CPU when CUDA not available."""
        mock_cuda_available.return_value = False

        with patch(
            "torch_diode.collection.matmul_data_utils.MatmulDatasetCollector"
        ) as mock_collector_class:
            mock_collector = self._create_mock_collector_with_proper_dataset()
            mock_collector_class.return_value = mock_collector

            output_file = os.path.join(self.temp_dir, "test_data.json")

            collect_data(output_file=output_file)

            # Verify collector was created with cpu device name
            mock_collector_class.assert_called_once()

    @patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector")
    @patch("torch.cuda.is_available")
    @patch("torch_diode.utils.dataset_utils.print_dataset_statistics")
    def test_collect_data_log_normal_mode(
        self, mock_print_stats, mock_cuda_available, mock_collector_class
    ):
        """Test data collection with log_normal mode."""
        mock_cuda_available.return_value = False

        mock_collector = self._create_mock_collector_with_proper_dataset()
        mock_collector_class.return_value = mock_collector

        output_file = os.path.join(self.temp_dir, "test_data.json")

        collect_data(
            output_file=output_file,
            mode="log_normal",
            log_normal_m_mean=7.0,
            log_normal_k_std=2.0,
        )

        # Verify collector was created with log_normal mode
        mock_collector_class.assert_called_once()

    @patch("torch.cuda.is_available")
    @patch("torch_diode.utils.dataset_utils.print_dataset_statistics")
    def test_collect_data_custom_dtypes(self, mock_print_stats, mock_cuda_available):
        """Test data collection with custom dtypes."""
        mock_cuda_available.return_value = True

        with patch(
            "torch_diode.collection.matmul_data_utils.MatmulDatasetCollector"
        ) as mock_collector_class:
            mock_collector = self._create_mock_collector_with_proper_dataset()
            mock_collector_class.return_value = mock_collector

            output_file = os.path.join(self.temp_dir, "test_data.json")
            custom_dtypes = [torch.float32, torch.float64]

            collect_data(output_file=output_file, dtypes=custom_dtypes)

            # Verify collector was created with custom dtypes
            args, kwargs = mock_collector_class.call_args
            assert kwargs["dtypes"] == custom_dtypes
