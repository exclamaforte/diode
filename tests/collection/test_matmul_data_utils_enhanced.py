"""
Enhanced tests for diode.collection.matmul_data_utils module to improve coverage.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open
import pytest
import torch
import msgpack

from torch_diode.collection.matmul_data_utils import (
    _collect_data_chunked,
    _create_validation_dataset_chunked,
    collect_data,
    create_validation_dataset,
    run_collector_example,
    run_matrix_multiplications,
)
from torch_diode.collection.matmul_dataset_collector import CollectionMode


class TestMatmulDataUtilsEnhanced:
    """Enhanced test class for matmul data utility functions."""

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
        mock_solution_dict_data = {mock_problem: mock_solution}  # Use mock_problem as key
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
        from torch_diode.collection.matmul_dataset_collector import MatmulDatasetCollector

        # Create a mock collector that will pass isinstance check
        mock_collector = Mock(spec=MatmulDatasetCollector)
        mock_dataset = self._create_mock_dataset_with_proper_structure()
        mock_collector.get_dataset.return_value = mock_dataset
        mock_collector.collect_data = Mock()
        mock_collector.save_to_file = Mock()
        mock_collector.save_table_to_file = Mock()
        mock_collector._generate_shapes_and_dtypes = Mock(return_value=[
            ((32, 64, 128), torch.float32, "mm"),
            ((64, 128, 256), torch.float16, "addmm"),
        ])
        mock_collector.hardware_name = "test_gpu"
        mock_collector.mode = CollectionMode.RANDOM
        mock_collector.operations = ["mm", "addmm"]
        mock_collector.operation_shape_set = None
        mock_collector.num_shapes = 10
        mock_collector.dtypes = [torch.float32]
        mock_collector.seed = 42
        mock_collector.min_size = 32
        mock_collector.max_size = 1024
        mock_collector.power_of_two = False
        mock_collector._is_collecting = False
        mock_collector.start_collection = Mock()
        mock_collector.stop_collection = Mock()
        mock_collector._run_matrix_multiplication = Mock()

        return mock_collector

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    @patch("torch_diode.collection.matmul_data_utils.print_dataset_statistics")
    def test_collect_data_operation_shape_set_mode(self, mock_print_stats, mock_device_name, mock_cuda_available):
        """Test data collection with operation_shape_set mode."""
        mock_cuda_available.return_value = True
        mock_device_name.return_value = "NVIDIA A100"
        
        # Create a mock operation shape set
        from torch_diode.types.matmul_types import OperationShapeSet
        mock_shape_set = Mock(spec=OperationShapeSet)

        with patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector") as mock_collector_class:
            mock_collector = self._create_mock_collector_with_proper_dataset()
            mock_collector_class.return_value = mock_collector

            output_file = os.path.join(self.temp_dir, "test_data.json")

            collect_data(
                output_file=output_file,
                mode="operation_shape_set",
                operation_shape_set=mock_shape_set,
            )

            # Verify collector was created with operation_shape_set mode
            mock_collector_class.assert_called_once()
            args, kwargs = mock_collector_class.call_args
            assert kwargs["mode"] == CollectionMode.OPERATION_SHAPE_SET
            assert kwargs["operation_shape_set"] == mock_shape_set

    @patch("torch.cuda.is_available")
    @patch("torch_diode.collection.matmul_data_utils.print_dataset_statistics")
    def test_collect_data_power_of_two_mode(self, mock_print_stats, mock_cuda_available):
        """Test data collection with power_of_two enabled."""
        mock_cuda_available.return_value = False

        with patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector") as mock_collector_class:
            mock_collector = self._create_mock_collector_with_proper_dataset()
            mock_collector_class.return_value = mock_collector

            output_file = os.path.join(self.temp_dir, "test_data.json")

            collect_data(
                output_file=output_file,
                power_of_two=True,
                search_space="DEFAULT"
            )

            # Verify collector was created with power_of_two enabled
            args, kwargs = mock_collector_class.call_args
            assert kwargs["power_of_two"] is True

    @patch("torch.cuda.is_available")
    @patch("torch_diode.collection.matmul_data_utils.print_dataset_statistics")
    def test_collect_data_custom_operations(self, mock_print_stats, mock_cuda_available):
        """Test data collection with custom operations."""
        mock_cuda_available.return_value = False

        with patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector") as mock_collector_class:
            mock_collector = self._create_mock_collector_with_proper_dataset()
            mock_collector_class.return_value = mock_collector

            output_file = os.path.join(self.temp_dir, "test_data.json")
            custom_operations = ["mm", "bmm"]

            collect_data(
                output_file=output_file,
                operations=custom_operations,
            )

            # Verify collector was created with custom operations
            args, kwargs = mock_collector_class.call_args
            assert kwargs["operations"] == custom_operations

    @patch("torch.cuda.is_available")
    @patch("torch_diode.collection.matmul_data_utils.print_dataset_statistics")
    def test_collect_data_log_normal_parameters(self, mock_print_stats, mock_cuda_available):
        """Test data collection with log normal distribution parameters."""
        mock_cuda_available.return_value = False

        with patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector") as mock_collector_class:
            mock_collector = self._create_mock_collector_with_proper_dataset()
            mock_collector_class.return_value = mock_collector

            output_file = os.path.join(self.temp_dir, "test_data.json")

            collect_data(
                output_file=output_file,
                mode="log_normal",
                log_normal_m_mean=7.0,
                log_normal_m_std=2.5,
                log_normal_n_mean=6.0,
                log_normal_n_std=1.8,
                log_normal_k_mean=6.5,
                log_normal_k_std=2.2,
            )

            # Verify collector was created with log normal parameters
            args, kwargs = mock_collector_class.call_args
            assert kwargs["log_normal_m_mean"] == 7.0
            assert kwargs["log_normal_m_std"] == 2.5
            assert kwargs["log_normal_n_mean"] == 6.0
            assert kwargs["log_normal_n_std"] == 1.8
            assert kwargs["log_normal_k_mean"] == 6.5
            assert kwargs["log_normal_k_std"] == 2.2

    @patch("torch_diode.collection.matmul_data_utils._collect_data_chunked")
    @patch("torch.cuda.is_available")
    def test_collect_data_chunked_with_parameters(self, mock_cuda_available, mock_chunked):
        """Test chunked data collection with all parameters."""
        mock_cuda_available.return_value = True
        mock_chunked.return_value = "chunked_output.json"

        with patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector") as mock_collector_class:
            mock_collector = self._create_mock_collector_with_proper_dataset()
            mock_collector_class.return_value = mock_collector

            output_file = os.path.join(self.temp_dir, "test_data.json")

            result = collect_data(
                output_file=output_file,
                chunk_size=25,
                search_mode="reduce-overhead",
                search_space="DEFAULT",
                file_format="msgpack"
            )

            assert result == "chunked_output.json"
            mock_chunked.assert_called_once()
            args, kwargs = mock_chunked.call_args
            assert kwargs["chunk_size"] == 25
            assert kwargs["search_mode"] == "reduce-overhead"
            assert kwargs["search_space"] == "DEFAULT"
            assert kwargs["file_format"] == "msgpack"

    @patch("torch_diode.collection.matmul_data_utils._create_validation_dataset_chunked")
    def test_create_validation_dataset_chunked_with_all_params(self, mock_chunked):
        """Test chunked validation dataset creation with all parameters."""
        mock_chunked.return_value = "chunked_validation.json"

        output_file = os.path.join(self.temp_dir, "validation.json")

        result = create_validation_dataset(
            output_file=output_file,
            chunk_size=15,
            mode="log_normal",
            num_shapes=50,
            dtypes=[torch.float32, torch.float64],
            seed=123,
            min_size=64,
            max_size=2048,
            power_of_two=True,
            search_mode="reduce-overhead",
            search_space="DEFAULT",
            file_format="msgpack",
            log_normal_m_mean=7.5,
            log_normal_m_std=2.8,
            log_normal_n_mean=6.2,
            log_normal_n_std=1.9,
            log_normal_k_mean=6.8,
            log_normal_k_std=2.4,
        )

        assert result == "chunked_validation.json"
        mock_chunked.assert_called_once()

    @patch("torch.cuda.is_available")
    def test_run_collector_example_cpu_device(self, mock_cuda_available):
        """Test collector example with CPU device."""
        mock_cuda_available.return_value = False

        with patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector") as mock_collector_class:
            with patch("torch_diode.collection.matmul_data_utils.run_matrix_multiplications") as mock_run_mm:
                mock_collector = self._create_mock_collector_with_proper_dataset()
                mock_collector.__enter__ = Mock(return_value=mock_collector)
                mock_collector.__exit__ = Mock(return_value=None)
                mock_collector_class.return_value = mock_collector

                run_collector_example(
                    output_dir=self.temp_dir,
                    use_context_manager=True,
                    num_shapes=3,
                    dtypes=[torch.float32]
                )

                mock_run_mm.assert_called_once()
                # Verify that CPU was used since CUDA is not available
                args, kwargs = mock_collector_class.call_args
                assert kwargs["hardware_name"] == "cpu"

    @patch("torch.cuda.is_available")
    def test_run_collector_example_custom_dtypes(self, mock_cuda_available):
        """Test collector example with custom dtypes."""
        mock_cuda_available.return_value = True

        with patch("torch.cuda.get_device_name") as mock_device_name:
            mock_device_name.return_value = "NVIDIA RTX 3080"
            
            with patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector") as mock_collector_class:
                with patch("torch_diode.collection.matmul_data_utils.run_matrix_multiplications") as mock_run_mm:
                    mock_collector = self._create_mock_collector_with_proper_dataset()
                    mock_collector_class.return_value = mock_collector

                    custom_dtypes = [torch.float64, torch.float32]
                    run_collector_example(
                        output_dir=self.temp_dir,
                        use_context_manager=False,
                        num_shapes=2,
                        dtypes=custom_dtypes
                    )

                    # Verify run_matrix_multiplications was called with custom dtypes
                    mock_run_mm.assert_called_once()
                    args, kwargs = mock_run_mm.call_args
                    assert args[1] == custom_dtypes  # dtypes are second argument

    def test_collect_data_chunked_mock_exists_files(self):
        """Test _collect_data_chunked with existing files."""
        with patch("os.path.exists") as mock_exists:
            with patch("builtins.open", mock_open(read_data='{"test": "data"}')) as mock_file:
                with patch("json.load") as mock_json_load:
                    with patch("torch_diode.types.matmul_dataset.Dataset.from_dict") as mock_from_dict:
                        # Mock existing files
                        mock_exists.side_effect = lambda path: path.endswith("_1.json")
                        
                        # Mock dataset structure
                        mock_dataset = Mock()
                        mock_hardware = Mock()
                        mock_operation = Mock()
                        mock_solution = {"solution1": Mock()}
                        mock_operation.solution = mock_solution
                        mock_hardware.operation = {"op1": mock_operation}
                        mock_dataset.hardware = {"hw1": mock_hardware}
                        mock_from_dict.return_value = mock_dataset
                        mock_json_load.return_value = {"test": "data"}
                        
                        # Mock collector
                        mock_collector = self._create_mock_collector_with_proper_dataset()
                        mock_collector._generate_shapes_and_dtypes.return_value = []  # No remaining work
                        
                        result = _collect_data_chunked(
                            collector=mock_collector,
                            output_file=os.path.join(self.temp_dir, "test.json"),
                            chunk_size=10,
                            search_mode="max-autotune",
                            search_space="DEFAULT",
                            file_format="json"
                        )
                        
                        # Should return the first existing chunk file
                        assert result.endswith("_1.json")

    def test_collect_data_chunked_mock_msgpack_format(self):
        """Test _collect_data_chunked with msgpack format and existing files."""
        with patch("os.path.exists") as mock_exists:
            with patch("builtins.open", mock_open()) as mock_file:
                with patch("msgpack.unpack") as mock_msgpack_unpack:
                    with patch("torch_diode.types.matmul_dataset.Dataset.from_dict") as mock_from_dict:
                        # Mock existing files
                        mock_exists.side_effect = lambda path: path.endswith("_1.msgpack")
                        
                        # Mock dataset structure
                        mock_dataset = Mock()
                        mock_hardware = Mock()
                        mock_operation = Mock()
                        mock_solution = {"solution1": Mock(), "solution2": Mock()}
                        mock_operation.solution = mock_solution
                        mock_hardware.operation = {"op1": mock_operation}
                        mock_dataset.hardware = {"hw1": mock_hardware}
                        mock_from_dict.return_value = mock_dataset
                        mock_msgpack_unpack.return_value = {"test": "data"}
                        
                        # Mock collector
                        mock_collector = self._create_mock_collector_with_proper_dataset()
                        mock_collector._generate_shapes_and_dtypes.return_value = []  # No remaining work
                        
                        base_file = os.path.join(self.temp_dir, "test.json")
                        result = _collect_data_chunked(
                            collector=mock_collector,
                            output_file=base_file,
                            chunk_size=10,
                            search_mode="max-autotune",
                            search_space="DEFAULT",  
                            file_format="msgpack"
                        )
                        
                        # Should return the first existing chunk file with msgpack extension
                        assert result.endswith("_1.msgpack")

    def test_collect_data_chunked_resumption_with_work_remaining(self):
        """Test _collect_data_chunked with resumption and remaining work."""
        with patch("os.path.exists") as mock_exists:
            with patch("builtins.open", mock_open()) as mock_file:
                with patch("json.load") as mock_json_load:
                    with patch("torch_diode.types.matmul_dataset.Dataset.from_dict") as mock_from_dict:
                        with patch("torch.set_grad_enabled"):
                            with patch("torch._inductor.config"):
                                with patch("torch.cuda.is_available", return_value=True):
                                    # Mock existing files
                                    mock_exists.side_effect = lambda path: path.endswith("_1.json")
                                    
                                    # Mock dataset structure with 1 operation completed
                                    mock_dataset = Mock()
                                    mock_hardware = Mock()
                                    mock_operation = Mock()
                                    mock_solution = {"solution1": Mock()}  # 1 completed operation
                                    mock_operation.solution = mock_solution
                                    mock_hardware.operation = {"op1": mock_operation}
                                    mock_dataset.hardware = {"hw1": mock_hardware}
                                    mock_from_dict.return_value = mock_dataset
                                    mock_json_load.return_value = {"test": "data"}
                                    
                                    # Mock collector with remaining work
                                    mock_collector = self._create_mock_collector_with_proper_dataset()
                                    remaining_work = [
                                        ((64, 128, 256), torch.float16, "mm"),
                                        ((128, 256, 512), torch.float32, "addmm"),
                                    ]
                                    mock_collector._generate_shapes_and_dtypes.return_value = [
                                        ((32, 64, 128), torch.float32, "mm"),  # This was already completed
                                        *remaining_work
                                    ]
                                    
                                    with patch("torch._dynamo.reset"):
                                        result = _collect_data_chunked(
                                            collector=mock_collector,
                                            output_file=os.path.join(self.temp_dir, "test.json"),
                                            chunk_size=10,
                                            search_mode="max-autotune",
                                            search_space="DEFAULT",
                                            file_format="json"
                                        )
                                        
                                        # Should return the first chunk file
                                        assert result.endswith("_1.json")
                                        
                                        # Should have started collection
                                        mock_collector.start_collection.assert_called()

    def test_create_validation_dataset_chunked_directory_handling(self):
        """Test _create_validation_dataset_chunked directory creation and handling."""
        with patch("os.makedirs") as mock_makedirs:
            with patch("os.path.exists") as mock_exists:
                with patch("torch.cuda.is_available", return_value=False):
                    with patch("torch.cuda.get_device_name", return_value="cpu"):
                        # Mock no existing files
                        mock_exists.return_value = False
                        
                        # Mock collector
                        mock_collector_class = Mock()
                        mock_collector = self._create_mock_collector_with_proper_dataset()
                        mock_collector_class.return_value = mock_collector
                        mock_collector._generate_shapes_and_dtypes.return_value = []
                        
                        with patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector", mock_collector_class):
                            with patch("torch.set_grad_enabled"):
                                with patch("torch._inductor.config"):
                                    # Test with directory path
                                    result = _create_validation_dataset_chunked(
                                        output_path=self.temp_dir,
                                        mode="random",
                                        num_shapes=2,
                                        dtypes=[torch.float32],
                                        seed=42,
                                        min_size=32,
                                        max_size=1024,
                                        power_of_two=False,
                                        search_mode="max-autotune",
                                        search_space="DEFAULT",
                                        file_format="json",
                                        chunk_size=1,
                                        log_normal_m_mean=6.5,
                                        log_normal_m_std=2.5,
                                        log_normal_n_mean=5.9,
                                        log_normal_n_std=1.7,
                                        log_normal_k_mean=6.2,
                                        log_normal_k_std=2.2,
                                    )
                                    
                                    # Should have created the directory
                                    mock_makedirs.assert_called_with(self.temp_dir, exist_ok=True)
                                    
                                    # Should return a path in the specified directory
                                    assert self.temp_dir in result


    @patch("torch.compile")
    @patch("torch.randn")
    def test_run_matrix_multiplications_multiple_search_modes(self, mock_randn, mock_compile):
        """Test run_matrix_multiplications with different search modes."""
        # Mock tensor creation - need multiple tensors for all matrix operations
        mock_tensor_a = Mock()
        mock_tensor_b = Mock()
        mock_tensor_c = Mock()
        mock_randn.side_effect = [mock_tensor_a, mock_tensor_b, mock_tensor_c] * 2  # For mm and addmm operations
        
        # Mock compiled function that returns a mock result
        mock_result = Mock()
        mock_compiled_fn = Mock(return_value=mock_result)
        mock_compile.return_value = mock_compiled_fn

        sizes = [(64, 128, 256)]
        dtypes = [torch.float32]

        # Test with different search mode
        run_matrix_multiplications(sizes, dtypes, device="cpu", search_mode="reduce-overhead")

        # Verify torch.compile was called with the correct search mode (should be called twice: mm and addmm)
        assert mock_compile.call_count == 2
        calls = mock_compile.call_args_list
        for call in calls:
            args, kwargs = call
            assert kwargs["mode"] == "reduce-overhead"
        
        # Verify the compiled functions were actually executed
        assert mock_compiled_fn.call_count == 2

    def test_collect_data_with_none_operations(self):
        """Test collect_data when operations is None (uses default)."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector") as mock_collector_class:
                mock_collector = self._create_mock_collector_with_proper_dataset()
                mock_collector_class.return_value = mock_collector

                output_file = os.path.join(self.temp_dir, "test_data.json")

                # Don't specify operations parameter (should use default)
                collect_data(output_file=output_file)

                # Verify collector was created with default operations
                args, kwargs = mock_collector_class.call_args
                assert kwargs["operations"] == ["mm", "addmm", "bmm"]

    def test_collect_data_with_none_dtypes_cuda(self):
        """Test collect_data when dtypes is None with CUDA available."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA A100"):
                with patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector") as mock_collector_class:
                    mock_collector = self._create_mock_collector_with_proper_dataset()
                    mock_collector_class.return_value = mock_collector

                    output_file = os.path.join(self.temp_dir, "test_data.json")

                    # Don't specify dtypes parameter (should use CUDA defaults)
                    collect_data(output_file=output_file)

                    # Verify collector was created with CUDA default dtypes
                    args, kwargs = mock_collector_class.call_args
                    assert kwargs["dtypes"] == [torch.float16, torch.bfloat16]

    def test_collect_data_with_none_dtypes_cpu(self):  
        """Test collect_data when dtypes is None with CPU only."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector") as mock_collector_class:
                mock_collector = self._create_mock_collector_with_proper_dataset()
                mock_collector_class.return_value = mock_collector

                output_file = os.path.join(self.temp_dir, "test_data.json")

                # Don't specify dtypes parameter (should use CPU defaults)
                collect_data(output_file=output_file)

                # Verify collector was created with CPU default dtypes
                args, kwargs = mock_collector_class.call_args
                assert kwargs["dtypes"] == [torch.float32]

    def test_chunked_validation_dataset_with_file_path(self):
        """Test _create_validation_dataset_chunked with file path instead of directory."""
        output_file = os.path.join(self.temp_dir, "validation_data.json")
        
        with patch("os.makedirs") as mock_makedirs:
            with patch("os.path.exists", return_value=False):
                with patch("torch.cuda.is_available", return_value=False):
                    with patch("torch.cuda.get_device_name", return_value="cpu"):
                        # Mock collector
                        mock_collector_class = Mock()
                        mock_collector = self._create_mock_collector_with_proper_dataset()
                        mock_collector_class.return_value = mock_collector
                        mock_collector._generate_shapes_and_dtypes.return_value = []
                        
                        with patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector", mock_collector_class):
                            with patch("torch.set_grad_enabled"):
                                with patch("torch._inductor.config"):
                                    result = _create_validation_dataset_chunked(
                                        output_path=output_file,
                                        mode="random",
                                        num_shapes=3,
                                        dtypes=[torch.float32],
                                        seed=42,
                                        min_size=32,
                                        max_size=512,
                                        power_of_two=False,
                                        search_mode="max-autotune",
                                        search_space="DEFAULT",
                                        file_format="json",
                                        chunk_size=1,
                                        log_normal_m_mean=6.5,
                                        log_normal_m_std=2.5,
                                        log_normal_n_mean=5.9,
                                        log_normal_n_std=1.7,
                                        log_normal_k_mean=6.2,
                                        log_normal_k_std=2.2,
                                    )
                                    
                                    # Should have created the parent directory
                                    expected_dir = os.path.dirname(output_file)
                                    if expected_dir == "":
                                        expected_dir = "."
                                    mock_makedirs.assert_called_with(expected_dir, exist_ok=True)
                                    
                                    # Should return a path with the base name from the file
                                    assert "validation_data" in result

    def test_chunked_collection_with_pytorch_config(self):
        """Test chunked collection properly configures PyTorch settings."""
        with patch("os.path.exists", return_value=False):
            with patch("torch.set_grad_enabled") as mock_set_grad:
                with patch("torch._inductor.config") as mock_config:
                    with patch("torch.cuda.is_available", return_value=True):
                        # Mock collector
                        mock_collector = self._create_mock_collector_with_proper_dataset()
                        mock_collector._generate_shapes_and_dtypes.return_value = [
                            ((32, 64, 128), torch.float32, "mm"),
                        ]
                        
                        with patch("torch._dynamo.reset"):
                            result = _collect_data_chunked(
                                collector=mock_collector,
                                output_file=os.path.join(self.temp_dir, "test.json"),
                                chunk_size=10,
                                search_mode="max-autotune",
                                search_space="EXHAUSTIVE",
                                file_format="json"
                            )
                            
                            # Verify PyTorch settings were configured
                            mock_set_grad.assert_called_with(False)
                            assert hasattr(mock_config, 'fx_graph_cache')
                            assert hasattr(mock_config, 'force_disable_caches')
                            assert hasattr(mock_config, 'max_autotune_gemm_backends')

    def test_run_collector_example_with_none_dtypes_cuda(self):
        """Test run_collector_example with None dtypes and CUDA available."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA A100"):
                with patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector") as mock_collector_class:
                    with patch("torch_diode.collection.matmul_data_utils.run_matrix_multiplications") as mock_run_mm:
                        mock_collector = self._create_mock_collector_with_proper_dataset()
                        mock_collector.__enter__ = Mock(return_value=mock_collector)
                        mock_collector.__exit__ = Mock(return_value=None)
                        mock_collector_class.return_value = mock_collector

                        # Call with dtypes=None (should use CUDA defaults)
                        run_collector_example(
                            output_dir=self.temp_dir,
                            use_context_manager=True,
                            num_shapes=2,
                            dtypes=None
                        )

                        # Verify run_matrix_multiplications was called with CUDA default dtypes
                        mock_run_mm.assert_called_once()
                        args, kwargs = mock_run_mm.call_args
                        assert args[1] == [torch.float16, torch.float32]  # dtypes are second argument

    def test_run_collector_example_with_none_dtypes_cpu(self):
        """Test run_collector_example with None dtypes and CPU only."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector") as mock_collector_class:
                with patch("torch_diode.collection.matmul_data_utils.run_matrix_multiplications") as mock_run_mm:
                    mock_collector = self._create_mock_collector_with_proper_dataset()
                    mock_collector_class.return_value = mock_collector

                    # Call with dtypes=None (should use CPU defaults)
                    run_collector_example(
                        output_dir=self.temp_dir,
                        use_context_manager=False,
                        num_shapes=2,
                        dtypes=None
                    )

                    # Verify run_matrix_multiplications was called with CPU default dtypes  
                    mock_run_mm.assert_called_once()
                    args, kwargs = mock_run_mm.call_args
                    assert args[1] == [torch.float32]  # dtypes are second argument
