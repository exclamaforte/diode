import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys
import torch
from collections import OrderedDict

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from torch_diode.collection.matmul_dataset_collector import MatmulDatasetCollector
from torch_diode.types.matmul_dataset import (
    Dataset,
    TimedConfig,
)
from torch_diode.types.matmul_types import (
    TritonGEMMConfig,
    MMShape,
    Table,
)


class TestMatmulDatasetCollector(unittest.TestCase):
    def setUp(self):
        # Create a collector instance for each test
        self.collector = MatmulDatasetCollector(hardware_name="test_gpu")

    def test_initialization(self):
        """Test that the collector initializes correctly."""
        self.assertEqual(self.collector.hardware_name, "test_gpu")
        self.assertIsInstance(self.collector.dataset, Dataset)
        self.assertFalse(self.collector._is_collecting)
        self.assertEqual(len(self.collector.dataset.hardware), 0)

    @patch('torch_diode.collection.matmul_dataset_collector.add_feedback_saver')
    def test_start_collection(self, mock_add_feedback_saver):
        """Test starting collection."""
        self.collector.start_collection()
        
        # Check that add_feedback_saver was called with the feedback handler
        mock_add_feedback_saver.assert_called_once()
        self.assertTrue(self.collector._is_collecting)
        
        # Starting collection again should not call add_feedback_saver again
        self.collector.start_collection()
        mock_add_feedback_saver.assert_called_once()

    @patch('torch_diode.collection.matmul_dataset_collector.clear_feedback_savers')
    def test_stop_collection(self, mock_clear_feedback_savers):
        """Test stopping collection."""
        # Set _is_collecting to True to simulate an active collection
        self.collector._is_collecting = True
        
        self.collector.stop_collection()
        
        # Check that clear_feedback_savers was called
        mock_clear_feedback_savers.assert_called_once()
        self.assertFalse(self.collector._is_collecting)
        
        # Stopping collection again should not call clear_feedback_savers again
        self.collector.stop_collection()
        mock_clear_feedback_savers.assert_called_once()

    def test_feedback_handler_mm(self):
        """Test the feedback handler with mm operation."""
        # Create mock input nodes
        mock_node0 = MagicMock()
        mock_node0.layout.size = [64, 128]
        mock_node0.layout.dtype = torch.float16
        
        mock_node1 = MagicMock()
        mock_node1.layout.size = [128, 32]
        mock_node1.layout.dtype = torch.float16
        
        input_nodes = [mock_node0, mock_node1]
        
        # Create mock TritonTemplateCaller
        mock_choice = MagicMock(spec=torch._inductor.select_algorithm.TritonTemplateCaller)
        mock_choice.log_info = {
            'tile_shape': '(32,16,64)',
            'num_stages': 2,
            'num_warps': 4,
            'GROUP_M': 8
        }
        
        # Create timings dictionary
        timings = {mock_choice: 0.001}
        
        # Call the feedback handler
        self.collector._feedback_handler(timings, "mm", input_nodes, None, 0.1)
        
        # Check that the dataset was updated correctly
        self.assertIn("test_gpu", self.collector.dataset.hardware)
        hardware = self.collector.dataset.hardware["test_gpu"]
        self.assertIn("mm", hardware.operation)
        operation = hardware.operation["mm"]
        
        # Check that we have one solution
        self.assertEqual(len(operation.solution), 1)
        
        # Get the problem and solution
        problem = next(iter(operation.solution.keys()))
        solution = operation.solution[problem]
        
        # Check problem dimensions
        self.assertEqual(problem.M, 64)
        self.assertEqual(problem.K, 128)
        self.assertEqual(problem.N, 32)
        self.assertEqual(problem.M_dtype, torch.float16)
        self.assertEqual(problem.K_dtype, torch.float16)
        
        # Check solution config
        self.assertEqual(len(solution.timed_configs), 1)
        timed_config = solution.timed_configs[0]
        config = timed_config.config
        self.assertEqual(config.block_m, 32)
        self.assertEqual(config.block_k, 16)
        self.assertEqual(config.block_n, 64)
        self.assertEqual(config.num_stages, 2)
        self.assertEqual(config.num_warps, 4)
        self.assertEqual(config.group_m, 8)
        self.assertEqual(timed_config.time, 0.001)

    def test_feedback_handler_addmm(self):
        """Test the feedback handler with addmm operation."""
        # Create mock input nodes
        mock_node0 = MagicMock()  # bias
        
        mock_node1 = MagicMock()  # input
        mock_node1.layout.size = [64, 128]
        mock_node1.layout.dtype = torch.float32
        
        mock_node2 = MagicMock()  # weight
        mock_node2.layout.size = [128, 32]
        mock_node2.layout.dtype = torch.float32
        
        input_nodes = [mock_node0, mock_node1, mock_node2]
        
        # Create mock TritonTemplateCaller
        mock_choice = MagicMock(spec=torch._inductor.select_algorithm.TritonTemplateCaller)
        mock_choice.log_info = {
            'tile_shape': '(16,32,64)',
            'num_stages': 3,
            'num_warps': 8,
            'GROUP_M': 4
        }
        
        # Create timings dictionary
        timings = {mock_choice: 0.002}
        
        # Call the feedback handler
        self.collector._feedback_handler(timings, "addmm", input_nodes, None, 0.1)
        
        # Check that the dataset was updated correctly
        self.assertIn("test_gpu", self.collector.dataset.hardware)
        hardware = self.collector.dataset.hardware["test_gpu"]
        self.assertIn("addmm", hardware.operation)
        operation = hardware.operation["addmm"]
        
        # Check that we have one solution
        self.assertEqual(len(operation.solution), 1)
        
        # Get the problem and solution
        problem = next(iter(operation.solution.keys()))
        solution = operation.solution[problem]
        
        # Check problem dimensions
        self.assertEqual(problem.M, 64)
        self.assertEqual(problem.K, 128)
        self.assertEqual(problem.N, 32)
        self.assertEqual(problem.M_dtype, torch.float32)
        self.assertEqual(problem.K_dtype, torch.float32)
        
        # Check solution config
        self.assertEqual(len(solution.timed_configs), 1)
        timed_config = solution.timed_configs[0]
        config = timed_config.config
        self.assertEqual(config.block_m, 16)
        self.assertEqual(config.block_k, 32)
        self.assertEqual(config.block_n, 64)
        self.assertEqual(config.num_stages, 3)
        self.assertEqual(config.num_warps, 8)
        self.assertEqual(config.group_m, 4)
        self.assertEqual(timed_config.time, 0.002)

    def test_to_table(self):
        """Test converting the dataset to a table."""
        # Create a problem
        problem = MMShape(
            B=1,
            M=128,
            N=64,
            K=256,
            M_dtype=torch.float16,
            K_dtype=torch.float16,
            out_dtype=torch.float16,
            out_size=(1, 128, 64),
            out_stride=(128 * 64, 64, 1),
        )
        
        # Create configs
        config1 = TritonGEMMConfig(
            name="mm_config",
            grid=1,
            block_m=32,
            block_n=32,
            block_k=16,
            group_m=8,
            num_stages=2,
            num_warps=4,
        )
        
        config2 = TritonGEMMConfig(
            name="mm_config",
            grid=1,
            block_m=64,
            block_n=64,
            block_k=32,
            group_m=4,
            num_stages=3,
            num_warps=8,
        )
        
        # Add timings to the dataset
        self.collector.dataset.add_timing(
            hardware_name="test_gpu",
            op_name="mm",
            problem=problem,
            config=config1,
            time=0.001,
        )
        
        self.collector.dataset.add_timing(
            hardware_name="test_gpu",
            op_name="mm",
            problem=problem,
            config=config2,
            time=0.0005,  # Faster
        )
        
        # Convert to table
        table = self.collector.to_table()
        
        # Check that the table was created correctly
        self.assertIn("test_gpu", table.hardware)
        hardware = table.hardware["test_gpu"]
        
        self.assertIn("mm", hardware.operation)
        operation = hardware.operation["mm"]
        
        self.assertIn(problem, operation.solution)
        solution = operation.solution[problem]
        
        # Check that all configs are included in the solution
        self.assertEqual(len(solution.config), 2)
        self.assertEqual(solution.config[0], config2)  # Faster config first
        self.assertEqual(solution.config[1], config1)

    @patch('builtins.open', new_callable=mock_open)
    def test_save_to_file(self, mock_file):
        """Test saving the dataset to a file."""
        # Add some data to the dataset
        problem = MMShape(
            B=1,
            M=128,
            N=64,
            K=256,
            M_dtype=torch.float16,
            K_dtype=torch.float16,
            out_dtype=torch.float16,
            out_size=(1, 128, 64),
            out_stride=(128 * 64, 64, 1),
        )
        
        config = TritonGEMMConfig(
            name="mm_config",
            grid=1,
            block_m=32,
            block_n=32,
            block_k=16,
            group_m=8,
            num_stages=2,
            num_warps=4,
        )
        
        self.collector.dataset.add_timing(
            hardware_name="test_gpu",
            op_name="mm",
            problem=problem,
            config=config,
            time=0.001,
        )
        
        # Save the dataset to a file
        self.collector.save_to_file("test_file.json")
        
        # Check that the file was opened and written to
        mock_file.assert_called_once_with("test_file.json", "w")
        mock_file().write.assert_called_once()
        
        # The write call should contain the serialized dataset
        serialized_dataset = self.collector.dataset.serialize()
        mock_file().write.assert_called_once_with(serialized_dataset)

    @patch('builtins.open', new_callable=mock_open, read_data='{"hardware": {}}')
    @patch('torch_diode.collection.matmul_dataset_collector.Dataset.deserialize')
    def test_load_from_file(self, mock_deserialize, mock_file):
        """Test loading the dataset from a file."""
        # Create a mock dataset to return from deserialize
        mock_dataset = MagicMock(spec=Dataset)
        mock_deserialize.return_value = mock_dataset
        
        # Load the dataset from a file
        self.collector.load_from_file("test_file.json")
        
        # Check that the file was opened and read from
        mock_file.assert_called_once_with("test_file.json", "r")
        mock_file().read.assert_called_once()
        
        # Check that deserialize was called with the file contents
        mock_deserialize.assert_called_once_with('{"hardware": {}}')
        
        # Check that the dataset was updated
        self.assertEqual(self.collector.dataset, mock_dataset)

    @patch('builtins.open', new_callable=mock_open)
    def test_save_table_to_file(self, mock_file):
        """Test saving the table to a file."""
        # Add some data to the dataset
        problem = MMShape(
            B=1,
            M=128,
            N=64,
            K=256,
            M_dtype=torch.float16,
            K_dtype=torch.float16,
            out_dtype=torch.float16,
            out_size=(1, 128, 64),
            out_stride=(128 * 64, 64, 1),
        )
        
        config = TritonGEMMConfig(
            name="mm_config",
            grid=1,
            block_m=32,
            block_n=32,
            block_k=16,
            group_m=8,
            num_stages=2,
            num_warps=4,
        )
        
        self.collector.dataset.add_timing(
            hardware_name="test_gpu",
            op_name="mm",
            problem=problem,
            config=config,
            time=0.001,
        )
        
        # Save the table to a file
        self.collector.save_table_to_file("test_table.json")
        
        # Check that the file was opened and written to
        mock_file.assert_called_once_with("test_table.json", "w")
        mock_file().write.assert_called_once()
        
        # The write call should contain the serialized table
        table = self.collector.to_table()
        serialized_table = table.serialize()
        mock_file().write.assert_called_once_with(serialized_table)

    @patch('torch_diode.collection.matmul_dataset_collector.add_feedback_saver')
    @patch('torch_diode.collection.matmul_dataset_collector.clear_feedback_savers')
    def test_context_manager(self, mock_clear, mock_add):
        """Test using the collector as a context manager."""
        with self.collector as collector:
            # Check that collection was started
            mock_add.assert_called_once()
            self.assertTrue(collector._is_collecting)
            
            # Check that the returned collector is the same as self.collector
            self.assertEqual(collector, self.collector)
        
        # Check that collection was stopped
        mock_clear.assert_called_once()
        self.assertFalse(self.collector._is_collecting)

    def test_estimate_memory_usage(self):
        """Test memory usage estimation for different operations and dtypes."""
        # Test mm operation with float32
        size = (1024, 512, 256)  # M, K, N
        memory_usage = self.collector._estimate_memory_usage(size, torch.float32, "mm", "cuda")
        
        # Expected: (M*K + K*N + M*N) * 4 bytes * 1.5 overhead
        # (1024*512 + 512*256 + 1024*256) * 4 * 1.5 = (524288 + 131072 + 262144) * 6 = 5505024
        expected = (1024 * 512 + 512 * 256 + 1024 * 256) * 4 * 1.5
        self.assertEqual(memory_usage, expected)
        
        # Test addmm operation with float16
        memory_usage = self.collector._estimate_memory_usage(size, torch.float16, "addmm", "cuda")
        
        # Expected: (M*K + K*N + M*N + M*N) * 2 bytes * 1.5 overhead
        expected = (1024 * 512 + 512 * 256 + 1024 * 256 + 1024 * 256) * 2 * 1.5
        self.assertEqual(memory_usage, expected)
        
        # Test bmm operation with float64
        memory_usage = self.collector._estimate_memory_usage(size, torch.float64, "bmm", "cuda")
        
        # Expected: (M*K + K*N + M*N) * 8 bytes * 1.5 overhead (same as mm for B=1)
        expected = (1024 * 512 + 512 * 256 + 1024 * 256) * 8 * 1.5
        self.assertEqual(memory_usage, expected)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    def test_check_memory_feasible_cuda(self, mock_memory_allocated, mock_device_properties, mock_cuda_available):
        """Test memory feasibility check for CUDA device."""
        # Mock GPU properties
        mock_props = MagicMock()
        mock_props.total_memory = 16 * 1024 * 1024 * 1024  # 16GB
        mock_device_properties.return_value = mock_props
        mock_memory_allocated.return_value = 2 * 1024 * 1024 * 1024  # 2GB allocated
        
        # Test feasible operation (small size)
        size = (128, 128, 128)
        is_feasible = self.collector._check_memory_feasible(size, torch.float32, "mm", "cuda")
        self.assertTrue(is_feasible)
        
        # Test infeasible operation (very large size)
        size = (50000, 50000, 50000)  # Would require ~1.5TB
        is_feasible = self.collector._check_memory_feasible(size, torch.float32, "mm", "cuda")
        self.assertFalse(is_feasible)

    def test_check_memory_feasible_cpu(self):
        """Test memory feasibility check for CPU device."""
        # Test feasible operation (reasonable size)
        size = (1024, 1024, 1024)
        is_feasible = self.collector._check_memory_feasible(size, torch.float32, "mm", "cpu")
        self.assertTrue(is_feasible)
        
        # Test infeasible operation (extremely large size)
        size = (100000, 100000, 100000)  # Would require ~120TB
        is_feasible = self.collector._check_memory_feasible(size, torch.float32, "mm", "cpu")
        self.assertFalse(is_feasible)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.randn')
    @patch('torch.compile')
    def test_run_matrix_multiplication_memory_skip(self, mock_compile, mock_randn, mock_memory_allocated, mock_device_properties, mock_cuda_available):
        """Test that _run_matrix_multiplication skips operations when memory is insufficient."""
        # Mock GPU properties with limited memory
        mock_props = MagicMock()
        mock_props.total_memory = 1 * 1024 * 1024 * 1024  # 1GB total
        mock_device_properties.return_value = mock_props
        mock_memory_allocated.return_value = 512 * 1024 * 1024  # 512MB allocated
        
        # Try to run a very large operation that should be skipped
        size = (10000, 10000, 10000)  # Would require ~1.1TB
        
        # This should not raise an exception and should skip the operation
        self.collector._run_matrix_multiplication(size, torch.float32, "mm", "cuda", "max-autotune")
        
        # Verify that torch.randn was never called (operation was skipped)
        mock_randn.assert_not_called()
        mock_compile.assert_not_called()

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.randn')
    @patch('torch.compile')
    def test_run_matrix_multiplication_memory_proceed(self, mock_compile, mock_randn, mock_memory_allocated, mock_device_properties, mock_cuda_available):
        """Test that _run_matrix_multiplication proceeds when memory is sufficient."""
        # Mock GPU properties with plenty of memory
        mock_props = MagicMock()
        mock_props.total_memory = 16 * 1024 * 1024 * 1024  # 16GB total
        mock_device_properties.return_value = mock_props
        mock_memory_allocated.return_value = 1 * 1024 * 1024 * 1024  # 1GB allocated
        
        # Mock tensor creation
        mock_tensor = MagicMock()
        mock_tensor.shape = (128, 128)
        mock_randn.return_value = mock_tensor
        
        # Mock compiled function
        mock_compiled_fn = MagicMock()
        mock_compile.return_value = mock_compiled_fn
        
        # Try to run a small operation that should proceed
        size = (128, 128, 128)
        
        # This should proceed and call torch.randn and torch.compile
        self.collector._run_matrix_multiplication(size, torch.float32, "mm", "cuda", "max-autotune")
        
        # Verify that torch.randn was called (operation proceeded)
        self.assertEqual(mock_randn.call_count, 2)  # Two tensors for mm operation
        mock_compile.assert_called_once()
        mock_compiled_fn.assert_called_once()

    @patch('torch_diode.collection.matmul_dataset_collector.logger')
    def test_memory_check_logging(self, mock_logger):
        """Test that memory check failures are properly logged."""
        # Test with a size that will definitely fail memory check
        size = (100000, 100000, 100000)
        
        # Call _check_memory_feasible which should log a warning
        is_feasible = self.collector._check_memory_feasible(size, torch.float32, "mm", "cpu")
        
        # Should return False
        self.assertFalse(is_feasible)
        
        # Should not log anything for CPU (only logs for CUDA)
        # Let's test CUDA logging with mocked CUDA functions
        with patch('torch.cuda.get_device_properties') as mock_props, \
            patch('torch.cuda.memory_allocated') as mock_allocated:
            
            mock_device_props = MagicMock()
            mock_device_props.total_memory = 1 * 1024 * 1024 * 1024  # 1GB
            mock_props.return_value = mock_device_props
            mock_allocated.return_value = 0
            
            is_feasible = self.collector._check_memory_feasible(size, torch.float32, "mm", "cuda")
            self.assertFalse(is_feasible)
            
            # Check that warning was logged
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args[0][0]
            self.assertIn("Skipping mm with size (100000, 100000, 100000)", warning_call)
            self.assertIn("estimated memory", warning_call)
            self.assertIn("exceeds safe limit", warning_call)


if __name__ == "__main__":
    unittest.main()
