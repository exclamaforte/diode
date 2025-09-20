import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys
import torch
from collections import OrderedDict

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from torch_diode.types.matmul_dataset import (
    TimedConfig,
    DatasetSolution,
    DatasetOperation,
    DatasetHardware,
    Dataset,
)
from torch_diode.types.matmul_types import (
    TritonGEMMConfig,
    MMShape,
    Table,
)


class TestMatmulDataset(unittest.TestCase):
    def setUp(self):
        # Create a dataset instance for each test
        self.dataset = Dataset(hardware=OrderedDict())
        
        # Create some test data
        self.hardware_name = "test_gpu"
        self.op_name = "mm"
        
        # Create a problem
        self.problem = MMShape(
            B=1,
            M=64,
            N=32,
            K=128,
            M_dtype=torch.float16,
            K_dtype=torch.float16,
            out_dtype=torch.float16,
            out_size=(1, 64, 32),
            out_stride=(64 * 32, 32, 1),
        )
        
        # Create configs
        self.config1 = TritonGEMMConfig(
            name="mm_config",
            grid=1,
            block_m=32,
            block_n=32,
            block_k=16,
            group_m=8,
            num_stages=2,
            num_warps=4,
        )
        
        self.config2 = TritonGEMMConfig(
            name="mm_config",
            grid=1,
            block_m=64,
            block_n=64,
            block_k=32,
            group_m=4,
            num_stages=3,
            num_warps=8,
        )
        
        # Create times
        self.time1 = 0.001  # 1ms
        self.time2 = 0.0005  # 0.5ms (faster)

    def test_initialization(self):
        """Test that the dataset initializes correctly."""
        self.assertIsInstance(self.dataset, Dataset)
        self.assertEqual(len(self.dataset.hardware), 0)

    def test_add_timing(self):
        """Test adding a timing to the dataset."""
        # Add a timing
        self.dataset.add_timing(
            hardware_name=self.hardware_name,
            op_name=self.op_name,
            problem=self.problem,
            config=self.config1,
            time=self.time1,
        )
        
        # Check that the hardware was added
        self.assertIn(self.hardware_name, self.dataset.hardware)
        hardware = self.dataset.hardware[self.hardware_name]
        
        # Check that the operation was added
        self.assertIn(self.op_name, hardware.operation)
        operation = hardware.operation[self.op_name]
        
        # Check that the problem was added
        self.assertIn(self.problem, operation.solution)
        solution = operation.solution[self.problem]
        
        # Check that the timed config was added
        self.assertEqual(len(solution.timed_configs), 1)
        timed_config = solution.timed_configs[0]
        self.assertEqual(timed_config.config, self.config1)
        self.assertEqual(timed_config.time, self.time1)
        
        # Add another timing for the same problem
        self.dataset.add_timing(
            hardware_name=self.hardware_name,
            op_name=self.op_name,
            problem=self.problem,
            config=self.config2,
            time=self.time2,
        )
        
        # Check that the timed config was added
        self.assertEqual(len(solution.timed_configs), 2)
        timed_config = solution.timed_configs[1]
        self.assertEqual(timed_config.config, self.config2)
        self.assertEqual(timed_config.time, self.time2)

    def test_to_table(self):
        """Test converting the dataset to a table."""
        # Add timings
        self.dataset.add_timing(
            hardware_name=self.hardware_name,
            op_name=self.op_name,
            problem=self.problem,
            config=self.config1,
            time=self.time1,
        )
        
        self.dataset.add_timing(
            hardware_name=self.hardware_name,
            op_name=self.op_name,
            problem=self.problem,
            config=self.config2,
            time=self.time2,
        )
        
        # Convert to table
        table = self.dataset.to_table()
        
        # Check that the table was created correctly
        self.assertIn(self.hardware_name, table.hardware)
        hardware = table.hardware[self.hardware_name]
        
        self.assertIn(self.op_name, hardware.operation)
        operation = hardware.operation[self.op_name]
        
        self.assertIn(self.problem, operation.solution)
        solution = operation.solution[self.problem]
        
        # Check that all configs are included in the solution
        self.assertEqual(len(solution.config), 2)
        self.assertIn(self.config1, solution.config)
        self.assertIn(self.config2, solution.config)
        
        # Check that the configs are in the correct order (fastest first)
        self.assertEqual(solution.config[0], self.config2)  # time2 is faster
        self.assertEqual(solution.config[1], self.config1)

    def test_get_fastest_configs(self):
        """Test getting the fastest configs."""
        # Add timings
        self.dataset.add_timing(
            hardware_name=self.hardware_name,
            op_name=self.op_name,
            problem=self.problem,
            config=self.config1,
            time=self.time1,
        )
        
        self.dataset.add_timing(
            hardware_name=self.hardware_name,
            op_name=self.op_name,
            problem=self.problem,
            config=self.config2,
            time=self.time2,
        )
        
        # Get fastest configs
        fastest_configs = self.dataset.get_fastest_configs()
        
        # Check that the fastest config was returned
        self.assertIn((self.hardware_name, self.op_name, self.problem), fastest_configs)
        fastest_config = fastest_configs[(self.hardware_name, self.op_name, self.problem)]
        self.assertEqual(fastest_config, self.config2)  # time2 is faster

    @patch('builtins.open', new_callable=mock_open)
    def test_serialize(self, mock_file):
        """Test serializing the dataset."""
        # Add a timing
        self.dataset.add_timing(
            hardware_name=self.hardware_name,
            op_name=self.op_name,
            problem=self.problem,
            config=self.config1,
            time=self.time1,
        )
        
        # Serialize the dataset
        serialized = self.dataset.serialize()
        
        # Check that the serialized data is a string
        self.assertIsInstance(serialized, str)
        
        # Check that the serialized data contains the expected values
        self.assertIn(self.hardware_name, serialized)
        self.assertIn(self.op_name, serialized)
        self.assertIn(str(self.time1), serialized)

    @patch('torch_diode.types.matmul_dataset.Dataset.from_dict')
    def test_deserialize(self, mock_from_dict):
        """Test deserializing the dataset."""
        # Create a mock dataset
        mock_dataset = MagicMock(spec=Dataset)
        mock_from_dict.return_value = mock_dataset
        
        # Deserialize a JSON string
        json_str = '{"hardware": {}}'
        dataset = Dataset.deserialize(json_str)
        
        # Check that from_dict was called with the parsed JSON
        mock_from_dict.assert_called_once()
        
        # Check that the deserialized dataset is the mock dataset
        self.assertEqual(dataset, mock_dataset)
        
        # Test deserializing invalid JSON
        mock_from_dict.side_effect = ValueError("Invalid JSON")
        dataset = Dataset.deserialize("invalid json")
        self.assertIsNone(dataset)


if __name__ == "__main__":
    unittest.main()
