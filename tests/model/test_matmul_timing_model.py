"""
Unit tests for the matrix multiplication timing prediction model.
"""

import unittest
import os
import sys
import torch
from collections import OrderedDict
import tempfile

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from diode.model.matmul_timing_model import MatmulTimingModel, DeepMatmulTimingModel
from diode.types.matmul_dataset import Dataset, TimedConfig, DatasetSolution, DatasetOperation, DatasetHardware
from diode.types.matmul_types import MMShape, TritonGEMMConfig
from diode.model.matmul_dataset_loader import MatmulTimingDataset, create_dataloaders


class TestMatmulTimingModel(unittest.TestCase):
    """
    Tests for the MatmulTimingModel class.
    """
    
    def setUp(self):
        """
        Set up the test environment.
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create a small dataset for testing
        self.dataset = self._create_test_dataset()
        
        # Create feature dimensions
        self.problem_feature_dim = 17  # Based on _extract_problem_features in MatmulTimingDataset
        self.config_feature_dim = 19   # Based on _extract_config_features in MatmulTimingDataset
        
        # Create a model
        self.model = MatmulTimingModel(
            problem_feature_dim=self.problem_feature_dim,
            config_feature_dim=self.config_feature_dim,
            hidden_dims=[32, 64, 32],
            dropout_rate=0.1,
        )
        
        # Create a deep model
        self.deep_model = DeepMatmulTimingModel(
            problem_feature_dim=self.problem_feature_dim,
            config_feature_dim=self.config_feature_dim,
            hidden_dim=32,
            num_layers=5,
            dropout_rate=0.1,
        )
    
    def _create_test_dataset(self):
        """
        Create a small dataset for testing.
        """
        # Create a dataset
        dataset = Dataset(hardware=OrderedDict())
        
        # Create a hardware entry
        hardware = DatasetHardware(operation=OrderedDict())
        dataset.hardware["test_gpu"] = hardware
        
        # Create an operation entry
        operation = DatasetOperation(solution=OrderedDict())
        hardware.operation["mm"] = operation
        
        # Create problems and solutions
        for i in range(10):
            # Create a problem
            problem = MMShape(
                B=1,
                M=64 * (i + 1),
                N=32 * (i + 1),
                K=128 * (i + 1),
                M_dtype=torch.float16,
                K_dtype=torch.float16,
                out_dtype=torch.float16,
                out_size=(1, 64 * (i + 1), 32 * (i + 1)),
                out_stride=(64 * 32 * (i + 1) ** 2, 32 * (i + 1), 1),
            )
            
            # Create a solution
            solution = DatasetSolution(timed_configs=[])
            operation.solution[problem] = solution
            
            # Create configs with different timings
            for j in range(5):
                # Create a config
                config = TritonGEMMConfig(
                    name="mm_config",
                    grid=1,
                    block_m=32 * (j + 1),
                    block_n=32 * (j + 1),
                    block_k=16 * (j + 1),
                    group_m=8,
                    num_stages=2,
                    num_warps=4,
                )
                
                # Create a timed config
                time = 0.001 * (j + 1) * (i + 1)  # Increase time with i and j
                timed_config = TimedConfig(config=config, time=time)
                solution.timed_configs.append(timed_config)
        
        return dataset
    
    def test_model_initialization(self):
        """
        Test that the model initializes correctly.
        """
        # Check that the model has the correct attributes
        self.assertEqual(self.model.problem_feature_dim, self.problem_feature_dim)
        self.assertEqual(self.model.config_feature_dim, self.config_feature_dim)
        self.assertEqual(self.model.input_dim, self.problem_feature_dim + self.config_feature_dim)
        self.assertEqual(len(self.model.hidden_dims), 3)
        self.assertEqual(self.model.hidden_dims[0], 32)
        self.assertEqual(self.model.hidden_dims[1], 64)
        self.assertEqual(self.model.hidden_dims[2], 32)
        self.assertEqual(self.model.dropout_rate, 0.1)
    
    def test_deep_model_initialization(self):
        """
        Test that the deep model initializes correctly.
        """
        # Check that the model has the correct attributes
        self.assertEqual(self.deep_model.problem_feature_dim, self.problem_feature_dim)
        self.assertEqual(self.deep_model.config_feature_dim, self.config_feature_dim)
        self.assertEqual(self.deep_model.input_dim, self.problem_feature_dim + self.config_feature_dim)
        self.assertEqual(self.deep_model.hidden_dim, 32)
        self.assertEqual(self.deep_model.num_layers, 5)
        self.assertEqual(self.deep_model.dropout_rate, 0.1)
        self.assertEqual(len(self.deep_model.hidden_layers), 5)
    
    def test_model_forward(self):
        """
        Test the forward pass of the model.
        """
        # Create random input tensors
        batch_size = 16
        problem_features = torch.randn(batch_size, self.problem_feature_dim)
        config_features = torch.randn(batch_size, self.config_feature_dim)
        
        # Forward pass
        outputs = self.model(problem_features, config_features)
        
        # Check the output shape
        self.assertEqual(outputs.shape, (batch_size, 1))
    
    def test_deep_model_forward(self):
        """
        Test the forward pass of the deep model.
        """
        # Create random input tensors
        batch_size = 16
        problem_features = torch.randn(batch_size, self.problem_feature_dim)
        config_features = torch.randn(batch_size, self.config_feature_dim)
        
        # Forward pass
        outputs = self.deep_model(problem_features, config_features)
        
        # Check the output shape
        self.assertEqual(outputs.shape, (batch_size, 1))
    
    def test_model_save_load(self):
        """
        Test saving and loading the model.
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            # Save the model
            self.model.save(tmp.name)
            
            # Load the model
            loaded_model = MatmulTimingModel.load(tmp.name)
            
            # Check that the loaded model has the same attributes
            self.assertEqual(loaded_model.problem_feature_dim, self.model.problem_feature_dim)
            self.assertEqual(loaded_model.config_feature_dim, self.model.config_feature_dim)
            self.assertEqual(loaded_model.input_dim, self.model.input_dim)
            self.assertEqual(loaded_model.hidden_dims, self.model.hidden_dims)
            self.assertEqual(loaded_model.dropout_rate, self.model.dropout_rate)
            
            # Check that the loaded model has the same parameters
            for p1, p2 in zip(self.model.parameters(), loaded_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
    
    def test_deep_model_save_load(self):
        """
        Test saving and loading the deep model.
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            # Save the model
            self.deep_model.save(tmp.name)
            
            # Load the model
            loaded_model = DeepMatmulTimingModel.load(tmp.name)
            
            # Check that the loaded model has the same attributes
            self.assertEqual(loaded_model.problem_feature_dim, self.deep_model.problem_feature_dim)
            self.assertEqual(loaded_model.config_feature_dim, self.deep_model.config_feature_dim)
            self.assertEqual(loaded_model.input_dim, self.deep_model.input_dim)
            self.assertEqual(loaded_model.hidden_dim, self.deep_model.hidden_dim)
            self.assertEqual(loaded_model.num_layers, self.deep_model.num_layers)
            self.assertEqual(loaded_model.dropout_rate, self.deep_model.dropout_rate)
            
            # Check that the loaded model has the same parameters
            for p1, p2 in zip(self.deep_model.parameters(), loaded_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
    
    def test_dataset_loader(self):
        """
        Test the dataset loader.
        """
        # Create a dataset
        timing_dataset = MatmulTimingDataset(
            dataset=self.dataset,
            hardware_name="test_gpu",
            op_name="mm",
            log_transform=True,
        )
        
        # Check that the dataset has the correct attributes
        self.assertEqual(timing_dataset.problem_feature_dim, self.problem_feature_dim)
        self.assertEqual(timing_dataset.config_feature_dim, self.config_feature_dim)
        self.assertEqual(len(timing_dataset), 50)  # 10 problems * 5 configs
        
        # Check that the dataset returns the correct items
        problem_features, config_features, timing = timing_dataset[0]
        self.assertEqual(problem_features.shape, (self.problem_feature_dim,))
        self.assertEqual(config_features.shape, (self.config_feature_dim,))
        self.assertEqual(timing.shape, (1,))
    
    def test_create_dataloaders(self):
        """
        Test creating dataloaders.
        """
        # Create dataloaders
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
            dataset=self.dataset,
            batch_size=16,
            train_ratio=0.7,
            val_ratio=0.15,
            hardware_name="test_gpu",
            op_name="mm",
            log_transform=True,
            num_workers=0,  # Use 0 workers for testing
            seed=42,
        )
        
        # Check that the dataloaders have the correct sizes
        self.assertEqual(len(train_dataloader.dataset), 35)  # 70% of 50
        self.assertEqual(len(val_dataloader.dataset), 7)    # 15% of 50
        self.assertEqual(len(test_dataloader.dataset), 8)   # 15% of 50
        
        # Check that the dataloaders return the correct items
        for problem_features, config_features, timing in train_dataloader:
            self.assertEqual(problem_features.shape[1], self.problem_feature_dim)
            self.assertEqual(config_features.shape[1], self.config_feature_dim)
            self.assertEqual(timing.shape[1], 1)
            break


if __name__ == "__main__":
    unittest.main()
