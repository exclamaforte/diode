"""
Dataset loader for matrix multiplication timing data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict

from diode.types.matmul_dataset import Dataset as MatmulDataset
from diode.types.matmul_types import MMProblem, TritonGEMMConfig

logger = logging.getLogger(__name__)

class MatmulTimingDataset(Dataset):
    """
    PyTorch Dataset for matrix multiplication timing data.
    
    This dataset extracts features from MMProblem and TritonGEMMConfig objects
    and converts them to tensors for training a neural network.
    """
    
    def __init__(
        self,
        dataset: MatmulDataset,
        hardware_name: Optional[str] = None,
        op_name: Optional[str] = None,
        log_transform: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            dataset: The MatmulDataset containing the timing data
            hardware_name: Optional hardware name to filter by
            op_name: Optional operation name to filter by
            log_transform: Whether to apply log transform to the timing values
        """
        self.dataset = dataset
        self.hardware_name = hardware_name
        self.op_name = op_name
        self.log_transform = log_transform
        
        # Extract the data
        self.problem_features = []
        self.config_features = []
        self.timings = []
        
        self._extract_data()
        
        # Convert to tensors
        self.problem_features = torch.tensor(self.problem_features, dtype=torch.float32)
        self.config_features = torch.tensor(self.config_features, dtype=torch.float32)
        self.timings = torch.tensor(self.timings, dtype=torch.float32).reshape(-1, 1)
        
        logger.info(f"Loaded {len(self)} samples from the dataset")
    
    def _extract_data(self) -> None:
        """
        Extract the data from the dataset.
        """
        # Iterate over the hardware
        for hw_name, hardware in self.dataset.hardware.items():
            # Skip if hardware_name is specified and doesn't match
            if self.hardware_name is not None and hw_name != self.hardware_name:
                continue
            
            # Iterate over the operations
            for op_name, operation in hardware.operation.items():
                # Skip if op_name is specified and doesn't match
                if self.op_name is not None and op_name != self.op_name:
                    continue
                
                # Iterate over the problems
                for problem, solution in operation.solution.items():
                    # Extract problem features
                    problem_feature = self._extract_problem_features(problem)
                    
                    # Iterate over the timed configs
                    for timed_config in solution.timed_configs:
                        # Extract config features
                        config_feature = self._extract_config_features(timed_config.config)
                        
                        # Extract timing
                        timing = timed_config.time
                        
                        # Apply log transform if specified
                        if self.log_transform:
                            timing = np.log(timing)
                        
                        # Add to the lists
                        self.problem_features.append(problem_feature)
                        self.config_features.append(config_feature)
                        self.timings.append(timing)
    
    def _extract_problem_features(self, problem: MMProblem) -> List[float]:
        """
        Extract features from an MMProblem.
        
        Args:
            problem: The MMProblem to extract features from
            
        Returns:
            List of features
        """
        # Extract features
        features = [
            problem.B,
            problem.M,
            problem.N,
            problem.K,
            # Convert dtypes to numeric values
            self._dtype_to_numeric(problem.M_dtype),
            self._dtype_to_numeric(problem.K_dtype),
            self._dtype_to_numeric(problem.out_dtype),
            # Add log-transformed features for better numerical stability
            np.log(max(1, problem.B)),
            np.log(max(1, problem.M)),
            np.log(max(1, problem.N)),
            np.log(max(1, problem.K)),
            # Add derived features
            problem.M * problem.K,  # Input matrix size
            problem.K * problem.N,  # Weight matrix size
            problem.M * problem.N,  # Output matrix size
            np.log(max(1, problem.M * problem.K)),  # Log input matrix size
            np.log(max(1, problem.K * problem.N)),  # Log weight matrix size
            np.log(max(1, problem.M * problem.N)),  # Log output matrix size
        ]
        
        return features
    
    def _extract_config_features(self, config: TritonGEMMConfig) -> List[float]:
        """
        Extract features from a TritonGEMMConfig.
        
        Args:
            config: The TritonGEMMConfig to extract features from
            
        Returns:
            List of features
        """
        # Extract features
        features = [
            config.grid,
            config.block_m,
            config.block_n,
            config.block_k,
            config.group_m,
            config.num_stages,
            config.num_warps,
            int(config.EVEN_K),
            int(config.ALLOW_TF32),
            int(config.USE_FAST_ACCUM),
            # Add log-transformed features for better numerical stability
            np.log(max(1, config.block_m)),
            np.log(max(1, config.block_n)),
            np.log(max(1, config.block_k)),
            # Add derived features
            config.block_m * config.block_k,  # Block input size
            config.block_k * config.block_n,  # Block weight size
            config.block_m * config.block_n,  # Block output size
            np.log(max(1, config.block_m * config.block_k)),  # Log block input size
            np.log(max(1, config.block_k * config.block_n)),  # Log block weight size
            np.log(max(1, config.block_m * config.block_n)),  # Log block output size
        ]
        
        return features
    
    def _dtype_to_numeric(self, dtype: torch.dtype) -> float:
        """
        Convert a torch dtype to a numeric value.
        
        Args:
            dtype: The torch dtype
            
        Returns:
            Numeric value representing the dtype
        """
        # Map dtypes to numeric values
        dtype_map = {
            torch.float16: 16.0,
            torch.float32: 32.0,
            torch.float64: 64.0,
            torch.bfloat16: 16.0,
            torch.int8: 8.0,
            torch.int16: 16.0,
            torch.int32: 32.0,
            torch.int64: 64.0,
        }
        
        return dtype_map.get(dtype, 0.0)
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            Number of samples in the dataset
        """
        return len(self.timings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (problem_features, config_features, timing)
        """
        return self.problem_features[idx], self.config_features[idx], self.timings[idx]
    
    @property
    def problem_feature_dim(self) -> int:
        """
        Get the dimension of the problem features.
        
        Returns:
            Dimension of the problem features
        """
        return self.problem_features.shape[1]
    
    @property
    def config_feature_dim(self) -> int:
        """
        Get the dimension of the config features.
        
        Returns:
            Dimension of the config features
        """
        return self.config_features.shape[1]


def create_dataloaders(
    dataset: MatmulDataset,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    hardware_name: Optional[str] = None,
    op_name: Optional[str] = None,
    log_transform: bool = True,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders from a MatmulDataset.
    
    Args:
        dataset: The MatmulDataset containing the timing data
        batch_size: Batch size for the dataloaders
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        hardware_name: Optional hardware name to filter by
        op_name: Optional operation name to filter by
        log_transform: Whether to apply log transform to the timing values
        num_workers: Number of workers for the dataloaders
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create the dataset
    full_dataset = MatmulTimingDataset(
        dataset=dataset,
        hardware_name=hardware_name,
        op_name=op_name,
        log_transform=log_transform,
    )
    
    # Calculate the sizes of the splits
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create the dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    logger.info(f"Created dataloaders with {train_size} training, {val_size} validation, and {test_size} test samples")
    
    return train_dataloader, val_dataloader, test_dataloader
