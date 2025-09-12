"""
Modern PyTorch data loader that finds and loads all data files from a directory.

This module provides a dataset class that can automatically discover and load
all JSON and MessagePack files from a given directory, combining them into
a single dataset for training.
"""

import glob
import logging
import os
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from diode.model.matmul_dataset_loader import MatmulTimingDataset
from diode.types.matmul_dataset import Dataset as MatmulDataset

logger = logging.getLogger(__name__)


class DirectoryMatmulDataset(Dataset):
    """
    PyTorch Dataset that loads all JSON and MessagePack files from a directory.
    
    This dataset automatically discovers all data files in a directory,
    loads them, and provides samples in random order for training.
    """
    
    def __init__(
        self,
        data_dir: str,
        hardware_name: Optional[str] = None,
        op_name: Optional[str] = None,
        log_transform: bool = True,
        file_extensions: Optional[List[str]] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the data files
            hardware_name: Optional hardware name to filter by
            op_name: Optional operation name to filter by
            log_transform: Whether to apply log transform to the timing values
            file_extensions: List of file extensions to look for (default: ['json', 'msgpack'])
        """
        self.data_dir = data_dir
        self.hardware_name = hardware_name
        self.op_name = op_name
        self.log_transform = log_transform
        
        if file_extensions is None:
            file_extensions = ['json', 'msgpack']
        self.file_extensions = file_extensions
        
        # Find all data files
        self.data_files = self._find_data_files()
        logger.info(f"Found {len(self.data_files)} data files in {data_dir}")
        
        if not self.data_files:
            raise ValueError(f"No data files found in directory: {data_dir}")
        
        # Load all datasets and combine them
        self.combined_dataset = self._load_and_combine_datasets()
        
        # Create the underlying MatmulTimingDataset
        self.timing_dataset = MatmulTimingDataset(
            dataset=self.combined_dataset,
            hardware_name=hardware_name,
            op_name=op_name,
            log_transform=log_transform,
        )
        
        logger.info(f"Loaded {len(self.timing_dataset)} samples from {len(self.data_files)} files")
    
    def _find_data_files(self) -> List[str]:
        """
        Find all data files in the directory with the specified extensions.
        
        Returns:
            List of file paths
        """
        data_files = []
        
        for ext in self.file_extensions:
            pattern = os.path.join(self.data_dir, f"*.{ext}")
            files = glob.glob(pattern)
            data_files.extend(files)
        
        # Sort for consistent ordering
        data_files.sort()
        
        return data_files
    
    def _load_and_combine_datasets(self) -> MatmulDataset:
        """
        Load all datasets and combine them into a single dataset.
        
        Returns:
            Combined MatmulDataset
        """
        combined_dataset = None
        
        for i, file_path in enumerate(self.data_files):
            logger.info(f"Loading file {i+1}/{len(self.data_files)}: {os.path.basename(file_path)}")
            
            try:
                dataset = self._load_single_file(file_path)
                if dataset is None:
                    logger.warning(f"Failed to load dataset from {file_path}")
                    continue
                
                if combined_dataset is None:
                    combined_dataset = dataset
                else:
                    # Merge the datasets
                    combined_dataset = self._merge_datasets(combined_dataset, dataset)
                    
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
                continue
        
        if combined_dataset is None:
            raise ValueError("No valid datasets could be loaded")
        
        return combined_dataset
    
    def _load_single_file(self, file_path: str) -> Optional[MatmulDataset]:
        """
        Load a single data file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            MatmulDataset or None if loading failed
        """
        try:
            if file_path.endswith('.msgpack'):
                with open(file_path, 'rb') as f:
                    data = f.read()
                return MatmulDataset.from_msgpack(data)
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = f.read()
                return MatmulDataset.deserialize(data)
            else:
                logger.warning(f"Unsupported file extension: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None
    
    def _merge_datasets(self, dataset1: MatmulDataset, dataset2: MatmulDataset) -> MatmulDataset:
        """
        Merge two MatmulDatasets into one.
        
        Args:
            dataset1: First dataset
            dataset2: Second dataset
            
        Returns:
            Merged dataset
        """
        # Create a new combined dataset
        combined = MatmulDataset(hardware={})
        
        # Copy hardware data from dataset1
        for hw_name, hw_data in dataset1.hardware.items():
            combined.hardware[hw_name] = hw_data
        
        # Merge hardware data from dataset2
        for hw_name, hw_data in dataset2.hardware.items():
            if hw_name in combined.hardware:
                # Merge operations for existing hardware
                combined_hw = combined.hardware[hw_name]
                
                # Handle both dict and object formats
                if isinstance(combined_hw, dict):
                    combined_operations = combined_hw.get("operation", {})
                else:
                    combined_operations = combined_hw.operation
                
                if isinstance(hw_data, dict):
                    new_operations = hw_data.get("operation", {})
                else:
                    new_operations = hw_data.operation
                
                # Merge operations
                for op_name, op_data in new_operations.items():
                    if op_name in combined_operations:
                        # Merge solutions for existing operation
                        if isinstance(combined_operations[op_name], dict):
                            combined_solutions = combined_operations[op_name].get("solution", {})
                        else:
                            combined_solutions = combined_operations[op_name].solution
                        
                        if isinstance(op_data, dict):
                            new_solutions = op_data.get("solution", {})
                        else:
                            new_solutions = op_data.solution
                        
                        # Add new solutions
                        combined_solutions.update(new_solutions)
                    else:
                        # Add new operation
                        combined_operations[op_name] = op_data
            else:
                # Add new hardware
                combined.hardware[hw_name] = hw_data
        
        return combined
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            Number of samples in the dataset
        """
        return len(self.timing_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (problem_features, config_features, timing)
        """
        return self.timing_dataset[idx]
    
    @property
    def problem_feature_dim(self) -> int:
        """
        Get the dimension of the problem features.
        
        Returns:
            Dimension of the problem features
        """
        return self.timing_dataset.problem_feature_dim
    
    @property
    def config_feature_dim(self) -> int:
        """
        Get the dimension of the config features.
        
        Returns:
            Dimension of the config features
        """
        return self.timing_dataset.config_feature_dim
    
    @property
    def configs(self):
        """
        Get the list of TritonGEMMConfig objects.
        
        Returns:
            List of TritonGEMMConfig objects from the underlying timing dataset
        """
        return self.timing_dataset.configs
    
    def get_file_info(self) -> List[Tuple[str, int]]:
        """
        Get information about the loaded files.
        
        Returns:
            List of tuples containing (filename, sample_count_estimate)
        """
        file_info = []
        for file_path in self.data_files:
            filename = os.path.basename(file_path)
            # We can't easily get exact sample counts per file after merging,
            # so we'll just return the filename
            file_info.append((filename, -1))  # -1 indicates unknown count
        return file_info


def create_directory_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    hardware_name: Optional[str] = None,
    op_name: Optional[str] = None,
    log_transform: bool = True,
    num_workers: int = 4,
    seed: int = 42,
    file_extensions: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders from all files in a directory.
    
    Args:
        data_dir: Directory containing the data files
        batch_size: Batch size for the dataloaders
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        hardware_name: Optional hardware name to filter by
        op_name: Optional operation name to filter by
        log_transform: Whether to apply log transform to the timing values
        num_workers: Number of workers for the dataloaders
        seed: Random seed for reproducibility
        file_extensions: List of file extensions to look for (default: ['json', 'msgpack'])
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    
    # Create the dataset from all files in the directory
    full_dataset = DirectoryMatmulDataset(
        data_dir=data_dir,
        hardware_name=hardware_name,
        op_name=op_name,
        log_transform=log_transform,
        file_extensions=file_extensions,
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
    
    logger.info(f"Created directory dataloaders with {train_size} training, {val_size} validation, and {test_size} test samples")
    logger.info(f"Loaded from files: {[info[0] for info in full_dataset.get_file_info()]}")
    
    return train_dataloader, val_dataloader, test_dataloader
