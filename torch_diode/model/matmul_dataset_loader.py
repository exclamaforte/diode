"""
Dataset loader for matrix multiplication timing data.
"""

import json
import logging
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from torch_diode.types.matmul_dataset import Dataset as MatmulDataset
from torch_diode.types.matmul_types import MMShape, TritonGEMMConfig
from torch_diode.utils.debug_config import type_assert
from torch_diode.utils.feature_extraction import (
    extract_config_features,
    extract_problem_features,
)

logger = logging.getLogger(__name__)


class MatmulTimingDataset:
    """
    PyTorch Dataset for matrix multiplication timing data.

    This dataset extracts features from MMShape and TritonGEMMConfig objects
    and converts them to tensors for training a neural network.
    """

    def __init__(
        self,
        dataset: MatmulDataset,
        hardware_name: Optional[str] = None,
        op_name: Optional[str] = None,
        log_transform: bool = True,
        debug: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            dataset: The MatmulDataset containing the timing data
            hardware_name: Optional hardware name to filter by
            op_name: Optional operation name to filter by
            log_transform: Whether to apply log transform to the timing values
            debug: Whether to enable debug logging and checks
        """
        type_assert(dataset is not None, "dataset cannot be None")
        type_assert(
            isinstance(dataset, MatmulDataset),
            f"dataset must be MatmulDataset, got {type(dataset)}",
        )
        type_assert(
            hardware_name is None or isinstance(hardware_name, str),
            f"hardware_name must be str or None, got {type(hardware_name)}",
        )
        type_assert(
            op_name is None or isinstance(op_name, str),
            f"op_name must be str or None, got {type(op_name)}",
        )
        type_assert(
            isinstance(log_transform, bool),
            f"log_transform must be bool, got {type(log_transform)}",
        )
        type_assert(isinstance(debug, bool), f"debug must be bool, got {type(debug)}")

        self.dataset = dataset
        self.hardware_name = hardware_name
        self.op_name = op_name
        self.log_transform = log_transform
        self.debug = debug

        # Extract the data
        self.problem_features = []
        self.config_features = []
        self.timings = []
        self.configs = []

        self._extract_data()

        # Convert to tensors
        self.problem_features = torch.tensor(self.problem_features, dtype=torch.float32)
        self.config_features = torch.tensor(self.config_features, dtype=torch.float32)
        self.timings = torch.tensor(self.timings, dtype=torch.float32).reshape(-1, 1)

        # Debug checks
        if self.debug:
            self._debug_data_quality()

        logger.info(f"Loaded {len(self)} samples from the dataset")

    def _extract_data(self) -> None:
        """
        Extract the data from the dataset.
        """
        # Iterate over the hardware
        hw_name = self.hardware_name
        if hw_name is not None:
            if hw_name not in self.dataset.hardware:
                logger.error(f"Hardware {hw_name} not found in the dataset")
                return
            hardware_items = {hw_name: self.dataset.hardware[hw_name]}
        else:
            # Process all hardware when no specific hardware is requested
            hardware_items = self.dataset.hardware

        # Process each hardware
        for hw_key, hardware in hardware_items.items():
            # Check if hardware is a DatasetHardware object or a dict
            if isinstance(hardware, dict) and "operation" in hardware:
                operations = hardware["operation"]
            else:
                operations = hardware.operation

            # Iterate over the operations
            for op_name, operation in operations.items():
                # Skip if op_name is specified and doesn't match
                if self.op_name is not None and op_name != self.op_name:
                    continue

                # Check if operation is a DatasetOperation object or a dict
                if isinstance(operation, dict) and "solution" in operation:
                    solutions = operation["solution"]
                else:
                    solutions = operation.solution

                # Iterate over the problems
                for problem_str, solution in solutions.items():
                    # Parse the problem if it's a string
                    if isinstance(problem_str, str):
                        try:
                            problem_dict = json.loads(problem_str)
                            problem = MMShape(
                                B=problem_dict["B"],
                                M=problem_dict["M"],
                                N=problem_dict["N"],
                                K=problem_dict["K"],
                                M_dtype=getattr(torch, problem_dict["M_dtype"]),
                                K_dtype=getattr(torch, problem_dict["K_dtype"]),
                                out_dtype=getattr(torch, problem_dict["out_dtype"]),
                                out_size=tuple(problem_dict["out_size"]),
                                out_stride=tuple(problem_dict["out_stride"]),
                            )
                        except (json.JSONDecodeError, KeyError, AttributeError) as e:
                            logger.error(f"Failed to parse problem: {e}")
                            continue
                    else:
                        problem = problem_str

                    # Extract problem features
                    problem_feature = self._extract_problem_features(problem)

                    # Check if solution is a DatasetSolution object or a dict
                    if isinstance(solution, dict) and "timed_configs" in solution:
                        timed_configs = solution["timed_configs"]
                    else:
                        timed_configs = solution.timed_configs

                    # Iterate over the timed configs
                    for timed_config in timed_configs:
                        # Parse the config if it's a dict with a string config
                        if isinstance(timed_config, dict):
                            config_str = timed_config.get("config", "")
                            time = timed_config.get("time", 0.0)

                            if isinstance(config_str, str):
                                try:
                                    config_dict = json.loads(config_str)
                                    config = TritonGEMMConfig(
                                        name=config_dict["name"],
                                        grid=config_dict["grid"],
                                        block_m=config_dict["block_m"],
                                        block_n=config_dict["block_n"],
                                        block_k=config_dict["block_k"],
                                        group_m=config_dict["group_m"],
                                        num_stages=config_dict["num_stages"],
                                        num_warps=config_dict["num_warps"],
                                        EVEN_K=config_dict.get("EVEN_K", False),
                                        ALLOW_TF32=config_dict.get("ALLOW_TF32", False),
                                        USE_FAST_ACCUM=config_dict.get(
                                            "USE_FAST_ACCUM", False
                                        ),
                                        ACC_TYPE=config_dict.get(
                                            "ACC_TYPE", "tl.float32"
                                        ),
                                    )
                                except (json.JSONDecodeError, KeyError) as e:
                                    logger.error(f"Failed to parse config: {e}")
                                    continue
                            else:
                                config = config_str
                        else:
                            config = timed_config.config
                            time = timed_config.time

                        # Extract config features
                        config_feature = self._extract_config_features(config)

                        # Extract timing
                        timing = time

                        # Check for invalid timing values before log transform
                        if timing <= 0:
                            if self.debug:
                                logger.warning(
                                    f"Invalid timing value: {timing} - skipping this sample"
                                )
                            continue  # Skip this sample

                        if not torch.isfinite(
                            torch.tensor(timing, dtype=torch.float32)
                        ):
                            if self.debug:
                                logger.warning(
                                    f"Non-finite timing value: {timing} - skipping this sample"
                                )
                            continue  # Skip this sample

                        # Apply log transform if specified
                        if self.log_transform:
                            timing = torch.log(
                                torch.tensor(timing, dtype=torch.float32)
                            )

                            # Check if log transform produced NaN/Inf
                            if not torch.isfinite(timing):
                                if self.debug:
                                    logger.warning(
                                        f"Log transform produced non-finite value for timing {time} - skipping this sample"
                                    )
                                continue  # Skip this sample

                        # Add to the lists
                        self.problem_features.append(problem_feature)
                        self.config_features.append(config_feature)
                        self.timings.append(timing)
                        self.configs.append(config)

    def _extract_problem_features(self, problem: MMShape) -> List[float]:
        """
        Extract features from an MMShape.

        Args:
            problem: The MMShape to extract features from

        Returns:
            List of features
        """
        return extract_problem_features(problem, return_tensors=True)

    def _extract_config_features(self, config: TritonGEMMConfig) -> List[float]:
        """
        Extract features from a TritonGEMMConfig.

        Args:
            config: The TritonGEMMConfig to extract features from

        Returns:
            List of features
        """
        return extract_config_features(config, return_tensors=True)

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

    def _debug_data_quality(self) -> None:
        """
        Debug method to check data quality for NaN/Inf values.
        """
        logger.info("=== DATA QUALITY DEBUG ===")

        # Check if tensors are empty
        if self.problem_features.numel() == 0:
            logger.warning("Problem features tensor is empty!")
            logger.info("=== END DATA QUALITY DEBUG ===")
            return

        if self.config_features.numel() == 0:
            logger.warning("Config features tensor is empty!")
            logger.info("=== END DATA QUALITY DEBUG ===")
            return

        if self.timings.numel() == 0:
            logger.warning("Timings tensor is empty!")
            logger.info("=== END DATA QUALITY DEBUG ===")
            return

        # Check problem features
        problem_nan_count = torch.isnan(self.problem_features).sum().item()
        problem_inf_count = torch.isinf(self.problem_features).sum().item()
        logger.info(
            f"Problem features: {problem_nan_count} NaN values, {problem_inf_count} Inf values"
        )

        if problem_nan_count > 0:
            nan_mask = torch.isnan(self.problem_features)
            nan_indices = torch.where(nan_mask)
            logger.warning(
                f"Problem feature NaN locations: {list(zip(nan_indices[0].tolist(), nan_indices[1].tolist()))[:10]}"
            )

        # Check config features
        config_nan_count = torch.isnan(self.config_features).sum().item()
        config_inf_count = torch.isinf(self.config_features).sum().item()
        logger.info(
            f"Config features: {config_nan_count} NaN values, {config_inf_count} Inf values"
        )

        if config_nan_count > 0:
            nan_mask = torch.isnan(self.config_features)
            nan_indices = torch.where(nan_mask)
            logger.warning(
                f"Config feature NaN locations: {list(zip(nan_indices[0].tolist(), nan_indices[1].tolist()))[:10]}"
            )

        # Check timings
        timing_nan_count = torch.isnan(self.timings).sum().item()
        timing_inf_count = torch.isinf(self.timings).sum().item()
        timing_negative_count = (self.timings <= 0).sum().item()
        logger.info(
            f"Timings: {timing_nan_count} NaN values, {timing_inf_count} Inf values, {timing_negative_count} negative/zero values"
        )

        if timing_nan_count > 0:
            nan_indices = torch.where(torch.isnan(self.timings))
            logger.warning(f"Timing NaN locations: {nan_indices[0].tolist()[:10]}")

        if timing_negative_count > 0:
            neg_indices = torch.where(self.timings <= 0)
            logger.warning(
                f"Negative/zero timing locations: {neg_indices[0].tolist()[:10]}"
            )

        # Statistics - only if tensors are not empty
        logger.info(
            f"Problem features - min: {self.problem_features.min():.6f}, max: {self.problem_features.max():.6f}"
        )
        logger.info(
            f"Config features - min: {self.config_features.min():.6f}, max: {self.config_features.max():.6f}"
        )
        logger.info(
            f"Timings - min: {self.timings.min():.6f}, max: {self.timings.max():.6f}"
        )

        # Check for extreme values
        problem_extreme_high = (self.problem_features > 1e6).sum().item()
        config_extreme_high = (self.config_features > 1e6).sum().item()
        logger.info(
            f"Extreme high values (>1e6): problem={problem_extreme_high}, config={config_extreme_high}"
        )

        logger.info("=== END DATA QUALITY DEBUG ===")


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
    debug: bool = False,
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
        debug: Whether to enable debug logging and checks

    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Create the dataset
    full_dataset = MatmulTimingDataset(
        dataset=dataset,
        hardware_name=hardware_name,
        op_name=op_name,
        log_transform=log_transform,
        debug=debug,
    )

    # Calculate the sizes of the splits
    dataset_size = len(full_dataset)

    # Check if dataset is empty
    if dataset_size == 0:
        logger.warning("Dataset is empty, returning None dataloaders")
        return None, None, None

    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Handle cases where split sizes are 0
    if train_size == 0:
        train_size = min(1, dataset_size)
        val_size = 0
        test_size = 0
    elif val_size == 0 and dataset_size > 1:
        val_size = 1
        test_size = max(0, dataset_size - train_size - val_size)

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

    logger.info(
        f"Created dataloaders with {train_size} training, {val_size} validation, and {test_size} test samples"
    )

    return train_dataloader, val_dataloader, test_dataloader
