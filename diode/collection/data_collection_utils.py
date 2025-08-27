"""
Data collection utility functions for matrix multiplication operations.
"""

import logging
import os
import time
import torch
from typing import List, Optional, Tuple

from diode.collection.matmul_dataset_collector import MatmulDatasetCollector, CollectionMode
from diode.types.matmul_dataset import Dataset as MatmulDataset
from diode.types.matmul_types import OperationShapeSet
from diode.utils.dataset_utils import generate_matrix_sizes, print_dataset_statistics

logger = logging.getLogger(__name__)


def run_matrix_multiplications(
    sizes: List[Tuple[int, int, int]],
    dtypes: List[torch.dtype],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    search_mode: str = "max-autotune",
) -> None:
    """
    Run matrix multiplication operations with the given sizes and dtypes.

    Args:
        sizes: List of (M, K, N) tuples
        dtypes: List of dtypes to test
        device: Device to run on
        search_mode: Search mode for torch.compile
    """
    for i, (M, K, N) in enumerate(sizes):
        for dtype in dtypes:
            # Create input matrices
            a = torch.randn(M, K, device=device, dtype=dtype)
            b = torch.randn(K, N, device=device, dtype=dtype)
            c = torch.randn(M, N, device=device, dtype=dtype)

            # Define functions to compile
            def mm_fn(x, y):
                return torch.mm(x, y)

            def addmm_fn(bias, x, y):
                return torch.addmm(bias, x, y)

            # Compile and run mm
            logger.info(
                f"[{i+1}/{len(sizes)}] Running mm with size ({M}, {K}) x ({K}, {N}) and dtype {dtype}"
            )
            compiled_mm = torch.compile(mm_fn, mode=search_mode)
            result_mm = compiled_mm(a, b)

            # Compile and run addmm
            logger.info(
                f"[{i+1}/{len(sizes)}] Running addmm with size ({M}, {K}) x ({K}, {N}) and dtype {dtype}"
            )
            compiled_addmm = torch.compile(addmm_fn, mode=search_mode)
            result_addmm = compiled_addmm(c, a, b)


def collect_data(
    output_file: str,
    mode: str = "random",
    operations: Optional[List[str]] = None,
    operation_shape_set: Optional[OperationShapeSet] = None,
    num_shapes: int = 100,
    dtypes: Optional[List[torch.dtype]] = None,
    seed: int = 42,
    min_size: int = 32,
    max_size: int = 4096,
    power_of_two: bool = False,
    include_rectangular: bool = True,
    include_odd_sizes: bool = True,
    search_mode: str = "max-autotune",
    search_space: str = "EXHAUSTIVE",
    file_format: str = "json",
) -> str:
    """
    Collect matrix multiplication timing data using the enhanced MatmulDatasetCollector.

    Args:
        output_file: Path to save the collected data
        mode: Collection mode ("random" or "operation_shape_set")
        operations: List of operations to collect data for (e.g., ['mm', 'addmm'])
        operation_shape_set: OperationShapeSet for operation_shape_set mode
        num_shapes: Number of matrix shapes to test (random mode)
        dtypes: List of dtypes to test
        seed: Random seed for reproducibility (random mode)
        min_size: Minimum matrix dimension (random mode)
        max_size: Maximum matrix dimension (random mode)
        power_of_two: Whether to generate only power-of-two sizes (random mode)
        include_rectangular: Whether to include rectangular matrices (random mode)
        include_odd_sizes: Whether to include odd-sized matrices (random mode)
        search_mode: Search mode for torch.compile
        search_space: Search space for autotuning (EXHAUSTIVE or DEFAULT)
        file_format: File format for saving (json or msgpack)

    Returns:
        Path to the saved dataset file
    """
    if dtypes is None:
        dtypes = (
            [torch.float16, torch.float32]
            if torch.cuda.is_available()
            else [torch.float32]
        )

    if operations is None:
        operations = ["mm", "addmm"]

    # Get the hardware name
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    logger.info(f"Collecting data on device: {device_name}")

    # Convert mode string to enum
    collection_mode = CollectionMode.RANDOM if mode == "random" else CollectionMode.OPERATION_SHAPE_SET

    # Create a collector with the enhanced parameters
    collector = MatmulDatasetCollector(
        hardware_name=device_name,
        mode=collection_mode,
        operations=operations,
        operation_shape_set=operation_shape_set,
        num_shapes=num_shapes,
        dtypes=dtypes,
        seed=seed,
        min_size=min_size,
        max_size=max_size,
        power_of_two=power_of_two,
        include_rectangular=include_rectangular,
        include_odd_sizes=include_odd_sizes,
    )

    # Use the collector's built-in collect_data method
    collector.collect_data(
        search_mode=search_mode,
        search_space=search_space,
    )

    # Save the collected dataset to a file
    if file_format == "msgpack":
        # Change extension to .msgpack if using msgpack format
        if output_file.endswith('.json'):
            output_file = output_file.replace('.json', '.msgpack')
        # Save as MessagePack
        dataset = collector.get_dataset()
        with open(output_file, 'wb') as f:
            f.write(dataset.to_msgpack())
        logger.info(f"Saved collected dataset to {output_file} (MessagePack format)")
    else:
        # Save as JSON (default)
        collector.save_to_file(output_file)
        logger.info(f"Saved collected dataset to {output_file} (JSON format)")

    # Print statistics about the collected data
    print_dataset_statistics(collector)

    return output_file


def create_validation_dataset(
    output_file: str,
    num_shapes: int = 30,
    dtypes: Optional[List[torch.dtype]] = None,
    seed: int = 43,  # Different seed from training
    min_size: int = 32,
    max_size: int = 4096,
    power_of_two: bool = False,
    include_rectangular: bool = True,
    include_odd_sizes: bool = True,
    search_mode: str = "max-autotune",
    search_space: str = "EXHAUSTIVE",
    file_format: str = "json",
) -> str:
    """
    Create a separate validation dataset for evaluating the model.

    Args:
        output_file: Path to save the validation data
        num_shapes: Number of matrix shapes to test
        dtypes: List of dtypes to test
        seed: Random seed for reproducibility (different from training)
        min_size: Minimum matrix dimension
        max_size: Maximum matrix dimension
        power_of_two: Whether to generate only power-of-two sizes
        include_rectangular: Whether to include rectangular matrices
        include_odd_sizes: Whether to include odd-sized matrices
        search_mode: Search mode for torch.compile
        search_space: Search space for autotuning (EXHAUSTIVE or DEFAULT)
        file_format: File format for saving (json or msgpack)

    Returns:
        Path to the saved validation dataset file
    """
    # Check if validation dataset already exists
    if os.path.exists(output_file):
        logger.info(f"Validation dataset already exists at {output_file}")
        return output_file

    logger.info(f"Creating validation dataset at {output_file}")

    # Use the same collection function but with different parameters
    return collect_data(
        output_file=output_file,
        num_shapes=num_shapes,
        dtypes=dtypes,
        seed=seed,
        min_size=min_size,
        max_size=max_size,
        power_of_two=power_of_two,
        include_rectangular=include_rectangular,
        include_odd_sizes=include_odd_sizes,
        search_mode=search_mode,
        search_space=search_space,
        file_format=file_format,
    )


def run_collector_example(
    output_dir: str = ".",
    use_context_manager: bool = True,
    num_shapes: int = 4,
    dtypes: Optional[List[torch.dtype]] = None,
) -> None:
    """
    Run an example demonstrating how to use the MatmulDatasetCollector class.

    Args:
        output_dir: Directory to save output files
        use_context_manager: Whether to use the collector as a context manager
        num_shapes: Number of matrix shapes to test
        dtypes: List of dtypes to test
    """
    if dtypes is None:
        dtypes = (
            [torch.float16, torch.float32]
            if torch.cuda.is_available()
            else [torch.float32]
        )

    # Get the hardware name
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device_name}")

    # Generate matrix sizes
    sizes = [
        (32, 64, 128),  # (M, K, N)
        (64, 128, 256),
        (128, 256, 512),
        (256, 512, 1024),
    ][:num_shapes]

    if use_context_manager:
        # Example using the collector as a context manager
        print("\nUsing the collector as a context manager")
        collector = MatmulDatasetCollector(hardware_name=device_name)

        # Use the collector as a context manager
        with collector:
            print("Running matrix multiplications...")
            run_matrix_multiplications(sizes, dtypes)

        # Save the collected dataset to a file
        dataset_file = os.path.join(
            output_dir, "matmul_dataset_context_manager.msgpack"
        )
        collector.save_to_file(dataset_file)
        print(f"Saved collected dataset to {dataset_file}")

        # Convert the dataset to a table and save it
        table_file = os.path.join(output_dir, "matmul_table_context_manager.msgpack")
        collector.save_table_to_file(table_file)
        print(f"Saved table to {table_file}")
    else:
        # Example using start_collection and stop_collection methods
        print("\nUsing start_collection and stop_collection methods")
        collector = MatmulDatasetCollector(hardware_name=device_name)
        collector.start_collection()

        # Run matrix multiplications
        print("Running matrix multiplications...")
        run_matrix_multiplications(sizes, dtypes)

        # Stop collection
        collector.stop_collection()

        # Save the collected dataset to a file
        dataset_file = os.path.join(output_dir, "matmul_dataset_explicit.msgpack")
        collector.save_to_file(dataset_file)
        print(f"Saved collected dataset to {dataset_file}")

        # Convert the dataset to a table and save it
        table_file = os.path.join(output_dir, "matmul_table_explicit.msgpack")
        collector.save_table_to_file(table_file)
        print(f"Saved table to {table_file}")

    # Print statistics about the collected data
    print_dataset_statistics(collector)
