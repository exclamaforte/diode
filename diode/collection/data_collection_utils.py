"""
Data collection utility functions for matrix multiplication operations.
"""

import json
import msgpack
import logging
import os
import time
from typing import List, Optional, Tuple

import torch

from diode.collection.matmul_dataset_collector import (
    CollectionMode,
    MatmulDatasetCollector,
)
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
    search_mode: str = "max-autotune",
    search_space: str = "EXHAUSTIVE",
    file_format: str = "json",
    chunk_size: Optional[int] = None,
    log_normal_m_mean: float = 6.5725472164323095,
    log_normal_m_std: float = 2.556199441605505,
    log_normal_n_mean: float = 5.913930073563466,
    log_normal_n_std: float = 1.66968141897024,
    log_normal_k_mean: float = 6.204916071423808,
    log_normal_k_std: float = 2.1646646856090177,
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
    if mode == "random":
        collection_mode = CollectionMode.RANDOM
    elif mode == "log_normal":
        collection_mode = CollectionMode.LOG_NORMAL
    else:
        collection_mode = CollectionMode.OPERATION_SHAPE_SET

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
        log_normal_m_mean=log_normal_m_mean,
        log_normal_m_std=log_normal_m_std,
        log_normal_n_mean=log_normal_n_mean,
        log_normal_n_std=log_normal_n_std,
        log_normal_k_mean=log_normal_k_mean,
        log_normal_k_std=log_normal_k_std,
    )

    # If chunking is disabled, use the original behavior
    if chunk_size is None:
        # Use the collector's built-in collect_data method
        collector.collect_data(
            search_mode=search_mode,
            search_space=search_space,
        )

        # Save the collected dataset to a file
        if file_format == "msgpack":
            # Change extension to .msgpack if using msgpack format
            if output_file.endswith(".json"):
                output_file = output_file.replace(".json", ".msgpack")
            # Save as MessagePack
            dataset = collector.get_dataset()
            with open(output_file, "wb") as f:
                f.write(dataset.to_msgpack())
            logger.info(
                f"Saved collected dataset to {output_file} (MessagePack format)"
            )
        else:
            # Save as JSON (default)
            collector.save_to_file(output_file)
            logger.info(f"Saved collected dataset to {output_file} (JSON format)")

        # Print statistics about the collected data
        print_dataset_statistics(collector)

        return output_file
    else:
        # Use chunked collection
        return _collect_data_chunked(
            collector=collector,
            output_file=output_file,
            chunk_size=chunk_size,
            search_mode=search_mode,
            search_space=search_space,
            file_format=file_format,
        )


def create_validation_dataset(
    output_file: str,
    mode: str = "random",
    num_shapes: int = 30,
    dtypes: Optional[List[torch.dtype]] = None,
    seed: int = 43,  # Different seed from training
    min_size: int = 32,
    max_size: int = 4096,
    power_of_two: bool = False,
    search_mode: str = "max-autotune",
    search_space: str = "EXHAUSTIVE",
    file_format: str = "json",
    log_normal_m_mean: float = 6.5725472164323095,
    log_normal_m_std: float = 2.556199441605505,
    log_normal_n_mean: float = 5.913930073563466,
    log_normal_n_std: float = 1.66968141897024,
    log_normal_k_mean: float = 6.204916071423808,
    log_normal_k_std: float = 2.1646646856090177,
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
        mode=mode,
        num_shapes=num_shapes,
        dtypes=dtypes,
        seed=seed,
        min_size=min_size,
        max_size=max_size,
        power_of_two=power_of_two,
        search_mode=search_mode,
        search_space=search_space,
        file_format=file_format,
        log_normal_m_mean=log_normal_m_mean,
        log_normal_m_std=log_normal_m_std,
        log_normal_n_mean=log_normal_n_mean,
        log_normal_n_std=log_normal_n_std,
        log_normal_k_mean=log_normal_k_mean,
        log_normal_k_std=log_normal_k_std,
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


def _collect_data_chunked(
    collector: MatmulDatasetCollector,
    output_file: str,
    chunk_size: int,
    search_mode: str,
    search_space: str,
    file_format: str,
) -> str:
    """
    Collect data in chunks, saving to separate files when chunk size is reached.

    Args:
        collector: The MatmulDatasetCollector instance
        output_file: Base output file path
        chunk_size: Number of shapes to collect before writing to a new file
        search_mode: Search mode for torch.compile
        search_space: Search space for autotuning
        file_format: File format for saving (json or msgpack)

    Returns:
        Path to the first chunk file
    """
    # Parse the output file to get base name and extension
    base_name, ext = os.path.splitext(output_file)
    if file_format == "msgpack" and ext == ".json":
        ext = ".msgpack"

    # Generate shapes and dtypes based on the collector's configuration
    shapes_and_dtypes = collector._generate_shapes_and_dtypes()
    total_operations = len(shapes_and_dtypes)

    logger.info(f"Collecting {total_operations} operations in chunks of {chunk_size}")

    # Set up PyTorch for compilation (similar to collector.collect_data)
    torch.set_grad_enabled(False)

    # Configure PyTorch inductor
    from torch._inductor import config

    config.fx_graph_cache = False
    config.force_disable_caches = True
    config.max_autotune_gemm_backends = "TRITON"
    config.triton.num_decompose_k_splits = 0

    # Set search space
    if search_space == "EXHAUSTIVE":
        config.max_autotune_gemm_search_space = "EXHAUSTIVE"
        logger.info("Set search space to EXHAUSTIVE")
    else:
        config.max_autotune_gemm_search_space = "DEFAULT"
        logger.info("Set search space to DEFAULT")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    chunk_num = 0
    current_chunk_count = 0
    first_chunk_file = None

    # Start collection
    collector.start_collection()

    try:
        for i, (size, dtype, op_name) in enumerate(shapes_and_dtypes):
            M, K, N = size
            logger.info(
                f"[{i+1}/{total_operations}] Running {op_name} with size "
                f"({M}, {K}) x ({K}, {N}) and dtype {dtype} (chunk {chunk_num + 1}, item {current_chunk_count + 1}/{chunk_size})"
            )

            # Clear compilation cache to avoid shape conflicts
            torch._dynamo.reset()

            # Run the matrix multiplication
            collector._run_matrix_multiplication(
                size, dtype, op_name, device, search_mode
            )

            current_chunk_count += 1

            # Check if we've reached the chunk size or if this is the last operation
            if current_chunk_count >= chunk_size or i == total_operations - 1:
                # Stop collection temporarily to save the current chunk
                collector.stop_collection()

                # Generate chunk filename
                chunk_filename = f"{base_name}_{chunk_num + 1}{ext}"
                if first_chunk_file is None:
                    first_chunk_file = chunk_filename

                # Save the current chunk
                if file_format == "msgpack":
                    dataset = collector.get_dataset()
                    with open(chunk_filename, "wb") as f:
                        f.write(dataset.to_msgpack())
                    logger.info(
                        f"Saved chunk {chunk_num + 1} to {chunk_filename} (MessagePack format)"
                    )
                else:
                    collector.save_to_file(chunk_filename)
                    logger.info(
                        f"Saved chunk {chunk_num + 1} to {chunk_filename} (JSON format)"
                    )

                # Print statistics for this chunk
                logger.info(f"Chunk {chunk_num + 1} statistics:")
                print_dataset_statistics(collector)

                # Reset for next chunk if not the last operation
                if i < total_operations - 1:
                    # Create a new collector for the next chunk
                    collector = MatmulDatasetCollector(
                        hardware_name=collector.hardware_name,
                        mode=collector.mode,
                        operations=collector.operations,
                        operation_shape_set=collector.operation_shape_set,
                        num_shapes=collector.num_shapes,
                        dtypes=collector.dtypes,
                        seed=collector.seed,
                        min_size=collector.min_size,
                        max_size=collector.max_size,
                        power_of_two=collector.power_of_two,
                    )

                    # Start collection for the next chunk
                    collector.start_collection()

                    chunk_num += 1
                    current_chunk_count = 0

    finally:
        # Ensure collection is stopped
        if collector._is_collecting:
            collector.stop_collection()

    logger.info(f"Chunked collection completed. Created {chunk_num + 1} chunk files.")
    return first_chunk_file or output_file


def convert_json_to_msgpack(
    input_files: List[str],
    output_dir: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Convert a list of JSON files to their equivalent MessagePack files.

    Args:
        input_files: List of JSON file paths to convert
        output_dir: Output directory for MessagePack files (default: same directory as input files)
        overwrite: Whether to overwrite existing MessagePack files if they exist
    """

    converted_count = 0
    error_count = 0

    for input_file in input_files:
        try:
            # Validate input file exists
            if not os.path.exists(input_file):
                logger.error(f"Input file does not exist: {input_file}")
                error_count += 1
                continue

            # Determine output file path
            if output_dir:
                # Use specified output directory
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                output_file = os.path.join(output_dir, f"{base_name}.msgpack")
            else:
                # Use same directory as input file
                base_name = os.path.splitext(input_file)[0]
                output_file = f"{base_name}.msgpack"

            # Load JSON data
            logger.info(f"Converting {input_file} to {output_file}")
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Save as MessagePack
            with open(output_file, "wb") as f:
                msgpack.pack(data, f)

            logger.info(f"Successfully converted {input_file} to {output_file}")
            converted_count += 1

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {input_file}: {e}")
            error_count += 1
        except Exception as e:
            logger.error(f"Error converting {input_file}: {e}")
            error_count += 1

    # Print summary
    logger.info(f"Conversion completed:")
    logger.info(f"  Converted: {converted_count} files")
    logger.info(f"  Errors: {error_count} files")
