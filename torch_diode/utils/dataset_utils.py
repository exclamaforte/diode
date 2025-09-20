"""
Dataset utility functions for matrix multiplication data collection and analysis.
"""

import logging
import random
import torch
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def print_dataset_statistics(
    dataset_or_collector: Any,
    hardware_name: Optional[str] = None,
    op_name: Optional[str] = None,
) -> None:
    """
    Print statistics about the dataset.

    Args:
        dataset_or_collector: The dataset or collector to print statistics for
        hardware_name: Optional hardware name to filter by
        op_name: Optional operation name to filter by
    """
    from torch_diode.collection.matmul_dataset_collector import MatmulDatasetCollector
    
    # Get the dataset from the collector if needed
    if isinstance(dataset_or_collector, MatmulDatasetCollector):
        dataset = dataset_or_collector.get_dataset()
    else:
        dataset = dataset_or_collector

    print("\nDataset Statistics:")
    print("------------------")

    # Count the number of hardware entries
    hardware_count = len(dataset.hardware)
    print(f"Number of hardware entries: {hardware_count}")

    # For each hardware, count operations and problems
    for hw_name, hardware in dataset.hardware.items():
        # Skip if hardware_name is specified and doesn't match
        if hardware_name is not None and hw_name != hardware_name:
            continue

        print(f"\nHardware: {hw_name}")

        op_count = len(hardware.operation)
        print(f"  Number of operations: {op_count}")

        for op_name_entry, operation in hardware.operation.items():
            # Skip if op_name is specified and doesn't match
            if op_name is not None and op_name_entry != op_name:
                continue

            problem_count = len(operation.solution)
            config_count = sum(
                len(solution.timed_configs) for solution in operation.solution.values()
            )
            print(
                f"  Operation '{op_name_entry}': {problem_count} problems, {config_count} configs"
            )

            # Print details of a few problems
            for i, (problem, solution) in enumerate(operation.solution.items()):
                if i >= 3:  # Limit to 3 problems for brevity
                    print(f"    ... and {problem_count - 3} more problems")
                    break

                print(
                    f"    Problem {i+1}: M={problem.M}, N={problem.N}, K={problem.K}, "
                    f"dtype={problem.M_dtype}, {len(solution.timed_configs)} configs"
                )

                # Print the fastest config for each problem
                if solution.timed_configs:
                    fastest_config = min(solution.timed_configs, key=lambda tc: tc.time)
                    print(
                        f"      Fastest config: block_m={fastest_config.config.block_m}, "
                        f"block_n={fastest_config.config.block_n}, "
                        f"block_k={fastest_config.config.block_k}, "
                        f"time={fastest_config.time*1000:.3f} ms"
                    )


def generate_matrix_sizes(
    num_shapes: int = 100,
    min_size: int = 32,
    max_size: int = 4096,
    power_of_two: bool = False,
    seed: int = 42,
) -> List[Tuple[int, int, int]]:
    """
    Generate a list of matrix sizes (M, K, N) for testing.

    Args:
        num_shapes: Number of shapes to generate
        min_size: Minimum matrix dimension
        max_size: Maximum matrix dimension
        power_of_two: Whether to generate only power-of-two sizes
        seed: Random seed for reproducibility

    Returns:
        List of (M, K, N) tuples
    """
    random.seed(seed)
    torch.manual_seed(seed)

    sizes = []

    # Generate random sizes to reach the desired number
    while len(sizes) < num_shapes:
        if power_of_two:
            # Generate power-of-two sizes
            m_pow = random.randint(
                int(torch.log2(torch.tensor(min_size, dtype=torch.float32))),
                int(torch.log2(torch.tensor(max_size, dtype=torch.float32))),
            )
            k_pow = random.randint(
                int(torch.log2(torch.tensor(min_size, dtype=torch.float32))),
                int(torch.log2(torch.tensor(max_size, dtype=torch.float32))),
            )
            n_pow = random.randint(
                int(torch.log2(torch.tensor(min_size, dtype=torch.float32))),
                int(torch.log2(torch.tensor(max_size, dtype=torch.float32))),
            )

            M = 2**m_pow
            K = 2**k_pow
            N = 2**n_pow
        else:
            # Generate random sizes
            M = random.randint(min_size, max_size)
            K = random.randint(min_size, max_size)
            N = random.randint(min_size, max_size)

        # Add the size if it's not already in the list
        if (M, K, N) not in sizes:
            sizes.append((M, K, N))

    return sizes[:num_shapes]  # Ensure we have exactly num_shapes
