import torch
from torch._inductor.select_algorithm import add_feedback_saver, clear_feedback_saver
from collections import OrderedDict
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Import the size_hints function from PyTorch inductor
import torch._inductor.config as inductor_config

from diode.types.matmul_types import (
    TritonGEMMConfig,
    MMShape,
    Table,
)
from diode.types.matmul_dataset import (
    Dataset,
    TimedConfig,
)

logger = logging.getLogger(__name__)

class MatmulDatasetCollector:
    """
    A class that hooks into the PyTorch feedback saver interface to collect
    matrix multiplication data with timing information and store it in a Dataset.
    """

    def __init__(self, hardware_name: str = "unknown"):
        """
        Initialize the MatmulDatasetCollector.

        Args:
            hardware_name: The name of the hardware being used.
        """
        self.hardware_name = hardware_name
        self.dataset = Dataset(hardware=OrderedDict())
        self._is_collecting = False

    def start_collection(self) -> None:
        """
        Start collecting data by hooking into the feedback saver interface.
        """
        if self._is_collecting:
            logger.warning("Collection is already in progress")
            return

        add_feedback_saver(self._feedback_handler)
        self._is_collecting = True
        logger.info("Started collecting matmul data")

    def stop_collection(self) -> None:
        """
        Stop collecting data by removing the feedback saver hook.
        """
        if not self._is_collecting:
            logger.warning("No collection in progress")
            return

        clear_feedback_saver()
        self._is_collecting = False
        logger.info("Stopped collecting matmul data")

    def _get_size_hints(self, mat1, mat2, m, n, k):
        """
        Get size hints for symbolic dimensions, similar to PyTorch inductor's get_size_hints.
        
        Args:
            mat1: First matrix
            mat2: Second matrix
            m, n, k: Matrix dimensions (may be symbolic)
            
        Returns:
            Tuple of (m, n, k) with integer values
        """
        from torch._inductor.virtualized import V
        
        # Handle m and k from mat1
        if not isinstance(m, int) or not isinstance(k, int):
            try:
                # Try to get size hints from the graph's sizevars
                (m, k) = V.graph.sizevars.size_hints(
                    mat1.layout.size, 
                    fallback=inductor_config.unbacked_symint_fallback
                )
            except (AttributeError, TypeError):
                # If that fails, use default values
                m = m if isinstance(m, int) else 1
                k = k if isinstance(k, int) else 1
        
        # Handle k and n from mat2
        if not isinstance(n, int) or not isinstance(k, int):
            try:
                # Try to get size hints from the graph's sizevars
                (k2, n) = V.graph.sizevars.size_hints(
                    mat2.layout.size, 
                    fallback=inductor_config.unbacked_symint_fallback
                )
                # Use k2 if k is not an int
                if not isinstance(k, int):
                    k = k2
            except (AttributeError, TypeError):
                # If that fails, use default values
                n = n if isinstance(n, int) else 1
                if not isinstance(k, int):
                    k = 1
        
        return m, n, k

    def _feedback_handler(self, timings: Dict, name: str, input_nodes: List, 
                         choices: Any, profiled_time: float) -> None:
        """
        Handle feedback from PyTorch's feedback saver interface.

        Args:
            timings: Dictionary mapping choices to benchmark times
            name: Name of the operation (e.g., "mm", "addmm")
            input_nodes: Input nodes for the operation
            choices: Available choices for the operation
            profiled_time: Time spent profiling
        """
        # Debug logging
        logger.debug(f"Feedback handler called with name: {name}, timings: {len(timings)}")
        
        # Only handle matrix multiplication operations
        if name not in ["mm", "addmm"]:
            logger.debug(f"Skipping operation: {name} (not mm or addmm)")
            return

        # Extract problem dimensions
        if name == "addmm":
            mat1 = input_nodes[1]
            mat2 = input_nodes[2]
            M, K, N = (
                mat1.layout.size[0],
                mat1.layout.size[1],
                mat2.layout.size[1],
            )
            M_dtype = mat1.layout.dtype
            K_dtype = mat2.layout.dtype
        elif name == "mm":
            mat1 = input_nodes[0]
            mat2 = input_nodes[1]
            M, K, N = (
                mat1.layout.size[0],
                mat1.layout.size[1],
                mat2.layout.size[1],
            )
            M_dtype = mat1.layout.dtype
            K_dtype = mat2.layout.dtype
        else:
            return
            
        # Get size hints for symbolic dimensions
        M, N, K = self._get_size_hints(mat1, mat2, M, N, K)

        # Create MMShape instance
        # Note: Some fields are approximated as we don't have all the information
        # from the feedback saver interface
        problem = MMShape(
            B=1,  # Batch size, assuming 1 for now
            M=M,
            N=N,
            K=K,
            M_dtype=M_dtype,
            K_dtype=K_dtype,
            out_dtype=M_dtype,  # Assuming output dtype is the same as input
            out_size=(1, M, N),  # Approximating output size
            out_stride=(M * N, N, 1),  # Approximating output stride
        )

        # Process each timing result
        for choice, bench_time in timings.items():
            # Only process TritonTemplateCaller choices
            if not isinstance(choice, torch._inductor.select_algorithm.TritonTemplateCaller):
                continue

            # Extract configuration details
            log_info = choice.log_info
            block_m, block_k, block_n = map(
                int, log_info.get('tile_shape', '(0,0,0)').strip('()').split(',')
            )
            
            # Create TritonGEMMConfig instance
            config = TritonGEMMConfig(
                name=f"{name}_config",
                grid=1,  # Default value, not available in feedback
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                group_m=log_info.get('GROUP_M', 1),
                num_stages=log_info.get('num_stages', 1),
                num_warps=log_info.get('num_warps', 4),
                # Other fields use default values
            )
            
            # Add the timing to the dataset
            self.dataset.add_timing(
                hardware_name=self.hardware_name,
                op_name=name,
                problem=problem,
                config=config,
                time=bench_time,
            )

    def get_dataset(self) -> Dataset:
        """
        Get the collected data as a Dataset.

        Returns:
            The Dataset containing all collected data.
        """
        return self.dataset

    def to_table(self) -> Table:
        """
        Convert the dataset to a table by selecting the fastest configuration for each problem.
        
        Returns:
            A Table with the fastest configuration for each problem.
        """
        return self.dataset.to_table()

    def save_to_file(self, file_path: str) -> None:
        """
        Save the collected data to a file.

        Args:
            file_path: Path to save the data to.
        """
        with open(file_path, 'w') as f:
            f.write(self.dataset.serialize())
        logger.info(f"Saved collected data to {file_path}")

    def load_from_file(self, file_path: str) -> None:
        """
        Load data from a file.

        Args:
            file_path: Path to load the data from.
        """
        with open(file_path, 'r') as f:
            content = f.read()
        
        dataset = Dataset.deserialize(content)
        if dataset:
            self.dataset = dataset
            logger.info(f"Loaded data from {file_path}")
        else:
            logger.error(f"Failed to load data from {file_path}")

    def save_table_to_file(self, file_path: str) -> None:
        """
        Convert the dataset to a table and save it to a file.

        Args:
            file_path: Path to save the table to.
        """
        table = self.to_table()
        with open(file_path, 'w') as f:
            f.write(table.serialize())
        logger.info(f"Saved table to {file_path}")

    def __enter__(self):
        """
        Context manager entry point.
        """
        self.start_collection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point.
        """
        self.stop_collection()
