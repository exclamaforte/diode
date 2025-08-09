import torch
from torch._inductor.select_algorithm import add_feedback_saver, clear_feedback_saver
from collections import OrderedDict
from torch.utils._ordered_set import OrderedSet
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from diode.types.matmul_types import (
    TritonGEMMConfig,
    MMProblem,
    Solution,
    Operation,
    Hardware,
    Table,
)

logger = logging.getLogger(__name__)

class MatmulCollector:
    """
    A class that hooks into the PyTorch feedback saver interface to collect
    matrix multiplication data and store it in structured types.
    """

    def __init__(self, hardware_name: str = "unknown"):
        """
        Initialize the MatmulCollector.

        Args:
            hardware_name: The name of the hardware being used.
        """
        self.hardware_name = hardware_name
        self.table = Table(hardware=OrderedDict())
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
        # Only handle matrix multiplication operations
        if name not in ["mm", "addmm"]:
            return

        # Extract problem dimensions
        if name == "addmm":
            M, K, N = (
                input_nodes[1].layout.size[0],
                input_nodes[1].layout.size[1],
                input_nodes[2].layout.size[1],
            )
            M_dtype = input_nodes[1].layout.dtype
            K_dtype = input_nodes[2].layout.dtype
        elif name == "mm":
            M, K, N = (
                input_nodes[0].layout.size[0],
                input_nodes[0].layout.size[1],
                input_nodes[1].layout.size[1],
            )
            M_dtype = input_nodes[0].layout.dtype
            K_dtype = input_nodes[1].layout.dtype
        else:
            return

        # Create MMProblem instance
        # Note: Some fields are approximated as we don't have all the information
        # from the feedback saver interface
        problem = MMProblem(
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
        configs = []
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
            
            configs.append(config)

        # Update the table with the new data
        self._update_table(name, problem, configs)

    def _update_table(self, op_name: str, problem: MMProblem, configs: List[TritonGEMMConfig]) -> None:
        """
        Update the table with new data.

        Args:
            op_name: Name of the operation
            problem: The matrix multiplication problem
            configs: List of configurations for the problem
        """
        # Get or create hardware entry
        if self.hardware_name not in self.table.hardware:
            self.table.hardware[self.hardware_name] = Hardware(operation=OrderedDict())
        
        hardware = self.table.hardware[self.hardware_name]
        
        # Get or create operation entry
        if op_name not in hardware.operation:
            hardware.operation[op_name] = Operation(name=op_name, solution=OrderedDict())
        
        operation = hardware.operation[op_name]
        
        # Get or create solution entry
        if problem not in operation.solution:
            operation.solution[problem] = Solution(name=op_name, config=[])
        
        # Update the solution with new configs
        solution = operation.solution[problem]
        
        # Add new configs that aren't already in the solution
        existing_configs = {hash(c) for c in solution.config}
        for config in configs:
            if hash(config) not in existing_configs:
                solution.config.append(config)
                existing_configs.add(hash(config))

    def get_table(self) -> Table:
        """
        Get the collected data as a Table.

        Returns:
            The Table containing all collected data.
        """
        return self.table

    def save_to_file(self, file_path: str) -> None:
        """
        Save the collected data to a file.

        Args:
            file_path: Path to save the data to.
        """
        with open(file_path, 'w') as f:
            f.write(self.table.serialize())
        logger.info(f"Saved collected data to {file_path}")

    def load_from_file(self, file_path: str) -> None:
        """
        Load data from a file.

        Args:
            file_path: Path to load the data from.
        """
        with open(file_path, 'r') as f:
            content = f.read()
        
        table = Table.deserialize(content)
        if table:
            self.table = table
            logger.info(f"Loaded data from {file_path}")
        else:
            logger.error(f"Failed to load data from {file_path}")

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
