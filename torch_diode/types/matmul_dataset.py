"""
Dataset class for storing matrix multiplication data with timing information.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, OrderedDict, Tuple
import json
import logging
import torch
from torch.utils._ordered_set import OrderedSet

from torch_diode.types.json_serializable import JSONSerializable
from torch_diode.types.matmul_types import (
    TritonGEMMConfig,
    MMShape,
    Solution,
    Operation,
    Hardware,
    Table,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class TimedConfig(JSONSerializable):
    """
    A configuration with its execution time.
    """
    config: TritonGEMMConfig
    time: float  # Execution time in seconds

    def __hash__(self) -> int:
        return hash((hash(self.config), self.time))


@dataclass(kw_only=True)
class DatasetSolution(JSONSerializable):
    """
    A solution with timed configurations.
    """
    timed_configs: List[TimedConfig]


@dataclass(kw_only=True)
class DatasetOperation(JSONSerializable):
    """
    An operation with solutions for different problems.
    """
    solution: OrderedDict[MMShape, DatasetSolution]


@dataclass(kw_only=True)
class DatasetHardware(JSONSerializable):
    """
    Hardware with operations.
    """
    operation: OrderedDict[str, DatasetOperation]


@dataclass(kw_only=True)
class Dataset(JSONSerializable):
    """
    Dataset for storing matrix multiplication data with timing information.
    """
    hardware: OrderedDict[str, DatasetHardware]

    def serialize(self) -> str:
        """
        Serialize the dataset to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def deserialize(cls, s: str):
        """
        Deserialize a JSON string to a Dataset.
        """
        try:
            return cls.from_dict(json.loads(s, object_pairs_hook=OrderedDict))
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error("Failed to deserialize dataset: %s", e)
            return None

    def add_timing(
        self,
        hardware_name: str,
        op_name: str,
        problem: MMShape,
        config: TritonGEMMConfig,
        time: float,
    ) -> None:
        """
        Add a timing for a configuration.
        
        Args:
            hardware_name: Name of the hardware
            op_name: Name of the operation (e.g., "mm", "addmm")
            problem: The matrix multiplication problem
            config: The configuration used
            time: The execution time in seconds
        """
        # Get or create hardware entry
        if hardware_name not in self.hardware:
            self.hardware[hardware_name] = DatasetHardware(operation=OrderedDict())
        
        hardware = self.hardware[hardware_name]
        
        # Get or create operation entry
        if op_name not in hardware.operation:
            hardware.operation[op_name] = DatasetOperation(solution=OrderedDict())
          
        operation = hardware.operation[op_name]
          
        # Get or create solution entry
        if problem not in operation.solution:
            operation.solution[problem] = DatasetSolution(timed_configs=[])
        
        solution = operation.solution[problem]
        
        # Add the timed config
        timed_config = TimedConfig(config=config, time=time)
        solution.timed_configs.append(timed_config)

    def to_table(self) -> Table:
        """
        Convert the dataset to a table by selecting the fastest configuration for each problem.
        
        Returns:
            A Table with the fastest configuration for each problem.
        """
        table = Table(hardware=OrderedDict())
        
        for hw_name, hardware in self.hardware.items():
            table_hw = Hardware(operation=OrderedDict())
            table.hardware[hw_name] = table_hw
            
            for op_name, operation in hardware.operation.items():
                table_op = Operation(solution=OrderedDict())
                table_hw.operation[op_name] = table_op
                  
                for problem, solution in operation.solution.items():
                    # Find the fastest configuration
                    if not solution.timed_configs:
                        continue
                      
                    # Sort by time (ascending)
                    sorted_configs = sorted(solution.timed_configs, key=lambda tc: tc.time)
                      
                    # Create a solution with the fastest config
                    table_solution = Solution(
                        config=[tc.config for tc in sorted_configs]
                    )
                      
                    table_op.solution[problem] = table_solution
        
        return table

    def get_fastest_configs(self) -> Dict[Tuple[str, str, MMShape], TritonGEMMConfig]:
        """
        Get the fastest configuration for each (hardware, operation, problem) tuple.
        
        Returns:
            A dictionary mapping (hardware_name, op_name, problem) to the fastest configuration.
        """
        result = {}
        
        for hw_name, hardware in self.hardware.items():
            for op_name, operation in hardware.operation.items():
                for problem, solution in operation.solution.items():
                    if not solution.timed_configs:
                        continue
                    
                    # Find the fastest configuration
                    fastest_config = min(solution.timed_configs, key=lambda tc: tc.time)
                    
                    result[(hw_name, op_name, problem)] = fastest_config.config
        
        return result
