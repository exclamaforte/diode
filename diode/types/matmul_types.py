import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict

import torch
from diode.types.json_serializable import JSONSerializable
from pydantic import Field, field_validator
from torch.utils._ordered_set import OrderedSet

logger = logging.getLogger(__name__)


class TritonGEMMConfig(JSONSerializable):
    name: str
    grid: int
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_stages: int
    num_warps: int
    EVEN_K: bool = False
    ALLOW_TF32: bool = False
    USE_FAST_ACCUM: bool = False
    ACC_TYPE: str = "tl.float32"
    
    # Mark as leaf class for backward compatibility with tests
    _is_leaf: bool = True

    @field_validator('name', 'ACC_TYPE')
    @classmethod
    def validate_string_fields(cls, v):
        if not isinstance(v, str):
            raise TypeError(f"Expected string, got {type(v).__name__}")
        return v

    @field_validator('grid', 'block_m', 'block_n', 'block_k', 'group_m', 'num_stages', 'num_warps')
    @classmethod
    def validate_int_fields(cls, v):
        if not isinstance(v, int) or isinstance(v, bool):
            raise TypeError(f"Expected int, got {type(v).__name__}")
        return v

    @field_validator('EVEN_K', 'ALLOW_TF32', 'USE_FAST_ACCUM')
    @classmethod
    def validate_bool_fields(cls, v):
        if not isinstance(v, bool):
            raise TypeError(f"Expected bool, got {type(v).__name__}")
        return v

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self.grid,
                self.block_m,
                self.block_n,
                self.block_k,
                self.group_m,
                self.num_stages,
                self.num_warps,
                self.EVEN_K,
                self.ALLOW_TF32,
                self.USE_FAST_ACCUM,
                self.ACC_TYPE,
            )
        )


class MMShape(JSONSerializable):
    B: int
    M: int
    M_dtype: torch.dtype
    N: int
    K: int
    K_dtype: torch.dtype
    out_dtype: torch.dtype
    out_size: Tuple[int, int, int]
    out_stride: Tuple[int, int, int]
    
    # Mark as leaf class for backward compatibility with tests
    _is_leaf: bool = True

    @field_validator('B', 'M', 'N', 'K')
    @classmethod
    def validate_int_fields(cls, v):
        if not isinstance(v, int) or isinstance(v, bool):
            raise TypeError(f"Expected int, got {type(v).__name__}")
        return v

    @field_validator('M_dtype', 'K_dtype', 'out_dtype')
    @classmethod
    def validate_dtype_fields(cls, v):
        if isinstance(v, str):
            # Try to convert string to torch.dtype
            try:
                return getattr(torch, v)
            except AttributeError:
                raise ValueError(f"Invalid torch dtype: {v}")
        elif not isinstance(v, torch.dtype):
            raise TypeError(f"Expected torch.dtype, got {type(v).__name__}")
        return v

    @field_validator('out_size', 'out_stride')
    @classmethod
    def ensure_tuple(cls, v):
        """Ensure that list values are converted to tuples to maintain hashability"""
        if isinstance(v, list):
            return tuple(v)
        return v

    def __hash__(self) -> int:
        return hash(
            (
                self.B,
                self.M,
                self.M_dtype,
                self.N,
                self.K_dtype,
                self.K,
                self.out_dtype,
                self.out_size,
                self.out_stride,
            )
        )



class Solution(JSONSerializable):
    name: str
    config: List[TritonGEMMConfig]
    
    @classmethod
    def model_validate(cls, obj: Any, **kwargs):
        """Override to call parse method for TritonGEMMConfig leaf classes"""
        if isinstance(obj, dict) and 'config' in obj:
            if isinstance(obj['config'], list):
                config_list = []
                for config_data in obj['config']:
                    # Use parse method for leaf classes
                    if hasattr(TritonGEMMConfig, '_is_leaf') and TritonGEMMConfig._is_leaf:
                        config_obj = TritonGEMMConfig.parse(json.dumps(config_data))
                    else:
                        config_obj = TritonGEMMConfig.model_validate(config_data)
                    config_list.append(config_obj)
                
                # Create a new dict with the converted config
                new_obj = dict(obj)
                new_obj['config'] = config_list
                
                # Use the standard pydantic validation with the converted data
                return super().model_validate(new_obj, **kwargs)
        
        return super().model_validate(obj, **kwargs)


class Operation(JSONSerializable):
    name: str
    solution: OrderedDict[MMShape, Solution] = Field(default_factory=OrderedDict)
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override to handle MMShape keys in solution dict"""
        # Get the basic data first
        data = {
            'name': self.name,
            'version': self.version
        }
        
        # Convert solution dict with MMShape keys to a serializable format
        if self.solution:
            solution_list = []
            for mmshape, solution in self.solution.items():
                # Serialize the MMShape key and Solution value
                solution_list.append({
                    'problem': mmshape.model_dump(),
                    'solution': solution.model_dump()
                })
            data['solution'] = solution_list
        else:
            data['solution'] = []
        
        return self._serialize_torch_dtypes(data)
    
    @classmethod
    def model_validate(cls, obj: Any, **kwargs):
        """Override to handle MMShape keys in solution dict"""
        if isinstance(obj, dict) and 'solution' in obj:
            if isinstance(obj['solution'], list):
                # Convert from serialized format back to dict with MMShape keys
                solution_dict = OrderedDict()
                for item in obj['solution']:
                    if isinstance(item, dict) and 'problem' in item and 'solution' in item:
                        # Use parse method for leaf classes
                        if hasattr(MMShape, '_is_leaf') and MMShape._is_leaf:
                            problem = MMShape.parse(json.dumps(item['problem']))
                        else:
                            problem = MMShape.model_validate(item['problem'])
                        solution = Solution.model_validate(item['solution'])
                        solution_dict[problem] = solution
                  
                # Create a new dict with the converted solution
                new_obj = dict(obj)
                new_obj['solution'] = solution_dict
                  
                # Use the standard pydantic validation with the converted data
                return super().model_validate(new_obj, **kwargs)
          
        return super().model_validate(obj, **kwargs)


class Hardware(JSONSerializable):
    operation: OrderedDict[str, Operation] = Field(default_factory=OrderedDict)
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override to handle Operation serialization"""
        data = {
            'version': self.version,
            'operation': {}
        }
        
        # Serialize each operation
        for op_name, operation in self.operation.items():
            data['operation'][op_name] = operation.model_dump()
        
        return self._serialize_torch_dtypes(data)
    
    @classmethod
    def model_validate(cls, obj: Any, **kwargs):
        """Override to handle Operation deserialization"""
        if isinstance(obj, dict) and 'operation' in obj:
            operations = OrderedDict()
            for op_name, op_data in obj['operation'].items():
                operations[op_name] = Operation.model_validate(op_data)
            
            # Create a new dict with the converted operations
            new_obj = dict(obj)
            new_obj['operation'] = operations
            
            # Use the standard pydantic validation with the converted data
            return super().model_validate(new_obj, **kwargs)
        
        return super().model_validate(obj, **kwargs)


class ShapeSet(JSONSerializable):
    """
    A collection of MMShape objects that can be serialized to JSON.
    This class is used to store and manage sets of matrix multiplication shapes.
    """

    shapes: List[MMShape] = Field(default_factory=list)

    def add_shape(self, shape: MMShape) -> None:
        """
        Add a shape to the collection.

        Args:
            shape: The MMShape to add
        """
        self.shapes.append(shape)


class OperationShapeSet(JSONSerializable):
    """
    A collection of ShapeSets organized by operation name.
    This class maps operation names (like 'mm', 'addmm', 'bmm') to their corresponding ShapeSets.
    """

    operations: Dict[str, ShapeSet] = Field(default_factory=dict)

    def add_shape(self, op_name: str, shape: MMShape) -> None:
        """
        Add a shape to the collection under a specific operation name.

        Args:
            op_name: The operation name (e.g., 'mm', 'addmm', 'bmm')
            shape: The MMShape to add
        """
        if op_name not in self.operations:
            self.operations[op_name] = ShapeSet()
        self.operations[op_name].add_shape(shape)

    def get_operation_names(self) -> List[str]:
        """
        Get all operation names in this collection.

        Returns:
            List of operation names
        """
        return list(self.operations.keys())

    def get_shapes_for_operation(self, op_name: str) -> List[MMShape]:
        """
        Get all shapes for a specific operation.

        Args:
            op_name: The operation name

        Returns:
            List of MMShape objects for the operation, or empty list if operation not found
        """
        if op_name in self.operations:
            return self.operations[op_name].shapes
        return []

    def serialize(self) -> str:
        """
        Serialize the OperationShapeSet to a JSON string.

        Returns:
            A JSON string representation of the OperationShapeSet
        """
        return self.model_dump_json(indent=2)

    @classmethod
    def deserialize(cls, s: str):
        """
        Deserialize a JSON string into an OperationShapeSet.

        Args:
            s: The JSON string to deserialize

        Returns:
            An OperationShapeSet object
        """
        try:
            return cls.model_validate_json(s)
        except Exception as e:
            logger.error("Failed to deserialize OperationShapeSet: %s", e)
            return None


class Table(JSONSerializable):
    hardware: OrderedDict[str, Hardware] = Field(default_factory=OrderedDict)
    set_cache: Dict[Tuple[str, str, MMShape], OrderedSet[TritonGEMMConfig]] = Field(
        default_factory=dict, exclude=True
    )

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override to handle Hardware serialization"""
        data = {
            'version': self.version,
            'hardware': {}
        }
        
        # Serialize each hardware
        for hw_name, hardware in self.hardware.items():
            data['hardware'][hw_name] = hardware.model_dump()
        
        return self._serialize_torch_dtypes(data)
    
    @classmethod
    def model_validate(cls, obj: Any, **kwargs):
        """Override to handle Hardware deserialization"""
        if isinstance(obj, dict) and 'hardware' in obj:
            hardware_dict = OrderedDict()
            for hw_name, hw_data in obj['hardware'].items():
                hardware_dict[hw_name] = Hardware.model_validate(hw_data)
            
            # Create a new dict with the converted hardware
            new_obj = dict(obj)
            new_obj['hardware'] = hardware_dict
            
            # Use the standard pydantic validation with the converted data
            return super().model_validate(new_obj, **kwargs)
        
        return super().model_validate(obj, **kwargs)

    def serialize(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def deserialize(cls, s: str):
        try:
            import json
            data = json.loads(s)
            return cls.model_validate(data)
        except Exception as e:
            logger.error("Failed to deserialize table: %s", e)
            return None

    def lookup(
        self, hardware: str, op_name: str, problem: MMShape
    ) -> Optional[List[TritonGEMMConfig]]:
        """
        Lookup the best TritonGEMMConfig for a given problem.
        """
        if hardware not in self.hardware:
            return None
        tmp = self.hardware[hardware].operation
        if op_name not in tmp:
            return None
        tmp = tmp[op_name].solution
        if problem not in tmp:
            return None
        return tmp[problem].config

    def lookup_set(
        self, hardware: str, op_name: str, problem: MMShape
    ) -> Optional[OrderedSet[TritonGEMMConfig]]:
        """
        Easier and faster to check membership in a set, but cache the sets for runtime.
        """
        if (hardware, op_name, problem) in self.set_cache:
            return self.set_cache[(hardware, op_name, problem)]
        problem_list = self.lookup(hardware, op_name, problem)
        problem_set = OrderedSet(problem_list) if problem_list is not None else None
        if problem_set is None:
            return None
        self.set_cache[(hardware, op_name, problem)] = problem_set
        return problem_set

    def filter(
        self,
        hardware: str,
        op_name: str,
        problem: MMShape,
        to_filter: List[TritonGEMMConfig],
    ) -> Optional[List[TritonGEMMConfig]]:
        """
        Filter a list of TritonGEMMConfig for a given problem.
        """
        problem_set = self.lookup_set(hardware, op_name, problem)
        if problem_set is None:
            return None
        ret = [x for x in to_filter if x in problem_set]
        if len(ret) == 0:
            return None
        return ret
