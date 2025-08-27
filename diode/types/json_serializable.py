import logging
from typing import Any, Dict, TypeVar
from collections import OrderedDict
import torch

import msgpack
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from typing_extensions import Self

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="JSONSerializable")


class JSONSerializable(BaseModel):
    """
    Base class that provides JSON and MessagePack serialization capabilities using Pydantic.
    This replaces the previous dataclass-based implementation with Pydantic models for
    automatic type checking, validation, and hierarchical type support.
    """

    # Incrementing version will invalidate all LUT entries, in the case of major perf update or
    # changes to the Ontology.
    version: int = 1

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to handle torch.dtype and other special types"""
        # Remove any JSON-specific parameters that model_dump doesn't accept
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['indent']}
        
        # Get the raw data before pydantic serialization to preserve OrderedDict
        raw_data = {}
        for field_name, field_info in self.__class__.model_fields.items():
            if hasattr(self, field_name):
                raw_data[field_name] = getattr(self, field_name)
        raw_data['version'] = self.version
        
        # Apply our custom serialization to the raw data
        return self._serialize_torch_dtypes(raw_data)

    def _serialize_torch_dtypes(self, obj: Any) -> Any:
        """Recursively serialize torch.dtype objects to strings and mark OrderedDict objects"""
        if isinstance(obj, torch.dtype):
            return str(obj).split(".")[-1]
        elif isinstance(obj, JSONSerializable):
            # If it's a JSONSerializable object, serialize it recursively
            return obj.model_dump()
        elif isinstance(obj, OrderedDict):
            # Create a special marker for OrderedDict
            result = {}
            result['__ordered_dict_marker__'] = True
            result['__ordered_dict_items__'] = [(k, self._serialize_torch_dtypes(v)) for k, v in obj.items()]
            return result
        elif isinstance(obj, dict):
            return {k: self._serialize_torch_dtypes(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_torch_dtypes(item) for item in obj]
        else:
            return obj

    @classmethod
    def model_validate(cls, obj: Any, **kwargs):
        """Override model_validate to handle torch.dtype deserialization"""
        if isinstance(obj, dict):
            obj = cls._deserialize_torch_dtypes(obj)
        return super().model_validate(obj, **kwargs)

    @classmethod
    def _deserialize_torch_dtypes(cls, obj: Any) -> Any:
        """Recursively deserialize torch.dtype strings back to torch.dtype objects and restore OrderedDict objects"""
        if isinstance(obj, dict):
            # Check if this is a marked OrderedDict
            if obj.get('__ordered_dict_marker__') is True and '__ordered_dict_items__' in obj:
                # Restore OrderedDict from marked format
                result = OrderedDict()
                for k, v in obj['__ordered_dict_items__']:
                    result[k] = cls._deserialize_torch_dtypes(v)
                return result
            else:
                # Regular dict processing
                result_type = OrderedDict if isinstance(obj, OrderedDict) else dict
                result = result_type()
                for k, v in obj.items():
                    # Check if this looks like a torch dtype field
                    if isinstance(k, str) and k.endswith('_dtype') and isinstance(v, str):
                        try:
                            result[k] = getattr(torch, v)
                        except AttributeError:
                            result[k] = v  # Keep as string if not a valid torch dtype
                    else:
                        result[k] = cls._deserialize_torch_dtypes(v)
                return result
        elif isinstance(obj, (list, tuple)):
            return [cls._deserialize_torch_dtypes(item) for item in obj]
        elif isinstance(obj, str):
            # Check if this string might be a torch dtype
            # Only convert strings that are actual torch dtypes
            torch_dtype_names = {
                'float32', 'float64', 'float16', 'bfloat16',
                'int8', 'int16', 'int32', 'int64',
                'uint8', 'uint16', 'uint32', 'uint64',
                'bool', 'complex64', 'complex128'
            }
            
            if obj in torch_dtype_names:
                try:
                    return getattr(torch, obj)
                except AttributeError:
                    return obj
            elif obj.startswith('torch.') and obj[6:] in torch_dtype_names:
                try:
                    return getattr(torch, obj[6:])  # Remove 'torch.' prefix
                except AttributeError:
                    return obj
            return obj
        else:
            return obj

    @classmethod
    def from_dict(cls, inp: Dict[str, Any] | str) -> Self:
        """
        Convert a dictionary representation to the object.
        """
        if isinstance(inp, str):
            return cls.model_validate_json(inp)
        return cls.model_validate(inp)

    def to_dict(self) -> OrderedDict[str, Any]:
        """
        Convert the object to a dictionary representation.
        """
        data = self.model_dump()
        return OrderedDict(data)

    def model_dump_json(self, **kwargs) -> str:
        """Override model_dump_json to use our custom serialization"""
        import json
        # Extract JSON-specific parameters
        indent = kwargs.pop('indent', None)
        # Use our custom model_dump method and then convert to JSON
        data = self.model_dump(**kwargs)
        return json.dumps(data, indent=indent)

    def __str__(self) -> str:
        """
        Return a JSON string representation of the object.
        """
        return self.model_dump_json()

    @classmethod
    def parse(cls, string: str) -> Self:
        """
        Parse the string representation of the object.
        """
        import json
        from pydantic import ValidationError
        try:
            # Parse JSON string to dict first
            data = json.loads(string)
            # Apply our custom deserialization
            data = cls._deserialize_torch_dtypes(data)
            # Then validate with pydantic
            return cls.model_validate(data)
        except ValidationError as e:
            # Convert ValidationError to ValueError for backward compatibility
            raise ValueError(f"Validation failed: {e}") from e
        except json.JSONDecodeError as e:
            # Convert JSON decode errors to ValueError
            raise ValueError(f"Invalid JSON: {e}") from e

    def to_msgpack(self) -> bytes:
        """
        Convert the object to MessagePack format.
        Returns bytes that can be written to a file or transmitted over a network.
        """
        try:
            return msgpack.packb(self.model_dump(), use_bin_type=True)
        except Exception as e:
            logger.error(
                "Failed to serialize %s to MessagePack: %s", self.__class__.__name__, e
            )
            raise ValueError(
                f"Failed to serialize {self.__class__.__name__} to MessagePack: {e}"
            ) from e

    @classmethod
    def from_msgpack(cls, data: bytes) -> Self:
        """
        Create an object from MessagePack data.
        Takes bytes and returns an instance of the class.
        """
        try:
            decoded_dict = msgpack.unpackb(data, raw=False, strict_map_key=False)
            return cls.model_validate(decoded_dict)
        except Exception as e:
            logger.error(
                "Failed to deserialize %s from MessagePack: %s", cls.__name__, e
            )
            raise ValueError(
                f"Failed to deserialize {cls.__name__} from MessagePack: {e}"
            ) from e

    def serialize_msgpack(self) -> bytes:
        """
        Serialize the object to MessagePack format.
        Alias for to_msgpack() for consistency with existing serialize() method.
        """
        return self.to_msgpack()

    @classmethod
    def deserialize_msgpack(cls, data: bytes) -> Self:
        """
        Deserialize an object from MessagePack format.
        Alias for from_msgpack() for consistency with existing deserialize() method.
        """
        return cls.from_msgpack(data)
