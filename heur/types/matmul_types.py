from dataclasses import dataclass, field
from typing import Any, OrderedDict, Optional, Tuple, Union, Self
from json_serializable import JSONSerializable
import json

@dataclass(kw_only=True)
class TritonGEMMConfig(JSONSerializable):
    _is_leaf: bool = True
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

    @classmethod
    def parse(cls, string: str) -> Self:
        d = json.loads(string, object_pairs_hook=OrderedDict)
        # validate types, yay python :P
        if "name" not in d:
            raise KeyError("Missing required field: name")
        if not isinstance(d["name"], str):
            raise TypeError(f"name must be a string, got {type(d['name'])}")
        if "grid" not in d:
            raise KeyError("Missing required field: grid")
        if not isinstance(d["grid"], int):
            raise TypeError(f"grid must be an int, got {type(d['grid'])}")
        if "block_m" not in d:
            raise KeyError("Missing required field: block_m")
        if not isinstance(d["block_m"], int):
            raise TypeError(f"block_m must be an int, got {type(d['block_m'])}")
        if "block_n" not in d:
            raise KeyError("Missing required field: block_n")
        if not isinstance(d["block_n"], int):
            raise TypeError(f"block_n must be an int, got {type(d['block_n'])}")
        if "block_k" not in d:
            raise KeyError("Missing required field: block_k")
        if not isinstance(d["block_k"], int):
            raise TypeError(f"block_k must be an int, got {type(d['block_k'])}")
        if "group_m" not in d:
            raise KeyError("Missing required field: group_m")
        if not isinstance(d["group_m"], int):
            raise TypeError(f"group_m must be an int, got {type(d['group_m'])}")
        if "num_stages" not in d:
            raise KeyError("Missing required field: num_stages")
        if not isinstance(d["num_stages"], int):
            raise TypeError(f"num_stages must be an int, got {type(d['num_stages'])}")
        if "num_warps" not in d:
            raise KeyError("Missing required field: num_warps")
        if not isinstance(d["num_warps"], int):
            raise TypeError(f"num_warps must be an int, got {type(d['num_warps'])}")
        if "EVEN_K" in d and not isinstance(d["EVEN_K"], bool):
            raise TypeError(f"EVEN_K must be a bool, got {type(d['EVEN_K'])}")
        if "ALLOW_TF32" in d and not isinstance(d["ALLOW_TF32"], bool):
            raise TypeError(f"ALLOW_TF32 must be a bool, got {type(d['ALLOW_TF32'])}")
        if "USE_FAST_ACCUM" in d and not isinstance(d["USE_FAST_ACCUM"], bool):
            raise TypeError(
                f"USE_FAST_ACCUM must be a bool, got {type(d['USE_FAST_ACCUM'])}"
            )
        if "ACC_TYPE" in d and not isinstance(d["ACC_TYPE"], str):
            raise TypeError(f"ACC_TYPE must be a string, got {type(d['ACC_TYPE'])}")
        return cls(**d)


@dataclass(kw_only=True)
class MMProblem(JSONSerializable):
    _is_leaf: bool = True
    B: int
    M: int
    M_dtype: torch.dtype
    N: int
    K: int
    K_dtype: torch.dtype
    out_dtype: torch.dtype
    out_size: tuple[int, int, int]
    out_stride: tuple[int, int, int]

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

    def __str__(self) -> str:
        """
        Return a string representation of the object.
        """
        d = asdict(self)
        d["M_dtype"] = str(d["M_dtype"]).split(".")[-1]
        d["K_dtype"] = str(d["K_dtype"]).split(".")[-1]
        d["out_dtype"] = str(d["out_dtype"]).split(".")[-1]
        d["out_size"] = list(d["out_size"])
        d["out_stride"] = list(d["out_stride"])
        d = OrderedDict((k, v) for k, v in d.items() if not k.startswith("_"))
        return json.dumps(d)

    @classmethod
    def parse(cls, string: str) -> Self:
        d = json.loads(string, object_pairs_hook=OrderedDict)
        # validate types, yay python :P
        if "B" not in d:
            raise KeyError("Missing required field: B")
        if not isinstance(d["B"], int):
            raise TypeError(f"B must be an int, got {type(d['B'])}")
        if "M" not in d:
            raise KeyError("Missing required field: M")
        if not isinstance(d["M"], int):
            raise TypeError(f"M must be an int, got {type(d['M'])}")
        if "N" not in d:
            raise KeyError("Missing required field: N")
        if not isinstance(d["N"], int):
            raise TypeError(f"N must be an int, got {type(d['N'])}")
        if "K" not in d:
            raise KeyError("Missing required field: K")
        if not isinstance(d["K"], int):
            raise TypeError(f"K must be an int, got {type(d['K'])}")
        if "M_dtype" not in d:
            raise KeyError("Missing required field: M_dtype")
        if not isinstance(d["M_dtype"], str):
            raise TypeError(f"M_dtype must be a string, got {type(d['M_dtype'])}")
        if "K_dtype" not in d:
            raise KeyError("Missing required field: K_dtype")
        if not isinstance(d["K_dtype"], str):
            raise TypeError(f"K_dtype must be a string, got {type(d['K_dtype'])}")
        if "out_dtype" not in d:
            raise KeyError("Missing required field: out_dtype")
        if not isinstance(d["out_dtype"], str):
            raise TypeError(f"out_dtype must be a string, got {type(d['out_dtype'])}")
        if "out_size" not in d:
            raise KeyError("Missing required field: out_size")
        if not isinstance(d["out_size"], list):
            raise TypeError(f"out_size must be a list, got {type(d['out_size'])}")
        if "out_stride" not in d:
            raise KeyError("Missing required field: out_stride")
        if not isinstance(d["out_stride"], list):
            raise TypeError(f"out_stride must be a list, got {type(d['out_stride'])}")

        # Validate torch dtype strings
        try:
            d["M_dtype"] = getattr(torch, d["M_dtype"])
        except AttributeError:
            raise ValueError(f"Invalid torch dtype: {d['M_dtype']}") from None
        try:
            d["K_dtype"] = getattr(torch, d["K_dtype"])
        except AttributeError:
            raise ValueError(f"Invalid torch dtype: {d['K_dtype']}") from None
        try:
            d["out_dtype"] = getattr(torch, d["out_dtype"])
        except AttributeError:
            raise ValueError(f"Invalid torch dtype: {d['out_dtype']}") from None

        d["out_size"] = tuple(d["out_size"])
        d["out_stride"] = tuple(d["out_stride"])
        return cls(**d)


@dataclass(kw_only=True)
class Solution(JSONSerializable):
    # like mm or addmm
    name: str
    # mapping
    config: list[TritonGEMMConfig]


@dataclass(kw_only=True)
class Operation(JSONSerializable):
    name: str
    solution: OrderedDict[MMProblem, Solution]


@dataclass(kw_only=True)
class Hardware(JSONSerializable):
    # like gfx942:sramecc+:xnack-
    operation: OrderedDict[str, Operation]


@dataclass(kw_only=True)
class Table(JSONSerializable):
    hardware: OrderedDict[str, Hardware]
    _set_cache: OrderedDict[
        tuple[str, str, MMProblem], OrderedSet[TritonGEMMConfig]
    ] = field(default_factory=OrderedDict)

    def serialize(self) -> str:
        foo = self.to_dict()
        return json.dumps(foo, indent=2)

    @classmethod
    def deserialize(cls, s: str) -> Optional[Self]:
        try:
            return cls.from_dict(json.loads(s, object_pairs_hook=OrderedDict))
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error("Failed to deserialize table: %s", e)
            return None

    def lookup(
        self, hardware: str, op_name: str, problem: MMProblem
    ) -> Optional[list[TritonGEMMConfig]]:
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
        self, hardware: str, op_name: str, problem: MMProblem
    ) -> Optional[OrderedSet[TritonGEMMConfig]]:
        """
        Easier and faster to check membership in a set, but cache the sets for runtime.
        """
        if (hardware, op_name, problem) in self._set_cache:
            return self._set_cache[(hardware, op_name, problem)]
        problem_list = self.lookup(hardware, op_name, problem)
        problem_set = OrderedSet(problem_list) if problem_list is not None else None
        if problem_set is None:
            return None
        self._set_cache[(hardware, op_name, problem)] = problem_set
        return problem_set

    def filter(
        self,
        hardware: str,
        op_name: str,
        problem: MMProblem,
        to_filter: list[TritonGEMMConfig],
    ) -> Optional[list[TritonGEMMConfig]]:
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
