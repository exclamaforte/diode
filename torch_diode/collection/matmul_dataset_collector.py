import logging
import time
from collections import OrderedDict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Import the size_hints function from PyTorch inductor
import torch._inductor.config as inductor_config
from torch._inductor.select_algorithm import add_feedback_saver, clear_feedback_savers

from torch_diode.types.matmul_dataset import Dataset
from torch_diode.types.matmul_types import (
    MMShape,
    OperationShapeSet,
    Table,
    TritonGEMMConfig,
)
from torch_diode.utils.dataset_utils import generate_matrix_sizes

logger = logging.getLogger(__name__)


class CollectionMode(Enum):
    """Enumeration for data collection modes."""

    RANDOM = "random"
    OPERATION_SHAPE_SET = "operation_shape_set"
    LOG_NORMAL = "log_normal"


class MatmulDatasetCollector:
    """
    A class that hooks into the PyTorch feedback saver interface to collect
    matrix multiplication data with timing information and store it in a Dataset.

    Supports two modes:
    1. RANDOM: Generate random matrix shapes and collect data for specified operations
    2. OPERATION_SHAPE_SET: Use predefined shapes from an OperationShapeSet
    """

    def __init__(
        self,
        hardware_name: str = "unknown",
        mode: CollectionMode = CollectionMode.RANDOM,
        operations: Optional[List[str]] = None,
        operation_shape_set: Optional[OperationShapeSet] = None,
        # Random mode parameters
        num_shapes: int = 100,
        dtypes: Optional[List[torch.dtype]] = None,
        seed: int = 42,
        min_size: int = 32,
        max_size: int = 4096,
        power_of_two: bool = False,
        # Log normal mode parameters
        log_normal_m_mean: float = 6.5725472164323095,
        log_normal_m_std: float = 2.556199441605505,
        log_normal_n_mean: float = 5.913930073563466,
        log_normal_n_std: float = 1.66968141897024,
        log_normal_k_mean: float = 6.204916071423808,
        log_normal_k_std: float = 2.1646646856090177,
    ):
        """
        Initialize the MatmulDatasetCollector.

        Args:
            hardware_name: The name of the hardware being used.
            mode: Collection mode (RANDOM or OPERATION_SHAPE_SET)
            operations: List of operations to collect data for (e.g., ['mm', 'addmm'])
            operation_shape_set: OperationShapeSet for OPERATION_SHAPE_SET mode
            num_shapes: Number of random shapes to generate (RANDOM mode)
            dtypes: List of dtypes to test
            seed: Random seed for reproducibility (RANDOM mode)
            min_size: Minimum matrix dimension (RANDOM mode)
            max_size: Maximum matrix dimension (RANDOM mode)
            power_of_two: Whether to generate only power-of-two sizes (RANDOM mode)
        """
        self.hardware_name = hardware_name
        self.mode = mode
        self.operations = operations or ["mm", "addmm"]
        self.operation_shape_set = operation_shape_set

        # Random mode parameters
        self.num_shapes = num_shapes
        self.dtypes = dtypes or (
            [torch.float16, torch.float32]
            if torch.cuda.is_available()
            else [torch.float32]
        )
        self.seed = seed
        self.min_size = min_size
        self.max_size = max_size
        self.power_of_two = power_of_two

        # Log normal mode parameters
        self.log_normal_m_mean = log_normal_m_mean
        self.log_normal_m_std = log_normal_m_std
        self.log_normal_n_mean = log_normal_n_mean
        self.log_normal_n_std = log_normal_n_std
        self.log_normal_k_mean = log_normal_k_mean
        self.log_normal_k_std = log_normal_k_std

        self.dataset = Dataset(hardware=OrderedDict())
        self._is_collecting = False

        # Validate parameters
        if mode == CollectionMode.OPERATION_SHAPE_SET and operation_shape_set is None:
            raise ValueError(
                "operation_shape_set must be provided when mode is OPERATION_SHAPE_SET"
            )

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

        clear_feedback_savers()
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
                    mat1.layout.size, fallback=inductor_config.unbacked_symint_fallback
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
                    mat2.layout.size, fallback=inductor_config.unbacked_symint_fallback
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

    def _feedback_handler(
        self,
        timings: Dict,
        name: str,
        input_nodes: List,
        choices: Any,
        profiled_time: float,
    ) -> None:
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
        logger.debug(
            f"Feedback handler called with name: {name}, timings: {len(timings)}"
        )

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
            if not isinstance(
                choice, torch._inductor.select_algorithm.TritonTemplateCaller
            ):
                continue

            # Extract configuration details
            log_info = choice.log_info
            block_m, block_k, block_n = map(
                int, log_info.get("tile_shape", "(0,0,0)").strip("()").split(",")
            )

            # Create TritonGEMMConfig instance
            config = TritonGEMMConfig(
                name=f"{name}_config",
                grid=1,  # Default value, not available in feedback
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                group_m=log_info.get("GROUP_M", 1),
                num_stages=log_info.get("num_stages", 1),
                num_warps=log_info.get("num_warps", 4),
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
        with open(file_path, "w") as f:
            f.write(self.dataset.serialize())
        logger.info(f"Saved collected data to {file_path}")

    def load_from_file(self, file_path: str) -> None:
        """
        Load data from a file.

        Args:
            file_path: Path to load the data from.
        """
        with open(file_path) as f:
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
        with open(file_path, "w") as f:
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

    def _round_small_dimension(self, dim: int) -> int:
        """
        round to nearest 8

        Args:
            dim: The dimension to potentially round

        Returns:
            The rounded dimension
        """
        # Round to nearest multiple of 8
        rounded = round(dim / 8) * 8
        # If rounding resulted in 0, use 8 instead
        return max(8, rounded)

    def _generate_log_normal_sizes(self) -> List[Tuple[int, int, int]]:
        """
        Generate matrix sizes using log normal distribution.

        Returns:
            List of (M, N, K) tuples sampled from log normal distributions
        """
        np.random.seed(self.seed)
        sizes = []

        for _ in range(self.num_shapes):
            # Sample M, N, K from separate log normal distributions
            m = int(np.random.lognormal(self.log_normal_m_mean, self.log_normal_m_std))
            n = int(np.random.lognormal(self.log_normal_n_mean, self.log_normal_n_std))
            k = int(np.random.lognormal(self.log_normal_k_mean, self.log_normal_k_std))

            # Round small dimensions to nearest multiple of 8
            m = self._round_small_dimension(m)
            n = self._round_small_dimension(n)
            k = self._round_small_dimension(k)

            sizes.append((m, k, n))

        return sizes

    def _generate_shapes_and_dtypes(
        self,
    ) -> List[Tuple[Tuple[int, int, int], torch.dtype, str]]:
        """
        Generate shapes and dtypes based on the collection mode.
        For sample efficiency, randomly samples one dtype and one operation per shape.

        Returns:
            List of tuples containing (shape, dtype, operation_name)
        """
        shapes_and_dtypes = []

        # Set up random number generator for consistent sampling
        rng = np.random.RandomState(self.seed)

        if self.mode == CollectionMode.RANDOM:
            # Generate random matrix sizes
            sizes = generate_matrix_sizes(
                num_shapes=self.num_shapes,
                seed=self.seed,
                min_size=self.min_size,
                max_size=self.max_size,
                power_of_two=self.power_of_two,
            )

            # Sample one dtype and one operation for each size
            for size in sizes:
                dtype = rng.choice(self.dtypes)
                op_name = rng.choice(self.operations)
                shapes_and_dtypes.append((size, dtype, op_name))

        elif self.mode == CollectionMode.LOG_NORMAL:
            # Generate log normal distributed matrix sizes
            sizes = self._generate_log_normal_sizes()

            # Sample one dtype and one operation for each size
            for size in sizes:
                dtype = rng.choice(self.dtypes)
                op_name = rng.choice(self.operations)
                shapes_and_dtypes.append((size, dtype, op_name))

        elif self.mode == CollectionMode.OPERATION_SHAPE_SET:
            # Use shapes from the OperationShapeSet
            if self.operation_shape_set is not None:
                # Collect all shapes from all operations first
                all_shapes = []
                for op_name in self.operations:
                    if op_name in self.operation_shape_set.operations:
                        shapes = self.operation_shape_set.get_shapes_for_operation(
                            op_name
                        )
                        for shape in shapes:
                            # Convert MMShape to (M, K, N) tuple and extract dtype
                            size = (shape.M, shape.K, shape.N)
                            dtype = shape.M_dtype  # Use M_dtype as the primary dtype
                            all_shapes.append((size, dtype))
                    else:
                        logger.warning(
                            f"Operation '{op_name}' not found in OperationShapeSet"
                        )

                # Sample one operation for each unique shape
                unique_shapes = list(set(all_shapes))
                for size, dtype in unique_shapes:
                    op_name = rng.choice(self.operations)
                    shapes_and_dtypes.append((size, dtype, op_name))
            else:
                logger.error(
                    "OperationShapeSet is None but mode is OPERATION_SHAPE_SET"
                )

        return shapes_and_dtypes

    def _estimate_memory_usage(
        self,
        size: Tuple[int, int, int],
        dtype: torch.dtype,
        op_name: str,
        device: str,
    ) -> float:
        """
        Estimate the memory usage in bytes for a matrix multiplication operation.

        Args:
            size: Matrix size as (M, K, N) tuple
            dtype: Data type for the matrices
            op_name: Operation name ('mm', 'addmm', or 'bmm')
            device: Device to run on

        Returns:
            Estimated memory usage in bytes
        """
        M, K, N = size

        # Get the size of the dtype in bytes
        if dtype == torch.float16:
            dtype_size = 2
        elif dtype == torch.float32:
            dtype_size = 4
        elif dtype == torch.float64:
            dtype_size = 8
        elif dtype == torch.int8:
            dtype_size = 1
        elif dtype == torch.int16:
            dtype_size = 2
        elif dtype == torch.int32:
            dtype_size = 4
        elif dtype == torch.int64:
            dtype_size = 8
        else:
            # Default to float32 size
            dtype_size = 4

        if op_name == "mm":
            # Two input matrices: (M, K) and (K, N), plus output (M, N)
            memory_usage = (M * K + K * N + M * N) * dtype_size
        elif op_name == "addmm":
            # Three input matrices: (M, K), (K, N), (M, N), plus output (M, N)
            memory_usage = (M * K + K * N + M * N + M * N) * dtype_size
        elif op_name == "bmm":
            # Two input matrices: (1, M, K) and (1, K, N), plus output (1, M, N)
            # Using B=1 as in the implementation
            memory_usage = (M * K + K * N + M * N) * dtype_size
        else:
            # Default estimation
            memory_usage = (M * K + K * N + M * N) * dtype_size

        # Add some overhead for intermediate computations
        memory_usage *= 1.5

        return memory_usage

    def _check_memory_feasible(
        self,
        size: Tuple[int, int, int],
        dtype: torch.dtype,
        op_name: str,
        device: str,
    ) -> bool:
        """
        Check if the matrix multiplication operation is feasible given available memory.

        Args:
            size: Matrix size as (M, K, N) tuple
            dtype: Data type for the matrices
            op_name: Operation name ('mm', 'addmm', or 'bmm')
            device: Device to run on

        Returns:
            True if the operation is feasible, False otherwise
        """
        # Estimate required memory for this operation
        estimated_memory = self._estimate_memory_usage(size, dtype, op_name, device)

        if device == "cpu":
            # For CPU, set a reasonable upper limit (e.g., 32GB)
            # This prevents extremely large operations that would crash the system
            cpu_memory_limit = 32 * 1024**3  # 32 GB in bytes
            is_feasible = estimated_memory <= cpu_memory_limit

            if not is_feasible:
                M, K, N = size
                logger.warning(
                    f"Skipping {op_name} with size ({M}, {K}, {N}) and dtype {dtype}: "
                    f"estimated memory {estimated_memory / (1024**3):.2f} GB exceeds "
                    f"CPU memory limit {cpu_memory_limit / (1024**3):.2f} GB"
                )

            return is_feasible

        try:
            # Get available GPU memory
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            available_memory = total_memory - allocated_memory

            is_feasible = estimated_memory <= available_memory

            if not is_feasible:
                M, K, N = size
                logger.warning(
                    f"Skipping {op_name} with size ({M}, {K}, {N}) and dtype {dtype}: "
                    f"estimated memory {estimated_memory / (1024**3):.2f} GB exceeds "
                    f"safe limit {available_memory / (1024**3):.2f} GB"
                )

            return is_feasible
        except Exception as e:
            logger.warning(
                f"Could not check GPU memory, proceeding with operation: {e}"
            )
            return True

    def _run_matrix_multiplication(
        self,
        size: Tuple[int, int, int],
        dtype: torch.dtype,
        op_name: str,
        device: str,
        search_mode: str,
    ) -> None:
        """
        Run a single matrix multiplication operation.

        Args:
            size: Matrix size as (M, K, N) tuple
            dtype: Data type for the matrices
            op_name: Operation name ('mm', 'addmm', or 'bmm')
            device: Device to run on
            search_mode: Search mode for torch.compile
        """
        M, K, N = size

        # Check if the operation is memory-feasible before proceeding
        if not self._check_memory_feasible(size, dtype, op_name, device):
            logger.info(
                f"Skipping {op_name} with size ({M}, {K}, {N}) due to memory constraints"
            )
            return

        try:
            if op_name == "mm":
                # Create input matrices for mm: (M, K) x (K, N) -> (M, N)
                a = torch.randn(M, K, device=device, dtype=dtype)
                b = torch.randn(K, N, device=device, dtype=dtype)

                # Validate tensor shapes before compilation
                logger.debug(f"Created tensors: a.shape={a.shape}, b.shape={b.shape}")
                assert a.shape == (
                    M,
                    K,
                ), f"Tensor a shape mismatch: expected ({M}, {K}), got {a.shape}"
                assert b.shape == (
                    K,
                    N,
                ), f"Tensor b shape mismatch: expected ({K}, {N}), got {b.shape}"

                def mm_fn(x, y):
                    return torch.mm(x, y)

                try:
                    compiled_fn = torch.compile(mm_fn, mode=search_mode)
                    result = compiled_fn(a, b)
                except Exception as compile_error:
                    logger.error(
                        f"Failed to compile/run {op_name} with size ({M}, {K}, {N}) and dtype {dtype} "
                        f"during exhaustive autotuning: {compile_error}"
                    )
                    return

            elif op_name == "addmm":
                # Create input matrices for addmm: bias(M, N) + (M, K) x (K, N) -> (M, N)
                a = torch.randn(M, K, device=device, dtype=dtype)
                b = torch.randn(K, N, device=device, dtype=dtype)
                c = torch.randn(M, N, device=device, dtype=dtype)

                def addmm_fn(bias, x, y):
                    return torch.addmm(bias, x, y)

                try:
                    compiled_fn = torch.compile(addmm_fn, mode=search_mode)
                    result = compiled_fn(c, a, b)
                except Exception as compile_error:
                    logger.error(
                        f"Failed to compile/run {op_name} with size ({M}, {K}, {N}) and dtype {dtype} "
                        f"during exhaustive autotuning: {compile_error}"
                    )
                    return

            elif op_name == "bmm":
                # Create input matrices for bmm: (B, M, K) x (B, K, N) -> (B, M, N)
                # For bmm, we'll use B=1 for simplicity
                B = 1
                a = torch.randn(B, M, K, device=device, dtype=dtype)
                b = torch.randn(B, K, N, device=device, dtype=dtype)

                def bmm_fn(x, y):
                    return torch.bmm(x, y)

                try:
                    compiled_fn = torch.compile(bmm_fn, mode=search_mode)
                    result = compiled_fn(a, b)
                except Exception as compile_error:
                    logger.error(
                        f"Failed to compile/run {op_name} with size ({M}, {K}, {N}) and dtype {dtype} "
                        f"during exhaustive autotuning: {compile_error}"
                    )
                    return

            else:
                logger.warning(f"Unsupported operation: {op_name}")
                return

        except Exception as e:
            logger.error(
                f"Error creating tensors for {op_name} with size ({M}, {K}, {N}) and dtype {dtype}: {e}"
            )
            raise

    def collect_data(
        self,
        search_mode: str = "max-autotune",
        search_space: str = "EXHAUSTIVE",
        device: Optional[str] = None,
    ) -> None:
        """
        Collect matrix multiplication timing data based on the configured mode.

        Args:
            search_mode: Search mode for torch.compile
            search_space: Search space for autotuning (EXHAUSTIVE or DEFAULT)
            device: Device to run on (defaults to cuda if available, else cpu)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Collecting data on device: {device}")
        logger.info(f"Collection mode: {self.mode.value}")
        logger.info(f"Operations: {self.operations}")

        # Set up PyTorch for compilation
        torch.set_grad_enabled(False)

        # Configure PyTorch inductor
        from torch._inductor import config

        config.fx_graph_cache = False
        config.force_disable_caches = True
        config.max_autotune_gemm_backends = "TRITON"
        config.triton.num_decompose_k_splits = 0

        # Set search space - this needs to be set before importing torch._inductor.config
        # but since it's already imported, we need to set it directly on the config object
        if search_space == "EXHAUSTIVE":
            config.max_autotune_gemm_search_space = "EXHAUSTIVE"
            logger.info("Set search space to EXHAUSTIVE")
        else:
            config.max_autotune_gemm_search_space = "DEFAULT"
            logger.info("Set search space to DEFAULT")

        # Generate shapes and dtypes
        shapes_and_dtypes = self._generate_shapes_and_dtypes()
        total_operations = len(shapes_and_dtypes)

        logger.info(f"Running {total_operations} matrix multiplications...")
        start_time = time.time()

        # Start collection
        self.start_collection()

        try:
            # Run matrix multiplications
            for i, (size, dtype, op_name) in enumerate(shapes_and_dtypes):
                M, K, N = size
                logger.info(
                    f"[{i + 1}/{total_operations}] Running {op_name} with size "
                    f"({M}, {K}) x ({K}, {N}) and dtype {dtype}"
                )

                # Clear compilation cache to avoid shape conflicts
                torch._dynamo.reset()

                self._run_matrix_multiplication(
                    size, dtype, op_name, device, search_mode
                )

        finally:
            # Stop collection
            self.stop_collection()

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"Collection completed in {elapsed_time:.2f} seconds")
