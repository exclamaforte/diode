"""
Simple script to collect matrix multiplication timing data using the feedback saver mechanism.
"""

import torch
import os
import sys
import logging
import time
import json
from collections import OrderedDict
from typing import Dict, List, Any

# Add the parent directory to the path so we can import the diode module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch._inductor.select_algorithm import add_feedback_saver, clear_feedback_saver
from diode.types.matmul_types import MMProblem, TritonGEMMConfig
from diode.types.matmul_dataset import Dataset, TimedConfig, DatasetSolution, DatasetOperation, DatasetHardware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def collect_data(output_file: str, num_shapes: int = 5):
    """
    Collect matrix multiplication timing data.
    
    Args:
        output_file: Path to save the collected data
        num_shapes: Number of matrix shapes to test
    """
    # Get the hardware name
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    logger.info(f"Collecting data on device: {device_name}")
    
    # Create a dataset
    dataset = Dataset(hardware=OrderedDict())
    
    # Set up PyTorch for compilation
    torch.set_grad_enabled(False)
    
    # Configure PyTorch inductor
    from torch._inductor import config
    config.fx_graph_cache = False
    config.force_disable_caches = True
    
    # Try to set environment variable for EXHAUSTIVE search, but fall back to DEFAULT if triton is not available
    try:
        import triton
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE"] = "DEFAULT"  # Using DEFAULT for faster testing
        logger.info("Set search space to DEFAULT")
    except ImportError:
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE"] = "DEFAULT"
        logger.info("Triton not available, using DEFAULT search space")
    
    # Define matrix sizes
    sizes = [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ][:num_shapes]
    
    # Define dtypes
    dtypes = [torch.float16, torch.float32] if torch.cuda.is_available() else [torch.float32]
    
    # Set up the feedback saver
    def feedback_handler(timings: Dict, name: str, input_nodes: List, choices: Any, profiled_time: float):
        logger.info(f"Feedback handler called with name: {name}, timings: {len(timings)}")
        
        # Only handle matrix multiplication operations
        if name not in ["mm", "addmm"]:
            logger.info(f"Skipping operation: {name} (not mm or addmm)")
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
        
        logger.info(f"Processing {name} with shape ({M}, {K}) x ({K}, {N}) and dtype {M_dtype}")
        
        # Create MMProblem instance
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
            # Get or create hardware entry
            if device_name not in dataset.hardware:
                dataset.hardware[device_name] = DatasetHardware(operation=OrderedDict())
            
            hardware = dataset.hardware[device_name]
            
            # Get or create operation entry
            if name not in hardware.operation:
                hardware.operation[name] = DatasetOperation(name=name, solution=OrderedDict())
            
            operation = hardware.operation[name]
            
            # Get or create solution entry
            if problem not in operation.solution:
                operation.solution[problem] = DatasetSolution(name=name, timed_configs=[])
            
            solution = operation.solution[problem]
            
            # Add the timed config
            timed_config = TimedConfig(config=config, time=bench_time)
            solution.timed_configs.append(timed_config)
            
            logger.info(f"Added timing for {name} with block_m={block_m}, block_n={block_n}, block_k={block_k}, time={bench_time*1000:.3f} ms")
    
    # Register the feedback handler
    add_feedback_saver(feedback_handler)
    logger.info("Registered feedback handler")
    
    # Run matrix multiplications with different sizes and dtypes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_shapes = len(sizes) * len(dtypes)
    
    logger.info(f"Running {total_shapes} matrix multiplications...")
    start_time = time.time()
    
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
            logger.info(f"[{i+1}/{len(sizes)}] Running mm with size ({M}, {K}) x ({K}, {N}) and dtype {dtype}")
            compiled_mm = torch.compile(mm_fn, mode="max-autotune")
            result_mm = compiled_mm(a, b)
            
            # Compile and run addmm
            logger.info(f"[{i+1}/{len(sizes)}] Running addmm with size ({M}, {K}) x ({K}, {N}) and dtype {dtype}")
            compiled_addmm = torch.compile(addmm_fn, mode="max-autotune")
            result_addmm = compiled_addmm(c, a, b)
    
    # Clear the feedback saver
    clear_feedback_saver()
    logger.info("Cleared feedback handler")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Collection completed in {elapsed_time:.2f} seconds")
    
    # Save the collected dataset to a file
    with open(output_file, 'w') as f:
        f.write(dataset.serialize())
    logger.info(f"Saved collected dataset to {output_file}")
    
    # Print statistics about the collected data
    print_dataset_statistics(dataset)

def print_dataset_statistics(dataset: Dataset) -> None:
    """
    Print statistics about the collected data.
    
    Args:
        dataset: The Dataset containing the collected data
    """
    print("\nDataset Statistics:")
    print("------------------")
    
    # Count the number of hardware entries
    hardware_count = len(dataset.hardware)
    print(f"Number of hardware entries: {hardware_count}")
    
    # For each hardware, count operations and problems
    for hw_name, hardware in dataset.hardware.items():
        print(f"\nHardware: {hw_name}")
        
        op_count = len(hardware.operation)
        print(f"  Number of operations: {op_count}")
        
        for op_name, operation in hardware.operation.items():
            problem_count = len(operation.solution)
            config_count = sum(len(solution.timed_configs) for solution in operation.solution.values())
            print(f"  Operation '{op_name}': {problem_count} problems, {config_count} configs")
            
            # Print details of a few problems
            for i, (problem, solution) in enumerate(operation.solution.items()):
                if i >= 3:  # Limit to 3 problems for brevity
                    print(f"    ... and {problem_count - 3} more problems")
                    break
                
                print(f"    Problem {i+1}: M={problem.M}, N={problem.N}, K={problem.K}, "
                      f"dtype={problem.M_dtype}, {len(solution.timed_configs)} configs")
                
                # Print the fastest config for each problem
                if solution.timed_configs:
                    fastest_config = min(solution.timed_configs, key=lambda tc: tc.time)
                    print(f"      Fastest config: block_m={fastest_config.config.block_m}, "
                          f"block_n={fastest_config.config.block_n}, "
                          f"block_k={fastest_config.config.block_k}, "
                          f"time={fastest_config.time*1000:.3f} ms")

if __name__ == "__main__":
    collect_data("simple_matmul_dataset.json")
