"""
Example script demonstrating how to use the MatmulCollector class.

This script shows how to collect matrix multiplication data from PyTorch operations
and store it in structured types.
"""

import torch
import platform
from diode.collection.matmul_collector import MatmulCollector

def run_example():
    # Get the hardware name (e.g., GPU model)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device_name}")
    
    # Create a collector instance with the hardware name
    collector = MatmulCollector(hardware_name=device_name)
    
    # Example 1: Using start_collection and stop_collection methods
    print("\nExample 1: Using start_collection and stop_collection methods")
    collector.start_collection()
    
    # Run some matrix multiplication operations
    print("Running matrix multiplications...")
    run_matrix_multiplications()
    
    # Stop collection
    collector.stop_collection()
    
    # Save the collected data to a file
    output_file = "matmul_data_example1.json"
    collector.save_to_file(output_file)
    print(f"Saved collected data to {output_file}")
    
    # Example 2: Using the collector as a context manager
    print("\nExample 2: Using the collector as a context manager")
    # Create a new collector for this example
    collector = MatmulCollector(hardware_name=device_name)
    
    # Use the collector as a context manager
    with collector:
        print("Running matrix multiplications...")
        run_matrix_multiplications()
    
    # Save the collected data to a file
    output_file = "matmul_data_example2.json"
    collector.save_to_file(output_file)
    print(f"Saved collected data to {output_file}")
    
    # Example 3: Loading data from a file
    print("\nExample 3: Loading data from a file")
    new_collector = MatmulCollector()
    new_collector.load_from_file(output_file)
    print(f"Loaded data from {output_file}")
    
    # Print some statistics about the collected data
    print_statistics(new_collector)

def run_matrix_multiplications():
    """Run various matrix multiplication operations to collect data."""
    # Set up PyTorch for compilation
    torch.set_grad_enabled(False)
    
    # Define matrix sizes to test
    sizes = [
        (32, 64, 128),   # (M, K, N)
        (64, 128, 256),
        (128, 256, 512),
        (256, 512, 1024),
    ]
    
    # Define dtypes to test
    dtypes = [torch.float16, torch.float32] if torch.cuda.is_available() else [torch.float32]
    
    # Run matrix multiplications with different sizes and dtypes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for dtype in dtypes:
        for M, K, N in sizes:
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
            print(f"Running mm with size ({M}, {K}) x ({K}, {N}) and dtype {dtype}")
            compiled_mm = torch.compile(mm_fn, mode="max-autotune")
            result_mm = compiled_mm(a, b)
            
            # Compile and run addmm
            print(f"Running addmm with size ({M}, {K}) x ({K}, {N}) and dtype {dtype}")
            compiled_addmm = torch.compile(addmm_fn, mode="max-autotune")
            result_addmm = compiled_addmm(c, a, b)

def print_statistics(collector):
    """Print statistics about the collected data."""
    table = collector.get_table()
    
    # Count the number of hardware entries
    hardware_count = len(table.hardware)
    print(f"Number of hardware entries: {hardware_count}")
    
    # For each hardware, count operations and problems
    for hw_name, hardware in table.hardware.items():
        print(f"\nHardware: {hw_name}")
        
        op_count = len(hardware.operation)
        print(f"  Number of operations: {op_count}")
        
        for op_name, operation in hardware.operation.items():
            problem_count = len(operation.solution)
            config_count = sum(len(solution.config) for solution in operation.solution.values())
            print(f"  Operation '{op_name}': {problem_count} problems, {config_count} configs")
            
            # Print details of a few problems
            for i, (problem, solution) in enumerate(operation.solution.items()):
                if i >= 3:  # Limit to 3 problems for brevity
                    print(f"    ... and {problem_count - 3} more problems")
                    break
                
                print(f"    Problem {i+1}: M={problem.M}, N={problem.N}, K={problem.K}, "
                      f"dtype={problem.M_dtype}, {len(solution.config)} configs")

if __name__ == "__main__":
    run_example()
