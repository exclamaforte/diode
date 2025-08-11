"""
Consolidated toolkit for matrix multiplication data collection, model training, and evaluation.

This script provides a unified interface for:
1. Collecting matrix multiplication timing data
2. Training neural network models on the collected data
3. Evaluating model performance
4. Visualizing results

All functionality is controlled through command-line flags.
"""

import torch
import numpy as np
import os
import sys
import logging
import argparse
import time
import random
from typing import List, Tuple, Optional, Dict, Any

# Add the parent directory to the path so we can import the diode module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diode.collection.matmul_dataset_collector import MatmulDatasetCollector
from diode.collection.matmul_collector import MatmulCollector
from diode.model.matmul_timing_model import MatmulTimingModel, DeepMatmulTimingModel
from diode.model.matmul_dataset_loader import MatmulTimingDataset, create_dataloaders
from diode.model.matmul_model_trainer import MatmulModelTrainer, train_model_from_dataset
from diode.types.matmul_dataset import Dataset as MatmulDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

###########################################
# Utility Functions
###########################################

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
            config_count = sum(len(solution.timed_configs) for solution in operation.solution.values())
            print(f"  Operation '{op_name_entry}': {problem_count} problems, {config_count} configs")
            
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

def print_collector_statistics(collector: MatmulCollector) -> None:
    """
    Print statistics about the collected data from MatmulCollector.
    
    Args:
        collector: The MatmulCollector containing the collected data
    """
    table = collector.get_table()
    
    print("\nCollector Statistics:")
    print("-------------------")
    
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

def plot_training_history(history: dict, save_path: Optional[str] = None) -> None:
    """
    Plot the training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot the loss
        ax1.plot(history["train_loss"], label="Train")
        ax1.plot(history["val_loss"], label="Validation")
        ax1.plot(history["test_loss"], label="Test")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(True)
        
        # Plot the learning rate
        ax2.plot(history["learning_rate"])
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate")
        ax2.grid(True)
        
        # Adjust the layout
        plt.tight_layout()
        
        # Save the plot
        if save_path is not None:
            plt.savefig(save_path)
            logger.info(f"Saved plot to {save_path}")
        
        # Show the plot
        plt.show()
    except ImportError:
        logger.warning("Matplotlib not available, skipping plot")

def generate_matrix_sizes(
    num_shapes: int = 100,
    min_size: int = 32,
    max_size: int = 4096,
    power_of_two: bool = False,
    include_rectangular: bool = True,
    include_odd_sizes: bool = True,
    seed: int = 42,
) -> List[Tuple[int, int, int]]:
    """
    Generate a list of matrix sizes (M, K, N) for testing.
    
    Args:
        num_shapes: Number of shapes to generate
        min_size: Minimum matrix dimension
        max_size: Maximum matrix dimension
        power_of_two: Whether to generate only power-of-two sizes
        include_rectangular: Whether to include rectangular matrices
        include_odd_sizes: Whether to include odd-sized matrices
        seed: Random seed for reproducibility
        
    Returns:
        List of (M, K, N) tuples
    """
    random.seed(seed)
    np.random.seed(seed)
    
    sizes = []
    
    # Add some common matrix sizes
    common_sizes = [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]
    
    sizes.extend(common_sizes)
    
    # Add rectangular matrices if requested
    if include_rectangular:
        rectangular_sizes = [
            (64, 128, 256),
            (128, 256, 512),
            (256, 512, 1024),
            (512, 1024, 2048),
        ]
        sizes.extend(rectangular_sizes)
    
    # Add odd sizes if requested
    if include_odd_sizes:
        odd_sizes = [
            (35, 75, 123),
            (111, 222, 333),
            (555, 333, 111),
        ]
        sizes.extend(odd_sizes)
    
    # Generate random sizes to reach the desired number
    while len(sizes) < num_shapes:
        if power_of_two:
            # Generate power-of-two sizes
            m_pow = random.randint(int(np.log2(min_size)), int(np.log2(max_size)))
            k_pow = random.randint(int(np.log2(min_size)), int(np.log2(max_size)))
            n_pow = random.randint(int(np.log2(min_size)), int(np.log2(max_size)))
            
            M = 2 ** m_pow
            K = 2 ** k_pow
            N = 2 ** n_pow
        else:
            # Generate random sizes
            M = random.randint(min_size, max_size)
            K = random.randint(min_size, max_size)
            N = random.randint(min_size, max_size)
        
        # Add the size if it's not already in the list
        if (M, K, N) not in sizes:
            sizes.append((M, K, N))
    
    return sizes[:num_shapes]  # Ensure we have exactly num_shapes

def analyze_worst_predictions(model, dataloader, device, top_n=10):
    """
    Analyze the worst predictions made by the model.
    
    Args:
        model: The trained model
        dataloader: The dataloader containing the validation data
        device: Device to run the model on
        top_n: Number of worst predictions to analyze
    """
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            problem_features, config_features, targets = batch
            problem_features = problem_features.to(device)
            config_features = config_features.to(device)
            targets = targets.to(device)
            
            outputs = model(problem_features, config_features)
            errors = torch.abs(outputs - targets).cpu().numpy()
            
            for i in range(len(errors)):
                all_errors.append({
                    'error': errors[i].item(),
                    'predicted': outputs[i].item(),
                    'actual': targets[i].item(),
                    'problem_features': problem_features[i].cpu().numpy(),
                    'config_features': config_features[i].cpu().numpy(),
                })
    
    # Sort by error (descending)
    all_errors.sort(key=lambda x: x['error'], reverse=True)
    
    # Print the top_n worst predictions
    print(f"\nTop {top_n} worst predictions:")
    print("----------------------------")
    
    for i, error_data in enumerate(all_errors[:top_n]):
        print(f"Error {i+1}:")
        print(f"  Predicted: {np.exp(error_data['predicted']):.6f} ms")
        print(f"  Actual: {np.exp(error_data['actual']):.6f} ms")
        print(f"  Error (log space): {error_data['error']:.6f}")
        print(f"  Error (ratio): {np.exp(error_data['predicted']) / np.exp(error_data['actual']):.2f}x")
        
        # Extract problem features (M, N, K) with better error handling
        try:
            # The problem features are in the order: B, M, N, K, ...
            # And the log-transformed versions are at indices 7, 8, 9
            # We use the log-transformed versions for better numerical stability
            M = int(np.exp(error_data['problem_features'][8]))
            N = int(np.exp(error_data['problem_features'][9]))
            K = int(np.exp(error_data['problem_features'][10]))
            
            # Check for unreasonable values that might indicate symbolic dimensions
            if M > 1e9 or N > 1e9 or K > 1e9:
                print(f"  Matrix size: (symbolic dimensions)")
            else:
                print(f"  Matrix size: ({M}, {K}) x ({K}, {N})")
        except (ValueError, OverflowError, IndexError):
            # Handle symbolic or invalid dimensions
            print(f"  Matrix size: (symbolic dimensions)")
        
        # Extract configuration features
        try:
            # The config features are in the order:
            # grid, block_m, block_n, block_k, group_m, num_stages, num_warps, EVEN_K, ALLOW_TF32, USE_FAST_ACCUM, ...
            config_features = error_data['config_features']
            block_m = int(config_features[1])
            block_n = int(config_features[2])
            block_k = int(config_features[3])
            group_m = int(config_features[4])
            num_stages = int(config_features[5])
            num_warps = int(config_features[6])
            even_k = bool(int(config_features[7]))
            allow_tf32 = bool(int(config_features[8]))
            use_fast_accum = bool(int(config_features[9]))
            
            print(f"  Config: block_m={block_m}, block_n={block_n}, block_k={block_k}, "
                  f"group_m={group_m}, num_stages={num_stages}, num_warps={num_warps}")
            print(f"          EVEN_K={even_k}, ALLOW_TF32={allow_tf32}, USE_FAST_ACCUM={use_fast_accum}")
        except (ValueError, IndexError):
            print(f"  Config: (unable to extract configuration)")
        
        print()

###########################################
# Data Collection Functions
###########################################

def run_matrix_multiplications(
    sizes: List[Tuple[int, int, int]],
    dtypes: List[torch.dtype],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    search_mode: str = "max-autotune",
) -> None:
    """
    Run matrix multiplication operations with the given sizes and dtypes.
    
    Args:
        sizes: List of (M, K, N) tuples
        dtypes: List of dtypes to test
        device: Device to run on
        search_mode: Search mode for torch.compile
    """
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
            compiled_mm = torch.compile(mm_fn, mode=search_mode)
            result_mm = compiled_mm(a, b)
            
            # Compile and run addmm
            logger.info(f"[{i+1}/{len(sizes)}] Running addmm with size ({M}, {K}) x ({K}, {N}) and dtype {dtype}")
            compiled_addmm = torch.compile(addmm_fn, mode=search_mode)
            result_addmm = compiled_addmm(c, a, b)

def collect_data(
    output_file: str,
    num_shapes: int = 100,
    dtypes: List[torch.dtype] = None,
    seed: int = 42,
    min_size: int = 32,
    max_size: int = 4096,
    power_of_two: bool = False,
    include_rectangular: bool = True,
    include_odd_sizes: bool = True,
    search_mode: str = "max-autotune",
    search_space: str = "EXHAUSTIVE",
) -> str:
    """
    Collect matrix multiplication timing data.
    
    Args:
        output_file: Path to save the collected data
        num_shapes: Number of matrix shapes to test
        dtypes: List of dtypes to test
        seed: Random seed for reproducibility
        min_size: Minimum matrix dimension
        max_size: Maximum matrix dimension
        power_of_two: Whether to generate only power-of-two sizes
        include_rectangular: Whether to include rectangular matrices
        include_odd_sizes: Whether to include odd-sized matrices
        search_mode: Search mode for torch.compile
        search_space: Search space for autotuning (EXHAUSTIVE or DEFAULT)
        
    Returns:
        Path to the saved dataset file
    """
    if dtypes is None:
        dtypes = [torch.float16, torch.float32] if torch.cuda.is_available() else [torch.float32]
    
    # Get the hardware name
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    logger.info(f"Collecting data on device: {device_name}")
    
    # Create a collector
    collector = MatmulDatasetCollector(hardware_name=device_name)
    
    # Generate matrix sizes
    sizes = generate_matrix_sizes(
        num_shapes=num_shapes, 
        seed=seed,
        min_size=min_size,
        max_size=max_size,
        power_of_two=power_of_two,
        include_rectangular=include_rectangular,
        include_odd_sizes=include_odd_sizes
    )
    logger.info(f"Generated {len(sizes)} matrix sizes for testing")
    
    # Set up PyTorch for compilation
    torch.set_grad_enabled(False)
    
    # Configure PyTorch inductor
    from torch._inductor import config
    config.fx_graph_cache = False
    config.force_disable_caches = True
    
    # Set search space
    if search_space == "EXHAUSTIVE":
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE"] = "EXHAUSTIVE"
        logger.info("Set search space to EXHAUSTIVE")
    else:
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE"] = "DEFAULT"
        logger.info("Set search space to DEFAULT")
    
    # Start collection
    collector.start_collection()
    logger.info("Started collection")
    
    # Run matrix multiplications
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_shapes = len(sizes) * len(dtypes)
    
    logger.info(f"Running {total_shapes} matrix multiplications...")
    start_time = time.time()
    
    run_matrix_multiplications(sizes, dtypes, device, search_mode)
    
    # Stop collection
    collector.stop_collection()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Collection completed in {elapsed_time:.2f} seconds")
    
    # Save the collected dataset to a file
    collector.save_to_file(output_file)
    logger.info(f"Saved collected dataset to {output_file}")
    
    # Print statistics about the collected data
    print_dataset_statistics(collector)
    
    return output_file

def create_validation_dataset(
    output_file: str,
    num_shapes: int = 30,
    dtypes: List[torch.dtype] = None,
    seed: int = 43,  # Different seed from training
    min_size: int = 32,
    max_size: int = 4096,
    power_of_two: bool = False,
    include_rectangular: bool = True,
    include_odd_sizes: bool = True,
    search_mode: str = "max-autotune",
    search_space: str = "EXHAUSTIVE",
) -> str:
    """
    Create a separate validation dataset for evaluating the model.
    
    Args:
        output_file: Path to save the validation data
        num_shapes: Number of matrix shapes to test
        dtypes: List of dtypes to test
        seed: Random seed for reproducibility (different from training)
        min_size: Minimum matrix dimension
        max_size: Maximum matrix dimension
        power_of_two: Whether to generate only power-of-two sizes
        include_rectangular: Whether to include rectangular matrices
        include_odd_sizes: Whether to include odd-sized matrices
        search_mode: Search mode for torch.compile
        search_space: Search space for autotuning (EXHAUSTIVE or DEFAULT)
        
    Returns:
        Path to the saved validation dataset file
    """
    # Check if validation dataset already exists
    if os.path.exists(output_file):
        logger.info(f"Validation dataset already exists at {output_file}")
        return output_file
    
    logger.info(f"Creating validation dataset at {output_file}")
    
    # Use the same collection function but with different parameters
    return collect_data(
        output_file=output_file,
        num_shapes=num_shapes,
        dtypes=dtypes,
        seed=seed,
        min_size=min_size,
        max_size=max_size,
        power_of_two=power_of_two,
        include_rectangular=include_rectangular,
        include_odd_sizes=include_odd_sizes,
        search_mode=search_mode,
        search_space=search_space,
    )

def run_collector_example(
    output_dir: str = ".",
    use_context_manager: bool = True,
    num_shapes: int = 4,
    dtypes: List[torch.dtype] = None,
) -> None:
    """
    Run an example demonstrating how to use the MatmulDatasetCollector class.
    
    Args:
        output_dir: Directory to save output files
        use_context_manager: Whether to use the collector as a context manager
        num_shapes: Number of matrix shapes to test
        dtypes: List of dtypes to test
    """
    if dtypes is None:
        dtypes = [torch.float16, torch.float32] if torch.cuda.is_available() else [torch.float32]
    
    # Get the hardware name
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device_name}")
    
    # Generate matrix sizes
    sizes = [
        (32, 64, 128),   # (M, K, N)
        (64, 128, 256),
        (128, 256, 512),
        (256, 512, 1024),
    ][:num_shapes]
    
    if use_context_manager:
        # Example using the collector as a context manager
        print("\nUsing the collector as a context manager")
        collector = MatmulDatasetCollector(hardware_name=device_name)
        
        # Use the collector as a context manager
        with collector:
            print("Running matrix multiplications...")
            run_matrix_multiplications(sizes, dtypes)
        
        # Save the collected dataset to a file
        dataset_file = os.path.join(output_dir, "matmul_dataset_context_manager.json")
        collector.save_to_file(dataset_file)
        print(f"Saved collected dataset to {dataset_file}")
        
        # Convert the dataset to a table and save it
        table_file = os.path.join(output_dir, "matmul_table_context_manager.json")
        collector.save_table_to_file(table_file)
        print(f"Saved table to {table_file}")
    else:
        # Example using start_collection and stop_collection methods
        print("\nUsing start_collection and stop_collection methods")
        collector = MatmulDatasetCollector(hardware_name=device_name)
        collector.start_collection()
        
        # Run matrix multiplications
        print("Running matrix multiplications...")
        run_matrix_multiplications(sizes, dtypes)
        
        # Stop collection
        collector.stop_collection()
        
        # Save the collected dataset to a file
        dataset_file = os.path.join(output_dir, "matmul_dataset_explicit.json")
        collector.save_to_file(dataset_file)
        print(f"Saved collected dataset to {dataset_file}")
        
        # Convert the dataset to a table and save it
        table_file = os.path.join(output_dir, "matmul_table_explicit.json")
        collector.save_table_to_file(table_file)
        print(f"Saved table to {table_file}")
    
    # Print statistics about the collected data
    print_dataset_statistics(collector)

def run_basic_collector_example(
    output_dir: str = ".",
    use_context_manager: bool = True,
    num_shapes: int = 4,
    dtypes: List[torch.dtype] = None,
) -> None:
    """
    Run an example demonstrating how to use the MatmulCollector class.
    
    Args:
        output_dir: Directory to save output files
        use_context_manager: Whether to use the collector as a context manager
        num_shapes: Number of matrix shapes to test
        dtypes: List of dtypes to test
    """
    if dtypes is None:
        dtypes = [torch.float16, torch.float32] if torch.cuda.is_available() else [torch.float32]
    
    # Get the hardware name
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device_name}")
    
    # Generate matrix sizes
    sizes = [
        (32, 64, 128),   # (M, K, N)
        (64, 128, 256),
        (128, 256, 512),
        (256, 512, 1024),
    ][:num_shapes]
    
    if use_context_manager:
        # Example using the collector as a context manager
        print("\nUsing the collector as a context manager")
        collector = MatmulCollector(hardware_name=device_name)
        
        # Use the collector as a context manager
        with collector:
            print("Running matrix multiplications...")
            run_matrix_multiplications(sizes, dtypes)
        
        # Save the collected data to a file
        output_file = os.path.join(output_dir, "matmul_data_context_manager.json")
        collector.save_to_file(output_file)
        print(f"Saved collected data to {output_file}")
    else:
        # Example using start_collection and stop_collection methods
        print("\nUsing start_collection and stop_collection methods")
        collector = MatmulCollector(hardware_name=device_name)
        collector.start_collection()
        
        # Run matrix multiplications
        print("Running matrix multiplications...")
        run_matrix_multiplications(sizes, dtypes)
        
        # Stop collection
        collector.stop_collection()
        
        # Save the collected data to a file
        output_file = os.path.join(output_dir, "matmul_data_explicit.json")
        collector.save_to_file(output_file)
        print(f"Saved collected data to {output_file}")
    
    # Example of loading data from a file
    print("\nLoading data from a file")
    new_collector = MatmulCollector()
    new_collector.load_from_file(output_file)
    print(f"Loaded data from {output_file}")
    
    # Print statistics about the collected data
    print_collector_statistics(new_collector)

###########################################
# Model Training and Evaluation Functions
###########################################

def train_model(
    dataset_path: str,
    model_path: str,
    model_type: str = "deep",
    batch_size: int = 64,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 10,
    hidden_dim: int = 128,
    num_layers: int = 10,
    hardware_name: Optional[str] = None,
    op_name: Optional[str] = None,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_dir: str = "logs",
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    Train a model on the collected data.
    
    Args:
        dataset_path: Path to the dataset file
        model_path: Path to save the trained model
        model_type: Type of model to train ("base" or "deep")
        batch_size: Batch size for training
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        patience: Number of epochs to wait for improvement before early stopping
        hidden_dim: Hidden dimension of the model
        num_layers: Number of layers in the model
        hardware_name: Optional hardware name to filter by
        op_name: Optional operation name to filter by
        seed: Random seed for reproducibility
        device: Device to train on
        log_dir: Directory to save logs
        
    Returns:
        Tuple of (trained model, training history)
    """
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(os.path.abspath(model_path)) if os.path.dirname(model_path) else ".", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Load the dataset
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset_json = f.read()
    
    dataset = MatmulDataset.deserialize(dataset_json)
    if dataset is None:
        logger.error(f"Failed to load dataset from {dataset_path}")
        return None, {}
    
    # Create dataloaders
    logger.info("Creating dataloaders")
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        hardware_name=hardware_name,
        op_name=op_name,
        log_transform=True,
        num_workers=4,
        seed=seed,
    )
    
    # Get the feature dimensions
    problem_feature_dim = train_dataloader.dataset.dataset.problem_feature_dim
    config_feature_dim = train_dataloader.dataset.dataset.config_feature_dim
    
    # Create the model
    logger.info(f"Creating {model_type} model with {problem_feature_dim} problem features and {config_feature_dim} config features")
    if model_type == "base":
        model = MatmulTimingModel(
            problem_feature_dim=problem_feature_dim,
            config_feature_dim=config_feature_dim,
        )
    else:  # "deep"
        model = DeepMatmulTimingModel(
            problem_feature_dim=problem_feature_dim,
            config_feature_dim=config_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    
    # Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {num_params} parameters")
    
    # Create the trainer
    trainer = MatmulModelTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
    )
    
    # Train the model
    logger.info(f"Training model for {num_epochs} epochs")
    history = trainer.train(
        num_epochs=num_epochs,
        patience=patience,
        checkpoint_path=model_path,
        verbose=True,
    )
    
    # Plot the training history
    history_plot_path = os.path.join(log_dir, f"matmul_timing_{model_type}_history.png")
    plot_training_history(history, history_plot_path)
    
    # Evaluate the model on the test set
    test_loss = trainer._evaluate(test_dataloader, "Test")
    rmse = np.sqrt(test_loss)
    
    logger.info(f"Test Loss (MSE): {test_loss:.6f}")
    logger.info(f"Test RMSE: {rmse:.6f}")
    logger.info(f"Test RMSE (exp): {np.exp(rmse):.6f}")
    
    return model, history

def validate_model(
    model_path: str,
    validation_dataset_path: str,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    hardware_name: Optional[str] = None,
    op_name: Optional[str] = None,
    top_n_worst: int = 10,
) -> None:
    """
    Validate a trained model on a separate validation dataset.
    
    Args:
        model_path: Path to the trained model
        validation_dataset_path: Path to the validation dataset
        batch_size: Batch size for validation
        device: Device to validate on
        hardware_name: Optional hardware name to filter by
        op_name: Optional operation name to filter by
        top_n_worst: Number of worst predictions to analyze
    """
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return
    
    # Check if validation dataset exists
    if not os.path.exists(validation_dataset_path):
        logger.error(f"Validation dataset not found at {validation_dataset_path}")
        return
    
    # Load the validation dataset
    logger.info(f"Loading validation dataset from {validation_dataset_path}")
    with open(validation_dataset_path, 'r') as f:
        dataset_json = f.read()
    
    dataset = MatmulDataset.deserialize(dataset_json)
    if dataset is None:
        logger.error(f"Failed to load validation dataset from {validation_dataset_path}")
        return
    
    # Create dataloaders (we only need the validation dataloader)
    logger.info("Creating validation dataloader")
    _, val_dataloader, _ = create_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        hardware_name=hardware_name,
        op_name=op_name,
        log_transform=True,
        num_workers=4,
        seed=42,  # Use a fixed seed for reproducibility
    )
    
    # Get the feature dimensions
    problem_feature_dim = val_dataloader.dataset.dataset.problem_feature_dim
    config_feature_dim = val_dataloader.dataset.dataset.config_feature_dim
    
    # Load the trained model weights
    logger.info(f"Loading model weights from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if the model was saved as a complete checkpoint or just state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        logger.info("Loading from checkpoint format")
        # Extract model parameters from checkpoint
        problem_feature_dim = checkpoint.get("problem_feature_dim", problem_feature_dim)
        config_feature_dim = checkpoint.get("config_feature_dim", config_feature_dim)
        hidden_dim = checkpoint.get("hidden_dim", 128)
        num_layers = checkpoint.get("num_layers", 10)
        model_type = checkpoint.get("model_type", "deep")
        
        # Recreate the model with the correct architecture
        if model_type == "base":
            model = MatmulTimingModel(
                problem_feature_dim=problem_feature_dim,
                config_feature_dim=config_feature_dim,
            )
        else:  # "deep"
            model = DeepMatmulTimingModel(
                problem_feature_dim=problem_feature_dim,
                config_feature_dim=config_feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
        
        # Load the state dict
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Direct state dict loading
        # Assume it's a deep model if we don't know
        model = DeepMatmulTimingModel(
            problem_feature_dim=problem_feature_dim,
            config_feature_dim=config_feature_dim,
        )
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Create a trainer just for evaluation
    trainer = MatmulModelTrainer(
        model=model,
        train_dataloader=None,
        val_dataloader=val_dataloader,
        test_dataloader=None,
        device=device,
    )
    
    # Evaluate the model on the validation dataset
    val_loss = trainer._evaluate(val_dataloader, "Validation")
    rmse = np.sqrt(val_loss)
    
    logger.info(f"Validation Loss (MSE): {val_loss:.6f}")
    logger.info(f"Validation RMSE: {rmse:.6f}")
    logger.info(f"Validation RMSE (exp): {np.exp(rmse):.6f}")
    
    # Analyze the worst predictions
    if top_n_worst > 0:
        logger.info(f"Analyzing worst {top_n_worst} predictions...")
        analyze_worst_predictions(model, val_dataloader, device, top_n=top_n_worst)

def run_model_example(
    dataset_path: str,
    model_type: str = "deep",
    batch_size: int = 64,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 10,
    log_dir: str = "logs",
    model_dir: str = "models",
    hardware_name: Optional[str] = None,
    op_name: Optional[str] = None,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Run an example demonstrating how to train and use a matrix multiplication timing prediction model.
    
    Args:
        dataset_path: Path to the dataset file
        model_type: Type of model to train ("base" or "deep")
        batch_size: Batch size for the dataloaders
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        patience: Number of epochs to wait for improvement before early stopping
        log_dir: Directory to save logs
        model_dir: Directory to save models
        hardware_name: Optional hardware name to filter by
        op_name: Optional operation name to filter by
        seed: Random seed for reproducibility
        device: Device to train on
    """
    # Create the directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Load the dataset
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset_json = f.read()
    
    dataset = MatmulDataset.deserialize(dataset_json)
    if dataset is None:
        logger.error(f"Failed to load dataset from {dataset_path}")
        return
    
    # Print dataset statistics
    print_dataset_statistics(dataset, hardware_name, op_name)
    
    # Train the model
    logger.info(f"Training {model_type} model")
    checkpoint_path = os.path.join(model_dir, f"matmul_timing_{model_type}_model.pt")
    
    model, history = train_model_from_dataset(
        dataset=dataset,
        model_type=model_type,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        log_dir=log_dir,
        checkpoint_path=checkpoint_path,
        hardware_name=hardware_name,
        op_name=op_name,
        log_transform=True,
        seed=seed,
        verbose=True,
        device=device,
    )
    
    # Plot the training history
    history_plot_path = os.path.join(log_dir, f"matmul_timing_{model_type}_history.png")
    plot_training_history(history, history_plot_path)
    
    # Evaluate the model on the test set
    logger.info("Making predictions on the test set")
    _, _, test_dataloader = create_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        hardware_name=hardware_name,
        op_name=op_name,
        log_transform=True,
        num_workers=4,
        seed=seed,
    )
    
    # Move the model to the device
    model = model.to(device)
    model.eval()
    
    # Initialize variables
    total_loss = 0.0
    criterion = torch.nn.MSELoss()
    
    # Evaluate on the test set
    with torch.no_grad():
        for problem_features, config_features, targets in test_dataloader:
            # Move the data to the device
            problem_features = problem_features.to(device)
            config_features = config_features.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(problem_features, config_features)
            
            # Calculate the loss
            loss = criterion(outputs, targets)
            
            # Update the total loss
            total_loss += loss.item()
    
    # Calculate the average loss
    avg_loss = total_loss / len(test_dataloader)
    
    # Calculate the RMSE
    rmse = np.sqrt(avg_loss)
    
    # Print the results
    print("\nModel Evaluation:")
    print("----------------")
    print(f"Test Loss (MSE): {avg_loss:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test RMSE (exp): {np.exp(rmse):.6f}")
    
    logger.info("Example completed")

###########################################
# Main Function
###########################################

def main():
    """
    Main function that parses command-line arguments and runs the appropriate mode.
    """
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="Matrix multiplication toolkit for data collection, model training, and evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add global arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run on")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run", required=True)
    
    # Collect data mode
    collect_parser = subparsers.add_parser("collect", help="Collect matrix multiplication timing data")
    collect_parser.add_argument("--output", type=str, default="matmul_dataset.json", 
                               help="Path to save the collected data")
    collect_parser.add_argument("--num-shapes", type=int, default=100, 
                               help="Number of matrix shapes to test")
    collect_parser.add_argument("--min-size", type=int, default=32, 
                               help="Minimum matrix dimension")
    collect_parser.add_argument("--max-size", type=int, default=4096, 
                               help="Maximum matrix dimension")
    collect_parser.add_argument("--power-of-two", action="store_true", 
                               help="Generate only power-of-two sizes")
    collect_parser.add_argument("--no-rectangular", action="store_true", 
                               help="Exclude rectangular matrices")
    collect_parser.add_argument("--no-odd-sizes", action="store_true", 
                               help="Exclude odd-sized matrices")
    collect_parser.add_argument("--search-mode", type=str, default="max-autotune", 
                               help="Search mode for torch.compile")
    collect_parser.add_argument("--search-space", type=str, default="EXHAUSTIVE", choices=["EXHAUSTIVE", "DEFAULT"],
                               help="Search space for autotuning")
    
    # Create validation dataset mode
    validate_data_parser = subparsers.add_parser("create-validation", 
                                               help="Create a separate validation dataset")
    validate_data_parser.add_argument("--output", type=str, default="matmul_validation_dataset.json", 
                                    help="Path to save the validation data")
    validate_data_parser.add_argument("--num-shapes", type=int, default=30, 
                                    help="Number of matrix shapes to test")
    validate_data_parser.add_argument("--min-size", type=int, default=32, 
                                    help="Minimum matrix dimension")
    validate_data_parser.add_argument("--max-size", type=int, default=4096, 
                                    help="Maximum matrix dimension")
    validate_data_parser.add_argument("--power-of-two", action="store_true", 
                                    help="Generate only power-of-two sizes")
    validate_data_parser.add_argument("--no-rectangular", action="store_true", 
                                    help="Exclude rectangular matrices")
    validate_data_parser.add_argument("--no-odd-sizes", action="store_true", 
                                    help="Exclude odd-sized matrices")
    validate_data_parser.add_argument("--search-mode", type=str, default="max-autotune", 
                                    help="Search mode for torch.compile")
    validate_data_parser.add_argument("--search-space", type=str, default="EXHAUSTIVE", choices=["EXHAUSTIVE", "DEFAULT"],
                                    help="Search space for autotuning")
    
    # Train model mode
    train_parser = subparsers.add_parser("train", help="Train a model on collected data")
    train_parser.add_argument("--dataset", type=str, required=True, 
                             help="Path to the dataset file")
    train_parser.add_argument("--model", type=str, default="matmul_model.pt", 
                             help="Path to save the trained model")
    train_parser.add_argument("--model-type", type=str, default="deep", choices=["base", "deep"], 
                             help="Type of model to train")
    train_parser.add_argument("--batch-size", type=int, default=64, 
                             help="Batch size for training")
    train_parser.add_argument("--num-epochs", type=int, default=100, 
                             help="Number of epochs to train for")
    train_parser.add_argument("--learning-rate", type=float, default=0.001, 
                             help="Learning rate for the optimizer")
    train_parser.add_argument("--weight-decay", type=float, default=1e-5, 
                             help="Weight decay for the optimizer")
    train_parser.add_argument("--patience", type=int, default=10, 
                             help="Number of epochs to wait for improvement before early stopping")
    train_parser.add_argument("--hidden-dim", type=int, default=128, 
                             help="Hidden dimension of the model")
    train_parser.add_argument("--num-layers", type=int, default=10, 
                             help="Number of layers in the model")
    train_parser.add_argument("--hardware-name", type=str, 
                             help="Hardware name to filter by")
    train_parser.add_argument("--op-name", type=str, 
                             help="Operation name to filter by")
    train_parser.add_argument("--log-dir", type=str, default="logs", 
                             help="Directory to save logs")
    
    # Validate model mode
    validate_model_parser = subparsers.add_parser("validate-model", 
                                                help="Validate a trained model on a separate validation dataset")
    validate_model_parser.add_argument("--model", type=str, required=True, 
                                      help="Path to the trained model")
    validate_model_parser.add_argument("--dataset", type=str, required=True, 
                                      help="Path to the validation dataset")
    validate_model_parser.add_argument("--batch-size", type=int, default=64, 
                                      help="Batch size for validation")
    validate_model_parser.add_argument("--hardware-name", type=str, 
                                      help="Hardware name to filter by")
    validate_model_parser.add_argument("--op-name", type=str, 
                                      help="Operation name to filter by")
    validate_model_parser.add_argument("--top-n-worst", type=int, default=10, 
                                      help="Number of worst predictions to analyze")
    
    # Collector example mode
    collector_example_parser = subparsers.add_parser("collector-example", 
                                                   help="Run an example demonstrating the MatmulDatasetCollector")
    collector_example_parser.add_argument("--output-dir", type=str, default=".", 
                                        help="Directory to save output files")
    collector_example_parser.add_argument("--use-context-manager", action="store_true", 
                                        help="Use the collector as a context manager")
    collector_example_parser.add_argument("--num-shapes", type=int, default=4, 
                                        help="Number of matrix shapes to test")
    
    # Basic collector example mode
    basic_collector_example_parser = subparsers.add_parser("collector-basic-example", 
                                                        help="Run an example demonstrating the MatmulCollector")
    basic_collector_example_parser.add_argument("--output-dir", type=str, default=".", 
                                             help="Directory to save output files")
    basic_collector_example_parser.add_argument("--use-context-manager", action="store_true", 
                                             help="Use the collector as a context manager")
    basic_collector_example_parser.add_argument("--num-shapes", type=int, default=4, 
                                             help="Number of matrix shapes to test")
    
    # Model example mode
    model_example_parser = subparsers.add_parser("model-example", 
                                               help="Run an example demonstrating model training and evaluation")
    model_example_parser.add_argument("--dataset", type=str, required=True, 
                                    help="Path to the dataset file")
    model_example_parser.add_argument("--model-type", type=str, default="deep", choices=["base", "deep"], 
                                    help="Type of model to train")
    model_example_parser.add_argument("--batch-size", type=int, default=64, 
                                    help="Batch size for the dataloaders")
    model_example_parser.add_argument("--num-epochs", type=int, default=100, 
                                    help="Number of epochs to train for")
    model_example_parser.add_argument("--learning-rate", type=float, default=0.001, 
                                    help="Learning rate for the optimizer")
    model_example_parser.add_argument("--weight-decay", type=float, default=1e-5, 
                                    help="Weight decay for the optimizer")
    model_example_parser.add_argument("--patience", type=int, default=10, 
                                    help="Number of epochs to wait for improvement before early stopping")
    model_example_parser.add_argument("--log-dir", type=str, default="logs", 
                                    help="Directory to save logs")
    model_example_parser.add_argument("--model-dir", type=str, default="models", 
                                    help="Directory to save models")
    model_example_parser.add_argument("--hardware-name", type=str, 
                                    help="Hardware name to filter by")
    model_example_parser.add_argument("--op-name", type=str, 
                                    help="Operation name to filter by")
    
    # Collect and train mode (combines collection and training)
    collect_train_parser = subparsers.add_parser("collect-and-train", 
                                               help="Collect data and train a model in one step")
    collect_train_parser.add_argument("--dataset", type=str, default="matmul_dataset.json", 
                                    help="Path to save the collected data")
    collect_train_parser.add_argument("--validation-dataset", type=str, default="matmul_validation_dataset.json", 
                                    help="Path to save the validation dataset")
    collect_train_parser.add_argument("--model", type=str, default="matmul_model.pt", 
                                    help="Path to save the trained model")
    collect_train_parser.add_argument("--num-shapes", type=int, default=100, 
                                    help="Number of matrix shapes to test")
    collect_train_parser.add_argument("--validation-shapes", type=int, default=30, 
                                    help="Number of matrix shapes for validation")
    collect_train_parser.add_argument("--min-size", type=int, default=32, 
                                    help="Minimum matrix dimension")
    collect_train_parser.add_argument("--max-size", type=int, default=4096, 
                                    help="Maximum matrix dimension")
    collect_train_parser.add_argument("--power-of-two", action="store_true", 
                                    help="Generate only power-of-two sizes")
    collect_train_parser.add_argument("--no-rectangular", action="store_true", 
                                    help="Exclude rectangular matrices")
    collect_train_parser.add_argument("--no-odd-sizes", action="store_true", 
                                    help="Exclude odd-sized matrices")
    collect_train_parser.add_argument("--search-mode", type=str, default="max-autotune", 
                                    help="Search mode for torch.compile")
    collect_train_parser.add_argument("--search-space", type=str, default="EXHAUSTIVE", choices=["EXHAUSTIVE", "DEFAULT"],
                                    help="Search space for autotuning")
    collect_train_parser.add_argument("--model-type", type=str, default="deep", choices=["base", "deep"], 
                                    help="Type of model to train")
    collect_train_parser.add_argument("--batch-size", type=int, default=64, 
                                    help="Batch size for training")
    collect_train_parser.add_argument("--num-epochs", type=int, default=100, 
                                    help="Number of epochs to train for")
    collect_train_parser.add_argument("--learning-rate", type=float, default=0.001, 
                                    help="Learning rate for the optimizer")
    collect_train_parser.add_argument("--weight-decay", type=float, default=1e-5, 
                                    help="Weight decay for the optimizer")
    collect_train_parser.add_argument("--patience", type=int, default=10, 
                                    help="Number of epochs to wait for improvement before early stopping")
    collect_train_parser.add_argument("--hidden-dim", type=int, default=128, 
                                    help="Hidden dimension of the model")
    collect_train_parser.add_argument("--num-layers", type=int, default=10, 
                                    help="Number of layers in the model")
    collect_train_parser.add_argument("--log-dir", type=str, default="logs", 
                                    help="Directory to save logs")
    collect_train_parser.add_argument("--skip-collection", action="store_true", 
                                    help="Skip data collection and use existing dataset")
    collect_train_parser.add_argument("--skip-validation", action="store_true", 
                                    help="Skip validation dataset creation")
    collect_train_parser.add_argument("--skip-training", action="store_true", 
                                    help="Skip model training")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Run the appropriate mode
    if args.mode == "collect":
        collect_data(
            output_file=args.output,
            num_shapes=args.num_shapes,
            seed=args.seed,
            min_size=args.min_size,
            max_size=args.max_size,
            power_of_two=args.power_of_two,
            include_rectangular=not args.no_rectangular,
            include_odd_sizes=not args.no_odd_sizes,
            search_mode=args.search_mode,
            search_space=args.search_space,
        )
    
    elif args.mode == "create-validation":
        create_validation_dataset(
            output_file=args.output,
            num_shapes=args.num_shapes,
            seed=args.seed,
            min_size=args.min_size,
            max_size=args.max_size,
            power_of_two=args.power_of_two,
            include_rectangular=not args.no_rectangular,
            include_odd_sizes=not args.no_odd_sizes,
            search_mode=args.search_mode,
            search_space=args.search_space,
        )
    
    elif args.mode == "train":
        train_model(
            dataset_path=args.dataset,
            model_path=args.model,
            model_type=args.model_type,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            hardware_name=args.hardware_name,
            op_name=args.op_name,
            seed=args.seed,
            device=args.device,
            log_dir=args.log_dir,
        )
    
    elif args.mode == "validate-model":
        validate_model(
            model_path=args.model,
            validation_dataset_path=args.dataset,
            batch_size=args.batch_size,
            device=args.device,
            hardware_name=args.hardware_name,
            op_name=args.op_name,
            top_n_worst=args.top_n_worst,
        )
    
    elif args.mode == "collector-example":
        run_collector_example(
            output_dir=args.output_dir,
            use_context_manager=args.use_context_manager,
            num_shapes=args.num_shapes,
        )
    
    elif args.mode == "collector-basic-example":
        run_basic_collector_example(
            output_dir=args.output_dir,
            use_context_manager=args.use_context_manager,
            num_shapes=args.num_shapes,
        )
    
    elif args.mode == "model-example":
        run_model_example(
            dataset_path=args.dataset,
            model_type=args.model_type,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            hardware_name=args.hardware_name,
            op_name=args.op_name,
            seed=args.seed,
            device=args.device,
        )
    
    elif args.mode == "collect-and-train":
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(args.dataset)) if os.path.dirname(args.dataset) else ".", exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.model)) if os.path.dirname(args.model) else ".", exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.validation_dataset)) if os.path.dirname(args.validation_dataset) else ".", exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        
        # Collect data if not skipped
        if not args.skip_collection:
            collect_data(
                output_file=args.dataset,
                num_shapes=args.num_shapes,
                seed=args.seed,
                min_size=args.min_size,
                max_size=args.max_size,
                power_of_two=args.power_of_two,
                include_rectangular=not args.no_rectangular,
                include_odd_sizes=not args.no_odd_sizes,
                search_mode=args.search_mode,
                search_space=args.search_space,
            )
        
        # Create validation dataset if not skipped
        if not args.skip_validation:
            create_validation_dataset(
                output_file=args.validation_dataset,
                num_shapes=args.validation_shapes,
                seed=args.seed + 1,  # Use a different seed for validation
                min_size=args.min_size,
                max_size=args.max_size,
                power_of_two=args.power_of_two,
                include_rectangular=not args.no_rectangular,
                include_odd_sizes=not args.no_odd_sizes,
                search_mode=args.search_mode,
                search_space=args.search_space,
            )
        
        # Train model if not skipped
        if not args.skip_training:
            train_model(
                dataset_path=args.dataset,
                model_path=args.model,
                model_type=args.model_type,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                patience=args.patience,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                seed=args.seed,
                device=args.device,
                log_dir=args.log_dir,
            )
            
            # Validate model on the validation dataset
            if not args.skip_validation and os.path.exists(args.validation_dataset):
                validate_model(
                    model_path=args.model,
                    validation_dataset_path=args.validation_dataset,
                    batch_size=args.batch_size,
                    device=args.device,
                )

if __name__ == "__main__":
    main()
