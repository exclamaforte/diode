"""
Example script demonstrating how to collect matrix multiplication data with EXHAUSTIVE search
and train a neural network model on the collected data.

This script:
1. Collects timing data for ~100 matrix multiplication shapes using max-autotune with EXHAUSTIVE search
2. Saves the collected data to a dataset file
3. Trains a neural network model on the collected data
4. Evaluates the model's performance
"""

import torch
import numpy as np
import os
import sys
import logging
import argparse
import time
import random
from typing import List, Tuple, Optional

# Add the parent directory to the path so we can import the diode module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diode.collection.matmul_dataset_collector import MatmulDatasetCollector
from diode.model.matmul_timing_model import DeepMatmulTimingModel
from diode.model.matmul_dataset_loader import create_dataloaders
from diode.model.matmul_model_trainer import MatmulModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_matrix_sizes(
    num_shapes: int = 100,
    min_size: int = 32,
    max_size: int = 4096,
    power_of_two: bool = False,
    seed: int = 42,
) -> List[Tuple[int, int, int]]:
    """
    Generate a list of matrix sizes (M, K, N) for testing.
    
    Args:
        num_shapes: Number of shapes to generate
        min_size: Minimum matrix dimension
        max_size: Maximum matrix dimension
        power_of_two: Whether to generate only power-of-two sizes
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
        # Non-square matrices
        (64, 128, 256),
        (128, 256, 512),
        (256, 512, 1024),
        (512, 1024, 2048),
        # Odd sizes
        (35, 75, 123),
        (111, 222, 333),
        (555, 333, 111),
    ]
    
    sizes.extend(common_sizes)
    
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

def collect_data(
    output_file: str,
    num_shapes: int = 100,
    dtypes: List[torch.dtype] = None,
    seed: int = 42,
) -> str:
    """
    Collect matrix multiplication timing data using EXHAUSTIVE search.
    
    Args:
        output_file: Path to save the collected data
        num_shapes: Number of matrix shapes to test
        dtypes: List of dtypes to test
        seed: Random seed for reproducibility
        
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
    sizes = generate_matrix_sizes(num_shapes=num_shapes, seed=seed)
    logger.info(f"Generated {len(sizes)} matrix sizes for testing")
    
    # Set up PyTorch for compilation
    torch.set_grad_enabled(False)
    
    # Configure PyTorch inductor
    from torch._inductor import config
    config.fx_graph_cache = False
    config.force_disable_caches = True
    
    # Try to set environment variable for EXHAUSTIVE search, but fall back to DEFAULT if triton is not available
    try:
        import triton
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE"] = "EXHAUSTIVE"
        logger.info("Set search space to EXHAUSTIVE")
    except ImportError:
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE"] = "DEFAULT"
        logger.info("Triton not available, falling back to DEFAULT search space")
    
    # Start collection
    collector.start_collection()
    logger.info("Started collection")
    
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

def print_dataset_statistics(collector: MatmulDatasetCollector) -> None:
    """
    Print statistics about the collected data.
    
    Args:
        collector: The MatmulDatasetCollector containing the collected data
    """
    dataset = collector.get_dataset()
    
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

def train_model(
    dataset_path: str,
    model_path: str,
    batch_size: int = 64,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 10,
    hidden_dim: int = 128,
    num_layers: int = 10,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Train a model on the collected data.
    
    Args:
        dataset_path: Path to the dataset file
        model_path: Path to save the trained model
        batch_size: Batch size for training
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        patience: Number of epochs to wait for improvement before early stopping
        hidden_dim: Hidden dimension of the model
        num_layers: Number of layers in the model
        seed: Random seed for reproducibility
        device: Device to train on
    """
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load the dataset
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset_json = f.read()
    
    from diode.types.matmul_dataset import Dataset as MatmulDataset
    dataset = MatmulDataset.deserialize(dataset_json)
    if dataset is None:
        logger.error(f"Failed to load dataset from {dataset_path}")
        return
    
    # Create dataloaders
    logger.info("Creating dataloaders")
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        log_transform=True,
        num_workers=4,
        seed=seed,
    )
    
    # Get the feature dimensions
    problem_feature_dim = train_dataloader.dataset.dataset.problem_feature_dim
    config_feature_dim = train_dataloader.dataset.dataset.config_feature_dim
    
    # Create the model
    logger.info(f"Creating model with {problem_feature_dim} problem features and {config_feature_dim} config features")
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
    
    # Plot the training history (if matplotlib is available)
    logger.info("Skipping plot generation (matplotlib not required for core functionality)")
    
    # Evaluate the model on the test set
    test_loss = trainer._evaluate(test_dataloader, "Test")
    rmse = np.sqrt(test_loss)
    
    logger.info(f"Test Loss (MSE): {test_loss:.6f}")
    logger.info(f"Test RMSE: {rmse:.6f}")
    logger.info(f"Test RMSE (exp): {np.exp(rmse):.6f}")

def main():
    """
    Main function.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Collect matrix multiplication data and train a model")
    parser.add_argument("--collect", action="store_true", help="Collect data")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--dataset", type=str, default="matmul_dataset_exhaustive.json", help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="matmul_model_exhaustive.pt", help="Path to save the model")
    parser.add_argument("--num-shapes", type=int, default=100, help="Number of matrix shapes to test")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for the optimizer")
    parser.add_argument("--patience", type=int, default=10, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension of the model")
    parser.add_argument("--num-layers", type=int, default=10, help="Number of layers in the model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.dataset)) if os.path.dirname(args.dataset) else ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.model)) if os.path.dirname(args.model) else ".", exist_ok=True)
    
    # Collect data if requested
    if args.collect:
        collect_data(
            output_file=args.dataset,
            num_shapes=args.num_shapes,
            seed=args.seed,
        )
    
    # Train model if requested
    if args.train:
        train_model(
            dataset_path=args.dataset,
            model_path=args.model,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            seed=args.seed,
            device=args.device,
        )

if __name__ == "__main__":
    main()
