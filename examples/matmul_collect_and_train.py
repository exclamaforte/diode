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
    min_size: int = 32,
    max_size: int = 4096,
    power_of_two: bool = False,
    include_rectangular: bool = True,
    include_odd_sizes: bool = True,
) -> str:
    """
    Collect matrix multiplication timing data using EXHAUSTIVE search.
    
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
    
    # Try to set environment variable for EXHAUSTIVE search, but fall back to DEFAULT if triton is not available
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE"] = "EXHAUSTIVE"
    logger.info("Set search space to EXHAUSTIVE")
    
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
    )

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

def validate_model(
    model_path: str,
    validation_dataset_path: str,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Validate a trained model on a separate validation dataset.
    
    Args:
        model_path: Path to the trained model
        validation_dataset_path: Path to the validation dataset
        batch_size: Batch size for validation
        device: Device to validate on
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
    
    from diode.types.matmul_dataset import Dataset as MatmulDataset
    dataset = MatmulDataset.deserialize(dataset_json)
    if dataset is None:
        logger.error(f"Failed to load validation dataset from {validation_dataset_path}")
        return
    
    # Create dataloaders (we only need the validation dataloader)
    logger.info("Creating validation dataloader")
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        log_transform=True,
        num_workers=4,
        seed=42,  # Use a fixed seed for reproducibility
    )
    
    # We'll use the validation dataloader for our validation
    
    # Get the feature dimensions
    problem_feature_dim = val_dataloader.dataset.dataset.problem_feature_dim
    config_feature_dim = val_dataloader.dataset.dataset.config_feature_dim
    
    # Create the model with the same architecture
    logger.info(f"Creating model with {problem_feature_dim} problem features and {config_feature_dim} config features")
    model = DeepMatmulTimingModel(
        problem_feature_dim=problem_feature_dim,
        config_feature_dim=config_feature_dim,
    )
    
    # Load the trained model weights
    logger.info(f"Loading model weights from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if the model was saved as a complete checkpoint or just state_dict
    if "model_state_dict" in checkpoint:
        logger.info("Loading from checkpoint format")
        # Extract model parameters from checkpoint
        problem_feature_dim = checkpoint.get("problem_feature_dim", problem_feature_dim)
        config_feature_dim = checkpoint.get("config_feature_dim", config_feature_dim)
        hidden_dim = checkpoint.get("hidden_dim", 128)
        num_layers = checkpoint.get("num_layers", 10)
        
        # Recreate the model with the correct architecture
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
    logger.info("Analyzing worst predictions...")
    analyze_worst_predictions(model, val_dataloader, device, top_n=10)

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
    
    # Ensure the model path is valid
    if not model_path:
        model_path = "matmul_model.pt"  # Default path
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    
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
    parser.add_argument("--validate", action="store_true", help="Validate model on separate validation dataset")
    parser.add_argument("--create-validation", action="store_true", help="Create a separate validation dataset")
    parser.add_argument("--dataset", type=str, default="matmul_dataset_exhaustive.json", help="Path to the dataset file")
    parser.add_argument("--validation-dataset", type=str, default="matmul_validation_dataset.json", help="Path to the validation dataset file")
    parser.add_argument("--model", type=str, default="matmul_model_exhaustive.pt", help="Path to save the model")
    parser.add_argument("--num-shapes", type=int, default=100, help="Number of matrix shapes to test")
    parser.add_argument("--validation-shapes", type=int, default=30, help="Number of matrix shapes for validation")
    parser.add_argument("--min-size", type=int, default=32, help="Minimum matrix dimension")
    parser.add_argument("--max-size", type=int, default=4096, help="Maximum matrix dimension")
    parser.add_argument("--power-of-two", action="store_true", help="Generate only power-of-two sizes")
    parser.add_argument("--no-rectangular", action="store_true", help="Exclude rectangular matrices")
    parser.add_argument("--no-odd-sizes", action="store_true", help="Exclude odd-sized matrices")
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
    if args.validation_dataset:
        os.makedirs(os.path.dirname(os.path.abspath(args.validation_dataset)) if os.path.dirname(args.validation_dataset) else ".", exist_ok=True)
    
    # Collect data if requested
    if args.collect:
        collect_data(
            output_file=args.dataset,
            num_shapes=args.num_shapes,
            seed=args.seed,
            min_size=args.min_size,
            max_size=args.max_size,
            power_of_two=args.power_of_two,
            include_rectangular=not args.no_rectangular,
            include_odd_sizes=not args.no_odd_sizes,
        )
    
    # Create validation dataset if requested
    if args.create_validation:
        create_validation_dataset(
            output_file=args.validation_dataset,
            num_shapes=args.validation_shapes,
            seed=args.seed + 1,  # Use a different seed for validation
            min_size=args.min_size,
            max_size=args.max_size,
            power_of_two=args.power_of_two,
            include_rectangular=not args.no_rectangular,
            include_odd_sizes=not args.no_odd_sizes,
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
    
    # Validate model if requested
    if args.validate:
        validate_model(
            model_path=args.model,
            validation_dataset_path=args.validation_dataset,
            batch_size=args.batch_size,
            device=args.device,
        )

if __name__ == "__main__":
    main()
