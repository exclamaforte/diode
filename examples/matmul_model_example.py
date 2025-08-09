"""
Example script demonstrating how to train and use a matrix multiplication timing prediction model.
"""

import torch
import numpy as np
import os
import logging
import argparse
from typing import Optional

from diode.types.matmul_dataset import Dataset as MatmulDataset
from diode.model.matmul_timing_model import MatmulTimingModel, DeepMatmulTimingModel
from diode.model.matmul_dataset_loader import MatmulTimingDataset, create_dataloaders
from diode.model.matmul_model_trainer import MatmulModelTrainer, train_model_from_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_example(
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
    Run the example.
    
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
    
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
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
    
    # Make predictions on the test set
    logger.info("Making predictions on the test set")
    evaluate_model(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        hardware_name=hardware_name,
        op_name=op_name,
        device=device,
    )
    
    logger.info("Example completed")

def print_dataset_statistics(
    dataset: MatmulDataset,
    hardware_name: Optional[str] = None,
    op_name: Optional[str] = None,
) -> None:
    """
    Print statistics about the dataset.
    
    Args:
        dataset: The dataset to print statistics for
        hardware_name: Optional hardware name to filter by
        op_name: Optional operation name to filter by
    """
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

def evaluate_model(
    model: torch.nn.Module,
    dataset: MatmulDataset,
    batch_size: int = 64,
    hardware_name: Optional[str] = None,
    op_name: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Evaluate the model on the test set.
    
    Args:
        model: The model to evaluate
        dataset: The dataset to evaluate on
        batch_size: Batch size for the dataloader
        hardware_name: Optional hardware name to filter by
        op_name: Optional operation name to filter by
        device: Device to evaluate on
    """
    # Create the dataloaders
    _, _, test_dataloader = create_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        hardware_name=hardware_name,
        op_name=op_name,
        log_transform=True,
        num_workers=4,
        seed=42,
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

def main():
    """
    Main function.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate a matrix multiplication timing prediction model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("--model-type", type=str, default="deep", choices=["base", "deep"], help="Type of model to train")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for the dataloaders")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for the optimizer")
    parser.add_argument("--patience", type=int, default=10, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--hardware-name", type=str, help="Hardware name to filter by")
    parser.add_argument("--op-name", type=str, help="Operation name to filter by")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    args = parser.parse_args()
    
    # Run the example
    run_example(
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

if __name__ == "__main__":
    main()
