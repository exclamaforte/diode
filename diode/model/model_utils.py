"""
Model utility functions for training and evaluation.
"""

import logging
import os
import torch
from typing import Dict, List, Optional, Tuple

from diode.types.matmul_dataset import Dataset as MatmulDataset
from diode.model.matmul_dataset_loader import create_dataloaders
from diode.model.matmul_model_trainer import (
    MatmulModelTrainer,
    train_model_from_dataset,
    analyze_worst_predictions,
)
from diode.model.matmul_timing_model import DeepMatmulTimingModel, MatmulTimingModel
from diode.utils.dataset_utils import print_dataset_statistics
from diode.utils.visualization_utils import plot_training_history

logger = logging.getLogger(__name__)


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

    # Create directories if they don't exist
    os.makedirs(
        (
            os.path.dirname(os.path.abspath(model_path))
            if os.path.dirname(model_path)
            else "."
        ),
        exist_ok=True,
    )
    os.makedirs(log_dir, exist_ok=True)

    # Load the dataset
    logger.info(f"Loading dataset from {dataset_path}")
    if dataset_path.endswith(".msgpack"):
        with open(dataset_path, "rb") as f:
            dataset_data = f.read()
        dataset = MatmulDataset.from_msgpack(dataset_data)
    else:
        with open(dataset_path, "r") as f:
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
    logger.info(
        f"Creating {model_type} model with {problem_feature_dim} problem features and {config_feature_dim} config features"
    )
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
    rmse = torch.sqrt(torch.tensor(test_loss))

    logger.info(f"Test Loss (MSE): {test_loss:.6f}")
    logger.info(f"Test RMSE: {rmse:.6f}")
    logger.info(f"Test RMSE (exp): {torch.exp(torch.tensor(rmse)):.6f}")

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
    with open(validation_dataset_path, "r") as f:
        dataset_json = f.read()

    dataset = MatmulDataset.deserialize(dataset_json)
    if dataset is None:
        logger.error(
            f"Failed to load validation dataset from {validation_dataset_path}"
        )
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
    rmse = torch.sqrt(torch.tensor(val_loss))

    logger.info(f"Validation Loss (MSE): {val_loss:.6f}")
    logger.info(f"Validation RMSE: {rmse:.6f}")
    logger.info(f"Validation RMSE (exp): {torch.exp(rmse):.6f}")

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
    if dataset_path.endswith(".msgpack"):
        with open(dataset_path, "rb") as f:
            dataset_data = f.read()
        dataset = MatmulDataset.from_msgpack(dataset_data)
    else:
        with open(dataset_path, "r") as f:
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
    rmse = torch.sqrt(torch.tensor(avg_loss))

    # Print the results
    print("\nModel Evaluation:")
    print("----------------")
    print(f"Test Loss (MSE): {avg_loss:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test RMSE (exp): {torch.exp(rmse):.6f}")

    logger.info("Example completed")
