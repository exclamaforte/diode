"""
Trainer for matrix multiplication timing prediction models.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch_diode.model.matmul_dataset_loader import create_dataloaders
from torch_diode.model.matmul_model_config import MatmulModelConfig
from torch_diode.model.matmul_timing_model import DeepMatmulTimingModel, MatmulTimingModel

from torch_diode.types.matmul_dataset import Dataset as MatmulDataset
from torch.utils.data import DataLoader

# Import matplotlib only when needed to avoid dependency issues
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MatmulModelTrainer:
    """
    Trainer for matrix multiplication timing prediction models.
    """

    def __init__(
        self,
        model: Union[MatmulTimingModel, DeepMatmulTimingModel],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        log_dir: str = "logs",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            test_dataloader: DataLoader for test data
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay for the optimizer
            log_dir: Directory to save logs and checkpoints
            device: Device to train on
        """
        self.model = model.to(device)

        # Ensure all parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.log_dir = log_dir
        self.device = device

        # Create the optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Create the loss function
        self.criterion = nn.MSELoss()

        # Create the log directory
        os.makedirs(log_dir, exist_ok=True)

        # Initialize training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "learning_rate": [],
        }

        logger.info(f"Initialized trainer with model: {type(model).__name__}")
        logger.info(f"Training on device: {device}")

    def train(
        self,
        num_epochs: int,
        patience: int = 20,
        checkpoint_path: Optional[str] = None,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
        scheduler_min_lr: float = 1e-6,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            num_epochs: Number of epochs to train for
            patience: Number of epochs to wait for improvement before early stopping
            checkpoint_path: Path to save the best model checkpoint
            scheduler_factor: Factor to reduce learning rate by
            scheduler_patience: Number of epochs to wait before reducing learning rate
            scheduler_min_lr: Minimum learning rate
            verbose: Whether to print progress

        Returns:
            Training history
        """
        # Create the learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr,
        )

        # Initialize variables for early stopping
        best_val_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0

        # Train the model
        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss = self._train_epoch(verbose=verbose)

            # Evaluate on validation set
            val_loss = self._evaluate(self.val_dataloader, "Validation")

            # Evaluate on test set
            test_loss = self._evaluate(self.test_dataloader, "Test")

            # Update the learning rate
            scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update the history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["test_loss"].append(test_loss)
            self.history["learning_rate"].append(current_lr)

            # Print the progress
            if verbose:
                print(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"Test Loss: {test_loss:.6f}, "
                    f"LR: {current_lr:.6f}"
                )

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                epochs_without_improvement = 0

                # Save the checkpoint
                if checkpoint_path is not None:
                    self.model.save(checkpoint_path)
                    if verbose:
                        print(f"Saved checkpoint to {checkpoint_path}")
            else:
                epochs_without_improvement += 1
                if verbose:
                    print(f"No improvement for {epochs_without_improvement} epochs")

                # Early stopping
                if epochs_without_improvement >= patience:
                    if verbose:
                        print(f"Early stopping after {epoch+1} epochs")
                    break

        # Restore the best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if verbose:
                print("Restored best model")

        return self.history

    def _train_epoch(self, verbose: bool = True) -> float:
        """
        Train the model for one epoch.

        Args:
            verbose: Whether to print progress

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)

        # Create progress bar
        if verbose:
            pbar = tqdm(total=num_batches, desc="Training")

        # Iterate over the batches
        for problem_features, config_features, targets in self.train_dataloader:
            # Move the data to the device and ensure they require gradients
            problem_features = problem_features.to(self.device).requires_grad_(True)
            config_features = config_features.to(self.device).requires_grad_(True)
            targets = targets.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(problem_features, config_features)

            # Calculate the loss manually to ensure it has a grad_fn
            loss = torch.mean((outputs - targets) ** 2)

            # Backward pass
            loss.backward()

            # Update the weights
            self.optimizer.step()

            # Update the total loss
            total_loss += loss.item()

            # Update the progress bar
            if verbose:
                pbar.update(1)

        # Close the progress bar
        if verbose:
            pbar.close()

        # Calculate the average loss
        avg_loss = total_loss / num_batches

        return avg_loss

    def _evaluate(self, dataloader: DataLoader, name: str = "Evaluation") -> float:
        """
        Evaluate the model on a dataset.

        Args:
            dataloader: DataLoader for the dataset
            name: Name of the dataset for logging

        Returns:
            Average loss on the dataset
        """
        logger.info(f"Starting {name} evaluation...")
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        num_samples = 0

        logger.info(f"{name} dataset contains {num_batches} batches")

        # Create progress bar for evaluation
        pbar = tqdm(total=num_batches, desc=f"{name} Evaluation")

        # Iterate over the batches
        with torch.no_grad():
            for batch_idx, (problem_features, config_features, targets) in enumerate(
                dataloader
            ):
                # Move the data to the device
                problem_features = problem_features.to(self.device)
                config_features = config_features.to(self.device)
                targets = targets.to(self.device)

                batch_size = problem_features.size(0)
                num_samples += batch_size

                # Forward pass
                outputs = self.model(problem_features, config_features)

                # Calculate the loss
                loss = self.criterion(outputs, targets)

                # Update the total loss
                total_loss += loss.item()

                # Update progress bar with current batch loss
                current_avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix(
                    {
                        "batch_loss": f"{loss.item():.6f}",
                        "avg_loss": f"{current_avg_loss:.6f}",
                        "samples": num_samples,
                    }
                )
                pbar.update(1)

                # Log progress every 10% of batches
                if (batch_idx + 1) % max(1, num_batches // 10) == 0:
                    progress_pct = ((batch_idx + 1) / num_batches) * 100
                    logger.info(
                        f"{name} evaluation progress: {progress_pct:.1f}% "
                        f"({batch_idx + 1}/{num_batches} batches, "
                        f"{num_samples} samples processed)"
                    )

        # Close progress bar
        pbar.close()

        # Calculate the average loss
        avg_loss = total_loss / num_batches

        logger.info(f"{name} evaluation completed:")
        logger.info(f"  - Processed {num_samples} samples in {num_batches} batches")
        logger.info(f"  - Average loss (MSE): {avg_loss:.6f}")
        logger.info(f"  - RMSE: {torch.sqrt(torch.tensor(avg_loss)):.6f}")
        logger.info(
            f"  - RMSE (exp): {torch.exp(torch.sqrt(torch.tensor(avg_loss))):.6f}"
        )

        return avg_loss

    def predict(
        self,
        problem_features: torch.Tensor,
        config_features: torch.Tensor,
        log_transform: bool = True,
    ) -> torch.Tensor:
        """
        Make predictions with the model.

        Args:
            problem_features: Problem features tensor
            config_features: Config features tensor
            log_transform: Whether the model was trained with log-transformed targets

        Returns:
            Predicted execution times
        """
        self.model.eval()

        # Move the data to the device
        problem_features = problem_features.to(self.device)
        config_features = config_features.to(self.device)

        # Make predictions
        with torch.no_grad():
            predictions = self.model(problem_features, config_features)

        # Convert from log space if needed
        if log_transform:
            predictions = torch.exp(predictions)

        return predictions

    def plot_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot the training history.

        Args:
            save_path: Path to save the plot
        """
        try:
            # Import matplotlib only when needed
            import matplotlib.pyplot as plt

            # Create the figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Plot the loss
            ax1.plot(self.history["train_loss"], label="Train")
            ax1.plot(self.history["val_loss"], label="Validation")
            ax1.plot(self.history["test_loss"], label="Test")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Loss")
            ax1.legend()
            ax1.grid(True)

            # Plot the learning rate
            ax2.plot(self.history["learning_rate"])
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


def get_model_save_path(
    heuristic_name: str,
    hardware_name: str,
    model_name: str,
    base_dir: Optional[str] = None,
) -> Path:
    """
    Get the path to save a model based on heuristic and hardware.

    Args:
        heuristic_name: Name of the heuristic (e.g., "matmul")
        hardware_name: Name of the hardware (e.g., "NVIDIA-H100", "AMD-MI250", "Intel-CPU")
        model_name: Name of the model file
        base_dir: Base directory for models. If None, uses the default
                 diode_models directory structure

    Returns:
        Path object for the model save location
    """
    if base_dir is None:
        # Use the default directory structure in diode_models
        try:
            import diode_models

            base_dir = Path(diode_models.__file__).parent
        except ImportError:
            # Fall back to the current directory if diode_models is not installed
            base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    # Create the directory structure: <base_dir>/<heuristic>/<hardware>/
    save_dir = base_dir / heuristic_name / hardware_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Return the full path including the model name
    return save_dir / model_name


def save_model_with_config(
    model: Union[MatmulTimingModel, DeepMatmulTimingModel],
    config: MatmulModelConfig,
    save_path: Union[str, Path],
) -> None:
    """
    Save a model with its configuration.

    Args:
        model: The model to save
        config: The model configuration
        save_path: Path to save the model (without extension)
    """
    save_path = Path(save_path)

    # Save the model
    model.save(str(save_path.with_suffix(".pt")))

    # Save the configuration
    config.save(save_path)

    logger.info(f"Saved model to {save_path.with_suffix('.pt')}")


def load_model_with_config(
    load_path: Union[str, Path],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[Union[MatmulTimingModel, DeepMatmulTimingModel], MatmulModelConfig]:
    """
    Load a model with its configuration.

    Args:
        load_path: Path to load the model from (without extension)
        device: Device to load the model to

    Returns:
        Tuple of (model, config)
    """
    load_path = Path(load_path)

    # Load the configuration
    try:
        config = MatmulModelConfig.load(load_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"No config file found for {load_path}")

    # Load the model
    model_path = load_path.with_suffix(".pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Create the model based on the config
    if config.model_type.lower() == "base":
        model = MatmulTimingModel(
            problem_feature_dim=config.problem_feature_dim,
            config_feature_dim=config.config_feature_dim,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate,
        )
    elif config.model_type.lower() == "deep":
        model = DeepMatmulTimingModel(
            problem_feature_dim=config.problem_feature_dim,
            config_feature_dim=config.config_feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout_rate=config.dropout_rate,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    # Load the state dict
    model.load_state_dict(
        torch.load(model_path, map_location=device)["model_state_dict"]
    )
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from {model_path}")

    return model, config


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
            errors = torch.abs(outputs - targets).cpu()

            for i in range(len(errors)):
                all_errors.append(
                    {
                        "error": errors[i].item(),
                        "predicted": outputs[i].item(),
                        "actual": targets[i].item(),
                        "problem_features": problem_features[i].cpu(),
                        "config_features": config_features[i].cpu(),
                    }
                )

    # Sort by error (descending)
    all_errors.sort(key=lambda x: x["error"], reverse=True)

    # Print the top_n worst predictions
    print(f"\nTop {top_n} worst predictions:")
    print("----------------------------")

    for i, error_data in enumerate(all_errors[:top_n]):
        print(f"Error {i+1}:")
        print(f"  Predicted: {torch.exp(torch.tensor(error_data['predicted'])):.6f} ms")
        print(f"  Actual: {torch.exp(torch.tensor(error_data['actual'])):.6f} ms")
        print(f"  Error (log space): {error_data['error']:.6f}")
        print(
            f"  Error (ratio): {torch.exp(torch.tensor(error_data['predicted'])) / torch.exp(torch.tensor(error_data['actual'])):.2f}x"
        )

        # Extract problem features (M, N, K) with better error handling
        try:
            # The problem features are in the order: B, M, N, K, ...
            # And the log-transformed versions are at indices 7, 8, 9
            # We use the log-transformed versions for better numerical stability
            M = int(torch.exp(torch.tensor(error_data["problem_features"][8])))
            N = int(torch.exp(torch.tensor(error_data["problem_features"][9])))
            K = int(torch.exp(torch.tensor(error_data["problem_features"][10])))

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
            config_features = error_data["config_features"]
            block_m = int(config_features[1])
            block_n = int(config_features[2])
            block_k = int(config_features[3])
            group_m = int(config_features[4])
            num_stages = int(config_features[5])
            num_warps = int(config_features[6])
            even_k = bool(int(config_features[7]))
            allow_tf32 = bool(int(config_features[8]))
            use_fast_accum = bool(int(config_features[9]))

            print(
                f"  Config: block_m={block_m}, block_n={block_n}, block_k={block_k}, "
                f"group_m={group_m}, num_stages={num_stages}, num_warps={num_warps}"
            )
            print(
                f"          EVEN_K={even_k}, ALLOW_TF32={allow_tf32}, USE_FAST_ACCUM={use_fast_accum}"
            )
        except (ValueError, IndexError):
            print(f"  Config: (unable to extract configuration)")

        print()


def train_model_from_dataset(
    dataset: MatmulDataset,
    config: Optional[MatmulModelConfig] = None,
    log_dir: str = "logs",
    checkpoint_path: Optional[str] = None,
    num_workers: int = 4,
    verbose: bool = True,
    save_model: bool = True,
) -> Tuple[
    Union[MatmulTimingModel, DeepMatmulTimingModel],
    Dict[str, List[float]],
    MatmulModelConfig,
]:
    """
    Train a model from a MatmulDataset.

    Args:
        dataset: The MatmulDataset containing the timing data
        config: Configuration for the model and training. If None, default config is used.
        log_dir: Directory to save logs and checkpoints
        checkpoint_path: Path to save the best model checkpoint
        num_workers: Number of workers for the dataloaders
        verbose: Whether to print progress
        save_model: Whether to save the model after training

    Returns:
        Tuple of (trained_model, training_history, config)
    """
    # Create or update the configuration
    if config is None:
        config = MatmulModelConfig()

    # Get hardware info from dataset if not specified in config
    if config.hardware_name == "unknown" and dataset.hardware_names:
        config.hardware_name = dataset.hardware_names[0]

        # Set a more granular hardware type based on hardware_name
        # This is a simple heuristic and might need to be updated based on actual hardware naming
        if "h100" in config.hardware_name.lower():
            config.hardware_type = "NVIDIA-H100"
        elif "a100" in config.hardware_name.lower():
            config.hardware_type = "NVIDIA-A100"
        elif "v100" in config.hardware_name.lower():
            config.hardware_type = "NVIDIA-V100"
        elif "mi" in config.hardware_name.lower():
            config.hardware_type = "AMD-MI250"
        elif "cpu" in config.hardware_name.lower():
            config.hardware_type = "Intel-CPU"
        else:
            config.hardware_type = config.hardware_name

    # Create the dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        dataset=dataset,
        batch_size=config.batch_size,
        hardware_name=config.hardware_name,
        op_name=config.op_name,
        log_transform=config.log_transform,
        num_workers=num_workers,
        seed=config.seed,
        debug=True,  # Enable debug mode to check data quality
    )

    # Check if dataloaders were created successfully
    if train_dataloader is None:
        logger.warning(
            "Dataset is empty or could not create dataloaders. Returning None."
        )
        return None, {}, config

    # Get the feature dimensions
    problem_feature_dim = train_dataloader.dataset.dataset.problem_feature_dim
    config_feature_dim = train_dataloader.dataset.dataset.config_feature_dim

    # Update the config with the feature dimensions
    config.problem_feature_dim = problem_feature_dim
    config.config_feature_dim = config_feature_dim

    # Create the model
    if config.model_type.lower() == "base":
        model = MatmulTimingModel(
            problem_feature_dim=problem_feature_dim,
            config_feature_dim=config_feature_dim,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate,
        )
    elif config.model_type.lower() == "deep":
        model = DeepMatmulTimingModel(
            problem_feature_dim=problem_feature_dim,
            config_feature_dim=config_feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout_rate=config.dropout_rate,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    # Create the trainer
    trainer = MatmulModelTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        log_dir=log_dir,
        device=config.device,
    )

    # Train the model
    history = trainer.train(
        num_epochs=config.num_epochs,
        patience=config.patience,
        checkpoint_path=checkpoint_path,
        verbose=verbose,
    )

    # Save the model with its configuration
    if save_model:
        # Generate a model name based on the hardware and model type
        model_name = f"matmul_{config.hardware_name}_{config.model_type}"

        # Get the save path
        save_path = get_model_save_path(
            heuristic_name=config.heuristic_name,
            hardware_name=config.hardware_name,
            model_name=model_name,
        )

        # Save the model with its configuration
        save_model_with_config(model, config, save_path)

        if verbose:
            print(f"Saved model to {save_path}.pt")
            print(f"Saved config to {save_path}.json")

    return model, history, config
