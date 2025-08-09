"""
Trainer for matrix multiplication timing prediction models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
import os
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from collections import OrderedDict
# Import matplotlib only when needed to avoid dependency issues
from tqdm import tqdm

from diode.types.matmul_dataset import Dataset as MatmulDataset
from diode.model.matmul_timing_model import MatmulTimingModel, DeepMatmulTimingModel
from diode.model.matmul_dataset_loader import MatmulTimingDataset, create_dataloaders

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
        patience: int = 10,
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
            verbose=verbose,
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
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"Test Loss: {test_loss:.6f}, "
                      f"LR: {current_lr:.6f}")
            
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
            # Move the data to the device
            problem_features = problem_features.to(self.device)
            config_features = config_features.to(self.device)
            targets = targets.to(self.device)
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(problem_features, config_features)
            
            # Calculate the loss
            loss = self.criterion(outputs, targets)
            
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
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        # Iterate over the batches
        with torch.no_grad():
            for problem_features, config_features, targets in dataloader:
                # Move the data to the device
                problem_features = problem_features.to(self.device)
                config_features = config_features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(problem_features, config_features)
                
                # Calculate the loss
                loss = self.criterion(outputs, targets)
                
                # Update the total loss
                total_loss += loss.item()
        
        # Calculate the average loss
        avg_loss = total_loss / num_batches
        
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


def train_model_from_dataset(
    dataset: MatmulDataset,
    model_type: str = "deep",
    batch_size: int = 64,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 10,
    log_dir: str = "logs",
    checkpoint_path: Optional[str] = None,
    hardware_name: Optional[str] = None,
    op_name: Optional[str] = None,
    log_transform: bool = True,
    num_workers: int = 4,
    seed: int = 42,
    verbose: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[Union[MatmulTimingModel, DeepMatmulTimingModel], Dict[str, List[float]]]:
    """
    Train a model from a MatmulDataset.
    
    Args:
        dataset: The MatmulDataset containing the timing data
        model_type: Type of model to train ("base" or "deep")
        batch_size: Batch size for the dataloaders
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        patience: Number of epochs to wait for improvement before early stopping
        log_dir: Directory to save logs and checkpoints
        checkpoint_path: Path to save the best model checkpoint
        hardware_name: Optional hardware name to filter by
        op_name: Optional operation name to filter by
        log_transform: Whether to apply log transform to the timing values
        num_workers: Number of workers for the dataloaders
        seed: Random seed for reproducibility
        verbose: Whether to print progress
        device: Device to train on
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Create the dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        hardware_name=hardware_name,
        op_name=op_name,
        log_transform=log_transform,
        num_workers=num_workers,
        seed=seed,
    )
    
    # Get the feature dimensions
    problem_feature_dim = train_dataloader.dataset.dataset.problem_feature_dim
    config_feature_dim = train_dataloader.dataset.dataset.config_feature_dim
    
    # Create the model
    if model_type.lower() == "base":
        model = MatmulTimingModel(
            problem_feature_dim=problem_feature_dim,
            config_feature_dim=config_feature_dim,
        )
    elif model_type.lower() == "deep":
        model = DeepMatmulTimingModel(
            problem_feature_dim=problem_feature_dim,
            config_feature_dim=config_feature_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create the trainer
    trainer = MatmulModelTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        log_dir=log_dir,
        device=device,
    )
    
    # Train the model
    history = trainer.train(
        num_epochs=num_epochs,
        patience=patience,
        checkpoint_path=checkpoint_path,
        verbose=verbose,
    )
    
    return model, history
