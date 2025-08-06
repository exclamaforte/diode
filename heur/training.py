"""Training utilities for ML heuristics."""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """Trainer class for heuristic models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize the trainer.

        Args:
            model: The model to train
            optimizer: The optimizer to use (default: Adam)
            loss_fn: The loss function to use (default: MSELoss for regression, BCELoss for binary classification)
            device: The device to use for training (default: cuda if available, else cpu)
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Default optimizer is Adam with lr=0.001
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=0.001)
        
        # Default loss function depends on output dimension
        if loss_fn is None:
            if hasattr(model, "output_dim") and model.output_dim == 1:
                self.loss_fn = nn.BCELoss()
            else:
                self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = loss_fn
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": {},
            "val_metrics": {},
        }
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        metrics: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, float]:
        """Train the model for one epoch.

        Args:
            train_loader: DataLoader for training data
            metrics: Dictionary of metric functions to evaluate during training

        Returns:
            Dictionary of training metrics for this epoch
        """
        self.model.train()
        total_loss = 0.0
        metric_values = {name: 0.0 for name in (metrics or {})}
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc="Training", leave=False) as pbar:
            for inputs, targets in pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                
                if metrics:
                    for name, metric_fn in metrics.items():
                        metric_values[name] += metric_fn(outputs, targets).item()
                
                # Update progress bar
                pbar.set_postfix(loss=loss.item())
        
        # Calculate average metrics
        avg_loss = total_loss / num_batches
        avg_metrics = {name: value / num_batches for name, value in metric_values.items()}
        avg_metrics["loss"] = avg_loss
        
        return avg_metrics
    
    def evaluate(
        self, 
        val_loader: DataLoader, 
        metrics: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, float]:
        """Evaluate the model on validation data.

        Args:
            val_loader: DataLoader for validation data
            metrics: Dictionary of metric functions to evaluate during validation

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        metric_values = {name: 0.0 for name in (metrics or {})}
        num_batches = len(val_loader)
        
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation", leave=False) as pbar:
                for inputs, targets in pbar:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    
                    # Update metrics
                    total_loss += loss.item()
                    
                    if metrics:
                        for name, metric_fn in metrics.items():
                            metric_values[name] += metric_fn(outputs, targets).item()
                    
                    # Update progress bar
                    pbar.set_postfix(loss=loss.item())
        
        # Calculate average metrics
        avg_loss = total_loss / num_batches
        avg_metrics = {name: value / num_batches for name, value in metric_values.items()}
        avg_metrics["loss"] = avg_loss
        
        return avg_metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: Optional[DataLoader] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        early_stopping_patience: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Train the model for multiple epochs.

        Args:
            train_loader: DataLoader for training data
            epochs: Number of epochs to train
            val_loader: Optional DataLoader for validation data
            metrics: Dictionary of metric functions to evaluate during training and validation
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            checkpoint_path: Path to save the best model checkpoint

        Returns:
            Dictionary containing training history
        """
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader, metrics)
            
            # Evaluate on validation set if provided
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, metrics)
                
                # Check for early stopping
                if early_stopping_patience is not None:
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        patience_counter = 0
                        
                        # Save checkpoint if path is provided
                        if checkpoint_path:
                            torch.save(self.model.state_dict(), checkpoint_path)
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"Early stopping at epoch {epoch+1}")
                            break
            
            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            if val_loader is not None:
                self.history["val_loss"].append(val_metrics["loss"])
            
            for name in (metrics or {}):
                if name not in self.history["train_metrics"]:
                    self.history["train_metrics"][name] = []
                self.history["train_metrics"][name].append(train_metrics[name])
                
                if val_loader is not None:
                    if name not in self.history["val_metrics"]:
                        self.history["val_metrics"][name] = []
                    self.history["val_metrics"][name].append(val_metrics[name])
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - train_loss: {train_metrics['loss']:.4f}", end="")
            
            if val_loader is not None:
                print(f" - val_loss: {val_metrics['loss']:.4f}", end="")
            
            for name in (metrics or {}):
                print(f" - train_{name}: {train_metrics[name]:.4f}", end="")
                if val_loader is not None:
                    print(f" - val_{name}: {val_metrics[name]:.4f}", end="")
            
            print()
        
        # Load best model if checkpoint was saved
        if checkpoint_path and early_stopping_patience is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))
        
        return self.history
