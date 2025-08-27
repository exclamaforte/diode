"""
Visualization utility functions for matrix multiplication data analysis.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


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
