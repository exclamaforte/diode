"""
Tests for diode.utils.visualization_utils module.
"""

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from diode.utils.visualization_utils import plot_training_history


class TestPlotTrainingHistory:
    """Test the plot_training_history function."""

    @patch("diode.utils.visualization_utils.plt")
    def test_plot_training_history_with_matplotlib(self, mock_plt):
        """Test plotting training history when matplotlib is available."""
        # Mock history data
        history = {
            "train_loss": [1.0, 0.8, 0.6, 0.4],
            "val_loss": [1.1, 0.9, 0.7, 0.5],
            "test_loss": [1.2, 1.0, 0.8, 0.6],
            "learning_rate": [0.001, 0.0008, 0.0006, 0.0004],
        }

        # Mock the subplots return value correctly
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        # Call the function
        plot_training_history(history)

        # Verify matplotlib functions were called
        mock_plt.subplots.assert_called_once_with(1, 2, figsize=(15, 5))

        # Verify ax1 (loss plot) calls
        assert mock_ax1.plot.call_count == 3  # train, val, test
        mock_ax1.set_xlabel.assert_called_with("Epoch")
        mock_ax1.set_ylabel.assert_called_with("Loss")
        mock_ax1.set_title.assert_called_with("Loss")
        mock_ax1.legend.assert_called_once()
        mock_ax1.grid.assert_called_with(True)

        # Verify ax2 (learning rate plot) calls
        mock_ax2.plot.assert_called_once_with(history["learning_rate"])
        mock_ax2.set_xlabel.assert_called_with("Epoch")
        mock_ax2.set_ylabel.assert_called_with("Learning Rate")
        mock_ax2.set_title.assert_called_with("Learning Rate")
        mock_ax2.grid.assert_called_with(True)

        # Verify layout and show
        mock_plt.tight_layout.assert_called_once()
        mock_plt.show.assert_called_once()

    @patch("diode.utils.visualization_utils.plt")
    def test_plot_training_history_with_save_path(self, mock_plt):
        """Test plotting training history with save path."""
        history = {
            "train_loss": [1.0, 0.8],
            "val_loss": [1.1, 0.9],
            "test_loss": [1.2, 1.0],
            "learning_rate": [0.001, 0.0008],
        }

        save_path = "/tmp/test_plot.png"

        # Mock the subplots return value correctly
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        plot_training_history(history, save_path=save_path)

        # Verify save was called
        mock_plt.savefig.assert_called_once_with(save_path)
        mock_plt.show.assert_called_once()

    @patch("diode.utils.visualization_utils.MATPLOTLIB_AVAILABLE", False)
    @patch("diode.utils.visualization_utils.logger")
    def test_plot_training_history_without_matplotlib(self, mock_logger):
        """Test plotting when matplotlib is not available."""
        history = {
            "train_loss": [1.0, 0.8],
            "val_loss": [1.1, 0.9],
            "test_loss": [1.2, 1.0],
            "learning_rate": [0.001, 0.0008],
        }

        # Should not raise an exception, just log a warning
        plot_training_history(history)
        mock_logger.warning.assert_called_with("Matplotlib not available, skipping plot")

    @patch("diode.utils.visualization_utils.plt")
    def test_plot_training_history_minimal_data(self, mock_plt):
        """Test plotting with minimal history data."""
        history = {
            "train_loss": [1.0],
            "val_loss": [1.1],
            "test_loss": [1.2],
            "learning_rate": [0.001],
        }

        # Mock the subplots return value correctly
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        plot_training_history(history)

        # Should still work with single data points
        assert mock_ax1.plot.call_count == 3
        mock_ax2.plot.assert_called_once()

    @patch("diode.utils.visualization_utils.plt")
    def test_plot_training_history_empty_data(self, mock_plt):
        """Test plotting with empty history data."""
        history = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "learning_rate": [],
        }

        # Mock the subplots return value correctly
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        plot_training_history(history)

        # Should still call plot functions even with empty data
        assert mock_ax1.plot.call_count == 3
        mock_ax2.plot.assert_called_once()

    @patch("diode.utils.visualization_utils.plt")
    @patch("diode.utils.visualization_utils.logger")
    def test_plot_training_history_logs_save_message(self, mock_logger, mock_plt):
        """Test that save message is logged."""
        history = {
            "train_loss": [1.0],
            "val_loss": [1.1],
            "test_loss": [1.2],
            "learning_rate": [0.001],
        }
        save_path = "/tmp/test.png"

        # Mock the subplots return value correctly
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        plot_training_history(history, save_path=save_path)

        mock_logger.info.assert_called_with(f"Saved plot to {save_path}")

    @patch("diode.utils.visualization_utils.MATPLOTLIB_AVAILABLE", False)
    @patch("diode.utils.visualization_utils.logger")
    def test_plot_training_history_logs_import_warning(self, mock_logger):
        """Test that import warning is logged when matplotlib unavailable."""
        history = {
            "train_loss": [1.0],
            "val_loss": [1.1],
            "test_loss": [1.2],
            "learning_rate": [0.001],
        }

        plot_training_history(history)

        mock_logger.warning.assert_called_with(
            "Matplotlib not available, skipping plot"
        )
