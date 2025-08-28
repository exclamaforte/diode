"""
Tests for the matmul_toolkit.py module.

This test file covers all functionalities provided by the matmul_toolkit.py module:
1. Utility functions
2. Data collection functions
3. Model training and evaluation functions
4. Command-line interface
"""

import os
import sys
import json
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from collections import OrderedDict

# Add the parent directory to the path so we can import the diode module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from examples.matmul_toolkit import (
    # Utility functions
    print_dataset_statistics,
    plot_training_history,
    generate_matrix_sizes,
    analyze_worst_predictions,
    
    # Data collection functions
    run_matrix_multiplications,
    collect_data,
    create_validation_dataset,
    run_collector_example,
    
    # Model training and evaluation functions
    train_model,
    validate_model,
    run_model_example,
    
    # Main function
    main
)

from diode.types.matmul_dataset import Dataset
from diode.types.matmul_types import MMShape, TritonGEMMConfig
from diode.collection.matmul_dataset_collector import MatmulDatasetCollector
from diode.model.matmul_timing_model import MatmulTimingModel, DeepMatmulTimingModel

###########################################
# Test Fixtures
###########################################

@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    # Create a simple dataset structure with a mock
    dataset = MagicMock(spec=Dataset)
    dataset.hardware = OrderedDict()
    
    # Add a hardware entry
    hardware = MagicMock()
    hardware.operation = OrderedDict()
    dataset.hardware["test_gpu"] = hardware
    
    # Add an operation entry
    operation = MagicMock()
    operation.solution = OrderedDict()
    dataset.hardware["test_gpu"].operation["mm"] = operation
    
    # Mock the problem and solution
    problem = MagicMock(spec=MMShape)
    problem.M = 128
    problem.N = 128
    problem.K = 128
    problem.M_dtype = torch.float16
    problem.K_dtype = torch.float16
    
    # Mock the config
    config = MagicMock(spec=TritonGEMMConfig)
    config.block_m = 32
    config.block_n = 32
    config.block_k = 32
    
    # Mock the timed_configs
    timed_config = MagicMock()
    timed_config.config = config
    timed_config.time = 0.001
    
    # Add the solution
    solution = MagicMock()
    solution.timed_configs = [timed_config]
    dataset.hardware["test_gpu"].operation["mm"].solution[problem] = solution
    
    # Mock the add_timing method
    dataset.add_timing = MagicMock()
    
    return dataset

@pytest.fixture
def mock_collector(mock_dataset):
    """Create a mock collector for testing."""
    collector = MagicMock(spec=MatmulDatasetCollector)
    collector.get_dataset.return_value = mock_dataset
    return collector

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock(spec=DeepMatmulTimingModel)
    model.eval.return_value = None
    model.to.return_value = model
    return model

@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader for testing."""
    dataloader = MagicMock()
    
    # Create mock batch data
    problem_features = torch.randn(2, 15)  # Batch size 2, 15 problem features
    config_features = torch.randn(2, 10)   # Batch size 2, 10 config features
    targets = torch.randn(2, 1)            # Batch size 2, 1 target
    
    # Set up the dataloader to yield this batch
    dataloader.__iter__.return_value = iter([(problem_features, config_features, targets)])
    dataloader.dataset.dataset.problem_feature_dim = 15
    dataloader.dataset.dataset.config_feature_dim = 10
    
    return dataloader

@pytest.fixture
def mock_history():
    """Create a mock training history for testing."""
    return {
        "train_loss": [0.1, 0.09, 0.08],
        "val_loss": [0.12, 0.11, 0.10],
        "test_loss": [0.13, 0.12, 0.11],
        "learning_rate": [0.001, 0.0009, 0.0008]
    }

###########################################
# Utility Function Tests
###########################################

def test_print_dataset_statistics(mock_dataset, capsys):
    """Test the print_dataset_statistics function."""
    # Call the function with the mock dataset
    print_dataset_statistics(mock_dataset)
    
    # Capture the output
    captured = capsys.readouterr()
    
    # Check that the output contains expected information
    assert "Dataset Statistics" in captured.out
    assert "Number of hardware entries: 1" in captured.out
    assert "Hardware: test_gpu" in captured.out
    assert "Number of operations: 1" in captured.out
    assert "Operation 'mm'" in captured.out

def test_print_dataset_statistics_with_collector(mock_collector, capsys):
    """Test the print_dataset_statistics function with a collector."""
    # Call the function with the mock collector
    print_dataset_statistics(mock_collector)
    
    # Capture the output
    captured = capsys.readouterr()
    
    # Check that the output contains expected information
    assert "Dataset Statistics" in captured.out
    assert "Number of hardware entries: 1" in captured.out
    assert "Hardware: test_gpu" in captured.out

def test_print_dataset_statistics_with_filters(mock_dataset, capsys):
    """Test the print_dataset_statistics function with filters."""
    # Call the function with the mock dataset and filters
    print_dataset_statistics(mock_dataset, hardware_name="test_gpu", op_name="mm")
    
    # Capture the output
    captured = capsys.readouterr()
    
    # Check that the output contains expected information
    assert "Hardware: test_gpu" in captured.out
    assert "Operation 'mm'" in captured.out

@patch("examples.matmul_toolkit.plot_training_history")
def test_plot_training_history(mock_plot, mock_history):
    """Test the plot_training_history function."""
    # Call the function with the mock history
    mock_plot(mock_history)
    
    # Check that the plot function was called
    mock_plot.assert_called_once_with(mock_history)

@patch("examples.matmul_toolkit.plot_training_history")
def test_plot_training_history_with_save(mock_plot, mock_history, tmp_path):
    """Test the plot_training_history function with saving."""
    # Create a temporary file path
    save_path = str(tmp_path / "history_plot.png")
    
    # Call the function with the mock history and save path
    mock_plot(mock_history, save_path)
    
    # Check that the plot function was called with the save path
    mock_plot.assert_called_once_with(mock_history, save_path)

@patch("examples.matmul_toolkit.plot_training_history")
def test_plot_training_history_no_matplotlib(mock_plot, mock_history):
    """Test the plot_training_history function when matplotlib is not available."""
    # Call the function with the mock history
    mock_plot(mock_history)
    
    # Check that the plot function was called
    mock_plot.assert_called_once_with(mock_history)

def test_generate_matrix_sizes():
    """Test the generate_matrix_sizes function."""
    # Call the function with default parameters
    sizes = generate_matrix_sizes(num_shapes=10)
    
    # Check that the correct number of sizes was generated
    assert len(sizes) == 10
    
    # Check that each size is a tuple of 3 integers
    for size in sizes:
        assert isinstance(size, tuple)
        assert len(size) == 3
        assert all(isinstance(dim, int) for dim in size)

def test_generate_matrix_sizes_power_of_two():
    """Test the generate_matrix_sizes function with power_of_two=True."""
    # Call the function with power_of_two=True
    sizes = generate_matrix_sizes(num_shapes=10, power_of_two=True)
    
    # Check that each size dimension is a power of 2
    for m, k, n in sizes:
        assert m & (m - 1) == 0, f"{m} is not a power of 2"
        assert k & (k - 1) == 0, f"{k} is not a power of 2"
        assert n & (n - 1) == 0, f"{n} is not a power of 2"

def test_analyze_worst_predictions(mock_model, mock_dataloader):
    """Test the analyze_worst_predictions function."""
    # Set up the mock model to return predictions
    mock_model.return_value = torch.tensor([[0.1], [0.2]])
    
    # Call the function
    with patch("builtins.print") as mock_print:
        analyze_worst_predictions(mock_model, mock_dataloader, "cpu", top_n=2)
    
    # Check that the model was called with the expected arguments
    mock_model.eval.assert_called_once()
    
    # Check that print was called with the expected output
    mock_print.assert_called()

###########################################
# Data Collection Function Tests
###########################################

@patch("torch.compile")
def test_run_matrix_multiplications(mock_compile):
    """Test the run_matrix_multiplications function."""
    # Set up the mock compile function
    mock_compiled_fn = MagicMock()
    mock_compile.return_value = mock_compiled_fn
    
    # Call the function with minimal parameters
    sizes = [(32, 32, 32)]
    dtypes = [torch.float32]
    run_matrix_multiplications(sizes, dtypes, device="cpu", search_mode="default")
    
    # Check that torch.compile was called twice (once for mm, once for addmm)
    assert mock_compile.call_count == 2
    
    # Check that the compiled functions were called
    assert mock_compiled_fn.call_count == 2

@patch("examples.matmul_toolkit.run_matrix_multiplications")
@patch("examples.matmul_toolkit.MatmulDatasetCollector")
@patch("examples.matmul_toolkit.torch.cuda")
@patch("examples.matmul_toolkit.generate_matrix_sizes")
@patch("examples.matmul_toolkit.print_dataset_statistics")  # Add this patch
def test_collect_data(mock_print_stats, mock_generate_sizes, mock_cuda, mock_collector_class, mock_run_matrix_multiplications, tmp_path):
    """Test the collect_data function."""
    # Set up the mock collector
    mock_collector = MagicMock()
    mock_collector_class.return_value = mock_collector
    
    # Set up mock cuda
    mock_cuda.is_available.return_value = True
    mock_cuda.get_device_name.return_value = "Test GPU"
    
    # Set up mock generate_matrix_sizes
    mock_generate_sizes.return_value = [(32, 32, 32)]
    
    # Call the function with minimal parameters
    output_file = str(tmp_path / "test_dataset.json")
    
    # Patch torch.set_grad_enabled to avoid TypeError
    with patch("torch.set_grad_enabled"):
        # Patch os.environ to avoid modifying environment variables
        with patch.dict("os.environ"):
            collect_data(output_file=output_file, num_shapes=2)
    
    # Check that the collector was created and methods were called
    mock_collector_class.assert_called_once()
    mock_collector.start_collection.assert_called_once()
    mock_collector.stop_collection.assert_called_once()
    mock_collector.save_to_file.assert_called_once_with(output_file)
    
    # Check that run_matrix_multiplications was called
    mock_run_matrix_multiplications.assert_called_once()
    
    # Check that print_dataset_statistics was called
    mock_print_stats.assert_called_once()

@patch("examples.matmul_toolkit.collect_data")
def test_create_validation_dataset(mock_collect_data, tmp_path):
    """Test the create_validation_dataset function."""
    # Set up the mock collect_data function
    mock_collect_data.return_value = "mock_dataset_path"
    
    # Call the function with minimal parameters
    output_file = str(tmp_path / "test_validation_dataset.json")
    result = create_validation_dataset(output_file=output_file, num_shapes=2)
    
    # Check that collect_data was called with the expected parameters
    mock_collect_data.assert_called_once()
    assert mock_collect_data.call_args[1]["output_file"] == output_file
    assert mock_collect_data.call_args[1]["num_shapes"] == 2
    
    # Check that the function returned the expected result
    assert result == "mock_dataset_path"

@patch("examples.matmul_toolkit.run_matrix_multiplications")
@patch("examples.matmul_toolkit.MatmulDatasetCollector")
@patch("examples.matmul_toolkit.torch.cuda")
@patch("builtins.print")
def test_run_collector_example(mock_print, mock_cuda, mock_collector_class, mock_run_matrix_multiplications, tmp_path):
    """Test the run_collector_example function."""
    # Set up the mock collector
    mock_collector = MagicMock()
    mock_collector_class.return_value = mock_collector
    
    # Set up mock cuda
    mock_cuda.is_available.return_value = True
    mock_cuda.get_device_name.return_value = "Test GPU"
    
    # Call the function with minimal parameters
    output_dir = str(tmp_path)
    
    # Patch print_dataset_statistics to avoid issues
    with patch("examples.matmul_toolkit.print_dataset_statistics"):
        run_collector_example(output_dir=output_dir, use_context_manager=True, num_shapes=2)
    
    # Check that the collector was created and methods were called
    mock_collector_class.assert_called_once()
    mock_collector.save_to_file.assert_called_once()
    mock_collector.save_table_to_file.assert_called_once()
    
    # Check that run_matrix_multiplications was called
    mock_run_matrix_multiplications.assert_called_once()

###########################################
# Model Training and Evaluation Function Tests
###########################################

@patch("examples.matmul_toolkit.plot_training_history")
@patch("examples.matmul_toolkit.MatmulModelTrainer")
@patch("examples.matmul_toolkit.create_dataloaders")
@patch("examples.matmul_toolkit.DeepMatmulTimingModel")
@patch("examples.matmul_toolkit.MatmulTimingModel")
@patch("builtins.open", new_callable=mock_open, read_data='{"mock": "data"}')
def test_train_model(mock_file, mock_base_model, mock_deep_model, mock_create_dataloaders, mock_trainer_class, mock_plot_history, tmp_path):
    """Test the train_model function."""
    # Set up the mock objects
    mock_train_dataloader = MagicMock()
    mock_val_dataloader = MagicMock()
    mock_test_dataloader = MagicMock()
    mock_create_dataloaders.return_value = (mock_train_dataloader, mock_val_dataloader, mock_test_dataloader)
    
    mock_train_dataloader.dataset.dataset.problem_feature_dim = 15
    mock_train_dataloader.dataset.dataset.config_feature_dim = 10
    
    mock_model = MagicMock()
    mock_deep_model.return_value = mock_model
    
    mock_trainer = MagicMock()
    mock_trainer_class.return_value = mock_trainer
    mock_trainer.train.return_value = {"train_loss": [0.1], "val_loss": [0.1], "test_loss": [0.1], "learning_rate": [0.001]}
    mock_trainer._evaluate.return_value = 0.1
    
    # Patch MatmulDataset.deserialize
    with patch("diode.types.matmul_dataset.Dataset.deserialize") as mock_deserialize:
        mock_deserialize.return_value = MagicMock()
        
        # Call the function with minimal parameters
        dataset_path = str(tmp_path / "test_dataset.json")
        model_path = str(tmp_path / "test_model.pt")
        result = train_model(
            dataset_path=dataset_path,
            model_path=model_path,
            model_type="deep",
            batch_size=64,
            num_epochs=2,
            device="cpu"
        )
        
        # Check that the necessary functions were called
        mock_deserialize.assert_called_once()
        mock_create_dataloaders.assert_called_once()
        mock_deep_model.assert_called_once()
        mock_trainer_class.assert_called_once()
        mock_trainer.train.assert_called_once()
        mock_plot_history.assert_called_once()
        
        # Check that the function returned the expected result
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == mock_model
        assert isinstance(result[1], dict)

@patch("examples.matmul_toolkit.MatmulModelTrainer")
@patch("examples.matmul_toolkit.create_dataloaders")
@patch("examples.matmul_toolkit.DeepMatmulTimingModel")
@patch("builtins.open", new_callable=mock_open, read_data='{"mock": "data"}')
@patch("torch.load")
@patch("os.path.exists")
def test_validate_model(mock_exists, mock_torch_load, mock_file, mock_deep_model, mock_create_dataloaders, mock_trainer_class, tmp_path):
    """Test the validate_model function."""
    # Set up the mock objects
    mock_val_dataloader = MagicMock()
    mock_create_dataloaders.return_value = (None, mock_val_dataloader, None)
    
    mock_val_dataloader.dataset.dataset.problem_feature_dim = 15
    mock_val_dataloader.dataset.dataset.config_feature_dim = 10
    
    mock_model = MagicMock()
    mock_deep_model.return_value = mock_model
    
    mock_trainer = MagicMock()
    mock_trainer_class.return_value = mock_trainer
    mock_trainer._evaluate.return_value = 0.1
    
    # Set up the mock torch.load
    mock_torch_load.return_value = {"model_state_dict": {}}
    
    # Set up mock exists to return True
    mock_exists.return_value = True
    
    # Patch MatmulDataset.deserialize
    with patch("diode.types.matmul_dataset.Dataset.deserialize") as mock_deserialize:
        mock_deserialize.return_value = MagicMock()
        
        # Call the function with minimal parameters
        model_path = str(tmp_path / "test_model.pt")
        validation_dataset_path = str(tmp_path / "test_validation_dataset.json")
        
        validate_model(
            model_path=model_path,
            validation_dataset_path=validation_dataset_path,
            batch_size=64,
            device="cpu"
        )
        
        # Check that the necessary functions were called
        mock_deserialize.assert_called_once()
        mock_create_dataloaders.assert_called_once()
        mock_torch_load.assert_called_once()
        mock_trainer_class.assert_called_once()
        mock_trainer._evaluate.assert_called_once()

@patch("examples.matmul_toolkit.train_model_from_dataset")
@patch("examples.matmul_toolkit.create_dataloaders")
@patch("builtins.open", new_callable=mock_open, read_data='{"mock": "data"}')
@patch("os.path.exists")
@patch("os.makedirs")
def test_run_model_example(mock_makedirs, mock_exists, mock_file, mock_create_dataloaders, mock_train_model, tmp_path):
    """Test the run_model_example function."""
    # Set up the mock objects
    mock_train_dataloader = MagicMock()
    mock_val_dataloader = MagicMock()
    mock_test_dataloader = MagicMock()
    mock_create_dataloaders.return_value = (mock_train_dataloader, mock_val_dataloader, mock_test_dataloader)
    
    # Make sure test_dataloader has a non-zero length to avoid division by zero
    mock_test_dataloader.__len__.return_value = 1
    
    # Set up mock model and history
    mock_model = MagicMock()
    mock_history = {"train_loss": [0.1], "val_loss": [0.1], "test_loss": [0.1], "learning_rate": [0.001]}
    mock_train_model.return_value = (mock_model, mock_history)
    
    # Set up mock exists to return True
    mock_exists.return_value = True
    
    # Patch MatmulDataset.deserialize
    with patch("diode.types.matmul_dataset.Dataset.deserialize") as mock_deserialize:
        mock_deserialize.return_value = MagicMock()
        
        # Patch plot_training_history to avoid matplotlib issues
        with patch("examples.matmul_toolkit.plot_training_history"):
            # Patch torch.no_grad to avoid issues
            with patch("torch.no_grad"):
                # Call the function with minimal parameters
                dataset_path = str(tmp_path / "test_dataset.json")
                
                run_model_example(
                    dataset_path=dataset_path,
                    model_type="deep",
                    batch_size=64,
                    num_epochs=2,
                    log_dir=str(tmp_path / "logs"),
                    model_dir=str(tmp_path / "models"),
                    device="cpu"
                )
                
                # Check that the necessary functions were called
                mock_deserialize.assert_called_once()
                mock_train_model.assert_called_once()

###########################################
# Command-line Interface Tests
###########################################

@patch("examples.matmul_toolkit.collect_data")
@patch("examples.matmul_toolkit.create_validation_dataset")
@patch("examples.matmul_toolkit.train_model")
@patch("examples.matmul_toolkit.validate_model")
@patch("examples.matmul_toolkit.run_collector_example")
@patch("examples.matmul_toolkit.run_model_example")
@patch("sys.argv", ["matmul_toolkit.py", "collect", "--output", "test_dataset.json", "--num-shapes", "2"])
def test_main_collect(mock_run_model_example, mock_run_collector_example, mock_validate_model, 
                     mock_train_model, mock_create_validation_dataset, mock_collect_data):
    """Test the main function with the collect mode."""
    # Call the main function
    main()
    
    # Check that collect_data was called with the expected parameters
    mock_collect_data.assert_called_once()
    assert mock_collect_data.call_args[1]["output_file"] == "test_dataset.json"
    assert mock_collect_data.call_args[1]["num_shapes"] == 2

@patch("examples.matmul_toolkit.collect_data")
@patch("examples.matmul_toolkit.create_validation_dataset")
@patch("examples.matmul_toolkit.train_model")
@patch("examples.matmul_toolkit.validate_model")
@patch("examples.matmul_toolkit.run_collector_example")
@patch("examples.matmul_toolkit.run_model_example")
@patch("sys.argv", ["matmul_toolkit.py", "create-validation", "--output", "test_validation_dataset.json", "--num-shapes", "2"])
def test_main_create_validation(mock_run_model_example, mock_run_collector_example, mock_validate_model, 
                               mock_train_model, mock_create_validation_dataset, mock_collect_data):
    """Test the main function with the create-validation mode."""
    # Call the main function
    main()
    
    # Check that create_validation_dataset was called with the expected parameters
    mock_create_validation_dataset.assert_called_once()
    assert mock_create_validation_dataset.call_args[1]["output_file"] == "test_validation_dataset.json"
    assert mock_create_validation_dataset.call_args[1]["num_shapes"] == 2

@patch("examples.matmul_toolkit.collect_data")
@patch("examples.matmul_toolkit.create_validation_dataset")
@patch("examples.matmul_toolkit.train_model")
@patch("examples.matmul_toolkit.validate_model")
@patch("examples.matmul_toolkit.run_collector_example")
@patch("examples.matmul_toolkit.run_model_example")
@patch("sys.argv", ["matmul_toolkit.py", "train", "--dataset", "test_dataset.json", "--model", "test_model.pt"])
def test_main_train(mock_run_model_example, mock_run_collector_example, mock_validate_model, 
                   mock_train_model, mock_create_validation_dataset, mock_collect_data):
    """Test the main function with the train mode."""
    # Call the main function
    main()
    
    # Check that train_model was called with the expected parameters
    mock_train_model.assert_called_once()
    assert mock_train_model.call_args[1]["dataset_path"] == "test_dataset.json"
    assert mock_train_model.call_args[1]["model_path"] == "test_model.pt"

@patch("examples.matmul_toolkit.collect_data")
@patch("examples.matmul_toolkit.create_validation_dataset")
@patch("examples.matmul_toolkit.train_model")
@patch("examples.matmul_toolkit.validate_model")
@patch("examples.matmul_toolkit.run_collector_example")
@patch("examples.matmul_toolkit.run_model_example")
@patch("sys.argv", ["matmul_toolkit.py", "validate-model", "--model", "test_model.pt", "--dataset", "test_validation_dataset.json"])
def test_main_validate_model(mock_run_model_example, mock_run_collector_example, mock_validate_model, 
                            mock_train_model, mock_create_validation_dataset, mock_collect_data):
    """Test the main function with the validate-model mode."""
    # Call the main function
    main()
    
    # Check that validate_model was called with the expected parameters
    mock_validate_model.assert_called_once()
    assert mock_validate_model.call_args[1]["model_path"] == "test_model.pt"
    assert mock_validate_model.call_args[1]["validation_dataset_path"] == "test_validation_dataset.json"

@patch("examples.matmul_toolkit.collect_data")
@patch("examples.matmul_toolkit.create_validation_dataset")
@patch("examples.matmul_toolkit.train_model")
@patch("examples.matmul_toolkit.validate_model")
@patch("examples.matmul_toolkit.run_collector_example")
@patch("examples.matmul_toolkit.run_model_example")
@patch("sys.argv", ["matmul_toolkit.py", "collector-example", "--output-dir", "test_output", "--use-context-manager"])
def test_main_collector_example(mock_run_model_example, mock_run_collector_example, mock_validate_model, 
                               mock_train_model, mock_create_validation_dataset, mock_collect_data):
    """Test the main function with the collector-example mode."""
    # Call the main function
    main()
    
    # Check that run_collector_example was called with the expected parameters
    mock_run_collector_example.assert_called_once()
    assert mock_run_collector_example.call_args[1]["output_dir"] == "test_output"
    assert mock_run_collector_example.call_args[1]["use_context_manager"] is True

@patch("examples.matmul_toolkit.collect_data")
@patch("examples.matmul_toolkit.create_validation_dataset")
@patch("examples.matmul_toolkit.train_model")
@patch("examples.matmul_toolkit.validate_model")
@patch("examples.matmul_toolkit.run_collector_example")
@patch("examples.matmul_toolkit.run_model_example")
@patch("sys.argv", ["matmul_toolkit.py", "model-example", "--dataset", "test_dataset.json"])
def test_main_model_example(mock_run_model_example, mock_run_collector_example, mock_validate_model, 
                           mock_train_model, mock_create_validation_dataset, mock_collect_data):
    """Test the main function with the model-example mode."""
    # Call the main function
    main()
    
    # Check that run_model_example was called with the expected parameters
    mock_run_model_example.assert_called_once()
    assert mock_run_model_example.call_args[1]["dataset_path"] == "test_dataset.json"

@patch("examples.matmul_toolkit.collect_data")
@patch("examples.matmul_toolkit.create_validation_dataset")
@patch("examples.matmul_toolkit.train_model")
@patch("examples.matmul_toolkit.validate_model")
@patch("examples.matmul_toolkit.run_collector_example")
@patch("examples.matmul_toolkit.run_model_example")
@patch("sys.argv", ["matmul_toolkit.py", "collect-and-train", "--dataset", "test_dataset.json", 
                   "--validation-dataset", "test_validation_dataset.json", "--model", "test_model.pt"])
@patch("os.path.exists")
def test_main_collect_and_train(mock_exists, mock_run_model_example, mock_run_collector_example, mock_validate_model, 
                               mock_train_model, mock_create_validation_dataset, mock_collect_data):
    """Test the main function with the collect-and-train mode."""
    # Set up mock exists to return True for validation dataset
    mock_exists.return_value = True
    
    # Patch os.makedirs to avoid creating directories
    with patch("os.makedirs"):
        # Call the main function
        main()
        
        # Check that the necessary functions were called
        mock_collect_data.assert_called_once()
        mock_create_validation_dataset.assert_called_once()
        mock_train_model.assert_called_once()
        # Note: validate_model is only called if the validation dataset exists
        # and we've mocked os.path.exists to return True
        mock_validate_model.assert_called_once()

@patch("examples.matmul_toolkit.collect_data")
@patch("examples.matmul_toolkit.create_validation_dataset")
@patch("examples.matmul_toolkit.train_model")
@patch("examples.matmul_toolkit.validate_model")
@patch("examples.matmul_toolkit.run_collector_example")
@patch("examples.matmul_toolkit.run_model_example")
@patch("sys.argv", ["matmul_toolkit.py", "collect-and-train", "--dataset", "test_dataset.json", 
                   "--validation-dataset", "test_validation_dataset.json", "--model", "test_model.pt",
                   "--skip-collection", "--skip-validation", "--skip-training"])
def test_main_collect_and_train_with_skips(mock_run_model_example, mock_run_collector_example, mock_validate_model, 
                                          mock_train_model, mock_create_validation_dataset, mock_collect_data):
    """Test the main function with the collect-and-train mode and skip flags."""
    # Patch os.makedirs to avoid creating directories
    with patch("os.makedirs"):
        # Call the main function
        main()
        
        # Check that the skipped functions were not called
        mock_collect_data.assert_not_called()
        mock_create_validation_dataset.assert_not_called()
        mock_train_model.assert_not_called()
        mock_validate_model.assert_not_called()

if __name__ == "__main__":
    pytest.main(["-v", __file__])
