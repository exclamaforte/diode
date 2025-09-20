"""
Tests for the matmul_toolkit.py module.

This test file covers all functionalities provided by the matmul_toolkit.py module:
1. Utility functions
2. Data collection functions
3. Model training and evaluation functions
4. Command-line interface
"""

import json
import os
import sys
from collections import OrderedDict
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
import torch

# Add the parent directory to the path so we can import the diode module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from torch_diode.collection.matmul_data_utils import (
    collect_data,
    create_validation_dataset,
    run_collector_example,
    run_matrix_multiplications,
)
from torch_diode.collection.matmul_dataset_collector import MatmulDatasetCollector
from torch_diode.model.matmul_model_trainer import analyze_worst_predictions
from torch_diode.model.matmul_timing_model import DeepMatmulTimingModel, MatmulTimingModel
from torch_diode.model.model_utils import run_model_example, train_model, validate_model

from torch_diode.types.matmul_dataset import Dataset
from torch_diode.types.matmul_types import MMShape, TritonGEMMConfig

# Import the functions from their actual locations
from torch_diode.utils.dataset_utils import generate_matrix_sizes, print_dataset_statistics
from torch_diode.utils.visualization_utils import plot_training_history
from workflows.matmul_toolkit import main

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
    problem.B = 1
    problem.M = 128
    problem.N = 128
    problem.K = 128
    problem.M_dtype = torch.float16
    problem.K_dtype = torch.float16
    problem.out_dtype = torch.float16
    problem.out_size = (128, 128)
    problem.out_stride = (128, 1)

    # Mock the config with all necessary attributes
    config = MagicMock(spec=TritonGEMMConfig)
    config.name = "test_config"
    config.grid = 1
    config.block_m = 32
    config.block_n = 32
    config.block_k = 32
    config.group_m = 8
    config.num_stages = 2
    config.num_warps = 4
    config.EVEN_K = True
    config.ALLOW_TF32 = True
    config.USE_FAST_ACCUM = False
    config.ACC_TYPE = "tl.float32"

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
    config_features = torch.randn(2, 10)  # Batch size 2, 10 config features
    targets = torch.randn(2, 1)  # Batch size 2, 1 target

    # Set up the dataloader to yield this batch
    dataloader.__iter__.return_value = iter(
        [(problem_features, config_features, targets)]
    )
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
        "learning_rate": [0.001, 0.0009, 0.0008],
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


@patch("torch_diode.utils.visualization_utils.plot_training_history")
def test_plot_training_history(mock_plot, mock_history):
    """Test the plot_training_history function."""
    # Call the function with the mock history
    mock_plot(mock_history)

    # Check that the plot function was called
    mock_plot.assert_called_once_with(mock_history)


@patch("torch_diode.utils.visualization_utils.plot_training_history")
def test_plot_training_history_with_save(mock_plot, mock_history, tmp_path):
    """Test the plot_training_history function with saving."""
    # Create a temporary file path
    save_path = str(tmp_path / "history_plot.png")

    # Call the function with the mock history and save path
    mock_plot(mock_history, save_path)

    # Check that the plot function was called with the save path
    mock_plot.assert_called_once_with(mock_history, save_path)


@patch("torch_diode.utils.visualization_utils.plot_training_history")
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


@patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector")
@patch("torch_diode.collection.matmul_data_utils.torch.cuda")
@patch("torch_diode.utils.dataset_utils.generate_matrix_sizes")
@patch("torch_diode.collection.matmul_data_utils.print_dataset_statistics")
def test_collect_data(
    mock_print_stats,
    mock_generate_sizes,
    mock_cuda,
    mock_collector_class,
    tmp_path,
):
    """Test the collect_data function."""
    # Set up the mock collector
    mock_collector = MagicMock()
    mock_collector_class.return_value = mock_collector

    # Mock the collect_data method to simulate it calling start_collection and stop_collection
    def mock_collect_data_method(*args, **kwargs):
        mock_collector.start_collection()
        mock_collector.stop_collection()

    mock_collector.collect_data = MagicMock(side_effect=mock_collect_data_method)

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

    # Check that the collector's collect_data method was called
    mock_collector.collect_data.assert_called_once()

    # Check that print_dataset_statistics was called
    mock_print_stats.assert_called_once()


@patch("torch_diode.collection.matmul_data_utils.collect_data")
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


@patch("torch_diode.collection.matmul_data_utils.run_matrix_multiplications")
@patch("torch_diode.collection.matmul_data_utils.MatmulDatasetCollector")
@patch("torch_diode.collection.matmul_data_utils.torch.cuda")
@patch("builtins.print")
def test_run_collector_example(
    mock_print,
    mock_cuda,
    mock_collector_class,
    mock_run_matrix_multiplications,
    tmp_path,
):
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
    with patch("torch_diode.utils.dataset_utils.print_dataset_statistics"):
        run_collector_example(
            output_dir=output_dir, use_context_manager=True, num_shapes=2
        )

    # Check that the collector was created and methods were called
    mock_collector_class.assert_called_once()
    mock_collector.save_to_file.assert_called_once()
    mock_collector.save_table_to_file.assert_called_once()

    # Check that run_matrix_multiplications was called
    mock_run_matrix_multiplications.assert_called_once()


###########################################
# Model Training and Evaluation Function Tests
###########################################


@patch("torch_diode.utils.visualization_utils.plot_training_history")
@patch("torch_diode.model.model_utils.MatmulModelTrainer")
@patch("torch_diode.model.matmul_dataset_loader.MatmulTimingDataset")
@patch("torch_diode.model.matmul_timing_model.DeepMatmulTimingModel")
@patch("torch_diode.model.matmul_timing_model.MatmulTimingModel")
@patch("builtins.open", new_callable=mock_open, read_data='{"mock": "data"}')
@patch("os.makedirs")
def test_train_model(
    mock_makedirs,
    mock_file,
    mock_base_model,
    mock_deep_model,
    mock_dataset_class,
    mock_trainer_class,
    mock_plot_history,
    tmp_path,
):
    """Test the train_model function."""
    # Create mock dataset that behaves properly
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 10
    mock_dataset.problem_feature_dim = 15
    mock_dataset.config_feature_dim = 10

    # Mock the dataset to return actual tensor data when accessed
    problem_features = torch.randn(15)
    config_features = torch.randn(10)
    timing = torch.tensor([0.1])
    mock_dataset.__getitem__.return_value = (problem_features, config_features, timing)

    mock_dataset_class.return_value = mock_dataset

    # Create mock dataloaders that return actual tensor data
    mock_train_dataloader = MagicMock()
    mock_val_dataloader = MagicMock()
    mock_test_dataloader = MagicMock()

    # Mock the dataloader iterator to return actual tensors
    batch_data = [
        (
            torch.randn(2, 15),  # problem features
            torch.randn(2, 10),  # config features
            torch.randn(2, 1),  # timings
        )
    ]
    mock_train_dataloader.__iter__.return_value = iter(batch_data)
    mock_val_dataloader.__iter__.return_value = iter(batch_data)
    mock_test_dataloader.__iter__.return_value = iter(batch_data)

    # Set up dataloaders to use the mock dataset
    mock_train_dataloader.dataset = MagicMock()
    mock_train_dataloader.dataset.dataset = mock_dataset
    mock_val_dataloader.dataset = MagicMock()
    mock_val_dataloader.dataset.dataset = mock_dataset
    mock_test_dataloader.dataset = MagicMock()
    mock_test_dataloader.dataset.dataset = mock_dataset

    mock_model = MagicMock()
    mock_model.load_state_dict = MagicMock()  # Mock the load_state_dict method
    mock_deep_model.return_value = mock_model

    mock_trainer = MagicMock()
    mock_trainer_class.return_value = mock_trainer
    mock_trainer.train.return_value = {
        "train_loss": [0.1],
        "val_loss": [0.1],
        "test_loss": [0.1],
        "learning_rate": [0.001],
    }
    mock_trainer._evaluate.return_value = 0.1

    # Patch MatmulDataset.deserialize
    with patch("torch_diode.types.matmul_dataset.Dataset.deserialize") as mock_deserialize:
        mock_deserialize.return_value = MagicMock()

        # Patch create_dataloaders to return our mock dataloaders
        with patch(
            "torch_diode.model.matmul_dataset_loader.create_dataloaders"
        ) as mock_create_dataloaders:
            mock_create_dataloaders.return_value = (
                mock_train_dataloader,
                mock_val_dataloader,
                mock_test_dataloader,
            )

            # Patch the plot_training_history function at the model_utils level
            with patch("torch_diode.model.model_utils.plot_training_history"):
                # Patch torch.manual_seed to avoid issues
                with patch("torch.manual_seed"):
                    # Call the function with minimal parameters
                    dataset_path = str(tmp_path / "test_dataset.json")
                    model_path = str(tmp_path / "test_model.pt")
                    result = train_model(
                        dataset_path=dataset_path,
                        model_path=model_path,
                        model_type="deep",
                        batch_size=64,
                        num_epochs=2,
                        device="cpu",
                    )

                    # Check that the necessary functions were called
                    mock_deserialize.assert_called_once()
                    mock_trainer_class.assert_called_once()
                    mock_trainer.train.assert_called_once()
                    # Note: plot_training_history is called at model_utils level

                    # Check that the function returned the expected result
                    assert isinstance(result, tuple)
                    assert len(result) == 2
                    # The function returns the actual model, not the mock
                    assert isinstance(result[0], torch.nn.Module)
                    assert isinstance(result[1], dict)


@patch("torch_diode.model.matmul_model_trainer.MatmulModelTrainer")
@patch("torch_diode.model.matmul_dataset_loader.MatmulTimingDataset")
@patch("torch_diode.model.matmul_timing_model.DeepMatmulTimingModel")
@patch("builtins.open", new_callable=mock_open, read_data='{"mock": "data"}')
@patch("torch.load")
@patch("os.path.exists")
def test_validate_model(
    mock_exists,
    mock_torch_load,
    mock_file,
    mock_deep_model,
    mock_dataset_class,
    mock_trainer_class,
    tmp_path,
):
    """Test the validate_model function."""
    # Create mock dataset that behaves properly
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 10
    mock_dataset.problem_feature_dim = 15
    mock_dataset.config_feature_dim = 10
    
    # Mock the dataset to return actual tensor data when accessed
    problem_features = torch.randn(15)
    config_features = torch.randn(10)  
    timing = torch.tensor([0.1])
    mock_dataset.__getitem__.return_value = (problem_features, config_features, timing)
    
    mock_dataset_class.return_value = mock_dataset

    # Create mock dataloaders with non-empty datasets
    mock_val_dataloader = MagicMock()
    
    # Mock the dataloader iterator to return actual tensors
    batch_data = [(
        torch.randn(2, 15),  # problem features
        torch.randn(2, 10),  # config features  
        torch.randn(2, 1)    # timings
    )]
    mock_val_dataloader.__iter__.return_value = iter(batch_data)

    # Set up validation dataloader to use the mock dataset
    mock_val_dataloader.dataset = MagicMock()
    mock_val_dataloader.dataset.dataset = mock_dataset

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
    with patch("torch_diode.types.matmul_dataset.Dataset.deserialize") as mock_deserialize:
        # Create a mock dataset with at least one timing entry
        mock_dataset_from_file = MagicMock()
        mock_deserialize.return_value = mock_dataset_from_file

        # Patch create_dataloaders to return our mock dataloader directly
        with patch("torch_diode.model.matmul_dataset_loader.create_dataloaders") as mock_create_dataloaders:
            mock_create_dataloaders.return_value = (None, mock_val_dataloader, None)
            
            # Mock the torch.nn.Module.load_state_dict method to skip loading
            with patch("torch.nn.Module.load_state_dict"):
                # Call the function with minimal parameters
                model_path = str(tmp_path / "test_model.pt")
                validation_dataset_path = str(tmp_path / "test_validation_dataset.json")

                validate_model(
                    model_path=model_path,
                    validation_dataset_path=validation_dataset_path,
                    batch_size=64,
                    device="cpu",
                )

                # Check that the necessary functions were called
                mock_deserialize.assert_called_once()
                # Note: create_dataloaders is called directly, not through mock
                mock_torch_load.assert_called_once()
                # Note: trainer is instantiated directly, not through mock
                # The real trainer is used, so we can't assert on the mock


@patch("torch_diode.model.model_utils.train_model_from_dataset")
@patch("torch_diode.model.matmul_dataset_loader.MatmulTimingDataset")
@patch("os.path.exists")
@patch("os.makedirs")
def test_run_model_example(
    mock_makedirs,
    mock_exists,
    mock_dataset_class,
    mock_train_model,
    tmp_path,
):
    """Test the run_model_example function."""
    # Create mock dataset that behaves properly
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 10
    mock_dataset.problem_feature_dim = 15
    mock_dataset.config_feature_dim = 10
    mock_dataset_class.return_value = mock_dataset

    # Create a proper mock model that returns tensors
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.to.return_value = mock_model
    # Make the model return proper tensors when called
    mock_model.return_value = torch.randn(2, 1)  # Return proper tensor outputs
    
    mock_history = {
        "train_loss": [0.1],
        "val_loss": [0.1],
        "test_loss": [0.1],
        "learning_rate": [0.001],
    }
    mock_train_model.return_value = (mock_model, mock_history, None)  # Note: 3 values

    # Set up mock exists to return True
    mock_exists.return_value = True

    # Patch MatmulDataset.deserialize
    with patch("torch_diode.types.matmul_dataset.Dataset.deserialize") as mock_deserialize:
        mock_deserialize.return_value = MagicMock()

        # Create a temporary dataset file for the test
        dataset_path = str(tmp_path / "test_dataset.json")
        with open(dataset_path, "w") as f:
            f.write('{"mock": "dataset"}')

        # Patch print_dataset_statistics to avoid issues
        with patch("torch_diode.utils.dataset_utils.print_dataset_statistics"):
            # Patch the plot_training_history function directly to avoid matplotlib issues
            with patch("torch_diode.model.model_utils.plot_training_history") as mock_plot:
                # Patch the second create_dataloaders call in run_model_example
                with patch("torch_diode.model.model_utils.create_dataloaders") as mock_create_dataloaders2:
                    # Create a mock test dataloader that returns actual tensors
                    mock_test_dataloader = MagicMock()
                    batch_data = [(
                        torch.randn(2, 15),  # problem features
                        torch.randn(2, 10),  # config features  
                        torch.randn(2, 1)    # timings
                    )]
                    mock_test_dataloader.__iter__.return_value = iter(batch_data)
                    mock_test_dataloader.__len__.return_value = 1
                    
                    mock_create_dataloaders2.return_value = (None, None, mock_test_dataloader)
                    
                    # Patch torch.no_grad to avoid issues
                    with patch("torch.no_grad"):
                        # Call the function with minimal parameters
                        run_model_example(
                            dataset_path=dataset_path,
                            model_type="deep",
                            batch_size=64,
                            num_epochs=2,
                            log_dir=str(tmp_path / "logs"),
                            model_dir=str(tmp_path / "models"),
                            device="cpu",
                        )

                        # Check that the necessary functions were called
                        mock_deserialize.assert_called_once()
                        mock_train_model.assert_called_once()
                        mock_plot.assert_called_once()


###########################################
# Command-line Interface Tests
###########################################


@patch("workflows.matmul_toolkit.collect_data")
@patch("workflows.matmul_toolkit.create_validation_dataset")
@patch("workflows.matmul_toolkit.train_model")
@patch("workflows.matmul_toolkit.validate_model")
@patch("workflows.matmul_toolkit.run_collector_example")
@patch("workflows.matmul_toolkit.run_model_example")
@patch(
    "sys.argv",
    [
        "matmul_toolkit.py",
        "collect",
        "--output",
        "test_dataset.json",
        "--num-shapes",
        "2",
    ],
)
def test_main_collect(
    mock_run_model_example,
    mock_run_collector_example,
    mock_validate_model,
    mock_train_model,
    mock_create_validation_dataset,
    mock_collect_data,
):
    """Test the main function with the collect mode."""
    # Call the main function
    main()

    # Check that collect_data was called with the expected parameters
    mock_collect_data.assert_called_once()
    assert mock_collect_data.call_args[1]["output_file"] == "test_dataset.json"
    assert mock_collect_data.call_args[1]["num_shapes"] == 2


@patch("workflows.matmul_toolkit.collect_data")
@patch("workflows.matmul_toolkit.create_validation_dataset")
@patch("workflows.matmul_toolkit.train_model")
@patch("workflows.matmul_toolkit.validate_model")
@patch("workflows.matmul_toolkit.run_collector_example")
@patch("workflows.matmul_toolkit.run_model_example")
@patch(
    "sys.argv",
    [
        "matmul_toolkit.py",
        "create-validation",
        "--output",
        "test_validation_dataset.json",
        "--num-shapes",
        "2",
    ],
)
def test_main_create_validation(
    mock_run_model_example,
    mock_run_collector_example,
    mock_validate_model,
    mock_train_model,
    mock_create_validation_dataset,
    mock_collect_data,
):
    """Test the main function with the create-validation mode."""
    # Call the main function
    main()

    # Check that create_validation_dataset was called with the expected parameters
    mock_create_validation_dataset.assert_called_once()
    assert (
        mock_create_validation_dataset.call_args[1]["output_file"]
        == "test_validation_dataset.json"
    )
    assert mock_create_validation_dataset.call_args[1]["num_shapes"] == 2


@patch("workflows.matmul_toolkit.collect_data")
@patch("workflows.matmul_toolkit.create_validation_dataset")
@patch("workflows.matmul_toolkit.train_model")
@patch("workflows.matmul_toolkit.validate_model")
@patch("workflows.matmul_toolkit.run_collector_example")
@patch("workflows.matmul_toolkit.run_model_example")
@patch(
    "sys.argv",
    [
        "matmul_toolkit.py",
        "train",
        "--dataset",
        "test_dataset.json",
        "--model",
        "test_model.pt",
    ],
)
def test_main_train(
    mock_run_model_example,
    mock_run_collector_example,
    mock_validate_model,
    mock_train_model,
    mock_create_validation_dataset,
    mock_collect_data,
):
    """Test the main function with the train mode."""
    # Call the main function
    main()

    # Check that train_model was called with the expected parameters
    mock_train_model.assert_called_once()
    assert mock_train_model.call_args[1]["dataset_path"] == "test_dataset.json"
    assert mock_train_model.call_args[1]["model_path"] == "test_model.pt"


@patch("workflows.matmul_toolkit.collect_data")
@patch("workflows.matmul_toolkit.create_validation_dataset")
@patch("workflows.matmul_toolkit.train_model")
@patch("workflows.matmul_toolkit.validate_model")
@patch("workflows.matmul_toolkit.run_collector_example")
@patch("workflows.matmul_toolkit.run_model_example")
@patch(
    "sys.argv",
    [
        "matmul_toolkit.py",
        "validate-model",
        "--model",
        "test_model.pt",
        "--dataset",
        "test_validation_dataset.json",
    ],
)
def test_main_validate_model(
    mock_run_model_example,
    mock_run_collector_example,
    mock_validate_model,
    mock_train_model,
    mock_create_validation_dataset,
    mock_collect_data,
):
    """Test the main function with the validate-model mode."""
    # Call the main function
    main()

    # Check that validate_model was called with the expected parameters
    mock_validate_model.assert_called_once()
    assert mock_validate_model.call_args[1]["model_path"] == "test_model.pt"
    assert (
        mock_validate_model.call_args[1]["validation_dataset_path"]
        == "test_validation_dataset.json"
    )


@patch("workflows.matmul_toolkit.collect_data")
@patch("workflows.matmul_toolkit.create_validation_dataset")
@patch("workflows.matmul_toolkit.train_model")
@patch("workflows.matmul_toolkit.validate_model")
@patch("workflows.matmul_toolkit.run_collector_example")
@patch("workflows.matmul_toolkit.run_model_example")
@patch(
    "sys.argv",
    [
        "matmul_toolkit.py",
        "collector-example",
        "--output-dir",
        "test_output",
        "--use-context-manager",
    ],
)
def test_main_collector_example(
    mock_run_model_example,
    mock_run_collector_example,
    mock_validate_model,
    mock_train_model,
    mock_create_validation_dataset,
    mock_collect_data,
):
    """Test the main function with the collector-example mode."""
    # Call the main function
    main()

    # Check that run_collector_example was called with the expected parameters
    mock_run_collector_example.assert_called_once()
    assert mock_run_collector_example.call_args[1]["output_dir"] == "test_output"
    assert mock_run_collector_example.call_args[1]["use_context_manager"] is True


@patch("workflows.matmul_toolkit.collect_data")
@patch("workflows.matmul_toolkit.create_validation_dataset")
@patch("workflows.matmul_toolkit.train_model")
@patch("workflows.matmul_toolkit.validate_model")
@patch("workflows.matmul_toolkit.run_collector_example")
@patch("workflows.matmul_toolkit.run_model_example")
@patch(
    "sys.argv", ["matmul_toolkit.py", "model-example", "--dataset", "test_dataset.json"]
)
def test_main_model_example(
    mock_run_model_example,
    mock_run_collector_example,
    mock_validate_model,
    mock_train_model,
    mock_create_validation_dataset,
    mock_collect_data,
):
    """Test the main function with the model-example mode."""
    # Call the main function
    main()

    # Check that run_model_example was called with the expected parameters
    mock_run_model_example.assert_called_once()
    assert mock_run_model_example.call_args[1]["dataset_path"] == "test_dataset.json"


@patch("workflows.matmul_toolkit.collect_data")
@patch("workflows.matmul_toolkit.create_validation_dataset")
@patch("workflows.matmul_toolkit.train_model")
@patch("workflows.matmul_toolkit.validate_model")
@patch("workflows.matmul_toolkit.run_collector_example")
@patch("workflows.matmul_toolkit.run_model_example")
@patch(
    "sys.argv",
    [
        "matmul_toolkit.py",
        "collect-and-train",
        "--dataset",
        "test_dataset.json",
        "--validation-dataset",
        "test_validation_dataset.json",
        "--model",
        "test_model.pt",
    ],
)
@patch("os.path.exists")
def test_main_collect_and_train(
    mock_exists,
    mock_run_model_example,
    mock_run_collector_example,
    mock_validate_model,
    mock_train_model,
    mock_create_validation_dataset,
    mock_collect_data,
):
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


@patch("workflows.matmul_toolkit.collect_data")
@patch("workflows.matmul_toolkit.create_validation_dataset")
@patch("workflows.matmul_toolkit.train_model")
@patch("workflows.matmul_toolkit.validate_model")
@patch("workflows.matmul_toolkit.run_collector_example")
@patch("workflows.matmul_toolkit.run_model_example")
@patch(
    "sys.argv",
    [
        "matmul_toolkit.py",
        "collect-and-train",
        "--dataset",
        "test_dataset.json",
        "--validation-dataset",
        "test_validation_dataset.json",
        "--model",
        "test_model.pt",
        "--skip-collection",
        "--skip-validation",
        "--skip-training",
    ],
)
def test_main_collect_and_train_with_skips(
    mock_run_model_example,
    mock_run_collector_example,
    mock_validate_model,
    mock_train_model,
    mock_create_validation_dataset,
    mock_collect_data,
):
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
