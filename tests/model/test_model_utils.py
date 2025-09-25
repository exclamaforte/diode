"""
Tests for diode.model.model_utils module.
"""

import json

# Enable debug flags for testing
try:
    from torch_diode.utils.debug_config import set_debug_flag

    set_debug_flag("ENABLE_TYPE_ASSERTS", True)
except ImportError:
    pass  # In case debug_config is not available yet
import os
import tempfile
from unittest.mock import Mock, mock_open, patch

import pytest
import torch

from torch_diode.model.model_utils import (
    run_model_example,
    train_model,
    train_model_from_directory,
    validate_max_autotune,
    validate_model,
)


class TestModelUtils:
    """Test model utility functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_dataloader(self, problem_dim=15, config_dim=25):
        """Create a mock dataloader with proper structure."""
        # Create a proper mock dataset and subset that works with PyTorch's dataloader
        from torch.utils.data import Dataset, Subset

        class MockTorchDataset(Dataset):
            def __init__(
                self, size=10, problem_dim=15, config_dim=25
            ):  # Increased size to ensure larger batches
                self.size = size
                self.problem_feature_dim = problem_dim
                self.config_feature_dim = config_dim
                self.data = [
                    (torch.randn(problem_dim), torch.randn(config_dim), torch.randn(1))
                    for _ in range(size)
                ]
                self.timing_dataset = Mock()
                self.timing_dataset.configs = [Mock() for _ in range(size)]
                self.configs = [Mock() for _ in range(size)]

            def __getitem__(self, idx):
                return self.data[idx % self.size]

            def __len__(self):
                return self.size

        # Create the actual dataset and subset
        mock_dataset = MockTorchDataset(
            size=10, problem_dim=problem_dim, config_dim=config_dim
        )
        mock_subset = Subset(mock_dataset, list(range(10)))  # Use all 10 items

        # Create the actual dataloader with larger batch size
        from torch.utils.data import DataLoader

        mock_dataloader = DataLoader(
            mock_subset,
            batch_size=4,
            num_workers=0,
            shuffle=False,  # Larger batch size
        )

        return mock_dataloader

    def _create_mock_model(self):
        """Create a mock model."""
        mock_model = Mock()

        # Create a callable that returns appropriate tensor shape based on input batch size
        def mock_forward(problem_features, config_features):
            batch_size = problem_features.shape[0]
            # Ensure the output is on the same device as the input
            device = (
                problem_features.device
                if hasattr(problem_features, "device")
                else "cpu"
            )
            output = torch.randn(batch_size, 1, requires_grad=True, device=device)
            return output

        mock_model.side_effect = mock_forward
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock()
        mock_model.load_state_dict = Mock()
        mock_model.save = Mock()
        # Mock parameters method to return an iterable of parameters on CPU
        mock_param = torch.nn.Parameter(torch.randn(10, 10, device="cpu"))
        mock_model.parameters = Mock(return_value=[mock_param])
        return mock_model

    def test_validate_max_autotune_missing_model(self):
        """Test validate_max_autotune with missing model file."""
        model_path = os.path.join(self.temp_dir, "nonexistent_model.pt")
        validation_path = os.path.join(self.temp_dir, "validation.json")
        solution_path = os.path.join(self.temp_dir, "solution.json")

        # Should exit early if model doesn't exist
        validate_max_autotune(model_path, validation_path, solution_path)
        # Test passes if no exception is raised

    def test_validate_max_autotune_missing_validation_dataset(self):
        """Test validate_max_autotune with missing validation dataset."""
        model_path = os.path.join(self.temp_dir, "model.pt")
        validation_path = os.path.join(self.temp_dir, "nonexistent_validation.json")
        solution_path = os.path.join(self.temp_dir, "solution.json")

        # Create dummy model file
        with open(model_path, "w") as f:
            f.write("dummy model")

        # Should exit early if validation dataset doesn't exist
        validate_max_autotune(model_path, validation_path, solution_path)
        # Test passes if no exception is raised

    def test_validate_max_autotune_missing_solution_file(self):
        """Test validate_max_autotune with missing solution file."""
        model_path = os.path.join(self.temp_dir, "model.pt")
        validation_path = os.path.join(self.temp_dir, "validation.json")
        solution_path = os.path.join(self.temp_dir, "nonexistent_solution.json")

        # Create dummy files
        with open(model_path, "w") as f:
            f.write("dummy model")
        with open(validation_path, "w") as f:
            f.write("{}")

        # Should exit early if solution file doesn't exist
        validate_max_autotune(model_path, validation_path, solution_path)
        # Test passes if no exception is raised

    @patch("builtins.open", mock_open(read_data='{"config": []}'))
    @patch("torch_diode.types.matmul_types.Solution.parse")
    def test_validate_max_autotune_invalid_solution(self, mock_parse):
        """Test validate_max_autotune with invalid solution format."""
        mock_parse.return_value = None

        model_path = os.path.join(self.temp_dir, "model.pt")
        validation_path = os.path.join(self.temp_dir, "validation.json")
        solution_path = os.path.join(self.temp_dir, "solution.json")

        # Create dummy files
        for path in [model_path, validation_path, solution_path]:
            with open(path, "w") as f:
                f.write("{}")

        # Should exit early if solution can't be parsed
        validate_max_autotune(model_path, validation_path, solution_path)
        # Test passes if no exception is raised

    @patch("builtins.open", mock_open())
    def test_validate_max_autotune_solution_load_exception(self):
        """Test validate_max_autotune with exception loading solution."""
        model_path = os.path.join(self.temp_dir, "model.pt")
        validation_path = os.path.join(self.temp_dir, "validation.json")
        solution_path = os.path.join(self.temp_dir, "solution.json")

        # Create dummy files
        for path in [model_path, validation_path, solution_path]:
            with open(path, "w") as f:
                f.write("{}")

        with patch("builtins.open", side_effect=Exception("File read error")):
            # Should handle exception gracefully
            validate_max_autotune(model_path, validation_path, solution_path)
            # Test passes if no exception is raised

    @patch("torch_diode.model.directory_dataset_loader.create_directory_dataloaders")
    @patch("os.path.isdir")
    def test_validate_max_autotune_directory_dataset_exception(
        self, mock_isdir, mock_create_loaders
    ):
        """Test validate_max_autotune with directory dataset loader exception."""
        mock_isdir.return_value = True
        mock_create_loaders.side_effect = Exception("Dataloader creation failed")

        model_path = os.path.join(self.temp_dir, "model.pt")
        validation_path = self.temp_dir
        solution_path = os.path.join(self.temp_dir, "solution.json")

        # Create dummy files
        with open(model_path, "w") as f:
            f.write("{}")
        with open(solution_path, "w") as f:
            f.write('{"config": []}')

        with patch("torch_diode.types.matmul_types.Solution.parse") as mock_parse:
            mock_solution = Mock()
            mock_solution.config = []
            mock_parse.return_value = mock_solution

            # Should handle exception gracefully
            validate_max_autotune(model_path, validation_path, solution_path)
            # Test passes if no exception is raised

    @patch("torch_diode.model.matmul_dataset_loader.create_dataloaders")
    @patch("torch_diode.types.matmul_dataset.Dataset.from_msgpack")
    @patch("os.path.isdir")
    def test_validate_max_autotune_msgpack_dataset_none(
        self, mock_isdir, mock_from_msgpack, mock_create_loaders
    ):
        """Test validate_max_autotune with msgpack dataset that returns None."""
        mock_isdir.return_value = False
        mock_from_msgpack.return_value = None

        model_path = os.path.join(self.temp_dir, "model.pt")
        validation_path = os.path.join(self.temp_dir, "validation.msgpack")
        solution_path = os.path.join(self.temp_dir, "solution.json")

        # Create dummy files
        with open(model_path, "w") as f:
            f.write("{}")
        with open(validation_path, "wb") as f:
            f.write(b"dummy msgpack")
        with open(solution_path, "w") as f:
            f.write('{"config": []}')

        with patch("torch_diode.types.matmul_types.Solution.parse") as mock_parse:
            mock_solution = Mock()
            mock_solution.config = []
            mock_parse.return_value = mock_solution

            # Should handle None dataset gracefully
            validate_max_autotune(model_path, validation_path, solution_path)
            # Test passes if no exception is raised

    @patch("torch_diode.model.matmul_dataset_loader.create_dataloaders")
    @patch("torch_diode.types.matmul_dataset.Dataset.deserialize")
    @patch("os.path.isdir")
    def test_validate_max_autotune_json_dataset_none(
        self, mock_isdir, mock_deserialize, mock_create_loaders
    ):
        """Test validate_max_autotune with JSON dataset that returns None."""
        mock_isdir.return_value = False
        mock_deserialize.return_value = None

        model_path = os.path.join(self.temp_dir, "model.pt")
        validation_path = os.path.join(self.temp_dir, "validation.json")
        solution_path = os.path.join(self.temp_dir, "solution.json")

        # Create dummy files
        for path in [model_path, validation_path, solution_path]:
            with open(path, "w") as f:
                f.write('{"config": []}')

        with patch("torch_diode.types.matmul_types.Solution.parse") as mock_parse:
            mock_solution = Mock()
            mock_solution.config = []
            mock_parse.return_value = mock_solution

            # Should handle None dataset gracefully
            validate_max_autotune(model_path, validation_path, solution_path)
            # Test passes if no exception is raised

    @patch("torch.load")
    def test_validate_max_autotune_model_load_exception(self, mock_torch_load):
        """Test validate_max_autotune with model loading exception."""
        mock_torch_load.side_effect = Exception("Model load failed")

        model_path = os.path.join(self.temp_dir, "model.pt")
        validation_path = os.path.join(self.temp_dir, "validation.json")
        solution_path = os.path.join(self.temp_dir, "solution.json")

        # Create dummy files
        for path in [model_path, validation_path, solution_path]:
            with open(path, "w") as f:
                f.write("{}")

        with patch("torch_diode.types.matmul_types.Solution.parse") as mock_parse:
            with patch(
                "torch_diode.model.matmul_dataset_loader.create_dataloaders"
            ) as mock_create_loaders:
                mock_solution = Mock()
                mock_solution.config = []
                mock_parse.return_value = mock_solution

                mock_dataloader = self._create_mock_dataloader()
                mock_create_loaders.return_value = (
                    mock_dataloader,
                    mock_dataloader,
                    mock_dataloader,
                )

                # Should handle model loading exception gracefully
                validate_max_autotune(model_path, validation_path, solution_path)
                # Test passes if no exception is raised

    def test_train_model_missing_dataset(self):
        """Test train_model with missing dataset file."""
        # Use a path that's guaranteed not to exist
        dataset_path = "/tmp/nonexistent_directory_12345/nonexistent_dataset.json"
        model_path = os.path.join(self.temp_dir, "model.pt")

        result = train_model(dataset_path, model_path)

        # Should return None and empty dict for missing dataset
        assert result == (None, {})

    @patch("torch_diode.types.matmul_dataset.Dataset.from_msgpack")
    def test_train_model_msgpack_dataset_none(self, mock_from_msgpack):
        """Test train_model with msgpack dataset that returns None."""
        mock_from_msgpack.return_value = None

        dataset_path = os.path.join(self.temp_dir, "dataset.msgpack")
        model_path = os.path.join(self.temp_dir, "model.pt")

        # Create dummy msgpack file
        with open(dataset_path, "wb") as f:
            f.write(b"dummy msgpack")

        result = train_model(dataset_path, model_path)

        # Should return None and empty dict for invalid dataset
        assert result == (None, {})

    @patch("torch_diode.types.matmul_dataset.Dataset.deserialize")
    def test_train_model_json_dataset_none(self, mock_deserialize):
        """Test train_model with JSON dataset that returns None."""
        mock_deserialize.return_value = None

        dataset_path = os.path.join(self.temp_dir, "dataset.json")
        model_path = os.path.join(self.temp_dir, "model.pt")

        # Create dummy JSON file
        with open(dataset_path, "w") as f:
            f.write("{}")

        result = train_model(dataset_path, model_path)

        # Should return None and empty dict for invalid dataset
        assert result == (None, {})

    @patch("torch_diode.model.model_utils.create_dataloaders")
    @patch("torch_diode.types.matmul_dataset.Dataset.deserialize")
    @patch("torch.manual_seed")
    @patch("os.makedirs")
    def test_train_model_base_model_type(
        self, mock_makedirs, mock_seed, mock_deserialize, mock_create_loaders
    ):
        """Test train_model with base model type."""
        # Mock dataset with proper structure
        mock_dataset = Mock()
        mock_dataset.hardware = {}  # Empty hardware dict to make iteration work
        mock_deserialize.return_value = mock_dataset

        # Create mock dataloaders that don't actually iterate - ensure they're not None
        mock_dataloader = self._create_mock_dataloader()
        mock_create_loaders.return_value = (
            mock_dataloader,
            mock_dataloader,
            mock_dataloader,
        )

        # Mock trainer completely to avoid actual training
        with patch(
            "torch_diode.model.model_utils.MatmulTimingModel"
        ) as mock_model_class:
            with patch("torch_diode.utils.visualization_utils.plot_training_history"):
                # Return the actual model and history without going through trainer
                mock_model = self._create_mock_model()
                mock_model.parameters.return_value = [
                    torch.randn(10, 10)
                ]  # Mock parameters for counting
                mock_model_class.return_value = mock_model

                dataset_path = os.path.join(self.temp_dir, "dataset.json")
                model_path = os.path.join(self.temp_dir, "model.pt")

                # Create dummy JSON file
                with open(dataset_path, "w") as f:
                    f.write("{}")

                # Mock the entire train_model function to avoid actual training
                with patch(
                    "torch_diode.model.model_utils.MatmulModelTrainer"
                ) as mock_trainer_class:
                    mock_trainer = Mock()
                    mock_trainer.train.return_value = {
                        "train_loss": [0.1, 0.05],
                        "val_loss": [0.15, 0.08],
                        "test_loss": [0.12, 0.06],
                        "learning_rate": [0.001, 0.0005],
                    }
                    mock_trainer._evaluate.return_value = 0.01
                    mock_trainer_class.return_value = mock_trainer

                    result = train_model(
                        dataset_path=dataset_path,
                        model_path=model_path,
                        model_type="base",
                        num_epochs=5,
                        learning_rate=0.01,
                        weight_decay=1e-4,
                        patience=10,
                        hardware_name="test_gpu",
                        op_name="mm",
                        seed=123,
                        device="cpu",
                        log_dir=self.temp_dir,
                    )

                    # Should return model and history
                    model, history = result
                    assert model is not None
                    assert isinstance(history, dict)

                    # Verify base model was created
                    mock_model_class.assert_called_once()

    @patch("torch_diode.model.model_utils.create_dataloaders")
    @patch("torch_diode.types.matmul_dataset.Dataset.deserialize")
    @patch("torch.manual_seed")
    @patch("os.makedirs")
    def test_train_model_deep_model_type(
        self, mock_makedirs, mock_seed, mock_deserialize, mock_create_loaders
    ):
        """Test train_model with deep model type."""
        # Mock dataset with proper structure
        mock_dataset = Mock()
        mock_dataset.hardware = {}  # Empty hardware dict to make iteration work
        mock_deserialize.return_value = mock_dataset

        # Create mock dataloaders that don't actually iterate
        mock_dataloader = self._create_mock_dataloader()
        mock_create_loaders.return_value = (
            mock_dataloader,
            mock_dataloader,
            mock_dataloader,
        )

        # Mock trainer completely to avoid actual training
        with patch(
            "torch_diode.model.model_utils.DeepMatmulTimingModel"
        ) as mock_model_class:
            with patch("torch_diode.utils.visualization_utils.plot_training_history"):
                mock_model = self._create_mock_model()
                mock_model.parameters.return_value = [
                    torch.randn(10, 10)
                ]  # Mock parameters for counting
                mock_model_class.return_value = mock_model

                dataset_path = os.path.join(self.temp_dir, "dataset.json")
                model_path = os.path.join(self.temp_dir, "model.pt")

                # Create dummy JSON file
                with open(dataset_path, "w") as f:
                    f.write("{}")

                # Mock the entire train_model function to avoid actual training
                with patch(
                    "torch_diode.model.model_utils.MatmulModelTrainer"
                ) as mock_trainer_class:
                    mock_trainer = Mock()
                    mock_trainer.train.return_value = {
                        "train_loss": [0.1, 0.05],
                        "val_loss": [0.15, 0.08],
                        "test_loss": [0.12, 0.06],
                        "learning_rate": [0.001, 0.0005],
                    }
                    mock_trainer._evaluate.return_value = 0.01
                    mock_trainer_class.return_value = mock_trainer

                    result = train_model(
                        dataset_path=dataset_path,
                        model_path=model_path,
                        model_type="deep",
                        hidden_dim=256,
                        num_layers=15,
                    )

                    # Should return model and history
                    model, history = result
                    assert model is not None
                    assert isinstance(history, dict)

                    # Verify deep model was created with custom parameters
                    mock_model_class.assert_called_once()
                    args, kwargs = mock_model_class.call_args
                    assert kwargs["hidden_dim"] == 256
                    assert kwargs["num_layers"] == 15

    def test_validate_model_missing_model(self):
        """Test validate_model with missing model file."""
        model_path = os.path.join(self.temp_dir, "nonexistent_model.pt")
        validation_path = os.path.join(self.temp_dir, "validation.json")

        # Should exit early if model doesn't exist
        validate_model(model_path, validation_path)
        # Test passes if no exception is raised

    def test_validate_model_missing_validation_dataset(self):
        """Test validate_model with missing validation dataset."""
        model_path = os.path.join(self.temp_dir, "model.pt")
        validation_path = os.path.join(self.temp_dir, "nonexistent_validation.json")

        # Create dummy model file
        with open(model_path, "w") as f:
            f.write("dummy model")

        # Should exit early if validation dataset doesn't exist
        validate_model(model_path, validation_path)
        # Test passes if no exception is raised

    @patch("torch_diode.model.directory_dataset_loader.create_directory_dataloaders")
    @patch("os.path.isdir")
    def test_validate_model_directory_dataset_exception(
        self, mock_isdir, mock_create_loaders
    ):
        """Test validate_model with directory dataset loader exception."""
        mock_isdir.return_value = True
        mock_create_loaders.side_effect = Exception("Dataloader creation failed")

        model_path = os.path.join(self.temp_dir, "model.pt")
        validation_path = self.temp_dir

        # Create dummy model file
        with open(model_path, "w") as f:
            f.write("dummy model")

        # Should handle exception gracefully
        validate_model(model_path, validation_path)
        # Test passes if no exception is raised

    @patch("torch_diode.model.matmul_dataset_loader.create_dataloaders")
    @patch("torch_diode.types.matmul_dataset.Dataset.from_msgpack")
    @patch("os.path.isdir")
    def test_validate_model_msgpack_dataset_none(
        self, mock_isdir, mock_from_msgpack, mock_create_loaders
    ):
        """Test validate_model with msgpack dataset that returns None."""
        mock_isdir.return_value = False
        mock_from_msgpack.return_value = None

        model_path = os.path.join(self.temp_dir, "model.pt")
        validation_path = os.path.join(self.temp_dir, "validation.msgpack")

        # Create dummy files
        with open(model_path, "w") as f:
            f.write("dummy model")
        with open(validation_path, "wb") as f:
            f.write(b"dummy msgpack")

        # Should handle None dataset gracefully
        validate_model(model_path, validation_path)
        # Test passes if no exception is raised

    @patch("torch_diode.model.matmul_dataset_loader.create_dataloaders")
    @patch("torch_diode.types.matmul_dataset.Dataset.deserialize")
    @patch("os.path.isdir")
    def test_validate_model_json_dataset_none(
        self, mock_isdir, mock_deserialize, mock_create_loaders
    ):
        """Test validate_model with JSON dataset that returns None."""
        mock_isdir.return_value = False
        mock_deserialize.return_value = None

        model_path = os.path.join(self.temp_dir, "model.pt")
        validation_path = os.path.join(self.temp_dir, "validation.json")

        # Create dummy files
        for path in [model_path, validation_path]:
            with open(path, "w") as f:
                f.write("{}")

        # Should handle None dataset gracefully
        validate_model(model_path, validation_path)
        # Test passes if no exception is raised

    @patch("torch.load")
    def test_validate_model_model_load_exception(self, mock_torch_load):
        """Test validate_model with model loading exception."""
        mock_torch_load.side_effect = Exception("Model load failed")

        model_path = os.path.join(self.temp_dir, "model.pt")
        validation_path = os.path.join(self.temp_dir, "validation.json")

        # Create dummy files
        for path in [model_path, validation_path]:
            with open(path, "w") as f:
                f.write("{}")

        with patch(
            "torch_diode.model.matmul_dataset_loader.create_dataloaders"
        ) as mock_create_loaders:
            mock_dataloader = self._create_mock_dataloader()
            mock_create_loaders.return_value = (
                mock_dataloader,
                mock_dataloader,
                mock_dataloader,
            )

            # Should handle model loading exception gracefully
            validate_model(model_path, validation_path)
            # Test passes if no exception is raised

    @patch("torch_diode.model.model_utils.create_dataloaders")
    @patch("torch_diode.types.matmul_dataset.Dataset.deserialize")
    @patch("torch.load")
    @patch("os.path.isdir")
    def test_validate_model_checkpoint_format_base_model(
        self, mock_isdir, mock_torch_load, mock_deserialize, mock_create_loaders
    ):
        """Test validate_model with checkpoint format and base model."""
        mock_isdir.return_value = False

        # Mock checkpoint with base model type
        mock_checkpoint = {
            "model_state_dict": {},
            "problem_feature_dim": 15,
            "config_feature_dim": 25,
            "hidden_dim": 128,
            "num_layers": 10,
            "model_type": "base",
        }
        mock_torch_load.return_value = mock_checkpoint

        # Mock dataset with proper structure
        mock_dataset = Mock()
        mock_dataset.hardware = {}
        mock_deserialize.return_value = mock_dataset

        # Mock dataloaders - ensure they're not None with correct dimensions
        mock_dataloader = self._create_mock_dataloader(problem_dim=15, config_dim=25)
        mock_create_loaders.return_value = (
            mock_dataloader,
            mock_dataloader,
            mock_dataloader,
        )

        with patch(
            "torch_diode.model.model_utils.MatmulTimingModel"
        ) as mock_model_class:
            with patch(
                "torch_diode.model.matmul_model_trainer.MatmulModelTrainer"
            ) as mock_trainer_class:
                with patch(
                    "torch_diode.model.matmul_model_trainer.analyze_worst_predictions"
                ):
                    mock_model = self._create_mock_model()
                    mock_model_class.return_value = mock_model

                    mock_trainer = Mock()
                    mock_trainer._evaluate = Mock(return_value=0.01)
                    mock_trainer_class.return_value = mock_trainer

                    model_path = os.path.join(self.temp_dir, "model.pt")
                    validation_path = os.path.join(self.temp_dir, "validation.json")

                    # Create dummy files
                    for path in [model_path, validation_path]:
                        with open(path, "w") as f:
                            f.write("{}")

                    # Mock os.makedirs to prevent log directory conflict
                    with patch("os.makedirs"):
                        validate_model(
                            model_path=model_path,
                            validation_dataset_path=validation_path,
                            top_n_worst=5,
                            hardware_name="test_gpu",
                            op_name="mm",
                            device="cpu",
                            batch_size=16,
                        )

                    # Verify base model was created
                    mock_model_class.assert_called_once()

    @patch("torch_diode.model.model_utils.create_dataloaders")
    @patch("torch_diode.types.matmul_dataset.Dataset.deserialize")
    @patch("torch.load")
    @patch("os.path.isdir")
    def test_validate_model_checkpoint_format_deep_model(
        self, mock_isdir, mock_torch_load, mock_deserialize, mock_create_loaders
    ):
        """Test validate_model with checkpoint format and deep model."""
        mock_isdir.return_value = False

        # Mock checkpoint with deep model type
        mock_checkpoint = {
            "model_state_dict": {},
            "problem_feature_dim": 15,
            "config_feature_dim": 25,
            "hidden_dim": 256,
            "num_layers": 15,
            "model_type": "deep",
        }
        mock_torch_load.return_value = mock_checkpoint

        # Mock dataset with proper structure
        mock_dataset = Mock()
        mock_dataset.hardware = {}
        mock_deserialize.return_value = mock_dataset

        # Mock dataloaders with matching dimensions
        mock_dataloader = self._create_mock_dataloader(problem_dim=15, config_dim=25)
        mock_create_loaders.return_value = (
            mock_dataloader,
            mock_dataloader,
            mock_dataloader,
        )

        with patch(
            "torch_diode.model.model_utils.DeepMatmulTimingModel"
        ) as mock_model_class:
            with patch(
                "torch_diode.model.matmul_model_trainer.MatmulModelTrainer"
            ) as mock_trainer_class:
                with patch(
                    "torch_diode.model.matmul_model_trainer.analyze_worst_predictions"
                ):
                    mock_model = self._create_mock_model()
                    mock_model_class.return_value = mock_model

                    mock_trainer = Mock()
                    mock_trainer._evaluate.return_value = 0.01
                    mock_trainer_class.return_value = mock_trainer

                    model_path = os.path.join(self.temp_dir, "model.pt")
                    validation_path = os.path.join(self.temp_dir, "validation.json")

                    # Create dummy files
                    for path in [model_path, validation_path]:
                        with open(path, "w") as f:
                            f.write("{}")

                    # Mock os.makedirs to prevent log directory conflict
                    with patch("os.makedirs"):
                        validate_model(
                            model_path=model_path,
                            validation_dataset_path=validation_path,
                            top_n_worst=0,  # Skip worst prediction analysis
                            device="cpu",  # Force CPU to avoid device mismatch
                        )

                    # Verify deep model was created with custom parameters
                    mock_model_class.assert_called_once()
                    args, kwargs = mock_model_class.call_args
                    assert kwargs["hidden_dim"] == 256
                    assert kwargs["num_layers"] == 15

    @patch("torch_diode.model.model_utils.create_dataloaders")
    @patch("torch_diode.types.matmul_dataset.Dataset.deserialize")
    @patch("torch.load")
    @patch("os.path.isdir")
    def test_validate_model_direct_state_dict(
        self, mock_isdir, mock_torch_load, mock_deserialize, mock_create_loaders
    ):
        """Test validate_model with direct state dict (not checkpoint format)."""
        mock_isdir.return_value = False

        # Mock direct state dict (not checkpoint format) with proper structure
        mock_state_dict = {
            "input_layer.0.weight": torch.randn(128, 30),
            "input_layer.0.bias": torch.randn(128),
            "output_layer.weight": torch.randn(1, 128),
            "output_layer.bias": torch.randn(1),
        }
        mock_torch_load.return_value = mock_state_dict

        # Mock dataset with proper structure
        mock_dataset = Mock()
        mock_dataset.hardware = {}
        mock_deserialize.return_value = mock_dataset

        # Mock dataloaders with correct dimensions
        mock_dataloader = self._create_mock_dataloader(problem_dim=15, config_dim=15)
        mock_create_loaders.return_value = (
            mock_dataloader,
            mock_dataloader,
            mock_dataloader,
        )

        with patch(
            "torch_diode.model.model_utils.DeepMatmulTimingModel"
        ) as mock_model_class:
            with patch(
                "torch_diode.model.matmul_model_trainer.MatmulModelTrainer"
            ) as mock_trainer_class:
                mock_model = self._create_mock_model()

                # Mock load_state_dict to accept any state dict without strict checking
                # By default, load_state_dict would fail, so we override it to succeed
                def mock_load_state_dict(state_dict, strict=True):
                    pass  # Just succeed without error

                mock_model.load_state_dict = Mock(side_effect=mock_load_state_dict)
                mock_model_class.return_value = mock_model

                mock_trainer = Mock()
                mock_trainer._evaluate.return_value = 0.01
                mock_trainer_class.return_value = mock_trainer

                model_path = os.path.join(self.temp_dir, "model.pt")
                validation_path = os.path.join(self.temp_dir, "validation.json")

                # Create dummy files
                for path in [model_path, validation_path]:
                    with open(path, "w") as f:
                        f.write("{}")

                with patch("os.makedirs"):
                    validate_model(model_path, validation_path)

                # Verify deep model was assumed (default for direct state dict)
                mock_model_class.assert_called_once()

    @patch("torch_diode.types.matmul_dataset.Dataset.deserialize")
    @patch("os.makedirs")
    def test_run_model_example_missing_dataset(self, mock_makedirs, mock_deserialize):
        """Test run_model_example with missing dataset file."""
        mock_deserialize.return_value = None

        dataset_path = os.path.join(self.temp_dir, "dataset.json")

        # Create dummy JSON file
        with open(dataset_path, "w") as f:
            f.write("{}")

        # Should exit early if dataset can't be loaded
        run_model_example(dataset_path)
        # Test passes if no exception is raised

    @patch("torch_diode.types.matmul_dataset.Dataset.from_msgpack")
    @patch("os.makedirs")
    def test_run_model_example_msgpack_dataset_none(
        self, mock_makedirs, mock_from_msgpack
    ):
        """Test run_model_example with msgpack dataset that returns None."""
        mock_from_msgpack.return_value = None

        dataset_path = os.path.join(self.temp_dir, "dataset.msgpack")

        # Create dummy msgpack file
        with open(dataset_path, "wb") as f:
            f.write(b"dummy msgpack")

        # Should exit early if dataset can't be loaded
        run_model_example(dataset_path)
        # Test passes if no exception is raised

    @patch("torch_diode.model.model_utils.print_dataset_statistics")
    @patch("torch_diode.model.model_utils.train_model_from_dataset")
    @patch("torch_diode.types.matmul_dataset.Dataset.deserialize")
    @patch("os.makedirs")
    def test_run_model_example_success(
        self, mock_makedirs, mock_deserialize, mock_train_from_dataset, mock_print_stats
    ):
        """Test run_model_example successful execution."""
        # Mock dataset with proper structure to avoid empty dataset issues
        mock_dataset = Mock()
        mock_dataset.hardware = {
            "gpu1": Mock(),
            "gpu2": Mock(),
        }  # Mock hardware dict with len() support
        # Add attributes to prevent empty dataset issues
        mock_dataset.entries = [Mock(), Mock()]  # Mock entries list
        mock_dataset.hardware_names = ["gpu1", "gpu2"]
        mock_deserialize.return_value = mock_dataset

        # Mock train_model_from_dataset to return a model, history, and config (not None)
        mock_model = self._create_mock_model()
        mock_history = {
            "train_loss": [0.1, 0.05],
            "val_loss": [0.15, 0.08],
            "test_loss": [0.12, 0.06],
            "learning_rate": [0.001, 0.0005],
        }
        mock_config = Mock()
        # Ensure the model is not None so the function doesn't exit early
        mock_train_from_dataset.return_value = (mock_model, mock_history, mock_config)

        # Mock the create_dataloaders function both in train_model_from_dataset and for test evaluation
        with patch(
            "torch_diode.model.matmul_model_trainer.create_dataloaders"
        ) as mock_create_dataloaders_trainer:
            with patch(
                "torch_diode.model.model_utils.create_dataloaders"
            ) as mock_create_dataloaders:
                with patch(
                    "torch_diode.utils.visualization_utils.plot_training_history"
                ):
                    with patch("torch.nn.MSELoss") as mock_mse:
                        # Create valid mock test dataloader
                        mock_dataloader = self._create_mock_dataloader()
                        # Mock both create_dataloaders calls
                        mock_create_dataloaders_trainer.return_value = (
                            mock_dataloader,
                            mock_dataloader,
                            mock_dataloader,
                        )
                        mock_create_dataloaders.return_value = (
                            None,
                            None,
                            mock_dataloader,
                        )

                        mock_criterion = Mock()
                        mock_criterion.return_value = torch.tensor(0.01)
                        mock_mse.return_value = mock_criterion

                        dataset_path = os.path.join(self.temp_dir, "dataset.json")

                        # Create dummy JSON file
                        with open(dataset_path, "w") as f:
                            f.write("{}")

                        run_model_example(
                            dataset_path=dataset_path,
                            model_type="deep",
                            batch_size=32,
                            num_epochs=10,
                            learning_rate=0.001,
                            weight_decay=1e-5,
                            patience=15,
                            log_dir=self.temp_dir,
                            model_dir=self.temp_dir,
                            hardware_name="test_gpu",
                            op_name="mm",
                            seed=42,
                            device="cpu",
                        )

                        # Verify key functions were called
                        mock_print_stats.assert_called_once()
                        mock_train_from_dataset.assert_called_once()

    @patch("torch_diode.model.directory_dataset_loader.create_directory_dataloaders")
    @patch("torch.manual_seed")
    @patch("os.makedirs")
    def test_train_model_from_directory_base_model(
        self, mock_makedirs, mock_seed, mock_create_loaders
    ):
        """Test train_model_from_directory with base model type."""
        # Mock the directory data loader to raise the expected exception
        mock_create_loaders.side_effect = ValueError(
            "No valid datasets after filtering/errors."
        )

        data_dir = self.temp_dir
        model_path = os.path.join(self.temp_dir, "model.pt")

        # Create a dummy JSON file in the directory with proper hardware structure
        dummy_data_file = os.path.join(self.temp_dir, "dummy_data.json")
        dummy_data = {"hardware": "test_gpu", "entries": [], "metadata": {}}
        with open(dummy_data_file, "w") as f:
            json.dump(dummy_data, f)

        # Expect ValueError to be raised when no valid datasets are found
        with pytest.raises(
            ValueError, match="No valid datasets after filtering/errors."
        ):
            train_model_from_directory(
                data_dir=data_dir,
                model_path=model_path,
                model_type="base",
                num_epochs=5,
                learning_rate=0.01,
                weight_decay=1e-4,
                patience=10,
                hidden_dim=64,
                num_layers=5,
                hardware_name="test_gpu",
                op_name="mm",
                seed=123,
                device="cpu",
                log_dir=self.temp_dir,
                file_extensions=["json", "msgpack"],
            )

    @patch("torch_diode.model.directory_dataset_loader.create_directory_dataloaders")
    @patch("torch.manual_seed")
    @patch("os.makedirs")
    def test_train_model_from_directory_deep_model(
        self, mock_makedirs, mock_seed, mock_create_loaders
    ):
        """Test train_model_from_directory with deep model type."""
        # Mock the directory data loader to raise the expected exception
        mock_create_loaders.side_effect = ValueError(
            "No valid datasets after filtering/errors."
        )

        data_dir = self.temp_dir
        model_path = os.path.join(self.temp_dir, "model.pt")

        # Create a dummy JSON file in the directory with proper hardware structure
        dummy_data_file = os.path.join(self.temp_dir, "dummy_data.json")
        dummy_data = {"hardware": "test_gpu", "entries": [], "metadata": {}}
        with open(dummy_data_file, "w") as f:
            json.dump(dummy_data, f)

        # Expect ValueError to be raised when no valid datasets are found
        with pytest.raises(
            ValueError, match="No valid datasets after filtering/errors."
        ):
            train_model_from_directory(
                data_dir=data_dir,
                model_path=model_path,
                model_type="deep",
                hidden_dim=512,
                num_layers=20,
            )

    @patch("torch_diode.model.model_utils.create_directory_dataloaders")
    @patch("torch.manual_seed")
    @patch("os.makedirs")
    def test_train_model_from_directory_successful_training(
        self, mock_makedirs, mock_seed, mock_create_loaders
    ):
        """Test train_model_from_directory that successfully gets past model creation."""
        # Create mock dataloaders that successfully return valid data
        mock_dataloader = self._create_mock_dataloader(problem_dim=20, config_dim=30)
        mock_create_loaders.return_value = (
            mock_dataloader,
            mock_dataloader,
            mock_dataloader,
        )

        data_dir = self.temp_dir
        model_path = os.path.join(self.temp_dir, "model.pt")

        # Create a dummy JSON file in the directory (not strictly needed since we're mocking loaders)
        dummy_data_file = os.path.join(self.temp_dir, "dummy_data.json")
        dummy_data = {"hardware": "test_gpu", "entries": [], "metadata": {}}
        with open(dummy_data_file, "w") as f:
            json.dump(dummy_data, f)

        # Mock the model classes and trainer to avoid actual training
        with patch(
            "torch_diode.model.model_utils.DeepMatmulTimingModel"
        ) as mock_model_class:
            with patch(
                "torch_diode.model.model_utils.MatmulModelTrainer"
            ) as mock_trainer_class:
                with patch(
                    "torch_diode.utils.visualization_utils.plot_training_history"
                ):
                    # Create mock model
                    mock_model = self._create_mock_model()
                    mock_model.parameters.return_value = [torch.randn(10, 10)]
                    mock_model_class.return_value = mock_model

                    # Create mock trainer
                    mock_trainer = Mock()
                    mock_trainer.train.return_value = {
                        "train_loss": [0.1, 0.05],
                        "val_loss": [0.15, 0.08],
                        "test_loss": [0.12, 0.06],
                        "learning_rate": [0.001, 0.0005],
                    }
                    mock_trainer._evaluate.return_value = 0.01
                    mock_trainer_class.return_value = mock_trainer

                    # Call the function - this should get past the model creation
                    result = train_model_from_directory(
                        data_dir=data_dir,
                        model_path=model_path,
                        model_type="deep",
                        num_epochs=5,
                        learning_rate=0.01,
                        weight_decay=1e-4,
                        patience=10,
                        hidden_dim=256,
                        num_layers=12,
                        hardware_name="test_gpu",
                        op_name="mm",
                        seed=123,
                        device="cpu",
                        log_dir=self.temp_dir,
                        file_extensions=["json", "msgpack"],
                    )

                    # Should return model and history
                    model, history = result
                    assert model is not None
                    assert isinstance(history, dict)

                    # Verify that create_directory_dataloaders was called
                    mock_create_loaders.assert_called_once()

                    # Verify that the model was created with correct parameters
                    mock_model_class.assert_called_once()
                    args, kwargs = mock_model_class.call_args
                    assert kwargs["problem_feature_dim"] == 20  # From mock dataloader
                    assert kwargs["config_feature_dim"] == 30  # From mock dataloader
                    assert kwargs["hidden_dim"] == 256
                    assert kwargs["num_layers"] == 12

                    # Verify trainer was created and trained
                    mock_trainer_class.assert_called_once()
                    mock_trainer.train.assert_called_once()

    @patch("torch_diode.model.model_utils.create_directory_dataloaders")
    @patch("torch.manual_seed")
    @patch("os.makedirs")
    def test_train_model_from_directory_base_model_successful(
        self, mock_makedirs, mock_seed, mock_create_loaders
    ):
        """Test train_model_from_directory with base model that successfully trains."""
        # Create mock dataloaders that successfully return valid data
        mock_dataloader = self._create_mock_dataloader(problem_dim=15, config_dim=25)
        mock_create_loaders.return_value = (
            mock_dataloader,
            mock_dataloader,
            mock_dataloader,
        )

        data_dir = self.temp_dir
        model_path = os.path.join(self.temp_dir, "model.pt")

        # Create a dummy JSON file in the directory
        dummy_data_file = os.path.join(self.temp_dir, "dummy_data.json")
        dummy_data = {"hardware": "test_gpu", "entries": [], "metadata": {}}
        with open(dummy_data_file, "w") as f:
            json.dump(dummy_data, f)

        # Mock the model classes and trainer to avoid actual training
        with patch(
            "torch_diode.model.model_utils.MatmulTimingModel"
        ) as mock_model_class:
            with patch(
                "torch_diode.model.model_utils.MatmulModelTrainer"
            ) as mock_trainer_class:
                with patch(
                    "torch_diode.utils.visualization_utils.plot_training_history"
                ):
                    # Create mock model
                    mock_model = self._create_mock_model()
                    mock_model.parameters.return_value = [torch.randn(5, 5)]
                    mock_model_class.return_value = mock_model

                    # Create mock trainer
                    mock_trainer = Mock()
                    mock_trainer.train.return_value = {
                        "train_loss": [0.2, 0.1],
                        "val_loss": [0.25, 0.12],
                        "test_loss": [0.22, 0.11],
                        "learning_rate": [0.001, 0.0005],
                    }
                    mock_trainer._evaluate.return_value = 0.02
                    mock_trainer_class.return_value = mock_trainer

                    # Call the function - this should get past the model creation
                    result = train_model_from_directory(
                        data_dir=data_dir,
                        model_path=model_path,
                        model_type="base",  # Test base model specifically
                        num_epochs=3,
                        learning_rate=0.005,
                        weight_decay=1e-3,
                        patience=5,
                        hardware_name="test_gpu",
                        op_name="mm",
                        seed=456,
                        device="cpu",
                        log_dir=self.temp_dir,
                    )

                    # Should return model and history
                    model, history = result
                    assert model is not None
                    assert isinstance(history, dict)

                    # Verify that the base model was created with correct parameters
                    mock_model_class.assert_called_once()
                    args, kwargs = mock_model_class.call_args
                    assert kwargs["problem_feature_dim"] == 15  # From mock dataloader
                    assert kwargs["config_feature_dim"] == 25  # From mock dataloader
                    # Base model doesn't have hidden_dim or num_layers parameters

                    # Verify trainer was created and trained
                    mock_trainer_class.assert_called_once()
                    mock_trainer.train.assert_called_once()
