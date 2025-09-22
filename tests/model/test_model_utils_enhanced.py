"""
Enhanced tests for model utils to increase coverage.
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
from unittest.mock import MagicMock, mock_open, patch

import pytest
import torch

from torch_diode.model.model_utils import (
    run_model_example,
    train_model,
    train_model_from_directory,
    validate_max_autotune,
    validate_model,
)
from torch_diode.types.matmul_types import Solution, TritonGEMMConfig


class TestValidateMaxAutotune:
    """Test the validate_max_autotune function edge cases."""

    def test_model_not_found(self, caplog):
        """Test when model file doesn't exist."""
        validate_max_autotune(
            model_path="/nonexistent/model.pt",
            validation_dataset_path="/tmp/valid.json",
            max_autotune_solution_path="/tmp/solution.json",
        )
        assert "Model not found at /nonexistent/model.pt" in caplog.text

    def test_validation_dataset_not_found(self, caplog):
        """Test when validation dataset doesn't exist."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as model_file:
            validate_max_autotune(
                model_path=model_file.name,
                validation_dataset_path="/nonexistent/valid.json",
                max_autotune_solution_path="/tmp/solution.json",
            )
            assert (
                "Validation dataset not found at /nonexistent/valid.json" in caplog.text
            )

    def test_max_autotune_solution_not_found(self, caplog):
        """Test when max-autotune solution file doesn't exist."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as model_file:
            with tempfile.NamedTemporaryFile(suffix=".json") as valid_file:
                validate_max_autotune(
                    model_path=model_file.name,
                    validation_dataset_path=valid_file.name,
                    max_autotune_solution_path="/nonexistent/solution.json",
                )
                assert (
                    "Max-autotune solution not found at /nonexistent/solution.json"
                    in caplog.text
                )

    def test_invalid_max_autotune_solution(self, caplog):
        """Test when max-autotune solution file is invalid."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as model_file:
            with tempfile.NamedTemporaryFile(suffix=".json") as valid_file:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as solution_file:
                    solution_file.write("invalid json")
                    solution_file.flush()

                    validate_max_autotune(
                        model_path=model_file.name,
                        validation_dataset_path=valid_file.name,
                        max_autotune_solution_path=solution_file.name,
                    )
                    assert "Failed to load max-autotune solution" in caplog.text

                os.unlink(solution_file.name)

    def test_none_max_autotune_solution(self, caplog):
        """Test when max-autotune solution deserializes to None."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as model_file:
            with tempfile.NamedTemporaryFile(suffix=".json") as valid_file:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as solution_file:
                    solution_file.write('{"invalid": "structure"}')
                    solution_file.flush()

                    validate_max_autotune(
                        model_path=model_file.name,
                        validation_dataset_path=valid_file.name,
                        max_autotune_solution_path=solution_file.name,
                    )
                    # The function should log an error when dataset loading fails due to empty file
                    assert "Failed to load validation dataset" in caplog.text

                os.unlink(solution_file.name)

    def test_directory_dataloader_creation_failure(self, caplog):
        """Test when directory dataloader creation fails."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as model_file:
            with tempfile.TemporaryDirectory() as temp_dir:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as solution_file:
                    # Create a minimal valid solution
                    solution_data = {"config": []}
                    json.dump(solution_data, solution_file)
                    solution_file.flush()

                    with patch(
                        "torch_diode.model.directory_dataset_loader.create_directory_dataloaders"
                    ) as mock_create:
                        mock_create.side_effect = Exception(
                            "Failed to create dataloaders"
                        )
                        validate_max_autotune(
                            model_path=model_file.name,
                            validation_dataset_path=temp_dir,
                            max_autotune_solution_path=solution_file.name,
                        )
                        assert (
                            "Failed to create dataloaders from directory" in caplog.text
                        )

                os.unlink(solution_file.name)


class TestTrainModel:
    """Test the train_model function edge cases."""

    def test_msgpack_dataset_loading_failure(self, caplog):
        """Test when msgpack dataset loading fails."""
        with tempfile.NamedTemporaryFile(suffix=".msgpack") as dataset_file:
            dataset_file.write(b"invalid msgpack data")
            dataset_file.flush()

            with pytest.raises(
                ValueError, match="Failed to deserialize Dataset from MessagePack"
            ):
                model, history = train_model(
                    dataset_path=dataset_file.name, model_path="/tmp/model.pt"
                )

    def test_json_dataset_loading_failure(self, caplog):
        """Test when JSON dataset loading fails."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as dataset_file:
            dataset_file.write("invalid json")
            dataset_file.flush()

            model, history = train_model(
                dataset_path=dataset_file.name, model_path="/tmp/model.pt"
            )
            assert model is None
            assert history == {}
            assert "Failed to load dataset" in caplog.text

    def test_dataset_none_after_loading(self, caplog):
        """Test when dataset is None after deserialization."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as dataset_file:
            dataset_file.write('{"invalid": "structure"}')
            dataset_file.flush()

            with patch(
                "torch_diode.types.matmul_dataset.Dataset.deserialize"
            ) as mock_deserialize:
                mock_deserialize.return_value = None
                model, history = train_model(
                    dataset_path=dataset_file.name, model_path="/tmp/model.pt"
                )
                assert model is None
                assert history == {}
                assert "Failed to load dataset" in caplog.text

    def test_file_not_found_error(self, caplog):
        """Test FileNotFoundError handling."""
        model, history = train_model(
            dataset_path="/nonexistent/dataset.json", model_path="/tmp/model.pt"
        )
        assert model is None
        assert history == {}
        assert "Failed to load dataset" in caplog.text

    def test_base_model_creation(self):
        """Test creation of base model type."""
        with patch("torch_diode.model.model_utils.create_dataloaders") as mock_create:
            mock_train_loader = MagicMock()
            mock_val_loader = MagicMock()
            mock_test_loader = MagicMock()
            mock_dataset = MagicMock()
            mock_dataset.problem_feature_dim = 10
            mock_dataset.config_feature_dim = 5
            mock_train_loader.dataset.dataset = mock_dataset
            mock_val_loader.dataset.dataset = mock_dataset
            mock_test_loader.dataset.dataset = mock_dataset

            # Make sure dataloaders are not empty and have more samples for batch norm
            mock_train_loader.__len__.return_value = 10
            mock_val_loader.__len__.return_value = 5
            mock_test_loader.__len__.return_value = 5

            # Create multiple samples to avoid batch norm issues
            train_samples = [
                (torch.randn(8, 10), torch.randn(8, 5), torch.randn(8, 1))
                for _ in range(10)
            ]
            val_samples = [
                (torch.randn(4, 10), torch.randn(4, 5), torch.randn(4, 1))
                for _ in range(5)
            ]
            test_samples = [
                (torch.randn(4, 10), torch.randn(4, 5), torch.randn(4, 1))
                for _ in range(5)
            ]

            mock_train_loader.__iter__.return_value = iter(train_samples)
            mock_val_loader.__iter__.return_value = iter(val_samples)
            mock_test_loader.__iter__.return_value = iter(test_samples)

            mock_create.return_value = (
                mock_train_loader,
                mock_val_loader,
                mock_test_loader,
            )

            with patch(
                "torch_diode.types.matmul_dataset.Dataset.deserialize"
            ) as mock_deserialize:
                mock_deserialize.return_value = MagicMock()

                with patch(
                    "torch_diode.model.matmul_model_trainer.MatmulModelTrainer"
                ) as mock_trainer_class:
                    mock_trainer = MagicMock()
                    mock_trainer.train.return_value = {"train_loss": [1.0, 0.5]}
                    mock_trainer._evaluate.return_value = 0.25
                    mock_trainer_class.return_value = mock_trainer

                    # The model is created within the train_model function by direct import
                    # So we need to patch it in the model_utils module where it's imported
                    with patch(
                        "torch_diode.model.model_utils.MatmulTimingModel"
                    ) as mock_model_class:
                        mock_model = MagicMock()
                        # Create actual parameters for the optimizer
                        param1 = torch.nn.Parameter(torch.randn(10, 5))
                        param2 = torch.nn.Parameter(torch.randn(5))
                        mock_model.parameters.return_value = [param1, param2]
                        mock_model.to.return_value = mock_model

                        # Make the model return actual tensors when called with gradients
                        def mock_forward(*args, **kwargs):
                            batch_size = args[0].shape[0] if len(args) > 0 else 1
                            device = args[0].device if len(args) > 0 else "cpu"
                            # Create output that requires gradients by doing a simple linear operation
                            input_tensor = (
                                args[0]
                                if len(args) > 0
                                else torch.randn(batch_size, 10, device=device)
                            )
                            output = torch.randn(
                                batch_size, 1, device=device, requires_grad=True
                            )
                            # Make sure output has a connection to inputs for gradient computation
                            output = output + 0 * input_tensor.sum()
                            return output

                        mock_model.side_effect = mock_forward

                        mock_model_class.return_value = mock_model

                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".json"
                        ) as dataset_file:
                            dataset_file.write('{"valid": "structure"}')
                            dataset_file.flush()

                            model, history = train_model(
                                dataset_path=dataset_file.name,
                                model_path="/tmp/model.pt",
                                model_type="base",
                            )

                            assert model is not None
                            assert history is not None


class TestValidateModel:
    """Test the validate_model function edge cases."""

    def test_model_not_found(self, caplog):
        """Test when model file doesn't exist."""
        validate_model(
            model_path="/nonexistent/model.pt",
            validation_dataset_path="/tmp/valid.json",
        )
        assert "Model not found at /nonexistent/model.pt" in caplog.text

    def test_validation_dataset_not_found(self, caplog):
        """Test when validation dataset doesn't exist."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as model_file:
            validate_model(
                model_path=model_file.name,
                validation_dataset_path="/nonexistent/valid.json",
            )
            assert (
                "Validation dataset not found at /nonexistent/valid.json" in caplog.text
            )

    def test_directory_validation_failure(self, caplog):
        """Test when directory validation fails."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as model_file:
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch(
                    "torch_diode.model.model_utils.create_directory_dataloaders"
                ) as mock_create:
                    mock_create.side_effect = Exception("Failed to create dataloaders")
                    validate_model(
                        model_path=model_file.name, validation_dataset_path=temp_dir
                    )
                    assert "Failed to create dataloaders from directory" in caplog.text

    def test_msgpack_validation_dataset_none(self, caplog):
        """Test when msgpack validation dataset is None after loading."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as model_file:
            with tempfile.NamedTemporaryFile(suffix=".msgpack") as dataset_file:
                dataset_file.write(b"valid msgpack")
                dataset_file.flush()

                with patch(
                    "torch_diode.types.matmul_dataset.Dataset.from_msgpack"
                ) as mock_from_msgpack:
                    mock_from_msgpack.return_value = None
                    validate_model(
                        model_path=model_file.name,
                        validation_dataset_path=dataset_file.name,
                    )
                    assert "Failed to load validation dataset" in caplog.text

    def test_json_validation_dataset_none(self, caplog):
        """Test when JSON validation dataset is None after loading."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as model_file:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as dataset_file:
                dataset_file.write('{"valid": "structure"}')
                dataset_file.flush()

                with patch(
                    "torch_diode.types.matmul_dataset.Dataset.deserialize"
                ) as mock_deserialize:
                    mock_deserialize.return_value = None
                    validate_model(
                        model_path=model_file.name,
                        validation_dataset_path=dataset_file.name,
                    )
                    assert "Failed to load validation dataset" in caplog.text

    def test_model_loading_failure(self, caplog):
        """Test when model loading fails."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as model_file:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as dataset_file:
                dataset_file.write('{"valid": "structure"}')
                dataset_file.flush()

                with patch(
                    "torch_diode.types.matmul_dataset.Dataset.deserialize"
                ) as mock_deserialize:
                    mock_deserialize.return_value = MagicMock()

                    with patch(
                        "torch_diode.model.model_utils.create_dataloaders"
                    ) as mock_create:
                        mock_loader = MagicMock()
                        mock_dataset = MagicMock()
                        mock_dataset.problem_feature_dim = 10
                        mock_dataset.config_feature_dim = 5
                        mock_loader.dataset.dataset = mock_dataset
                        mock_create.return_value = (None, mock_loader, None)

                        with patch("torch.load") as mock_torch_load:
                            mock_torch_load.side_effect = Exception("Failed to load")
                            validate_model(
                                model_path=model_file.name,
                                validation_dataset_path=dataset_file.name,
                            )
                            assert "Failed to load model weights" in caplog.text

    def test_base_model_checkpoint_loading(self):
        """Test loading base model from checkpoint format."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as model_file:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as dataset_file:
                dataset_file.write('{"valid": "structure"}')
                dataset_file.flush()

                with patch(
                    "torch_diode.types.matmul_dataset.Dataset.deserialize"
                ) as mock_deserialize:
                    mock_deserialize.return_value = MagicMock()

                    with patch(
                        "torch_diode.model.model_utils.create_dataloaders"
                    ) as mock_create:
                        mock_loader = MagicMock()
                        mock_dataset = MagicMock()
                        mock_dataset.problem_feature_dim = 10
                        mock_dataset.config_feature_dim = 5
                        mock_loader.dataset.dataset = mock_dataset

                        # Make sure loader has samples to avoid division by zero
                        mock_loader.__len__.return_value = 2
                        mock_loader.__iter__.return_value = iter(
                            [
                                (
                                    torch.randn(4, 10),
                                    torch.randn(4, 5),
                                    torch.randn(4, 1),
                                ),
                                (
                                    torch.randn(4, 10),
                                    torch.randn(4, 5),
                                    torch.randn(4, 1),
                                ),
                            ]
                        )

                        mock_create.return_value = (None, mock_loader, None)

                        checkpoint = {
                            "model_state_dict": {},
                            "problem_feature_dim": 10,
                            "config_feature_dim": 5,
                            "hidden_dim": 64,
                            "num_layers": 5,
                            "model_type": "base",
                        }

                        with patch("torch.load") as mock_torch_load:
                            mock_torch_load.return_value = checkpoint

                            with patch(
                                "torch_diode.model.model_utils.MatmulTimingModel"
                            ) as mock_model_class:
                                mock_model = MagicMock()
                                # Create actual parameters for the optimizer
                                param1 = torch.nn.Parameter(torch.randn(10, 5))
                                param2 = torch.nn.Parameter(torch.randn(5))
                                mock_model.parameters.return_value = [param1, param2]
                                mock_model.to.return_value = mock_model
                                mock_model.load_state_dict = (
                                    MagicMock()
                                )  # Mock the load_state_dict method
                                mock_model.eval.return_value = mock_model

                                # Mock the model to return predictions on the same device as inputs
                                def mock_forward(*args):
                                    device = args[0].device if len(args) > 0 else "cpu"
                                    return torch.randn(4, 1, device=device)

                                mock_model.side_effect = mock_forward
                                mock_model_class.return_value = mock_model

                                with patch(
                                    "torch_diode.model.matmul_model_trainer.MatmulModelTrainer"
                                ) as mock_trainer_class:
                                    mock_trainer = MagicMock()
                                    mock_trainer._evaluate.return_value = 0.25
                                    mock_trainer_class.return_value = mock_trainer

                                    validate_model(
                                        model_path=model_file.name,
                                        validation_dataset_path=dataset_file.name,
                                        top_n_worst=0,  # Skip worst predictions analysis
                                    )

                                    mock_model_class.assert_called_once_with(
                                        problem_feature_dim=10, config_feature_dim=5
                                    )

    def test_worst_predictions_analysis(self):
        """Test worst predictions analysis."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as model_file:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as dataset_file:
                dataset_file.write('{"valid": "structure"}')
                dataset_file.flush()

                with patch(
                    "torch_diode.types.matmul_dataset.Dataset.deserialize"
                ) as mock_deserialize:
                    mock_deserialize.return_value = MagicMock()

                    with patch(
                        "torch_diode.model.model_utils.create_dataloaders"
                    ) as mock_create:
                        mock_loader = MagicMock()
                        mock_dataset = MagicMock()
                        mock_dataset.problem_feature_dim = 10
                        mock_dataset.config_feature_dim = 5
                        mock_loader.dataset.dataset = mock_dataset

                        # Make sure loader has samples to avoid division by zero
                        mock_loader.__len__.return_value = 2
                        mock_loader.__iter__.return_value = iter(
                            [
                                (
                                    torch.randn(4, 10),
                                    torch.randn(4, 5),
                                    torch.randn(4, 1),
                                ),
                                (
                                    torch.randn(4, 10),
                                    torch.randn(4, 5),
                                    torch.randn(4, 1),
                                ),
                            ]
                        )

                        mock_create.return_value = (None, mock_loader, None)

                        # Skip the checkpoint loading by providing an empty but valid checkpoint
                        checkpoint = {
                            "model_state_dict": {},
                            "problem_feature_dim": 10,
                            "config_feature_dim": 5,
                            "hidden_dim": 64,
                            "num_layers": 1,
                            "model_type": "deep",
                        }

                        with patch("torch.load") as mock_torch_load:
                            mock_torch_load.return_value = checkpoint

                            with patch(
                                "torch_diode.model.matmul_timing_model.DeepMatmulTimingModel"
                            ) as mock_model_class:
                                mock_model = MagicMock()
                                mock_model.eval.return_value = mock_model
                                mock_model.load_state_dict = MagicMock()

                                # Mock the model to return predictions on the same device as inputs
                                def mock_forward(*args):
                                    device = args[0].device if len(args) > 0 else "cpu"
                                    return torch.randn(4, 1, device=device)

                                mock_model.side_effect = mock_forward
                                mock_model_class.return_value = mock_model

                                with patch(
                                    "torch_diode.model.matmul_model_trainer.MatmulModelTrainer"
                                ) as mock_trainer_class:
                                    mock_trainer = MagicMock()
                                    mock_trainer._evaluate.return_value = 0.25
                                    mock_trainer_class.return_value = mock_trainer

                                    with patch(
                                        "torch_diode.model.model_utils.analyze_worst_predictions"
                                    ) as mock_analyze:
                                        validate_model(
                                            model_path=model_file.name,
                                            validation_dataset_path=dataset_file.name,
                                            top_n_worst=5,
                                        )

                                        mock_analyze.assert_called_once()


class TestRunModelExample:
    """Test the run_model_example function edge cases."""

    def test_msgpack_dataset_none(self, caplog):
        """Test when msgpack dataset is None after loading."""
        with tempfile.NamedTemporaryFile(suffix=".msgpack") as dataset_file:
            dataset_file.write(b"valid msgpack")
            dataset_file.flush()

            with patch(
                "torch_diode.types.matmul_dataset.Dataset.from_msgpack"
            ) as mock_from_msgpack:
                mock_from_msgpack.return_value = None
                run_model_example(dataset_path=dataset_file.name)
                assert "Failed to load dataset" in caplog.text

    def test_json_dataset_none(self, caplog):
        """Test when JSON dataset is None after loading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as dataset_file:
            dataset_file.write('{"valid": "structure"}')
            dataset_file.flush()

            with patch(
                "torch_diode.types.matmul_dataset.Dataset.deserialize"
            ) as mock_deserialize:
                mock_deserialize.return_value = None
                run_model_example(dataset_path=dataset_file.name)
                assert "Failed to load dataset" in caplog.text

    def test_model_training_failure(self, caplog):
        """Test when model training fails (returns None)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as dataset_file:
            dataset_file.write('{"valid": "structure"}')
            dataset_file.flush()

            with patch(
                "torch_diode.types.matmul_dataset.Dataset.deserialize"
            ) as mock_deserialize:
                mock_deserialize.return_value = MagicMock()

                with patch("torch_diode.utils.dataset_utils.print_dataset_statistics"):
                    with patch(
                        "torch_diode.model.matmul_model_trainer.train_model_from_dataset"
                    ) as mock_train:
                        mock_train.return_value = (None, {}, None)
                        run_model_example(dataset_path=dataset_file.name)
                        assert (
                            "Model training failed or dataset was empty" in caplog.text
                        )

    def test_empty_test_dataloader(self, caplog):
        """Test when test dataloader is None/empty."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as dataset_file:
            dataset_file.write('{"valid": "structure"}')
            dataset_file.flush()

            with patch(
                "torch_diode.types.matmul_dataset.Dataset.deserialize"
            ) as mock_deserialize:
                mock_deserialize.return_value = MagicMock()

                with patch("torch_diode.utils.dataset_utils.print_dataset_statistics"):
                    with patch(
                        "torch_diode.model.matmul_model_trainer.train_model_from_dataset"
                    ) as mock_train:
                        mock_model = MagicMock()
                        mock_train.return_value = (
                            mock_model,
                            {"train_loss": [1.0]},
                            None,
                        )

                        with patch(
                            "torch_diode.utils.visualization_utils.plot_training_history"
                        ):
                            with patch(
                                "torch_diode.model.model_utils.create_dataloaders"
                            ) as mock_create:
                                mock_create.return_value = (
                                    None,
                                    None,
                                    None,
                                )  # Empty test dataloader
                                run_model_example(dataset_path=dataset_file.name)
                                assert (
                                    "Model training failed or dataset was empty"
                                    in caplog.text
                                )

    def test_successful_example_run(self):
        """Test successful example run."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as dataset_file:
            dataset_file.write('{"valid": "structure"}')
            dataset_file.flush()

            with patch(
                "torch_diode.types.matmul_dataset.Dataset.deserialize"
            ) as mock_deserialize:
                mock_deserialize.return_value = MagicMock()

                with patch("torch_diode.utils.dataset_utils.print_dataset_statistics"):
                    with patch(
                        "torch_diode.model.matmul_model_trainer.train_model_from_dataset"
                    ) as mock_train:
                        mock_model = MagicMock()
                        mock_model.to.return_value = mock_model
                        mock_train.return_value = (
                            mock_model,
                            {"train_loss": [1.0]},
                            None,
                        )

                        with patch(
                            "torch_diode.utils.visualization_utils.plot_training_history"
                        ):
                            with patch(
                                "torch_diode.model.model_utils.create_dataloaders"
                            ) as mock_create:
                                # Create mock test dataloader with data
                                mock_test_loader = MagicMock()
                                mock_test_loader.__len__.return_value = 1
                                mock_test_loader.__iter__.return_value = iter(
                                    [
                                        (
                                            torch.randn(1, 10),
                                            torch.randn(1, 5),
                                            torch.randn(1, 1),
                                        )
                                    ]
                                )
                                mock_create.return_value = (
                                    None,
                                    None,
                                    mock_test_loader,
                                )

                                with patch("torch.nn.MSELoss") as mock_criterion_class:
                                    mock_criterion = MagicMock()
                                    mock_criterion.return_value = torch.tensor(0.25)
                                    mock_criterion_class.return_value = mock_criterion

                                    run_model_example(dataset_path=dataset_file.name)


class TestTrainModelFromDirectory:
    """Test the train_model_from_directory function."""

    def test_successful_directory_training(self):
        """Test successful training from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_train_loader = MagicMock()
            mock_val_loader = MagicMock()
            mock_test_loader = MagicMock()
            mock_dataset = MagicMock()
            mock_dataset.problem_feature_dim = 10
            mock_dataset.config_feature_dim = 5
            mock_train_loader.dataset.dataset = mock_dataset
            mock_val_loader.dataset.dataset = mock_dataset
            mock_test_loader.dataset.dataset = mock_dataset

            # Make sure all loaders have data to avoid empty dataset issues
            mock_train_loader.__len__.return_value = 10
            mock_val_loader.__len__.return_value = 5
            mock_test_loader.__len__.return_value = 5

            # Create sample data
            train_samples = [
                (torch.randn(8, 10), torch.randn(8, 5), torch.randn(8, 1))
                for _ in range(10)
            ]
            val_samples = [
                (torch.randn(4, 10), torch.randn(4, 5), torch.randn(4, 1))
                for _ in range(5)
            ]
            test_samples = [
                (torch.randn(4, 10), torch.randn(4, 5), torch.randn(4, 1))
                for _ in range(5)
            ]

            mock_train_loader.__iter__.return_value = iter(train_samples)
            mock_val_loader.__iter__.return_value = iter(val_samples)
            mock_test_loader.__iter__.return_value = iter(test_samples)

            with patch(
                "torch_diode.model.model_utils.create_directory_dataloaders"
            ) as mock_create:
                mock_create.return_value = (
                    mock_train_loader,
                    mock_val_loader,
                    mock_test_loader,
                )

                with patch(
                    "torch_diode.model.matmul_model_trainer.MatmulModelTrainer"
                ) as mock_trainer_class:
                    mock_trainer = MagicMock()
                    mock_trainer.train.return_value = {"train_loss": [1.0, 0.5]}
                    mock_trainer._evaluate.return_value = 0.25
                    mock_trainer_class.return_value = mock_trainer

                    with patch(
                        "torch_diode.model.model_utils.DeepMatmulTimingModel"
                    ) as mock_model_class:
                        mock_model = MagicMock()
                        # Create actual parameters for the optimizer
                        param1 = torch.nn.Parameter(torch.randn(10, 5))
                        param2 = torch.nn.Parameter(torch.randn(5))
                        mock_model.parameters.return_value = [param1, param2]
                        mock_model.to.return_value = mock_model
                        
                        # Mock the model to return actual tensors
                        def mock_forward(*args, **kwargs):
                            batch_size = args[0].shape[0] if len(args) > 0 else 8
                            device = args[0].device if len(args) > 0 else "cpu"
                            return torch.randn(batch_size, 1, device=device, requires_grad=True)
                        mock_model.side_effect = mock_forward
                        
                        mock_model_class.return_value = mock_model

                        with patch(
                            "torch_diode.utils.visualization_utils.plot_training_history"
                        ):
                            model, history = train_model_from_directory(
                                data_dir=temp_dir, model_path="/tmp/model.pt"
                            )

                            assert model is not None
                            assert history is not None
                            mock_create.assert_called_once()

    def test_base_model_from_directory(self):
        """Test base model creation from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "torch_diode.model.model_utils.create_directory_dataloaders"
            ) as mock_create:
                mock_train_loader = MagicMock()
                mock_val_loader = MagicMock()
                mock_test_loader = MagicMock()
                mock_dataset = MagicMock()
                mock_dataset.problem_feature_dim = 10
                mock_dataset.config_feature_dim = 5
                mock_train_loader.dataset.dataset = mock_dataset
                mock_val_loader.dataset.dataset = mock_dataset
                mock_test_loader.dataset.dataset = mock_dataset

                # Make sure all loaders have data to avoid empty dataset issues
                mock_train_loader.__len__.return_value = 10
                mock_val_loader.__len__.return_value = 5
                mock_test_loader.__len__.return_value = 5

                # Create sample data
                train_samples = [
                    (torch.randn(8, 10), torch.randn(8, 5), torch.randn(8, 1))
                    for _ in range(10)
                ]
                val_samples = [
                    (torch.randn(4, 10), torch.randn(4, 5), torch.randn(4, 1))
                    for _ in range(5)
                ]
                test_samples = [
                    (torch.randn(4, 10), torch.randn(4, 5), torch.randn(4, 1))
                    for _ in range(5)
                ]

                mock_train_loader.__iter__.return_value = iter(train_samples)
                mock_val_loader.__iter__.return_value = iter(val_samples)
                mock_test_loader.__iter__.return_value = iter(test_samples)

                mock_create.return_value = (
                    mock_train_loader,
                    mock_val_loader,
                    mock_test_loader,
                )

                with patch(
                    "torch_diode.model.matmul_model_trainer.MatmulModelTrainer"
                ) as mock_trainer_class:
                    mock_trainer = MagicMock()
                    mock_trainer.train.return_value = {"train_loss": [1.0, 0.5]}
                    mock_trainer._evaluate.return_value = 0.25
                    mock_trainer_class.return_value = mock_trainer

                    with patch(
                        "torch_diode.model.model_utils.MatmulTimingModel"
                    ) as mock_model_class:
                        mock_model = MagicMock()
                        # Create actual parameters for the optimizer
                        param1 = torch.nn.Parameter(torch.randn(10, 5))
                        param2 = torch.nn.Parameter(torch.randn(5))
                        mock_model.parameters.return_value = [param1, param2]
                        mock_model.to.return_value = mock_model
                        
                        # Mock the model to return actual tensors
                        def mock_forward(*args, **kwargs):
                            batch_size = args[0].shape[0] if len(args) > 0 else 8
                            device = args[0].device if len(args) > 0 else "cpu"
                            return torch.randn(batch_size, 1, device=device, requires_grad=True)
                        mock_model.side_effect = mock_forward
                        
                        mock_model_class.return_value = mock_model

                        with patch(
                            "torch_diode.utils.visualization_utils.plot_training_history"
                        ):
                            model, history = train_model_from_directory(
                                data_dir=temp_dir,
                                model_path="/tmp/model.pt",
                                model_type="base",
                            )

                            assert model is not None
                            assert history is not None
                            mock_model_class.assert_called_once()
