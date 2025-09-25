"""
Tests for the validate_max_autotune function in model_utils.py.
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
from unittest.mock import Mock, patch

import torch

from torch_diode.model.model_utils import validate_max_autotune


class TestValidateMaxAutotune:
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_dummy_model_file(self):
        """Create a dummy model checkpoint file."""
        model_path = os.path.join(self.temp_dir, "model.pt")

        # Create a minimal checkpoint
        checkpoint = {
            "model_state_dict": {},
            "model_type": "deep",
            "problem_feature_dim": 4,
            "config_feature_dim": 6,
            "hidden_dim": 128,
            "num_layers": 10,
        }
        torch.save(checkpoint, model_path)
        return model_path

    def create_dummy_dataset_file(self):
        """Create a dummy dataset file."""
        dataset_path = os.path.join(self.temp_dir, "dataset.json")

        # Create minimal dataset JSON
        dataset_data = {"hardware": {}, "metadata": {"version": "1.0"}}
        with open(dataset_path, "w") as f:
            json.dump(dataset_data, f)
        return dataset_path

    def create_dummy_max_autotune_file(self, format_type="single"):
        """Create a dummy max-autotune solution file."""
        ma_path = os.path.join(self.temp_dir, "max_autotune.json")

        if format_type == "single":
            # Single solution format
            ma_data = {
                "config": [
                    {
                        "name": "test_config",
                        "grid": 1024,
                        "block_m": 64,
                        "block_n": 128,
                        "block_k": 32,
                        "group_m": 8,
                        "num_stages": 3,
                        "num_warps": 4,
                    }
                ]
            }
        else:
            # Per-op format
            ma_data = {
                "mm": {
                    "config": [
                        {
                            "name": "test_config",
                            "grid": 1024,
                            "block_m": 64,
                            "block_n": 128,
                            "block_k": 32,
                            "group_m": 8,
                            "num_stages": 3,
                            "num_warps": 4,
                        }
                    ]
                }
            }

        with open(ma_path, "w") as f:
            json.dump(ma_data, f)
        return ma_path

    def test_missing_model_file(self, caplog):
        """Test behavior when model file is missing."""
        nonexistent_model = "/nonexistent/model.pt"
        dataset_path = self.create_dummy_dataset_file()
        ma_path = self.create_dummy_max_autotune_file()

        validate_max_autotune(
            model_path=nonexistent_model,
            validation_dataset_path=dataset_path,
            max_autotune_solution_path=ma_path,
        )

        assert "Model not found" in caplog.text

    def test_missing_dataset_file(self, caplog):
        """Test behavior when dataset file is missing."""
        model_path = self.create_dummy_model_file()
        nonexistent_dataset = "/nonexistent/dataset.json"
        ma_path = self.create_dummy_max_autotune_file()

        validate_max_autotune(
            model_path=model_path,
            validation_dataset_path=nonexistent_dataset,
            max_autotune_solution_path=ma_path,
        )

        assert "Validation dataset not found" in caplog.text

    def test_missing_max_autotune_file(self, caplog):
        """Test behavior when max-autotune file is missing."""
        model_path = self.create_dummy_model_file()
        dataset_path = self.create_dummy_dataset_file()
        nonexistent_ma = "/nonexistent/max_autotune.json"

        validate_max_autotune(
            model_path=model_path,
            validation_dataset_path=dataset_path,
            max_autotune_solution_path=nonexistent_ma,
        )

        assert "Max-autotune solution not found" in caplog.text

    def test_invalid_max_autotune_json(self, caplog):
        """Test behavior when max-autotune JSON is malformed."""
        model_path = self.create_dummy_model_file()
        dataset_path = self.create_dummy_dataset_file()

        # Create malformed JSON file
        ma_path = os.path.join(self.temp_dir, "bad_max_autotune.json")
        with open(ma_path, "w") as f:
            f.write("invalid json {")

        validate_max_autotune(
            model_path=model_path,
            validation_dataset_path=dataset_path,
            max_autotune_solution_path=ma_path,
        )

        assert "Failed to load max-autotune solution JSON" in caplog.text

    @patch("torch_diode.model.model_utils.create_directory_dataloaders")
    def test_directory_dataset_loading_failure(self, mock_create_dataloaders, caplog):
        """Test behavior when directory dataset loading fails."""
        model_path = self.create_dummy_model_file()
        dataset_dir = os.path.join(self.temp_dir, "dataset_dir")
        os.makedirs(dataset_dir)
        ma_path = self.create_dummy_max_autotune_file()

        # Mock the directory dataloader creation to raise an exception
        mock_create_dataloaders.side_effect = Exception("Mock directory loading error")

        validate_max_autotune(
            model_path=model_path,
            validation_dataset_path=dataset_dir,
            max_autotune_solution_path=ma_path,
        )

        assert "Failed to create dataloaders from directory" in caplog.text

    @patch("torch_diode.model.model_utils.MatmulDataset")
    def test_single_file_dataset_loading_failure(self, mock_dataset_class, caplog):
        """Test behavior when single file dataset loading fails."""
        model_path = self.create_dummy_model_file()
        dataset_path = self.create_dummy_dataset_file()
        ma_path = self.create_dummy_max_autotune_file()

        # Mock the dataset deserialization to return None
        mock_dataset_class.deserialize.return_value = None

        validate_max_autotune(
            model_path=model_path,
            validation_dataset_path=dataset_path,
            max_autotune_solution_path=ma_path,
        )

        assert "Failed to load validation dataset" in caplog.text

    @patch("torch_diode.model.model_utils.create_directory_dataloaders")
    @patch("torch_diode.model.model_utils.DeepMatmulTimingModel")
    def test_successful_directory_loading(
        self, mock_model_class, mock_create_dataloaders
    ):
        """Test successful directory-based dataset loading path."""
        model_path = self.create_dummy_model_file()
        dataset_dir = os.path.join(self.temp_dir, "dataset_dir")
        os.makedirs(dataset_dir)
        ma_path = self.create_dummy_max_autotune_file()

        # Mock successful dataloader creation
        mock_dataloader = Mock()
        mock_dataloader.dataset.dataset.problem_feature_dim = 4
        mock_dataloader.dataset.dataset.config_feature_dim = 6
        mock_dataloader.dataset.dataset.timing_dataset = Mock()
        mock_dataloader.dataset.dataset.timing_dataset.problem_features = torch.zeros(
            (0, 4)
        )
        mock_dataloader.dataset.dataset.timing_dataset.timings = torch.zeros(0)
        mock_dataloader.dataset.dataset.timing_dataset.configs = []
        mock_dataloader.dataset.dataset.timing_dataset.log_transform = True

        # Mock dataset structure for shape extraction
        mock_dataloader.dataset.dataset.dataset = Mock()
        mock_dataloader.dataset.dataset.dataset.hardware = {}

        mock_create_dataloaders.return_value = (None, mock_dataloader, None)

        # Mock model creation and loading
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        # This should complete without error
        validate_max_autotune(
            model_path=model_path,
            validation_dataset_path=dataset_dir,
            max_autotune_solution_path=ma_path,
        )

        # Verify dataloader was created for directory
        mock_create_dataloaders.assert_called_once()

    @patch("torch_diode.model.model_utils.create_dataloaders")
    @patch("torch_diode.model.model_utils.MatmulDataset")
    @patch("torch_diode.model.model_utils.DeepMatmulTimingModel")
    def test_msgpack_dataset_loading(
        self, mock_model_class, mock_dataset_class, mock_create_dataloaders
    ):
        """Test msgpack dataset loading path."""
        model_path = self.create_dummy_model_file()

        # Create a dummy msgpack file
        dataset_path = os.path.join(self.temp_dir, "dataset.msgpack")
        with open(dataset_path, "wb") as f:
            f.write(b"dummy msgpack data")

        ma_path = self.create_dummy_max_autotune_file()

        # Mock dataset loading
        mock_dataset = Mock()
        mock_dataset_class.from_msgpack.return_value = mock_dataset

        # Mock dataloader creation
        mock_dataloader = Mock()
        mock_dataloader.dataset.dataset.problem_feature_dim = 4
        mock_dataloader.dataset.dataset.config_feature_dim = 6
        mock_dataloader.dataset.dataset.timing_dataset = Mock()
        mock_dataloader.dataset.dataset.timing_dataset.problem_features = torch.zeros(
            (0, 4)
        )
        mock_dataloader.dataset.dataset.timing_dataset.timings = torch.zeros(0)
        mock_dataloader.dataset.dataset.timing_dataset.configs = []
        mock_dataloader.dataset.dataset.timing_dataset.log_transform = True

        # Mock dataset structure for shape extraction
        mock_dataloader.dataset.dataset.dataset = Mock()
        mock_dataloader.dataset.dataset.dataset.hardware = {}

        mock_create_dataloaders.return_value = (None, mock_dataloader, None)

        # Mock model creation
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        validate_max_autotune(
            model_path=model_path,
            validation_dataset_path=dataset_path,
            max_autotune_solution_path=ma_path,
        )

        # Verify msgpack loading was attempted
        mock_dataset_class.from_msgpack.assert_called_once()

    def test_per_op_max_autotune_format(self):
        """Test per-op max-autotune format parsing."""
        model_path = self.create_dummy_model_file()
        dataset_path = self.create_dummy_dataset_file()
        ma_path = self.create_dummy_max_autotune_file(format_type="per_op")

        # Mock dependencies to prevent full execution
        with patch(
            "torch_diode.model.model_utils.create_dataloaders"
        ) as mock_create_dataloaders:
            mock_dataloader = Mock()
            mock_dataloader.dataset.dataset.problem_feature_dim = 4
            mock_dataloader.dataset.dataset.config_feature_dim = 6
            mock_dataloader.dataset.dataset.timing_dataset = Mock()
            mock_dataloader.dataset.dataset.timing_dataset.problem_features = (
                torch.zeros((0, 4))
            )
            mock_dataloader.dataset.dataset.timing_dataset.timings = torch.zeros(0)
            mock_dataloader.dataset.dataset.timing_dataset.configs = []
            mock_dataloader.dataset.dataset.timing_dataset.log_transform = True
            mock_dataloader.dataset.dataset.dataset = Mock()
            mock_dataloader.dataset.dataset.dataset.hardware = {}

            mock_create_dataloaders.return_value = (None, mock_dataloader, None)

            with patch("torch_diode.model.model_utils.DeepMatmulTimingModel"):
                with patch(
                    "torch_diode.model.model_utils.MatmulDataset.deserialize"
                ) as mock_deserialize:
                    mock_deserialize.return_value = Mock()

                    # This should complete without error, testing per-op format parsing
                    validate_max_autotune(
                        model_path=model_path,
                        validation_dataset_path=dataset_path,
                        max_autotune_solution_path=ma_path,
                    )

    def test_base_model_loading(self):
        """Test loading of base model type."""
        # Create checkpoint with base model type
        model_path = os.path.join(self.temp_dir, "base_model.pt")
        checkpoint = {
            "model_state_dict": {},
            "model_type": "base",
            "problem_feature_dim": 4,
            "config_feature_dim": 6,
        }
        torch.save(checkpoint, model_path)

        dataset_path = self.create_dummy_dataset_file()
        ma_path = self.create_dummy_max_autotune_file()

        with patch(
            "torch_diode.model.model_utils.create_dataloaders"
        ) as mock_create_dataloaders:
            mock_dataloader = Mock()
            mock_dataloader.dataset.dataset.problem_feature_dim = 4
            mock_dataloader.dataset.dataset.config_feature_dim = 6
            mock_dataloader.dataset.dataset.timing_dataset = Mock()
            mock_dataloader.dataset.dataset.timing_dataset.problem_features = (
                torch.zeros((0, 4))
            )
            mock_dataloader.dataset.dataset.timing_dataset.timings = torch.zeros(0)
            mock_dataloader.dataset.dataset.timing_dataset.configs = []
            mock_dataloader.dataset.dataset.timing_dataset.log_transform = True
            mock_dataloader.dataset.dataset.dataset = Mock()
            mock_dataloader.dataset.dataset.dataset.hardware = {}

            mock_create_dataloaders.return_value = (None, mock_dataloader, None)

            with patch(
                "torch_diode.model.model_utils.MatmulTimingModel"
            ) as mock_base_model:
                with patch(
                    "torch_diode.model.model_utils.MatmulDataset.deserialize"
                ) as mock_deserialize:
                    mock_deserialize.return_value = Mock()
                    mock_model = Mock()
                    mock_base_model.return_value = mock_model

                    validate_max_autotune(
                        model_path=model_path,
                        validation_dataset_path=dataset_path,
                        max_autotune_solution_path=ma_path,
                    )

                    # Verify base model was created
                    mock_base_model.assert_called_once()

    def test_direct_state_dict_loading(self):
        """Test loading model from direct state dict (not checkpoint format)."""
        # Create model file with direct state dict
        model_path = os.path.join(self.temp_dir, "direct_model.pt")
        state_dict = {"layer.weight": torch.randn(10, 5)}
        torch.save(state_dict, model_path)

        dataset_path = self.create_dummy_dataset_file()
        ma_path = self.create_dummy_max_autotune_file()

        with patch(
            "torch_diode.model.model_utils.create_dataloaders"
        ) as mock_create_dataloaders:
            mock_dataloader = Mock()
            mock_dataloader.dataset.dataset.problem_feature_dim = 4
            mock_dataloader.dataset.dataset.config_feature_dim = 6
            mock_dataloader.dataset.dataset.timing_dataset = Mock()
            mock_dataloader.dataset.dataset.timing_dataset.problem_features = (
                torch.zeros((0, 4))
            )
            mock_dataloader.dataset.dataset.timing_dataset.timings = torch.zeros(0)
            mock_dataloader.dataset.dataset.timing_dataset.configs = []
            mock_dataloader.dataset.dataset.timing_dataset.log_transform = True
            mock_dataloader.dataset.dataset.dataset = Mock()
            mock_dataloader.dataset.dataset.dataset.hardware = {}

            mock_create_dataloaders.return_value = (None, mock_dataloader, None)

            with patch(
                "torch_diode.model.model_utils.DeepMatmulTimingModel"
            ) as mock_deep_model:
                with patch(
                    "torch_diode.model.model_utils.MatmulDataset.deserialize"
                ) as mock_deserialize:
                    mock_deserialize.return_value = Mock()
                    mock_model = Mock()
                    mock_deep_model.return_value = mock_model

                    validate_max_autotune(
                        model_path=model_path,
                        validation_dataset_path=dataset_path,
                        max_autotune_solution_path=ma_path,
                    )

                    # Verify deep model was created for direct state dict
                    mock_deep_model.assert_called_once()

    def test_hardware_and_op_filtering(self):
        """Test hardware_name and op_name filtering parameters."""
        model_path = self.create_dummy_model_file()
        dataset_path = self.create_dummy_dataset_file()
        ma_path = self.create_dummy_max_autotune_file()

        with patch(
            "torch_diode.model.model_utils.create_dataloaders"
        ) as mock_create_dataloaders:
            mock_dataloader = Mock()
            mock_dataloader.dataset.dataset.problem_feature_dim = 4
            mock_dataloader.dataset.dataset.config_feature_dim = 6
            mock_dataloader.dataset.dataset.timing_dataset = Mock()
            mock_dataloader.dataset.dataset.timing_dataset.problem_features = (
                torch.zeros((0, 4))
            )
            mock_dataloader.dataset.dataset.timing_dataset.timings = torch.zeros(0)
            mock_dataloader.dataset.dataset.timing_dataset.configs = []
            mock_dataloader.dataset.dataset.timing_dataset.log_transform = True
            mock_dataloader.dataset.dataset.dataset = Mock()
            mock_dataloader.dataset.dataset.dataset.hardware = {}

            mock_create_dataloaders.return_value = (None, mock_dataloader, None)

            with patch("torch_diode.model.model_utils.DeepMatmulTimingModel"):
                with patch(
                    "torch_diode.model.model_utils.MatmulDataset.deserialize"
                ) as mock_deserialize:
                    mock_deserialize.return_value = Mock()

                    validate_max_autotune(
                        model_path=model_path,
                        validation_dataset_path=dataset_path,
                        max_autotune_solution_path=ma_path,
                        hardware_name="test_hardware",
                        op_name="test_op",
                    )

                    # Verify hardware and op filtering was passed to dataloader creation
                    call_args = mock_create_dataloaders.call_args
                    assert call_args[1]["hardware_name"] == "test_hardware"
                    assert call_args[1]["op_name"] == "test_op"
