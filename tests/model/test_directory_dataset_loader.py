"""
Tests for diode.model.directory_dataset_loader module.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from diode.model.directory_dataset_loader import (
    _get_operations,
    _get_solutions,
    _is_cuda_available,
    _median_time_us,
    _pin_memory_default,
    _safe_getattr_or_dict,
    _safe_setattr_or_dict,
    _set_operations,
    _set_solutions,
    create_directory_dataloaders,
    DirectoryMatmulDataset,
    LazyDirectoryMatmulDataset,
)
from diode.types.matmul_dataset import Dataset as MatmulDataset
from torch.utils.data import DataLoader


class TestUtilityFunctions:
    """Test utility functions."""

    def test_is_cuda_available_true(self):
        """Test _is_cuda_available when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            assert _is_cuda_available() is True

    def test_is_cuda_available_false(self):
        """Test _is_cuda_available when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            assert _is_cuda_available() is False

    def test_is_cuda_available_exception(self):
        """Test _is_cuda_available when exception occurs."""
        with patch("torch.cuda.is_available", side_effect=Exception):
            assert _is_cuda_available() is False

    def test_pin_memory_default_true(self):
        """Test _pin_memory_default when CUDA is available."""
        with patch(
            "diode.model.directory_dataset_loader._is_cuda_available", return_value=True
        ):
            assert _pin_memory_default() is True

    def test_pin_memory_default_false(self):
        """Test _pin_memory_default when CUDA is not available."""
        with patch(
            "diode.model.directory_dataset_loader._is_cuda_available",
            return_value=False,
        ):
            assert _pin_memory_default() is False

    def test_safe_getattr_or_dict_with_dict(self):
        """Test _safe_getattr_or_dict with dictionary."""
        data = {"test_attr": "value"}
        assert _safe_getattr_or_dict(data, "test_attr", "default") == "value"
        assert _safe_getattr_or_dict(data, "missing", "default") == "default"

    def test_safe_getattr_or_dict_with_object(self):
        """Test _safe_getattr_or_dict with object."""

        # Test with a simple object that has the attribute
        class TestObj:
            def __init__(self):
                self.test_attr = "value"

        obj = TestObj()
        assert _safe_getattr_or_dict(obj, "test_attr", "default") == "value"
        assert _safe_getattr_or_dict(obj, "missing", "default") == "default"

    def test_safe_setattr_or_dict_with_dict(self):
        """Test _safe_setattr_or_dict with dictionary."""
        data = {}
        _safe_setattr_or_dict(data, "test_attr", "value")
        assert data["test_attr"] == "value"

    def test_safe_setattr_or_dict_with_object(self):
        """Test _safe_setattr_or_dict with object."""
        obj = Mock()
        _safe_setattr_or_dict(obj, "test_attr", "value")
        assert obj.test_attr == "value"

    def test_get_operations_dict(self):
        """Test _get_operations with dictionary."""
        hw_obj = {"operation": {"op1": "data"}}
        assert _get_operations(hw_obj) == {"op1": "data"}

    def test_get_operations_object(self):
        """Test _get_operations with object."""
        hw_obj = Mock()
        hw_obj.operation = {"op1": "data"}
        assert _get_operations(hw_obj) == {"op1": "data"}

    def test_set_operations_dict(self):
        """Test _set_operations with dictionary."""
        hw_obj = {}
        _set_operations(hw_obj, {"op1": "data"})
        assert hw_obj["operation"] == {"op1": "data"}

    def test_set_operations_object(self):
        """Test _set_operations with object."""
        hw_obj = Mock()
        _set_operations(hw_obj, {"op1": "data"})
        assert hw_obj.operation == {"op1": "data"}

    def test_get_solutions_dict(self):
        """Test _get_solutions with dictionary."""
        op_obj = {"solution": {"sol1": "data"}}
        assert _get_solutions(op_obj) == {"sol1": "data"}

    def test_get_solutions_object(self):
        """Test _get_solutions with object."""
        op_obj = Mock()
        op_obj.solution = {"sol1": "data"}
        assert _get_solutions(op_obj) == {"sol1": "data"}

    def test_set_solutions_dict(self):
        """Test _set_solutions with dictionary."""
        op_obj = {}
        _set_solutions(op_obj, {"sol1": "data"})
        assert op_obj["solution"] == {"sol1": "data"}

    def test_set_solutions_object(self):
        """Test _set_solutions with object."""
        op_obj = Mock()
        _set_solutions(op_obj, {"sol1": "data"})
        assert op_obj.solution == {"sol1": "data"}

    def test_median_time_us_dict_success(self):
        """Test _median_time_us with dictionary."""
        sol = {"stats": {"median_us": 10.5}}
        assert _median_time_us(sol) == 10.5

    def test_median_time_us_dict_missing_stats(self):
        """Test _median_time_us with dictionary missing stats."""
        sol = {"other": "data"}
        assert _median_time_us(sol) == float("inf")

    def test_median_time_us_dict_missing_median(self):
        """Test _median_time_us with dictionary missing median_us."""
        sol = {"stats": {"other": "data"}}
        assert _median_time_us(sol) == float("inf")

    def test_median_time_us_object_success(self):
        """Test _median_time_us with object."""
        sol = Mock()
        sol.stats = Mock()
        sol.stats.median_us = 10.5
        assert _median_time_us(sol) == 10.5

    def test_median_time_us_object_no_stats(self):
        """Test _median_time_us with object without stats."""
        sol = Mock()
        sol.stats = None
        assert _median_time_us(sol) == float("inf")

    def test_median_time_us_exception(self):
        """Test _median_time_us with exception."""
        sol = Mock()
        sol.stats = Mock()
        sol.stats.median_us = Mock(side_effect=Exception)
        assert _median_time_us(sol) == float("inf")


class TestDirectoryMatmulDataset:
    """Test DirectoryMatmulDataset class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = {
            "hardware": {
                "hw1": {
                    "operation": {
                        "op1": {"solution": {"sol1": {"stats": {"median_us": 10.0}}}}
                    }
                }
            }
        }

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_file(self, filename, data=None):
        """Create a test data file."""
        if data is None:
            data = self.sample_data

        file_path = Path(self.temp_dir) / filename
        if filename.endswith(".json"):
            with open(file_path, "w") as f:
                json.dump(data, f)
        elif filename.endswith(".msgpack"):
            # Mock msgpack file creation
            with open(file_path, "wb") as f:
                f.write(b"mock_msgpack_data")
        return str(file_path)

    def test_init_no_files(self):
        """Test initialization with no data files."""
        with pytest.raises(ValueError, match="No data files found"):
            DirectoryMatmulDataset(self.temp_dir)

    @patch("diode.model.directory_dataset_loader.MatmulTimingDataset")
    def test_init_basic(self, mock_timing_dataset):
        """Test basic initialization."""
        # Create a test file
        self.create_test_file("test.json")

        # Mock the MatmulDataset.deserialize method
        mock_dataset = Mock(spec=MatmulDataset)
        mock_dataset.hardware = self.sample_data["hardware"]

        with patch.object(MatmulDataset, "deserialize", return_value=mock_dataset):
            # Mock timing dataset
            mock_timing_dataset.return_value = Mock()
            mock_timing_dataset.return_value.__len__ = Mock(return_value=1)

            dataset = DirectoryMatmulDataset(self.temp_dir)

            assert dataset.data_dir == self.temp_dir
            assert len(dataset.data_files) == 1
            assert dataset.data_files[0].endswith("test.json")

    def test_find_data_files(self):
        """Test _find_data_files method."""
        # Create test files
        self.create_test_file("test1.json")
        self.create_test_file("test2.msgpack")
        Path(self.temp_dir, "test.txt").write_text("not a data file")

        with patch("diode.model.directory_dataset_loader.MatmulTimingDataset"):
            with patch.object(
                MatmulDataset, "deserialize", return_value=Mock(hardware={})
            ):
                with pytest.raises(ValueError):  # No valid datasets after filtering
                    DirectoryMatmulDataset(self.temp_dir)

    @patch("diode.model.directory_dataset_loader.MatmulTimingDataset")
    def test_load_single_file_json(self, mock_timing_dataset):
        """Test _load_single_file with JSON file."""
        file_path = self.create_test_file("test.json")

        mock_dataset = Mock(spec=MatmulDataset)
        mock_dataset.hardware = self.sample_data["hardware"]

        with patch.object(
            MatmulDataset, "deserialize", return_value=mock_dataset
        ) as mock_deserialize:
            mock_timing_dataset.return_value = Mock()
            mock_timing_dataset.return_value.__len__ = Mock(return_value=1)

            dataset = DirectoryMatmulDataset(self.temp_dir)
            result = dataset._load_single_file(file_path)

            # deserialize is called during init and during _load_single_file
            assert mock_deserialize.call_count >= 1

    @patch("diode.model.directory_dataset_loader.MatmulTimingDataset")
    def test_load_single_file_msgpack(self, mock_timing_dataset):
        """Test _load_single_file with MessagePack file."""
        file_path = self.create_test_file("test.msgpack")

        mock_dataset = Mock(spec=MatmulDataset)
        mock_dataset.hardware = self.sample_data["hardware"]

        with patch.object(
            MatmulDataset, "from_msgpack", return_value=mock_dataset
        ) as mock_from_msgpack:
            mock_timing_dataset.return_value = Mock()
            mock_timing_dataset.return_value.__len__ = Mock(return_value=1)

            dataset = DirectoryMatmulDataset(self.temp_dir)
            result = dataset._load_single_file(file_path)

            # from_msgpack is called during init and during _load_single_file
            assert mock_from_msgpack.call_count >= 1

    def test_load_single_file_unsupported(self):
        """Test _load_single_file with unsupported file."""
        dataset = Mock()  # Create a mock dataset object
        dataset._load_single_file = DirectoryMatmulDataset._load_single_file.__get__(
            dataset, DirectoryMatmulDataset
        )

        result = dataset._load_single_file("test.txt")
        assert result is None

    def test_load_single_file_exception(self):
        """Test _load_single_file with exception."""
        dataset = Mock()
        dataset._load_single_file = DirectoryMatmulDataset._load_single_file.__get__(
            dataset, DirectoryMatmulDataset
        )

        # File doesn't exist, should handle exception
        result = dataset._load_single_file("nonexistent.json")
        assert result is None

    def test_merge_datasets(self):
        """Test _merge_datasets method."""
        # Create two datasets to merge
        dataset_a = MatmulDataset(
            hardware={
                "hw1": {
                    "operation": {
                        "op1": {"solution": {"sol1": {"stats": {"median_us": 10.0}}}}
                    }
                }
            }
        )

        dataset_b = MatmulDataset(
            hardware={
                "hw1": {
                    "operation": {
                        "op1": {
                            "solution": {
                                "sol1": {
                                    "stats": {"median_us": 5.0}
                                },  # Faster solution
                                "sol2": {"stats": {"median_us": 15.0}},
                            }
                        }
                    }
                }
            }
        )

        # Create a real DirectoryMatmulDataset instance (but we won't fully initialize it)
        dataset = object.__new__(DirectoryMatmulDataset)
        result = dataset._merge_datasets(dataset_a, dataset_b)

        # Check that the faster solution was kept
        hw1_op1_sols = result.hardware["hw1"]["operation"]["op1"]["solution"]
        assert len(hw1_op1_sols) == 2
        assert hw1_op1_sols["sol1"]["stats"]["median_us"] == 5.0  # Faster one kept
        assert hw1_op1_sols["sol2"]["stats"]["median_us"] == 15.0

    @patch("diode.model.directory_dataset_loader.MatmulTimingDataset")
    def test_dataset_interface_methods(self, mock_timing_dataset):
        """Test dataset interface methods."""
        self.create_test_file("test.json")

        mock_dataset = Mock(spec=MatmulDataset)
        mock_dataset.hardware = self.sample_data["hardware"]

        mock_timing = Mock()
        mock_timing.__len__ = Mock(return_value=5)
        mock_timing.__getitem__ = Mock(return_value=("tensor1", "tensor2", "tensor3"))
        mock_timing.problem_feature_dim = 10
        mock_timing.config_feature_dim = 20
        mock_timing.configs = ["config1", "config2"]
        mock_timing_dataset.return_value = mock_timing

        with patch.object(MatmulDataset, "deserialize", return_value=mock_dataset):
            dataset = DirectoryMatmulDataset(self.temp_dir)

            assert len(dataset) == 5
            assert dataset[0] == ("tensor1", "tensor2", "tensor3")
            assert dataset.problem_feature_dim == 10
            assert dataset.config_feature_dim == 20
            assert dataset.configs == ["config1", "config2"]

            file_info = dataset.get_file_info()
            assert len(file_info) == 1
            assert file_info[0][0] == "test.json"
            assert file_info[0][1] == -1  # Sentinel value


class TestLazyDirectoryMatmulDataset:
    """Test LazyDirectoryMatmulDataset class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = {
            "hardware": {
                "hw1": {
                    "operation": {
                        "op1": {"solution": {"sol1": {"stats": {"median_us": 10.0}}}}
                    }
                }
            }
        }

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_file(self, filename, data=None):
        """Create a test data file."""
        if data is None:
            data = self.sample_data

        file_path = Path(self.temp_dir) / filename
        if filename.endswith(".json"):
            with open(file_path, "w") as f:
                json.dump(data, f)
        elif filename.endswith(".msgpack"):
            with open(file_path, "wb") as f:
                f.write(b"mock_msgpack_data")
        return str(file_path)

    def test_init_no_files(self):
        """Test initialization with no data files."""
        with pytest.raises(ValueError, match="No data files found"):
            LazyDirectoryMatmulDataset(self.temp_dir)

    @patch("diode.model.directory_dataset_loader.MatmulTimingDataset")
    def test_init_no_samples(self, mock_timing_dataset):
        """Test initialization with no samples after filtering."""
        self.create_test_file("test.json", {"hardware": {}})  # Empty hardware

        mock_dataset = Mock(spec=MatmulDataset)
        mock_dataset.hardware = {}

        with patch.object(MatmulDataset, "deserialize", return_value=mock_dataset):
            mock_timing_dataset.return_value = Mock()
            mock_timing_dataset.return_value.__len__ = Mock(return_value=0)

            with pytest.raises(ValueError, match="No samples after filtering"):
                LazyDirectoryMatmulDataset(self.temp_dir)

    @patch("diode.model.directory_dataset_loader.MatmulTimingDataset")
    def test_init_basic(self, mock_timing_dataset):
        """Test basic initialization."""
        self.create_test_file("test.json")

        mock_dataset = Mock(spec=MatmulDataset)
        mock_dataset.hardware = self.sample_data["hardware"]

        with patch.object(MatmulDataset, "deserialize", return_value=mock_dataset):
            mock_timing = Mock()
            mock_timing.__len__ = Mock(return_value=5)
            mock_timing_dataset.return_value = mock_timing

            dataset = LazyDirectoryMatmulDataset(self.temp_dir)

            assert dataset.data_dir == self.temp_dir
            assert len(dataset.data_files) == 1
            assert len(dataset) == 5

    @patch("diode.model.directory_dataset_loader.MatmulTimingDataset")
    def test_getitem(self, mock_timing_dataset):
        """Test __getitem__ method."""
        self.create_test_file("test.json")

        mock_dataset = Mock(spec=MatmulDataset)
        mock_dataset.hardware = self.sample_data["hardware"]

        mock_timing = Mock()
        mock_timing.__len__ = Mock(return_value=3)
        mock_timing.__getitem__ = Mock(return_value=("tensor1", "tensor2", "tensor3"))
        mock_timing_dataset.return_value = mock_timing

        with patch.object(MatmulDataset, "deserialize", return_value=mock_dataset):
            dataset = LazyDirectoryMatmulDataset(self.temp_dir)

            result = dataset[1]
            assert result == ("tensor1", "tensor2", "tensor3")

    @patch("diode.model.directory_dataset_loader.MatmulTimingDataset")
    def test_getitem_out_of_bounds(self, mock_timing_dataset):
        """Test __getitem__ with out of bounds index."""
        self.create_test_file("test.json")

        mock_dataset = Mock(spec=MatmulDataset)
        mock_dataset.hardware = self.sample_data["hardware"]

        mock_timing = Mock()
        mock_timing.__len__ = Mock(return_value=3)
        mock_timing_dataset.return_value = mock_timing

        with patch.object(MatmulDataset, "deserialize", return_value=mock_dataset):
            dataset = LazyDirectoryMatmulDataset(self.temp_dir)

            with pytest.raises(IndexError):
                dataset[5]

    @patch("diode.model.directory_dataset_loader.MatmulTimingDataset")
    def test_properties(self, mock_timing_dataset):
        """Test property methods."""
        self.create_test_file("test.json")

        mock_dataset = Mock(spec=MatmulDataset)
        mock_dataset.hardware = self.sample_data["hardware"]

        mock_timing = Mock()
        mock_timing.__len__ = Mock(return_value=3)
        mock_timing.problem_feature_dim = 10
        mock_timing.config_feature_dim = 20
        mock_timing.configs = ["config1"]
        mock_timing_dataset.return_value = mock_timing

        with patch.object(MatmulDataset, "deserialize", return_value=mock_dataset):
            dataset = LazyDirectoryMatmulDataset(self.temp_dir)

            assert dataset.problem_feature_dim == 10
            assert dataset.config_feature_dim == 20
            assert dataset.configs == ["config1"]

    @patch("diode.model.directory_dataset_loader.MatmulTimingDataset")
    def test_get_file_info(self, mock_timing_dataset):
        """Test get_file_info method."""
        self.create_test_file("test.json")

        mock_dataset = Mock(spec=MatmulDataset)
        mock_dataset.hardware = self.sample_data["hardware"]

        mock_timing = Mock()
        mock_timing.__len__ = Mock(return_value=3)
        mock_timing_dataset.return_value = mock_timing

        with patch.object(MatmulDataset, "deserialize", return_value=mock_dataset):
            dataset = LazyDirectoryMatmulDataset(self.temp_dir)

            file_info = dataset.get_file_info()
            assert len(file_info) == 1
            assert file_info[0][0] == "test.json"
            assert file_info[0][1] == 3


class TestCreateDirectoryDataloaders:
    """Test create_directory_dataloaders function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_invalid_ratios(self):
        """Test with invalid train/val ratios."""
        with pytest.raises(ValueError, match="train_ratio and val_ratio must be in"):
            create_directory_dataloaders(self.temp_dir, train_ratio=0.0)

        with pytest.raises(ValueError, match="train_ratio and val_ratio must be in"):
            create_directory_dataloaders(self.temp_dir, train_ratio=1.0)

        with pytest.raises(ValueError, match="train_ratio \\+ val_ratio must be < 1.0"):
            create_directory_dataloaders(self.temp_dir, train_ratio=0.8, val_ratio=0.3)

    @patch("diode.model.directory_dataset_loader.DirectoryMatmulDataset")
    def test_basic_creation_eager(self, mock_dataset_class):
        """Test basic dataloader creation with eager dataset."""
        # Mock dataset with proper __len__ implementation
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.get_file_info = Mock(return_value=[("test.json", 100)])
        mock_dataset_class.return_value = mock_dataset

        # Mock torch.utils.data.random_split with properly sized mock datasets
        mock_train_ds = Mock()
        mock_val_ds = Mock()
        mock_test_ds = Mock()

        # Add __len__ methods to the split datasets
        mock_train_ds.__len__ = Mock(return_value=80)
        mock_val_ds.__len__ = Mock(return_value=10)
        mock_test_ds.__len__ = Mock(return_value=10)

        with patch(
            "torch.utils.data.random_split",
            return_value=(mock_train_ds, mock_val_ds, mock_test_ds),
        ):
            train_dl, val_dl, test_dl = create_directory_dataloaders(
                self.temp_dir, batch_size=32, train_ratio=0.8, val_ratio=0.1
            )

            assert isinstance(train_dl, DataLoader)
            assert isinstance(val_dl, DataLoader)
            assert isinstance(test_dl, DataLoader)

    @patch("diode.model.directory_dataset_loader.LazyDirectoryMatmulDataset")
    def test_basic_creation_lazy(self, mock_dataset_class):
        """Test basic dataloader creation with lazy dataset."""
        # Mock dataset with proper __len__ implementation
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.get_file_info = Mock(return_value=[("test.json", 100)])
        mock_dataset_class.return_value = mock_dataset

        # Mock torch.utils.data.random_split with properly sized mock datasets
        mock_train_ds = Mock()
        mock_val_ds = Mock()
        mock_test_ds = Mock()

        # Add __len__ methods to the split datasets
        mock_train_ds.__len__ = Mock(return_value=80)
        mock_val_ds.__len__ = Mock(return_value=10)
        mock_test_ds.__len__ = Mock(return_value=10)

        with patch(
            "torch.utils.data.random_split",
            return_value=(mock_train_ds, mock_val_ds, mock_test_ds),
        ):
            train_dl, val_dl, test_dl = create_directory_dataloaders(
                self.temp_dir,
                batch_size=32,
                train_ratio=0.8,
                val_ratio=0.1,
                use_lazy=True,
            )

            assert isinstance(train_dl, DataLoader)
            assert isinstance(val_dl, DataLoader)
            assert isinstance(test_dl, DataLoader)

    @patch("diode.model.directory_dataset_loader.DirectoryMatmulDataset")
    @patch("torch.utils.data.DistributedSampler")
    def test_distributed_creation(self, mock_sampler_class, mock_dataset_class):
        """Test dataloader creation with distributed training."""
        # Mock dataset with proper __len__ implementation
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.get_file_info = Mock(return_value=[("test.json", 100)])
        mock_dataset_class.return_value = mock_dataset

        # Mock distributed sampler
        mock_sampler = Mock()
        mock_sampler_class.return_value = mock_sampler

        # Mock torch.utils.data.random_split with properly sized mock datasets
        mock_train_ds = Mock()
        mock_val_ds = Mock()
        mock_test_ds = Mock()

        # Add __len__ methods to the split datasets
        mock_train_ds.__len__ = Mock(return_value=80)
        mock_val_ds.__len__ = Mock(return_value=10)
        mock_test_ds.__len__ = Mock(return_value=10)

        with patch(
            "torch.utils.data.random_split",
            return_value=(mock_train_ds, mock_val_ds, mock_test_ds),
        ):
            # Patch DataLoader in the specific module where it's imported
            with patch("diode.model.directory_dataset_loader.DataLoader") as mock_dataloader:
                # Mock DataLoader to return our test instance
                mock_dataloader.return_value = Mock(spec=DataLoader)
                
                train_dl, val_dl, test_dl = create_directory_dataloaders(
                    self.temp_dir,
                    batch_size=32,
                    train_ratio=0.8,
                    val_ratio=0.1,
                    distributed=True,
                    rank=0,
                    world_size=2,
                )

                # Should create distributed samplers - check that DataLoader was called with samplers
                assert mock_dataloader.call_count == 3  # train, val, test
                # Verify that sampler was passed to DataLoader calls
                for call in mock_dataloader.call_args_list:
                    args, kwargs = call
                    assert 'sampler' in kwargs

    @patch("diode.model.directory_dataset_loader.DirectoryMatmulDataset")
    def test_invalid_split_sizes(self, mock_dataset_class):
        """Test with invalid split sizes."""
        # Mock dataset with very small size
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=2)  # Too small to split
        mock_dataset_class.return_value = mock_dataset

        with pytest.raises(ValueError, match="Split sizes invalid"):
            create_directory_dataloaders(
                self.temp_dir, batch_size=32, train_ratio=0.8, val_ratio=0.1
            )

    @patch("diode.model.directory_dataset_loader.DirectoryMatmulDataset")
    def test_file_info_exception(self, mock_dataset_class):
        """Test when get_file_info raises exception."""
        # Mock dataset with proper __len__ implementation
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.get_file_info = Mock(side_effect=Exception("Test error"))
        mock_dataset_class.return_value = mock_dataset

        # Mock torch.utils.data.random_split with properly sized mock datasets
        mock_train_ds = Mock()
        mock_val_ds = Mock()
        mock_test_ds = Mock()

        # Set proper lengths for the split datasets
        mock_train_ds.__len__ = Mock(return_value=80)
        mock_val_ds.__len__ = Mock(return_value=10)
        mock_test_ds.__len__ = Mock(return_value=10)

        with patch(
            "torch.utils.data.random_split",
            return_value=(mock_train_ds, mock_val_ds, mock_test_ds),
        ):
            # Should not raise exception, just log it
            train_dl, val_dl, test_dl = create_directory_dataloaders(
                self.temp_dir, batch_size=32, train_ratio=0.8, val_ratio=0.1
            )

            assert isinstance(train_dl, DataLoader)
