"""
Simple tests for diode.collection.matmul_data_utils module to improve coverage.
"""


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

from torch_diode.collection.matmul_data_utils import (
    collect_data,
    create_validation_dataset,
    run_collector_example,
    run_matrix_multiplications,
)
from torch_diode.collection.matmul_dataset_collector import CollectionMode


class TestMatmulDataUtilsSimple:
    """Simple test class for matmul data utility functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_collector(self):
        """Create a mock collector."""
        mock_collector = Mock()
        mock_collector.get_dataset.return_value = Mock()
        mock_collector.collect_data = Mock()
        mock_collector.save_to_file = Mock()
        mock_collector.save_table_to_file = Mock()
        return mock_collector

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    @patch("torch_diode.collection.matmul_data_utils.print_dataset_statistics")
    def test_collect_data_basic_success(
        self, mock_print_stats, mock_device_name, mock_cuda_available
    ):
        """Test basic collect_data success."""
        mock_cuda_available.return_value = True
        mock_device_name.return_value = "NVIDIA A100"

        with patch(
            "torch_diode.collection.matmul_data_utils.MatmulDatasetCollector"
        ) as mock_collector_class:
            mock_collector = self._create_mock_collector()
            mock_collector_class.return_value = mock_collector

            output_file = os.path.join(self.temp_dir, "test_data.json")

            collect_data(output_file=output_file)

            # Verify collector was created
            mock_collector_class.assert_called_once()
            # Verify collect_data was called
            mock_collector.collect_data.assert_called_once()

    @patch("torch.cuda.is_available")
    @patch("torch_diode.collection.matmul_data_utils.print_dataset_statistics")
    def test_collect_data_cpu_fallback(self, mock_print_stats, mock_cuda_available):
        """Test collect_data with CPU fallback."""
        mock_cuda_available.return_value = False

        with patch(
            "torch_diode.collection.matmul_data_utils.MatmulDatasetCollector"
        ) as mock_collector_class:
            mock_collector = self._create_mock_collector()
            mock_collector_class.return_value = mock_collector

            output_file = os.path.join(self.temp_dir, "test_data.json")

            collect_data(output_file=output_file)

            # Verify collector was created with CPU
            args, kwargs = mock_collector_class.call_args
            assert kwargs["hardware_name"] == "cpu"

    @patch("torch.cuda.is_available")
    @patch("torch_diode.collection.matmul_data_utils.print_dataset_statistics")
    def test_collect_data_with_custom_parameters(
        self, mock_print_stats, mock_cuda_available
    ):
        """Test collect_data with custom parameters."""
        mock_cuda_available.return_value = False

        with patch(
            "torch_diode.collection.matmul_data_utils.MatmulDatasetCollector"
        ) as mock_collector_class:
            mock_collector = self._create_mock_collector()
            mock_collector_class.return_value = mock_collector

            output_file = os.path.join(self.temp_dir, "test_data.json")

            collect_data(
                output_file=output_file,
                mode="random",
                num_shapes=50,
                dtypes=[torch.float32, torch.float16],
                seed=123,
                min_size=64,
                max_size=2048,
                power_of_two=True,
                operations=["mm", "addmm"],
            )

            # Verify parameters were passed
            args, kwargs = mock_collector_class.call_args
            assert kwargs["mode"] == CollectionMode.RANDOM
            assert kwargs["num_shapes"] == 50
            assert kwargs["seed"] == 123
            assert kwargs["min_size"] == 64
            assert kwargs["max_size"] == 2048
            assert kwargs["power_of_two"] is True
            assert kwargs["operations"] == ["mm", "addmm"]

    @patch("torch_diode.collection.matmul_data_utils.print_dataset_statistics")
    def test_create_validation_dataset_basic(self, mock_print_stats):
        """Test basic create_validation_dataset."""
        with patch(
            "torch_diode.collection.matmul_data_utils.MatmulDatasetCollector"
        ) as mock_collector_class:
            mock_collector = self._create_mock_collector()
            mock_collector_class.return_value = mock_collector

            output_file = os.path.join(self.temp_dir, "validation.json")

            create_validation_dataset(output_file=output_file)

            # Verify collector was created
            mock_collector_class.assert_called_once()

    @patch("torch_diode.collection.matmul_data_utils.print_dataset_statistics")
    def test_create_validation_dataset_with_parameters(self, mock_print_stats):
        """Test create_validation_dataset with custom parameters."""
        with patch(
            "torch_diode.collection.matmul_data_utils.MatmulDatasetCollector"
        ) as mock_collector_class:
            mock_collector = self._create_mock_collector()
            mock_collector_class.return_value = mock_collector

            output_file = os.path.join(self.temp_dir, "validation.json")

            create_validation_dataset(
                output_file=output_file,
                mode="log_normal",
                num_shapes=25,
                dtypes=[torch.float64],
                seed=456,
                min_size=32,
                max_size=1024,
                power_of_two=False,
            )

            # Verify parameters were passed
            args, kwargs = mock_collector_class.call_args
            assert kwargs["mode"] == CollectionMode.LOG_NORMAL
            assert kwargs["num_shapes"] == 25
            assert kwargs["seed"] == 456

    @patch("torch.cuda.is_available")
    @patch("torch_diode.collection.matmul_data_utils.print_dataset_statistics")
    def test_run_collector_example_basic(self, mock_print_stats, mock_cuda_available):
        """Test basic run_collector_example."""
        mock_cuda_available.return_value = True

        with patch("torch.cuda.get_device_name") as mock_device_name:
            mock_device_name.return_value = "NVIDIA RTX 3080"

            with patch(
                "torch_diode.collection.matmul_data_utils.MatmulDatasetCollector"
            ) as mock_collector_class:
                with patch(
                    "torch_diode.collection.matmul_data_utils.run_matrix_multiplications"
                ) as mock_run_mm:
                    mock_collector = self._create_mock_collector()
                    mock_collector_class.return_value = mock_collector

                    run_collector_example(
                        output_dir=self.temp_dir,
                        use_context_manager=False,
                        num_shapes=5,
                        dtypes=[torch.float32],
                    )

                    # Verify functions were called
                    mock_collector_class.assert_called_once()
                    mock_run_mm.assert_called_once()

    @patch("torch.randn")
    @patch("torch.compile")
    def test_run_matrix_multiplications_basic(self, mock_compile, mock_randn):
        """Test basic run_matrix_multiplications."""
        # Mock tensor creation
        mock_tensor_a = Mock()
        mock_tensor_b = Mock()
        mock_tensor_c = Mock()
        # Return different tensors for different calls
        mock_randn.side_effect = [
            mock_tensor_a,
            mock_tensor_b,
            mock_tensor_c,
        ] * 2  # 2 iterations for 2 dtypes

        # Mock compiled function that returns a mock result
        mock_result = Mock()
        mock_compiled_fn = Mock(return_value=mock_result)
        mock_compile.return_value = mock_compiled_fn

        sizes = [(32, 64, 128)]
        dtypes = [torch.float32]

        # This should not hang - properly mock the execution
        run_matrix_multiplications(sizes, dtypes, device="cpu")

        # Verify torch.compile was called (should be called twice: once for mm, once for addmm)
        assert mock_compile.call_count == 2
        # Verify the compiled functions were actually called
        assert mock_compiled_fn.call_count == 2

    @patch("torch_diode.collection.matmul_data_utils.print_dataset_statistics")
    def test_collect_data_with_none_operations(self, mock_print_stats):
        """Test collect_data when operations is None."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch(
                "torch_diode.collection.matmul_data_utils.MatmulDatasetCollector"
            ) as mock_collector_class:
                mock_collector = self._create_mock_collector()
                mock_collector_class.return_value = mock_collector

                output_file = os.path.join(self.temp_dir, "test_data.json")

                collect_data(output_file=output_file, operations=None)

                # Verify default operations were used
                args, kwargs = mock_collector_class.call_args
                assert kwargs["operations"] == ["mm", "addmm", "bmm"]

    @patch("torch_diode.collection.matmul_data_utils.print_dataset_statistics")
    def test_collect_data_with_none_dtypes_cuda(self, mock_print_stats):
        """Test collect_data when dtypes is None with CUDA."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA A100"):
                with patch(
                    "torch_diode.collection.matmul_data_utils.MatmulDatasetCollector"
                ) as mock_collector_class:
                    mock_collector = self._create_mock_collector()
                    mock_collector_class.return_value = mock_collector

                    output_file = os.path.join(self.temp_dir, "test_data.json")

                    collect_data(output_file=output_file, dtypes=None)

                    # Verify CUDA default dtypes were used
                    args, kwargs = mock_collector_class.call_args
                    assert kwargs["dtypes"] == [torch.float16, torch.bfloat16]

    @patch("torch_diode.collection.matmul_data_utils.print_dataset_statistics")
    def test_collect_data_with_none_dtypes_cpu(self, mock_print_stats):
        """Test collect_data when dtypes is None with CPU."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch(
                "torch_diode.collection.matmul_data_utils.MatmulDatasetCollector"
            ) as mock_collector_class:
                mock_collector = self._create_mock_collector()
                mock_collector_class.return_value = mock_collector

                output_file = os.path.join(self.temp_dir, "test_data.json")

                collect_data(output_file=output_file, dtypes=None)

                # Verify CPU default dtypes were used
                args, kwargs = mock_collector_class.call_args
                assert kwargs["dtypes"] == [torch.float32]

    @patch("torch_diode.collection.matmul_data_utils.print_dataset_statistics")
    def test_collect_data_operation_shape_set_mode(self, mock_print_stats):
        """Test collect_data with operation_shape_set mode."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch(
                "torch_diode.collection.matmul_data_utils.MatmulDatasetCollector"
            ) as mock_collector_class:
                mock_collector = self._create_mock_collector()
                mock_collector_class.return_value = mock_collector

                # Create a mock operation shape set
                mock_shape_set = Mock()

                output_file = os.path.join(self.temp_dir, "test_data.json")

                collect_data(
                    output_file=output_file,
                    mode="operation_shape_set",
                    operation_shape_set=mock_shape_set,
                )

                # Verify collector was created with operation_shape_set mode
                args, kwargs = mock_collector_class.call_args
                assert kwargs["mode"] == CollectionMode.OPERATION_SHAPE_SET
                assert kwargs["operation_shape_set"] == mock_shape_set

    @patch("torch_diode.collection.matmul_data_utils.print_dataset_statistics")
    def test_collect_data_log_normal_parameters(self, mock_print_stats):
        """Test collect_data with log normal parameters."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch(
                "torch_diode.collection.matmul_data_utils.MatmulDatasetCollector"
            ) as mock_collector_class:
                mock_collector = self._create_mock_collector()
                mock_collector_class.return_value = mock_collector

                output_file = os.path.join(self.temp_dir, "test_data.json")

                collect_data(
                    output_file=output_file,
                    mode="log_normal",
                    log_normal_m_mean=7.0,
                    log_normal_m_std=2.5,
                    log_normal_n_mean=6.0,
                    log_normal_n_std=1.8,
                    log_normal_k_mean=6.5,
                    log_normal_k_std=2.2,
                )

                # Verify log normal parameters were passed
                args, kwargs = mock_collector_class.call_args
                assert kwargs["log_normal_m_mean"] == 7.0
                assert kwargs["log_normal_m_std"] == 2.5
                assert kwargs["log_normal_n_mean"] == 6.0
                assert kwargs["log_normal_n_std"] == 1.8
                assert kwargs["log_normal_k_mean"] == 6.5
                assert kwargs["log_normal_k_std"] == 2.2
