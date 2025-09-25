"""
Tests for the _debug_data_quality method in MatmulTimingDataset.
"""

from collections import OrderedDict

# Enable debug flags for testing
try:
    from torch_diode.utils.debug_config import set_debug_flag

    set_debug_flag("ENABLE_TYPE_ASSERTS", True)
except ImportError:
    pass  # In case debug_config is not available yet
from unittest.mock import patch

import torch

from torch_diode.model.matmul_dataset_loader import MatmulTimingDataset
from torch_diode.types.matmul_dataset import (
    Dataset,
    DatasetHardware,
    DatasetOperation,
    DatasetSolution,
    TimedConfig,
)
from torch_diode.types.matmul_types import MMShape, TritonGEMMConfig


class TestDebugDataQuality:
    """Test the _debug_data_quality method with various data quality issues."""

    def create_mock_dataset(self, timing_values):
        """Create a mock dataset with specified timing values."""
        # Create mock problem
        problem = MMShape(
            B=1,
            M=64,
            N=32,
            K=16,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1, 64, 32),
            out_stride=(2048, 32, 1),
        )

        # Create mock configs and timed configs
        timed_configs = []
        for i, time_val in enumerate(timing_values):
            config = TritonGEMMConfig(
                name=f"test_config_{i}",
                grid=1,
                block_m=16,
                block_n=16,
                block_k=16,
                group_m=1,
                num_stages=2,
                num_warps=4,
            )
            timed_config = TimedConfig(config=config, time=time_val)
            timed_configs.append(timed_config)

        # Create dataset structure
        solution = DatasetSolution(timed_configs=timed_configs)
        operation = DatasetOperation(solution=OrderedDict({problem: solution}))
        hardware = DatasetHardware(operation=OrderedDict({"mm": operation}))
        dataset = Dataset(hardware=OrderedDict({"test_gpu": hardware}))

        return dataset

    def test_debug_data_quality_normal_data(self, caplog):
        """Test _debug_data_quality with normal, clean data."""
        import logging

        caplog.set_level(logging.INFO)

        timing_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        dataset = self.create_mock_dataset(timing_values)

        matmul_dataset = MatmulTimingDataset(
            dataset=dataset, hardware_name="test_gpu", debug=True
        )

        # With log transform enabled, the timing values will be negative (log of < 1)
        # So we expect warnings about negative/zero timing locations
        warnings = [
            record for record in caplog.records if record.levelname == "WARNING"
        ]

        # Check that info logs were created
        info_logs = [record for record in caplog.records if record.levelname == "INFO"]
        assert any("DATA QUALITY DEBUG" in record.message for record in info_logs)
        assert any(
            "0 NaN values, 0 Inf values" in record.message for record in info_logs
        )

        # Check that we got the expected number of samples
        assert len(matmul_dataset.timings) == 5

    def test_debug_data_quality_nan_values(self, caplog):
        """Test _debug_data_quality with NaN values in timing data."""
        timing_values = [0.1, float("nan"), 0.3, float("nan"), 0.5]
        dataset = self.create_mock_dataset(timing_values)

        matmul_dataset = MatmulTimingDataset(
            dataset=dataset, hardware_name="test_gpu", debug=True
        )

        # Check that warnings were logged for NaN values
        warnings = [
            record for record in caplog.records if record.levelname == "WARNING"
        ]
        nan_warnings = [w for w in warnings if "Non-finite timing value" in w.message]
        assert len(nan_warnings) == 2  # Two NaN values should be skipped

        # Check that the dataset only contains valid timing values
        assert len(matmul_dataset.timings) == 3  # Only 3 valid values should remain

    def test_debug_data_quality_inf_values(self, caplog):
        """Test _debug_data_quality with infinity values in timing data."""
        timing_values = [0.1, float("inf"), 0.3, float("-inf"), 0.5]
        dataset = self.create_mock_dataset(timing_values)

        matmul_dataset = MatmulTimingDataset(
            dataset=dataset, hardware_name="test_gpu", debug=True
        )

        # Check that warnings were logged for Inf values
        warnings = [
            record for record in caplog.records if record.levelname == "WARNING"
        ]
        # Positive infinity should be caught by "Non-finite timing value" check
        inf_warnings = [w for w in warnings if "Non-finite timing value" in w.message]
        # Negative infinity should be caught by "Invalid timing value" check
        invalid_warnings = [w for w in warnings if "Invalid timing value" in w.message]

        assert len(inf_warnings) == 1  # Only positive inf caught by non-finite check
        assert len(invalid_warnings) == 1  # Negative inf caught by invalid check

        # Check that the dataset only contains valid timing values
        assert len(matmul_dataset.timings) == 3  # Only 3 valid values should remain

    def test_debug_data_quality_negative_zero_values(self, caplog):
        """Test _debug_data_quality with negative and zero timing values."""
        timing_values = [0.1, -0.1, 0.0, -5.0, 0.5]
        dataset = self.create_mock_dataset(timing_values)

        matmul_dataset = MatmulTimingDataset(
            dataset=dataset, hardware_name="test_gpu", debug=True
        )

        # Check that warnings were logged for invalid timing values
        warnings = [
            record for record in caplog.records if record.levelname == "WARNING"
        ]
        invalid_warnings = [w for w in warnings if "Invalid timing value" in w.message]
        assert len(invalid_warnings) == 3  # Three invalid values should be skipped

        # Check that the dataset only contains valid timing values
        assert len(matmul_dataset.timings) == 2  # Only 2 valid values should remain

    def test_debug_data_quality_mixed_issues(self, caplog):
        """Test _debug_data_quality with mixed data quality issues."""
        timing_values = [0.1, float("nan"), -0.1, float("inf"), 0.0, 0.5, float("-inf")]
        dataset = self.create_mock_dataset(timing_values)

        matmul_dataset = MatmulTimingDataset(
            dataset=dataset, hardware_name="test_gpu", debug=True
        )

        # Check that appropriate warnings were logged
        warnings = [
            record for record in caplog.records if record.levelname == "WARNING"
        ]

        # Should have warnings for various issues
        nan_inf_warnings = [
            w for w in warnings if "Non-finite timing value" in w.message
        ]
        invalid_warnings = [w for w in warnings if "Invalid timing value" in w.message]

        # Only positive inf is caught by non-finite, everything else by invalid check
        assert len(nan_inf_warnings) == 2  # nan, inf
        assert len(invalid_warnings) == 3  # -0.1, 0.0, -inf

        # Check that the dataset only contains valid timing values
        assert len(matmul_dataset.timings) == 2  # Only 2 valid values should remain

    def test_debug_data_quality_empty_tensors(self, caplog):
        """Test _debug_data_quality with empty dataset."""
        # Create an empty dataset
        hardware = DatasetHardware(operation=OrderedDict())
        dataset = Dataset(hardware=OrderedDict({"empty_gpu": hardware}))

        matmul_dataset = MatmulTimingDataset(
            dataset=dataset, hardware_name="empty_gpu", debug=True
        )

        # Check that warnings were logged for empty tensors
        warnings = [
            record for record in caplog.records if record.levelname == "WARNING"
        ]
        empty_warnings = [w for w in warnings if "tensor is empty" in w.message]
        assert len(empty_warnings) >= 1  # Should warn about empty tensors

    def test_debug_data_quality_log_transform_issues(self, caplog):
        """Test _debug_data_quality with log transform producing NaN/Inf."""
        # Include a very small positive value that might cause issues with log transform
        timing_values = [1e-20, 0.1, 0.2]
        dataset = self.create_mock_dataset(timing_values)

        matmul_dataset = MatmulTimingDataset(
            dataset=dataset,
            hardware_name="test_gpu",
            log_transform=True,
            debug=True,  # Enable log transform
        )

        # Check that the dataset was created (log transform should handle small values)
        assert len(matmul_dataset.timings) >= 1

    @patch("torch_diode.model.matmul_dataset_loader.logger")
    def test_debug_data_quality_feature_nan_detection(self, mock_logger):
        """Test _debug_data_quality detection of NaN values in features."""
        timing_values = [0.1, 0.2, 0.3]
        dataset = self.create_mock_dataset(timing_values)

        # Create dataset and manually inject NaN values
        matmul_dataset = MatmulTimingDataset(
            dataset=dataset,
            hardware_name="test_gpu",
            debug=False,  # Don't trigger debug in constructor
        )

        # Manually inject NaN values into features for testing
        matmul_dataset.problem_features[0][0] = float("nan")
        matmul_dataset.config_features[1][0] = float("nan")

        # Now call debug method
        matmul_dataset._debug_data_quality()

        # Verify that NaN detection was logged
        mock_logger.warning.assert_called()
        warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if "NaN locations" in str(call)
        ]
        assert len(warning_calls) >= 1

    @patch("torch_diode.model.matmul_dataset_loader.logger")
    def test_debug_data_quality_extreme_values(self, mock_logger):
        """Test _debug_data_quality detection of extreme values."""
        timing_values = [0.1, 0.2, 0.3]
        dataset = self.create_mock_dataset(timing_values)

        # Create dataset
        matmul_dataset = MatmulTimingDataset(
            dataset=dataset,
            hardware_name="test_gpu",
            debug=False,  # Don't trigger debug in constructor
        )

        # Manually inject extreme values for testing
        matmul_dataset.problem_features[0][0] = 1e10  # Extreme high value
        matmul_dataset.config_features[0][0] = 2e6  # Extreme high value

        # Now call debug method
        matmul_dataset._debug_data_quality()

        # Verify that extreme value detection was logged
        mock_logger.info.assert_called()
        info_calls = [
            call
            for call in mock_logger.info.call_args_list
            if "Extreme high values" in str(call)
        ]
        assert len(info_calls) >= 1

    def test_debug_data_quality_statistics_logging(self, caplog):
        """Test that _debug_data_quality logs proper statistics."""
        import logging

        caplog.set_level(logging.INFO)

        timing_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        dataset = self.create_mock_dataset(timing_values)

        matmul_dataset = MatmulTimingDataset(
            dataset=dataset, hardware_name="test_gpu", debug=True
        )

        # Check that statistics were logged
        info_logs = [record for record in caplog.records if record.levelname == "INFO"]

        # Should have min/max statistics for features and timings
        stats_logs = [
            log for log in info_logs if "min:" in log.message and "max:" in log.message
        ]
        # Should have at least some statistics logged (may not be exactly 3)
        assert len(stats_logs) >= 1

        # Verify that debug logging occurred
        debug_logs = [log for log in info_logs if "DATA QUALITY DEBUG" in log.message]
        assert len(debug_logs) >= 1

    def test_debug_data_quality_no_debug_flag(self, caplog):
        """Test that _debug_data_quality is not called when debug=False."""
        timing_values = [0.1, float("nan"), 0.3]  # Include NaN to test
        dataset = self.create_mock_dataset(timing_values)

        matmul_dataset = MatmulTimingDataset(
            dataset=dataset,
            hardware_name="test_gpu",
            debug=False,  # Debug disabled
        )

        # Check that no debug logs were created
        debug_logs = [
            record
            for record in caplog.records
            if "DATA QUALITY DEBUG" in record.message
        ]
        assert len(debug_logs) == 0

        # But NaN values should still be filtered out
        assert len(matmul_dataset.timings) == 2  # Only valid values remain

    def test_debug_data_quality_with_log_transform_disabled(self, caplog):
        """Test _debug_data_quality with log transform disabled."""
        import logging

        caplog.set_level(logging.INFO)

        timing_values = [0.1, 0.2, 0.3]
        dataset = self.create_mock_dataset(timing_values)

        matmul_dataset = MatmulTimingDataset(
            dataset=dataset,
            hardware_name="test_gpu",
            log_transform=False,
            debug=True,  # Disable log transform
        )

        # Check that debug ran successfully
        info_logs = [record for record in caplog.records if record.levelname == "INFO"]
        debug_logs = [log for log in info_logs if "DATA QUALITY DEBUG" in log.message]
        assert len(debug_logs) >= 1

        # Values should not be log-transformed but converted to tensor values
        # The timing values should be in their original form but as tensor values
        original_timings = torch.tensor(timing_values, dtype=torch.float32).reshape(
            -1, 1
        )
        # Use torch.allclose for floating point comparison
        assert torch.allclose(matmul_dataset.timings, original_timings, atol=1e-6)

    def test_create_mock_dataset_structure(self):
        """Test that our mock dataset creation works correctly."""
        timing_values = [0.1, 0.2, 0.3]
        dataset = self.create_mock_dataset(timing_values)

        # Verify dataset structure
        assert "test_gpu" in dataset.hardware
        assert "mm" in dataset.hardware["test_gpu"].operation

        solutions = dataset.hardware["test_gpu"].operation["mm"].solution
        assert len(solutions) == 1

        # Get the first (and only) solution
        solution = next(iter(solutions.values()))
        assert len(solution.timed_configs) == 3

        # Verify timing values
        times = [tc.time for tc in solution.timed_configs]
        assert times == timing_values

    def test_debug_data_quality_integration(self, caplog):
        """Integration test for _debug_data_quality with realistic problematic data."""
        import logging

        caplog.set_level(logging.INFO)

        # Create a dataset with various realistic issues
        timing_values = [
            0.001,  # Normal small value
            0.1,  # Normal value
            float("nan"),  # NaN from failed measurement
            float("inf"),  # Inf from division by zero
            -0.001,  # Negative time (measurement error)
            0.0,  # Zero time (measurement error)
            1e-10,  # Very small but valid time
            10.0,  # Large but valid time
        ]

        dataset = self.create_mock_dataset(timing_values)

        matmul_dataset = MatmulTimingDataset(
            dataset=dataset, hardware_name="test_gpu", log_transform=True, debug=True
        )

        # Should have 4 valid values: 0.001, 0.1, 1e-10, 10.0
        assert len(matmul_dataset.timings) == 4

        # Check that various issues were detected and logged
        warnings = [
            record for record in caplog.records if record.levelname == "WARNING"
        ]
        info_logs = [record for record in caplog.records if record.levelname == "INFO"]

        # Should have warnings for problematic values
        assert len(warnings) >= 3  # At least for nan, inf, negative/zero values

        # Should have comprehensive debug info
        debug_start = any(
            "=== DATA QUALITY DEBUG ===" in log.message for log in info_logs
        )
        debug_end = any(
            "=== END DATA QUALITY DEBUG ===" in log.message for log in info_logs
        )
        assert debug_start and debug_end
