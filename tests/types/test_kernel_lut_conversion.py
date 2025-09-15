# Owner(s): ["module: diode"]
"""
Tests for graceful handling of malformed JSON in kernel LUT.
"""

import json
import logging
import os
import tempfile
import threading
from collections import OrderedDict
from io import StringIO

import torch
import unittest
from unittest import TestCase

from diode.types.matmul_types import (
    Hardware,
    MMShape,
    Operation,
    Solution,
    Table,
    TritonGEMMConfig,
)
from diode.types.kernel_lut import convert_triton_configs_to_gemm_configs


def run_tests():
    unittest.main()




class TestTritonConfigConversion(TestCase):
    """Tests for converting triton.runtime.autotuner.Config objects to TritonGEMMConfig."""

    def setUp(self):
        """Set up mock triton config objects for testing."""
        # Create mock triton config objects that mimic the structure shown in the debug output
        super().setUp()
        self.mock_triton_config1 = type(
            "MockTritonConfig",
            (),
            {
                "kwargs": {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 16, "GROUP_M": 8},
                "num_stages": 3,
                "num_warps": 4,
                "maxnreg": None,
                "num_buffers_warp_spec": None,
                "num_consumer_groups": None,
                "num_ctas": None,
                "pre_hook": None,
                "reg_dec_producer": None,
                "reg_inc_consumer": None,
            },
        )()

        self.mock_triton_config2 = type(
            "MockTritonConfig",
            (),
            {
                "kwargs": {
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_K": 32,
                    "GROUP_M": 16,
                    "EVEN_K": True,
                    "ALLOW_TF32": True,
                    "USE_FAST_ACCUM": False,
                    "ACC_TYPE": "tl.float16",
                },
                "num_stages": 2,
                "num_warps": 8,
            },
        )()

        self.mock_triton_config3 = type(
            "MockTritonConfig",
            (),
            {
                "kwargs": {
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "BLOCK_K": 64,
                    "GROUP_M": 32,
                },
                "num_stages": 4,
                "num_warps": 16,
            },
        )()

        # Config with minimal kwargs (testing defaults)
        self.mock_triton_config_minimal = type(
            "MockTritonConfig",
            (),
            {
                "kwargs": {},
                "num_stages": 1,
                "num_warps": 2,
            },
        )()

        # Config with missing attributes (testing robustness)
        self.mock_triton_config_incomplete = type(
            "MockTritonConfig",
            (),
            {
                "kwargs": {"BLOCK_M": 32, "BLOCK_N": 32},
                # Missing num_stages and num_warps
            },
        )()

        self.triton_configs = [self.mock_triton_config1, self.mock_triton_config2]
        self.gemm_configs = convert_triton_configs_to_gemm_configs(
            self.triton_configs, name_prefix="lut_test"
        )
        # Create a problem
        self.problem = MMShape(
            B=1024,
            M=1024,
            M_dtype=torch.float32,
            N=1024,
            K=512,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1024, 1024, 512),
            out_stride=(1024 * 512, 512, 1),
        )

        # Create a solution with converted configs
        self.solution = Solution(config=self.gemm_configs)

        # Create operation
        self.operation = Operation(
            solution=OrderedDict([(self.problem, self.solution)])
        )

        # Create hardware and table
        self.hardware = Hardware(operation=OrderedDict([("mm", self.operation)]))
        self.valid_table = Table(hardware=OrderedDict([("test_gpu", self.hardware)]))

    def test_convert_single_config(self):
        """Test converting a single triton config to TritonGEMMConfig."""
        triton_configs = [self.mock_triton_config1]
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        self.assertEqual(len(gemm_configs), 1)
        config = gemm_configs[0]

        # Check type
        self.assertIsInstance(config, TritonGEMMConfig)

        # Check values from kwargs
        self.assertEqual(config.block_m, 16)
        self.assertEqual(config.block_n, 32)
        self.assertEqual(config.block_k, 16)
        self.assertEqual(config.group_m, 8)

        # Check values from direct attributes
        self.assertEqual(config.num_stages, 3)
        self.assertEqual(config.num_warps, 4)

        # Check defaults
        self.assertEqual(config.grid, 1)
        self.assertFalse(config.EVEN_K)
        self.assertFalse(config.ALLOW_TF32)
        self.assertFalse(config.USE_FAST_ACCUM)
        self.assertEqual(config.ACC_TYPE, "tl.float32")

        # Check generated name
        self.assertEqual(config.name, "triton_config_0")

    def test_convert_configs_with_edge_case_values(self):
        """Test conversion with edge case values."""
        triton_configs = [
            self.mock_triton_config1,
            self.mock_triton_config2,
            self.mock_triton_config3,
        ]
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        for i, config in enumerate(gemm_configs):
            with self.subTest(config_index=i, config_name=config.name):
                self.assertEqual(len(gemm_configs), 3)

                if i == 0:
                    # Check first config
                    self.assertEqual(config.name, "triton_config_0")
                    self.assertEqual(config.block_m, 16)
                    self.assertEqual(config.block_n, 32)
                    self.assertEqual(config.num_stages, 3)
                    self.assertEqual(config.num_warps, 4)
                elif i == 1:
                    # Check second config (with optional parameters)
                    self.assertEqual(config.name, "triton_config_1")
                    self.assertEqual(config.block_m, 64)
                    self.assertEqual(config.block_n, 64)
                    self.assertEqual(config.block_k, 32)
                    self.assertEqual(config.group_m, 16)
                    self.assertEqual(config.num_stages, 2)
                    self.assertEqual(config.num_warps, 8)
                    self.assertTrue(config.EVEN_K)
                    self.assertTrue(config.ALLOW_TF32)
                    self.assertFalse(config.USE_FAST_ACCUM)
                    self.assertEqual(config.ACC_TYPE, "tl.float16")
                elif i == 2:
                    # Check third config
                    self.assertEqual(config.name, "triton_config_2")
                    self.assertEqual(config.block_m, 128)
                    self.assertEqual(config.block_n, 128)
                    self.assertEqual(config.num_stages, 4)
                    self.assertEqual(config.num_warps, 16)

    def test_convert_with_custom_name_prefix(self):
        """Test converting configs with custom name prefix."""
        triton_configs = [self.mock_triton_config1, self.mock_triton_config2]
        gemm_configs = convert_triton_configs_to_gemm_configs(
            triton_configs, name_prefix="custom_config"
        )

        self.assertEqual(len(gemm_configs), 2)
        self.assertEqual(gemm_configs[0].name, "custom_config_0")
        self.assertEqual(gemm_configs[1].name, "custom_config_1")

    def test_convert_empty_list(self):
        """Test converting empty list of triton configs."""
        triton_configs = []
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        self.assertEqual(len(gemm_configs), 0)
        self.assertIsInstance(gemm_configs, list)

    def test_convert_config_with_defaults(self):
        """Test converting config with minimal kwargs (testing default values)."""
        triton_configs = [self.mock_triton_config_minimal]
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        self.assertEqual(len(gemm_configs), 1)
        config = gemm_configs[0]

        # Should use default values for missing kwargs
        self.assertEqual(config.block_m, 64)  # Default fallback
        self.assertEqual(config.block_n, 64)  # Default fallback
        self.assertEqual(config.block_k, 32)  # Default fallback
        self.assertEqual(config.group_m, 8)  # Default fallback

        # Should use actual values from config object
        self.assertEqual(config.num_stages, 1)
        self.assertEqual(config.num_warps, 2)

    def test_convert_config_with_missing_attributes(self):
        """Test converting config with missing attributes (testing robustness)."""
        triton_configs = [self.mock_triton_config_incomplete]
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        self.assertEqual(len(gemm_configs), 1)
        config = gemm_configs[0]

        # Should use values from kwargs where available
        self.assertEqual(config.block_m, 32)
        self.assertEqual(config.block_n, 32)

        # Should use defaults for missing kwargs
        self.assertEqual(config.block_k, 32)  # Default fallback
        self.assertEqual(config.group_m, 8)  # Default fallback

        # Should use defaults for missing attributes
        self.assertEqual(config.num_stages, 2)  # Default fallback
        self.assertEqual(config.num_warps, 4)  # Default fallback

    def test_convert_config_all_optional_parameters(self):
        """Test that all optional parameters are correctly extracted."""
        # Create a config with all optional parameters set
        mock_config_full = type(
            "MockTritonConfig",
            (),
            {
                "kwargs": {
                    "BLOCK_M": 96,
                    "BLOCK_N": 96,
                    "BLOCK_K": 48,
                    "GROUP_M": 12,
                    "EVEN_K": True,
                    "ALLOW_TF32": False,
                    "USE_FAST_ACCUM": True,
                    "ACC_TYPE": "tl.bfloat16",
                },
                "num_stages": 5,
                "num_warps": 12,
            },
        )()

        triton_configs = [mock_config_full]
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        self.assertEqual(len(gemm_configs), 1)
        config = gemm_configs[0]

        # Check all parameters are correctly set
        self.assertEqual(config.block_m, 96)
        self.assertEqual(config.block_n, 96)
        self.assertEqual(config.block_k, 48)
        self.assertEqual(config.group_m, 12)
        self.assertEqual(config.num_stages, 5)
        self.assertEqual(config.num_warps, 12)
        self.assertTrue(config.EVEN_K)
        self.assertFalse(config.ALLOW_TF32)
        self.assertTrue(config.USE_FAST_ACCUM)
        self.assertEqual(config.ACC_TYPE, "tl.bfloat16")

    def test_converted_configs_are_serializable(self):
        """Test that converted configs can be serialized/deserialized."""
        triton_configs = [self.mock_triton_config1, self.mock_triton_config2]
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        # Test serialization of individual configs
        for config in gemm_configs:
            config_dict = config.to_dict()
            self.assertIsInstance(config_dict, OrderedDict)

            # Test round-trip serialization
            reconstructed = TritonGEMMConfig.from_dict(config_dict)
            self.assertEqual(config.name, reconstructed.name)
            self.assertEqual(config.block_m, reconstructed.block_m)
            self.assertEqual(config.block_n, reconstructed.block_n)
            self.assertEqual(config.block_k, reconstructed.block_k)
            self.assertEqual(config.group_m, reconstructed.group_m)
            self.assertEqual(config.num_stages, reconstructed.num_stages)
            self.assertEqual(config.num_warps, reconstructed.num_warps)

    def test_converted_configs_in_kernel_lut_structures(self):
        """Test that converted configs work within kernel LUT structures."""

        # Test lookup functionality
        result = self.valid_table.lookup("test_gpu", "mm", self.problem)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "lut_test_0")
        self.assertEqual(result[1].name, "lut_test_1")

        # Test serialization of the complete structure
        serialized = self.valid_table.serialize()
        self.assertIsInstance(serialized, str)

        # Test deserialization
        reconstructed_table = Table.deserialize(serialized)
        reconstructed_result = reconstructed_table.lookup(
            "test_gpu", "mm", self.problem
        )
        self.assertIsNotNone(reconstructed_result)
        self.assertEqual(len(reconstructed_result), 2)

    def test_convert_configs_preserve_uniqueness(self):
        """Test that converted configs maintain uniqueness for hashing."""
        # Create real triton configs for this test
        import triton
        triton_config1 = triton.runtime.autotuner.Config(
            kwargs={"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 16, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        )
        triton_config2 = triton.runtime.autotuner.Config(
            kwargs={"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 16},
            num_stages=2,
            num_warps=8,
        )

        triton_configs = [triton_config1, triton_config2, triton_config1]
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        self.assertEqual(len(gemm_configs), 3)

        # Even though two configs have the same source, they should have different names
        self.assertEqual(gemm_configs[0].name, "triton_config_0")
        self.assertEqual(gemm_configs[1].name, "triton_config_1")
        self.assertEqual(gemm_configs[2].name, "triton_config_2")

        # But the first and third should have the same parameters (except name)
        self.assertEqual(gemm_configs[0].block_m, gemm_configs[2].block_m)
        self.assertEqual(gemm_configs[0].block_n, gemm_configs[2].block_n)
        self.assertEqual(gemm_configs[0].num_stages, gemm_configs[2].num_stages)
        self.assertEqual(gemm_configs[0].num_warps, gemm_configs[2].num_warps)

        # Test that they can be used in sets (hash works)
        config_set = set(gemm_configs)
        self.assertEqual(
            len(config_set), 3
        )  # All should be unique due to different names

    def test_convert_configs_with_edge_case_values_(self):
        """Test conversion with edge case values."""
        # Config with very large values
        mock_config_large = type(
            "MockTritonConfig",
            (),
            {
                "kwargs": {
                    "BLOCK_M": 512,
                    "BLOCK_N": 512,
                    "BLOCK_K": 256,
                    "GROUP_M": 64,
                },
                "num_stages": 10,
                "num_warps": 32,
            },
        )()

        # Config with very small values
        mock_config_small = type(
            "MockTritonConfig",
            (),
            {
                "kwargs": {"BLOCK_M": 8, "BLOCK_N": 8, "BLOCK_K": 4, "GROUP_M": 1},
                "num_stages": 1,
                "num_warps": 1,
            },
        )()

        triton_configs = [mock_config_large, mock_config_small]
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        self.assertEqual(len(gemm_configs), 2)

        # Check large values
        large_config = gemm_configs[0]
        self.assertEqual(large_config.block_m, 512)
        self.assertEqual(large_config.block_n, 512)
        self.assertEqual(large_config.block_k, 256)
        self.assertEqual(large_config.group_m, 64)
        self.assertEqual(large_config.num_stages, 10)
        self.assertEqual(large_config.num_warps, 32)

        # Check small values
        small_config = gemm_configs[1]
        self.assertEqual(small_config.block_m, 8)
        self.assertEqual(small_config.block_n, 8)
        self.assertEqual(small_config.block_k, 4)
        self.assertEqual(small_config.group_m, 1)
        self.assertEqual(small_config.num_stages, 1)
        self.assertEqual(small_config.num_warps, 1)

    def test_convert_configs_type_safety(self):
        """Test that converted configs have correct types for all fields."""
        triton_configs = [self.mock_triton_config2]  # Config with all optional params
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        self.assertEqual(len(gemm_configs), 1)
        config = gemm_configs[0]

        # Type checking for all fields
        self.assertIsInstance(config.name, str)
        self.assertIsInstance(config.grid, int)
        self.assertIsInstance(config.block_m, int)
        self.assertIsInstance(config.block_n, int)
        self.assertIsInstance(config.block_k, int)
        self.assertIsInstance(config.group_m, int)
        self.assertIsInstance(config.num_stages, int)
        self.assertIsInstance(config.num_warps, int)
        self.assertIsInstance(config.EVEN_K, bool)
        self.assertIsInstance(config.ALLOW_TF32, bool)
        self.assertIsInstance(config.USE_FAST_ACCUM, bool)
        self.assertIsInstance(config.ACC_TYPE, str)
        self.assertIsInstance(config.version, int)

    def test_convert_configs_with_none_kwargs(self):
        """Test conversion when kwargs is None or missing."""
        mock_config_no_kwargs = type(
            "MockTritonConfig",
            (),
            {
                "kwargs": None,
                "num_stages": 2,
                "num_warps": 4,
            },
        )()

        mock_config_missing_kwargs = type(
            "MockTritonConfig",
            (),
            {
                "num_stages": 3,
                "num_warps": 8,
            },
        )()

        triton_configs = [mock_config_no_kwargs, mock_config_missing_kwargs]
        gemm_configs = convert_triton_configs_to_gemm_configs(triton_configs)

        self.assertEqual(len(gemm_configs), 2)

        # Both should use default values for block sizes
        for config in gemm_configs:
            self.assertEqual(config.block_m, 64)  # Default
            self.assertEqual(config.block_n, 64)  # Default
            self.assertEqual(config.block_k, 32)  # Default
            self.assertEqual(config.group_m, 8)  # Default

        # But should preserve the actual num_stages and num_warps
        self.assertEqual(gemm_configs[0].num_stages, 2)
        self.assertEqual(gemm_configs[0].num_warps, 4)
        self.assertEqual(gemm_configs[1].num_stages, 3)
        self.assertEqual(gemm_configs[1].num_warps, 8)


if __name__ == "__main__":
    run_tests()

        