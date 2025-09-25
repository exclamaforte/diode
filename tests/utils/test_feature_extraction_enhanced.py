"""
Enhanced tests for feature extraction utilities.
"""

import torch

from torch_diode.types.matmul_types import MMShape, TritonGEMMConfig
from torch_diode.utils.feature_extraction import (
    extract_config_features,
    extract_config_features_compat,
    extract_problem_features,
    extract_problem_features_compat,
)


class TestFeatureExtractionEnhanced:
    def test_extract_problem_features_compat_success(self):
        """Test extract_problem_features_compat with valid MMShape."""
        mm_shape = MMShape(
            M=128,
            N=256,
            K=512,
            B=1,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1, 128, 256),
            out_stride=(32768, 256, 1),
        )

        features = extract_problem_features_compat(mm_shape)

        assert isinstance(features, torch.Tensor)
        assert features.dtype == torch.float32
        assert features.shape == (4,)
        assert features[0] == 128.0  # M
        assert features[1] == 256.0  # N
        assert features[2] == 512.0  # K
        assert features[3] == 1.0  # B

    def test_extract_problem_features_compat_exception(self):
        """Test extract_problem_features_compat with invalid input."""

        # Mock object that will raise exception when accessing attributes
        class BadMMShape:
            @property
            def M(self):
                raise AttributeError("Test exception")

        features = extract_problem_features_compat(BadMMShape())

        assert isinstance(features, torch.Tensor)
        assert features.dtype == torch.float32
        assert features.shape == (4,)
        assert torch.allclose(features, torch.zeros(4, dtype=torch.float32))

    def test_extract_config_features_compat_success(self):
        """Test extract_config_features_compat with valid TritonGEMMConfig."""
        config = TritonGEMMConfig(
            name="test_config",
            grid=1024,
            block_m=64,
            block_n=128,
            block_k=32,
            group_m=8,
            num_stages=3,
            num_warps=4,
        )

        features = extract_config_features_compat(config)

        assert isinstance(features, torch.Tensor)
        assert features.dtype == torch.float32
        assert features.shape == (6,)
        assert features[0] == 64.0  # block_m
        assert features[1] == 128.0  # block_n
        assert features[2] == 32.0  # block_k
        assert features[3] == 8.0  # group_m
        assert features[4] == 3.0  # num_stages
        assert features[5] == 4.0  # num_warps

    def test_extract_config_features_compat_exception(self):
        """Test extract_config_features_compat with invalid input."""

        # Mock object that will raise exception when accessing attributes
        class BadConfig:
            @property
            def block_m(self):
                raise AttributeError("Test exception")

        features = extract_config_features_compat(BadConfig())

        assert isinstance(features, torch.Tensor)
        assert features.dtype == torch.float32
        assert features.shape == (6,)
        assert torch.allclose(features, torch.zeros(6, dtype=torch.float32))

    def test_extract_problem_features_tensor_return(self):
        """Test extract_problem_features with return_tensors=False."""
        mm_shape = MMShape(
            M=128,
            N=256,
            K=512,
            B=2,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(2, 128, 256),
            out_stride=(32768, 256, 1),
        )

        features = extract_problem_features(mm_shape, return_tensors=False)

        assert isinstance(features, torch.Tensor)
        assert features.dtype == torch.float32
        assert features.shape == (17,)  # Updated to expect 17 comprehensive features
        assert features[0] == 128.0  # M
        assert features[1] == 256.0  # N
        assert features[2] == 512.0  # K
        assert features[3] == 2.0  # B

    def test_extract_problem_features_list_return(self):
        """Test extract_problem_features with return_tensors=True."""
        mm_shape = MMShape(
            M=64,
            N=128,
            K=256,
            B=4,
            M_dtype=torch.float32,
            K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(4, 64, 128),
            out_stride=(8192, 128, 1),
        )

        features = extract_problem_features(mm_shape, return_tensors=True)

        assert isinstance(features, list)
        assert len(features) == 17  # Updated to expect 17 comprehensive features
        assert features[0] == 64.0  # M
        assert features[1] == 128.0  # N
        assert features[2] == 256.0  # K
        assert features[3] == 4.0  # B

    def test_extract_problem_features_exception_tensor(self):
        """Test extract_problem_features exception handling with tensor return."""

        class BadMMShape:
            @property
            def M(self):
                raise AttributeError("Test exception")

        features = extract_problem_features(BadMMShape(), return_tensors=False)

        assert isinstance(features, torch.Tensor)
        assert features.dtype == torch.float32
        assert features.shape == (17,)  # Updated to expect 17 comprehensive features
        assert torch.allclose(features, torch.zeros(17, dtype=torch.float32))

    def test_extract_problem_features_exception_list(self):
        """Test extract_problem_features exception handling with list return."""

        class BadMMShape:
            @property
            def M(self):
                raise AttributeError("Test exception")

        features = extract_problem_features(BadMMShape(), return_tensors=True)

        assert isinstance(features, list)
        assert len(features) == 4
        assert features == [0.0, 0.0, 0.0, 0.0]

    def test_extract_config_features_tensor_return(self):
        """Test extract_config_features with return_tensors=False."""
        config = TritonGEMMConfig(
            name="test_config",
            grid=512,
            block_m=32,
            block_n=64,
            block_k=16,
            group_m=4,
            num_stages=2,
            num_warps=8,
        )

        features = extract_config_features(config, return_tensors=False)

        assert isinstance(features, torch.Tensor)
        assert features.dtype == torch.float32
        assert features.shape == (6,)
        assert features[0] == 32.0  # block_m
        assert features[1] == 64.0  # block_n
        assert features[2] == 16.0  # block_k
        assert features[3] == 4.0  # group_m
        assert features[4] == 2.0  # num_stages
        assert features[5] == 8.0  # num_warps

    def test_extract_config_features_list_return(self):
        """Test extract_config_features with return_tensors=True."""
        config = TritonGEMMConfig(
            name="test_config",
            grid=256,
            block_m=16,
            block_n=32,
            block_k=64,
            group_m=2,
            num_stages=5,
            num_warps=6,
        )

        features = extract_config_features(config, return_tensors=True)

        assert isinstance(features, list)
        assert len(features) == 6
        assert features == [16.0, 32.0, 64.0, 2.0, 5.0, 6.0]

    def test_extract_config_features_exception_tensor(self):
        """Test extract_config_features exception handling with tensor return."""

        class BadConfig:
            @property
            def block_m(self):
                raise AttributeError("Test exception")

        features = extract_config_features(BadConfig(), return_tensors=False)

        assert isinstance(features, torch.Tensor)
        assert features.dtype == torch.float32
        assert features.shape == (6,)
        assert torch.allclose(features, torch.zeros(6, dtype=torch.float32))

    def test_extract_config_features_exception_list(self):
        """Test extract_config_features exception handling with list return."""

        class BadConfig:
            @property
            def block_m(self):
                raise AttributeError("Test exception")

        features = extract_config_features(BadConfig(), return_tensors=True)

        assert isinstance(features, list)
        assert len(features) == 6
        assert features == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
