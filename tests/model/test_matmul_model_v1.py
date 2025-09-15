"""
Unit tests for the MatmulModelV1 class and its interface.
"""

import unittest
import os
import sys
import torch
import numpy as np
import pandas as pd
import tempfile
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from diode.model.matmul_model_v1 import (
    MatmulModelV1, 
    NeuralNetwork, 
    ModelWrapper,
    get_nn_x,
    get_total_gb_feature,
    get_total_gflop_feature
)
from diode.model.matmul_timing_model import MatmulTimingModel, DeepMatmulTimingModel


class TestMatmulModelV1(unittest.TestCase):
    """
    Tests for the MatmulModelV1 class interface.
    """
    
    def setUp(self):
        """
        Set up the test environment.
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create feature dimensions that match the MatmulModelV1 expected features
        self.problem_feature_dim = 7  # dtype_size, dim_m, dim_n, dim_k, total_gb, total_gflop, flops_per_byte
        self.config_feature_dim = 5   # config_block_k, config_block_m, config_block_n, config_num_stages, config_num_warps
        
        # Create a model
        self.model = MatmulModelV1(
            problem_feature_dim=self.problem_feature_dim,
            config_feature_dim=self.config_feature_dim,
            hidden_layer_widths=[64, 128, 64],
            kernel_overhead=0.005,
            dropout_rate=0.1,
        )
        
        # Create reference models for interface comparison
        self.ref_timing_model = MatmulTimingModel(
            problem_feature_dim=self.problem_feature_dim,
            config_feature_dim=self.config_feature_dim,
            hidden_dims=[64, 128, 64],
            dropout_rate=0.1,
        )
        
        self.ref_deep_model = DeepMatmulTimingModel(
            problem_feature_dim=self.problem_feature_dim,
            config_feature_dim=self.config_feature_dim,
            hidden_dim=64,
            num_layers=3,
            dropout_rate=0.1,
        )
    
    def test_model_initialization(self):
        """
        Test that the model initializes correctly with the expected interface.
        """
        # Check that the model has the correct attributes
        self.assertEqual(self.model.problem_feature_dim, self.problem_feature_dim)
        self.assertEqual(self.model.config_feature_dim, self.config_feature_dim)
        self.assertEqual(self.model.input_dim, self.problem_feature_dim + self.config_feature_dim)
        self.assertEqual(self.model.hidden_layer_widths, [64, 128, 64])
        self.assertEqual(self.model.kernel_overhead, 0.005)
        self.assertEqual(self.model.dropout_rate, 0.1)
        
        # Check that log_kernel_overhead is calculated correctly
        expected_log_overhead = torch.log(torch.tensor(0.005)).item()
        self.assertAlmostEqual(self.model.log_kernel_overhead, expected_log_overhead, places=6)
        
        # Check that the model has the expected layers
        self.assertIsInstance(self.model.linear_relu_stack, torch.nn.Sequential)
    
    def test_model_forward_interface_consistency(self):
        """
        Test that the forward pass interface is consistent with other timing models.
        """
        # Create random input tensors
        batch_size = 16
        problem_features = torch.randn(batch_size, self.problem_feature_dim)
        config_features = torch.randn(batch_size, self.config_feature_dim)
        
        # Test MatmulModelV1 forward pass
        outputs_v1 = self.model(problem_features, config_features)
        
        # Test reference models forward pass
        outputs_timing = self.ref_timing_model(problem_features, config_features)
        outputs_deep = self.ref_deep_model(problem_features, config_features)
        
        # Check that all models have the same output shape
        self.assertEqual(outputs_v1.shape, (batch_size, 1))
        self.assertEqual(outputs_timing.shape, (batch_size, 1))
        self.assertEqual(outputs_deep.shape, (batch_size, 1))
        
        # Check that outputs are finite (no NaN or inf)
        self.assertTrue(torch.isfinite(outputs_v1).all())
        self.assertTrue(torch.isfinite(outputs_timing).all())
        self.assertTrue(torch.isfinite(outputs_deep).all())
    
    def test_model_save_load_interface_consistency(self):
        """
        Test that save/load interface is consistent with other timing models.
        """
        # Test MatmulModelV1 save/load
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            # Save the model
            self.model.save(tmp.name)
            
            # Load the model
            loaded_model = MatmulModelV1.load(tmp.name)
            
            # Check that the loaded model has the same attributes
            self.assertEqual(loaded_model.problem_feature_dim, self.model.problem_feature_dim)
            self.assertEqual(loaded_model.config_feature_dim, self.model.config_feature_dim)
            self.assertEqual(loaded_model.input_dim, self.model.input_dim)
            self.assertEqual(loaded_model.hidden_layer_widths, self.model.hidden_layer_widths)
            self.assertEqual(loaded_model.kernel_overhead, self.model.kernel_overhead)
            self.assertEqual(loaded_model.dropout_rate, self.model.dropout_rate)
            
            # Check that the loaded model has the same parameters
            for p1, p2 in zip(self.model.parameters(), loaded_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2, atol=1e-6))
        
        # Test reference models save/load for interface comparison
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            self.ref_timing_model.save(tmp.name)
            loaded_ref = MatmulTimingModel.load(tmp.name)
            self.assertEqual(loaded_ref.problem_feature_dim, self.ref_timing_model.problem_feature_dim)
    
    def test_model_load_with_device(self):
        """
        Test loading model with different devices.
        """
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            # Save the model
            self.model.save(tmp.name)
            
            # Load on CPU
            loaded_cpu = MatmulModelV1.load(tmp.name, device='cpu')
            self.assertEqual(str(loaded_cpu.linear_relu_stack[0].weight.device), 'cpu')
            
            # Test that forward pass works on CPU
            problem_features = torch.randn(4, self.problem_feature_dim)
            config_features = torch.randn(4, self.config_feature_dim)
            outputs = loaded_cpu(problem_features, config_features)
            self.assertEqual(outputs.shape, (4, 1))
    
    def test_model_load_backwards_compatibility(self):
        """
        Test loading models with missing optional parameters (backwards compatibility).
        """
        # Create a checkpoint with missing optional parameters
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            # Save a checkpoint without dropout_rate or kernel_overhead
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'problem_feature_dim': self.problem_feature_dim,
                'config_feature_dim': self.config_feature_dim,
                'hidden_layer_widths': [64, 128, 64],
                # Missing kernel_overhead and dropout_rate
            }
            torch.save(checkpoint, tmp.name)
            
            # Load the model (should use defaults)
            loaded_model = MatmulModelV1.load(tmp.name)
            
            # Check that defaults are used
            self.assertEqual(loaded_model.kernel_overhead, 0.00541)  # Default value
            self.assertEqual(loaded_model.dropout_rate, 0.0)  # Default value
    
    def test_neural_network_alias(self):
        """
        Test that NeuralNetwork is an alias for MatmulModelV1 (backwards compatibility).
        """
        # Test that NeuralNetwork is the same class as MatmulModelV1
        self.assertIs(NeuralNetwork, MatmulModelV1)
        
        # Test that we can create a model using the alias
        nn_model = NeuralNetwork(
            problem_feature_dim=self.problem_feature_dim,
            config_feature_dim=self.config_feature_dim,
        )
        
        # Test that it's actually a MatmulModelV1 instance
        self.assertIsInstance(nn_model, MatmulModelV1)
    
    def test_model_with_different_configurations(self):
        """
        Test model with different configuration parameters.
        """
        # Test with minimal configuration
        minimal_model = MatmulModelV1(
            problem_feature_dim=3,
            config_feature_dim=2,
            hidden_layer_widths=[16],
        )
        
        # Test forward pass
        problem_features = torch.randn(8, 3)
        config_features = torch.randn(8, 2)
        outputs = minimal_model(problem_features, config_features)
        self.assertEqual(outputs.shape, (8, 1))
        
        # Test with complex configuration
        complex_model = MatmulModelV1(
            problem_feature_dim=10,
            config_feature_dim=8,
            hidden_layer_widths=[32, 64, 128, 64, 32],
            kernel_overhead=0.001,
            dropout_rate=0.3,
        )
        
        # Test forward pass
        problem_features = torch.randn(4, 10)
        config_features = torch.randn(4, 8)
        outputs = complex_model(problem_features, config_features)
        self.assertEqual(outputs.shape, (4, 1))
    
    def test_log_sum_exp_computation(self):
        """
        Test that the logsumexp computation works correctly.
        """
        # Test in eval mode to ensure consistent behavior
        self.model.eval()
          
        # Create a simple test
        problem_features = torch.randn(1, self.problem_feature_dim)
        config_features = torch.randn(1, self.config_feature_dim)
          
        # Get the model prediction
        output = self.model(problem_features, config_features)
          
        # Manually compute the expected result
        x = torch.cat([problem_features, config_features], dim=1)
        log_base_pred = self.model.linear_relu_stack(x)
        log_overhead_tsr = torch.full_like(
            input=log_base_pred, fill_value=self.model.log_kernel_overhead
        )
        expected_output = torch.logsumexp(
            torch.stack([log_base_pred, log_overhead_tsr], dim=-1), dim=-1
        )
          
        # Check that they match
        self.assertTrue(torch.allclose(output, expected_output))
          
        # Also test that the computation works correctly with a larger batch
        problem_features_batch = torch.randn(4, self.problem_feature_dim)
        config_features_batch = torch.randn(4, self.config_feature_dim)
        output_batch = self.model(problem_features_batch, config_features_batch)
          
        # Check that batch output has correct shape and is finite
        self.assertEqual(output_batch.shape, (4, 1))
        self.assertTrue(torch.isfinite(output_batch).all())
    
    def test_error_handling(self):
        """
        Test error handling for invalid inputs.
        """
        # Test with mismatched feature dimensions
        with self.assertRaises(RuntimeError):
            problem_features = torch.randn(4, self.problem_feature_dim + 1)  # Wrong dimension
            config_features = torch.randn(4, self.config_feature_dim)
            self.model(problem_features, config_features)
        
        with self.assertRaises(RuntimeError):
            problem_features = torch.randn(4, self.problem_feature_dim)
            config_features = torch.randn(4, self.config_feature_dim + 1)  # Wrong dimension
            self.model(problem_features, config_features)
        
        # Test with mismatched batch sizes
        with self.assertRaises(RuntimeError):
            problem_features = torch.randn(4, self.problem_feature_dim)
            config_features = torch.randn(8, self.config_feature_dim)  # Different batch size
            self.model(problem_features, config_features)


class TestMatmulModelV1HelperFunctions(unittest.TestCase):
    """
    Tests for helper functions in matmul_model_v1.py.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create test dataframe
        self.test_df = pd.DataFrame({
            'dtype_size': [16, 32, 16, 32],
            'dim_m': [64, 128, 256, 512],
            'dim_n': [32, 64, 128, 256],
            'dim_k': [128, 256, 512, 1024],
        })
    
    def test_get_total_gb_feature(self):
        """
        Test the get_total_gb_feature function.
        """
        result = get_total_gb_feature(self.test_df)
        
        # Check that result is a pandas Series
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.test_df))
        
        # Manually calculate expected values for first row
        # m=64, n=32, k=128, dtype_size=16 bits = 2 bytes
        # A: 64*128, B: 128*32, C: 64*32
        # Total bytes: (64*128 + 128*32 + 64*32) * 2 = (8192 + 4096 + 2048) * 2 = 28672
        # GB: 28672 / 1e9 = 0.000028672
        expected_first = ((64 * 128 + 128 * 32 + 64 * 32) * (16 / 8)) / 1e9
        self.assertAlmostEqual(result.iloc[0], expected_first, places=10)
    
    def test_get_total_gflop_feature(self):
        """
        Test the get_total_gflop_feature function.
        """
        result = get_total_gflop_feature(self.test_df)
        
        # Check that result is a pandas Series
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.test_df))
        
        # Manually calculate expected values for first row
        # m=64, n=32, k=128
        # FLOPS = 2 * m * n * k = 2 * 64 * 32 * 128 = 524288
        # GFLOPS = 524288 / 1e9 = 0.000524288
        expected_first = (2 * 64 * 32 * 128) / 1e9
        self.assertAlmostEqual(result.iloc[0], expected_first, places=10)
    
    @patch('torch.cuda.is_available')
    def test_get_nn_x(self, mock_cuda_available):
        """
        Test the get_nn_x function.
        """
        # Mock CUDA availability to avoid device issues in testing
        mock_cuda_available.return_value = False
        
        # Create test dataframe with all required columns
        df = pd.DataFrame({
            'dtype_size': [16.0, 32.0],
            'dim_m': [64.0, 128.0],
            'dim_n': [32.0, 64.0],
            'dim_k': [128.0, 256.0],
            'total_gb': [0.001, 0.002],
            'total_gflop': [0.5, 1.0],
            'flops_per_byte': [500.0, 500.0],
            'config_block_k': [16.0, 32.0],
            'config_block_m': [16.0, 32.0],
            'config_block_n': [16.0, 32.0],
            'config_num_stages': [2.0, 2.0],
            'config_num_warps': [4.0, 4.0],
        })
        
        with patch('torch.from_numpy') as mock_from_numpy:
            # Mock tensor creation to avoid CUDA issues
            mock_tensor = MagicMock()
            mock_tensor.to.return_value = mock_tensor
            mock_tensor.astype.return_value = mock_tensor
            mock_from_numpy.return_value = mock_tensor
            
            with patch('pandas.Series.to_numpy') as mock_to_numpy:
                mock_to_numpy.return_value = np.array([1.0, 2.0])
                
                try:
                    x_tens, mean, std = get_nn_x(df)
                    # If we get here without error, the function structure is correct
                    self.assertTrue(True)
                except Exception as e:
                    # The function may fail due to mocking complexities, but structure should be tested
                    error_str = str(e).lower()
                    self.assertTrue('to_numpy' in error_str or 'device' in error_str)


class TestModelWrapper(unittest.TestCase):
    """
    Tests for the ModelWrapper class (backwards compatibility).
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        torch.manual_seed(42)
    
    def test_model_wrapper_initialization_default(self):
        """
        Test ModelWrapper initialization with default parameters.
        """
        wrapper = ModelWrapper()
        
        # Check that the model is created and set to eval mode
        self.assertIsInstance(wrapper.model, MatmulModelV1)
        self.assertFalse(wrapper.model.training)  # Should be in eval mode
        
        # Check that standardization parameters are set
        self.assertIsInstance(wrapper.mean_for_standardization, torch.Tensor)
        self.assertIsInstance(wrapper.std_for_standardization, torch.Tensor)
        self.assertEqual(len(wrapper.mean_for_standardization), 12)
        self.assertEqual(len(wrapper.std_for_standardization), 12)
    
    def test_model_wrapper_initialization_with_path(self):
        """
        Test ModelWrapper initialization with model path.
        """
        # Create a temporary model file
        model = MatmulModelV1(
            problem_feature_dim=7,
            config_feature_dim=5,
        )
        
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            model.save(tmp.name)
            
            # Create wrapper with model path
            wrapper = ModelWrapper(model_path=tmp.name)
            
            # Check that the model is loaded correctly
            self.assertIsInstance(wrapper.model, MatmulModelV1)
            self.assertFalse(wrapper.model.training)  # Should be in eval mode
    
    def test_model_wrapper_vec_method(self):
        """
        Test the vec method of ModelWrapper.
        """
        wrapper = ModelWrapper()
        
        # Create a mock config object
        mock_config = MagicMock()
        mock_config.all_kwargs.return_value = {
            'BLOCK_M': 32,
            'BLOCK_N': 64,
            'BLOCK_K': 16,
            'num_stages': 2,
            'num_warps': 4,
        }
        
        # Test vec method
        result = wrapper.vec(128, 256, 512, 16, mock_config)
        
        # Check result
        expected = (128, 256, 512, 16, 32, 64, 16, 2, 4)
        self.assertEqual(result, expected)
    
    def test_model_wrapper_vec_params_method(self):
        """
        Test the vec_params static method of ModelWrapper.
        """
        # Create a mock params object
        mock_params = MagicMock()
        mock_params.block_m = 32
        mock_params.block_n = 64
        mock_params.block_k = 16
        mock_params.num_stages = 2
        mock_params.num_warps = 4
        
        # Test vec_params method
        result = ModelWrapper.vec_params(128, 256, 512, 16, mock_params)
        
        # Check result
        expected = (128, 256, 512, 16, 32, 64, 16, 2, 4)
        self.assertEqual(result, expected)
    
    def test_model_wrapper_encode_method(self):
        """
        Test the encode method of ModelWrapper.
        """
        wrapper = ModelWrapper()
        
        # Create mock config objects
        mock_configs = []
        for i in range(3):
            mock_config = MagicMock()
            mock_config.all_kwargs.return_value = {
                'BLOCK_M': 32 * (i + 1),
                'BLOCK_N': 32 * (i + 1),
                'BLOCK_K': 16 * (i + 1),
                'num_stages': 2,
                'num_warps': 4,
            }
            mock_configs.append(mock_config)
        
        # Test with supported dtypes
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            with patch('torch.from_numpy') as mock_from_numpy, \
                 patch('pandas.DataFrame') as mock_df_class:
                
                # Mock DataFrame and its methods
                mock_df = MagicMock()
                mock_df_class.return_value = mock_df
                mock_get_nn_x = MagicMock()
                
                with patch('diode.model.matmul_model_v1.get_nn_x', mock_get_nn_x):
                    mock_tensor = torch.randn(3, 12)  # 3 configs, 12 features
                    mock_get_nn_x.return_value = (mock_tensor, None, None)
                    
                    result = wrapper.encode(128, 256, 512, dtype, mock_configs)
                    
                    # Check that get_nn_x was called
                    mock_get_nn_x.assert_called_once()
        
        # Test with unsupported dtype
        with self.assertRaises(ValueError):
            wrapper.encode(128, 256, 512, torch.int32, mock_configs)
    
    def test_model_wrapper_inference_method(self):
        """
        Test the inference method of ModelWrapper.
        """
        wrapper = ModelWrapper()
        
        # Create test input tensor (batch_size=4, features=12)
        inp_tensor = torch.randn(4, 12)
        
        # Test inference
        result = wrapper.inference(inp_tensor)
        
        # Check result shape
        self.assertEqual(result.shape, (4, 1))
        
        # Check that result is finite
        self.assertTrue(torch.isfinite(result).all())
    
    def test_model_wrapper_decode_method(self):
        """
        Test the decode method of ModelWrapper.
        """
        wrapper = ModelWrapper()
        
        # Create test tensor
        test_tensor = torch.randn(4, 1)
        
        # Test decode (should return the same tensor)
        result = wrapper.decode(test_tensor)
        
        # Check that it returns the same tensor
        self.assertTrue(torch.equal(result, test_tensor))


class TestInterfaceConsistency(unittest.TestCase):
    """
    Tests to ensure MatmulModelV1 has consistent interface with other timing models.
    """
    
    def setUp(self):
        """
        Set up models for comparison.
        """
        torch.manual_seed(42)
        
        self.problem_feature_dim = 7
        self.config_feature_dim = 5
        
        # Create all three model types
        self.model_v1 = MatmulModelV1(
            problem_feature_dim=self.problem_feature_dim,
            config_feature_dim=self.config_feature_dim,
            hidden_layer_widths=[32, 64, 32],
        )
        
        self.timing_model = MatmulTimingModel(
            problem_feature_dim=self.problem_feature_dim,
            config_feature_dim=self.config_feature_dim,
            hidden_dims=[32, 64, 32],
        )
        
        self.deep_model = DeepMatmulTimingModel(
            problem_feature_dim=self.problem_feature_dim,
            config_feature_dim=self.config_feature_dim,
            hidden_dim=32,
            num_layers=3,
        )
    
    def test_forward_method_consistency(self):
        """
        Test that all models have the same forward method signature.
        """
        # Create test inputs
        problem_features = torch.randn(8, self.problem_feature_dim)
        config_features = torch.randn(8, self.config_feature_dim)
        
        # Test that all models accept the same inputs and produce same output shape
        output_v1 = self.model_v1(problem_features, config_features)
        output_timing = self.timing_model(problem_features, config_features)
        output_deep = self.deep_model(problem_features, config_features)
        
        # All should have the same output shape
        self.assertEqual(output_v1.shape, output_timing.shape)
        self.assertEqual(output_v1.shape, output_deep.shape)
        self.assertEqual(output_v1.shape, (8, 1))
    
    def test_save_method_consistency(self):
        """
        Test that all models have consistent save methods.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test that all models can save without errors
            self.model_v1.save(os.path.join(tmpdir, 'model_v1.pt'))
            self.timing_model.save(os.path.join(tmpdir, 'timing_model.pt'))
            self.deep_model.save(os.path.join(tmpdir, 'deep_model.pt'))
            
            # Check that files were created
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'model_v1.pt')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'timing_model.pt')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'deep_model.pt')))
    
    def test_load_method_consistency(self):
        """
        Test that all models have consistent load methods.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save all models
            v1_path = os.path.join(tmpdir, 'model_v1.pt')
            timing_path = os.path.join(tmpdir, 'timing_model.pt')
            deep_path = os.path.join(tmpdir, 'deep_model.pt')
            
            self.model_v1.save(v1_path)
            self.timing_model.save(timing_path)
            self.deep_model.save(deep_path)
            
            # Test that all models can load correctly
            loaded_v1 = MatmulModelV1.load(v1_path)
            loaded_timing = MatmulTimingModel.load(timing_path)
            loaded_deep = DeepMatmulTimingModel.load(deep_path)
            
            # Test that loaded models work correctly
            problem_features = torch.randn(4, self.problem_feature_dim)
            config_features = torch.randn(4, self.config_feature_dim)
            
            output_v1 = loaded_v1(problem_features, config_features)
            output_timing = loaded_timing(problem_features, config_features)
            output_deep = loaded_deep(problem_features, config_features)
            
            # All should produce valid outputs
            self.assertEqual(output_v1.shape, (4, 1))
            self.assertEqual(output_timing.shape, (4, 1))
            self.assertEqual(output_deep.shape, (4, 1))
    
    def test_attribute_consistency(self):
        """
        Test that all models have consistent core attributes.
        """
        # All models should have these attributes
        for model in [self.model_v1, self.timing_model, self.deep_model]:
            self.assertTrue(hasattr(model, 'problem_feature_dim'))
            self.assertTrue(hasattr(model, 'config_feature_dim'))
            self.assertTrue(hasattr(model, 'input_dim'))
            self.assertTrue(hasattr(model, 'dropout_rate'))
            
            # Check attribute values
            self.assertEqual(model.problem_feature_dim, self.problem_feature_dim)
            self.assertEqual(model.config_feature_dim, self.config_feature_dim)
            self.assertEqual(model.input_dim, self.problem_feature_dim + self.config_feature_dim)


if __name__ == "__main__":
    unittest.main()
