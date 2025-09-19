"""
Enhanced tests for MatmulDatasetCollector to increase coverage.
"""

import logging
import os
import tempfile
import pytest
import numpy as np
import torch
from collections import OrderedDict
from unittest.mock import patch, MagicMock, call

from diode.collection.matmul_dataset_collector import (
    MatmulDatasetCollector,
    CollectionMode,
)
from diode.types.matmul_types import MMShape, OperationShapeSet


class TestMatmulDatasetCollectorInit:
    """Test MatmulDatasetCollector initialization edge cases."""

    def test_init_with_operation_shape_set_mode_without_shape_set(self):
        """Test initialization with OPERATION_SHAPE_SET mode but no shape set provided."""
        with pytest.raises(ValueError, match="operation_shape_set must be provided"):
            MatmulDatasetCollector(
                mode=CollectionMode.OPERATION_SHAPE_SET,
                operation_shape_set=None
            )

    def test_init_with_cuda_unavailable(self):
        """Test initialization when CUDA is unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            collector = MatmulDatasetCollector()
            assert collector.dtypes == [torch.float32]

    def test_init_with_cuda_available(self):
        """Test initialization when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            collector = MatmulDatasetCollector()
            assert torch.float16 in collector.dtypes
            assert torch.float32 in collector.dtypes

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_dtypes = [torch.float32, torch.bfloat16]
        operations = ["bmm", "mm"]
        
        collector = MatmulDatasetCollector(
            hardware_name="custom_gpu",
            mode=CollectionMode.LOG_NORMAL,
            operations=operations,
            num_shapes=50,
            dtypes=custom_dtypes,
            min_size=64,
            max_size=2048,
            power_of_two=True,
            log_normal_m_mean=7.0,
            log_normal_m_std=2.0
        )
        
        assert collector.hardware_name == "custom_gpu"
        assert collector.mode == CollectionMode.LOG_NORMAL
        assert collector.operations == operations
        assert collector.num_shapes == 50
        assert collector.dtypes == custom_dtypes
        assert collector.min_size == 64
        assert collector.max_size == 2048
        assert collector.power_of_two is True
        assert collector.log_normal_m_mean == 7.0
        assert collector.log_normal_m_std == 2.0


class TestMatmulDatasetCollectorCollection:
    """Test collection start/stop functionality."""

    def test_start_collection_when_already_collecting(self, caplog):
        """Test starting collection when already in progress."""
        collector = MatmulDatasetCollector()
        
        with patch('diode.collection.matmul_dataset_collector.add_feedback_saver'):
            collector.start_collection()
            assert collector._is_collecting is True
            
            # Try to start again
            collector.start_collection()
            assert "Collection is already in progress" in caplog.text

    def test_stop_collection_when_not_collecting(self, caplog):
        """Test stopping collection when not in progress."""
        collector = MatmulDatasetCollector()
        
        collector.stop_collection()
        assert "No collection in progress" in caplog.text

    def test_context_manager(self):
        """Test using collector as context manager."""
        collector = MatmulDatasetCollector()
          
        with patch('diode.collection.matmul_dataset_collector.add_feedback_saver') as mock_add:
            with patch('diode.collection.matmul_dataset_collector.clear_feedback_savers') as mock_clear:
                with collector:
                    assert collector._is_collecting is True
                    mock_add.assert_called_once()
                  
                assert collector._is_collecting is False
                mock_clear.assert_called_once()


class TestMatmulDatasetCollectorSizeHints:
    """Test the _get_size_hints method."""

    def test_get_size_hints_with_ints(self):
        """Test _get_size_hints when dimensions are already integers."""
        collector = MatmulDatasetCollector()
        mat1 = MagicMock()
        mat2 = MagicMock()
        
        m, n, k = collector._get_size_hints(mat1, mat2, 128, 256, 64)
        assert m == 128
        assert n == 256
        assert k == 64

    def test_get_size_hints_with_symbolic(self):
        """Test _get_size_hints with symbolic dimensions."""
        collector = MatmulDatasetCollector()
        mat1 = MagicMock()
        mat2 = MagicMock()
        mat1.layout.size = [MagicMock(), MagicMock()]
        mat2.layout.size = [MagicMock(), MagicMock()]
          
        # Mock V.graph.sizevars.size_hints to return specific values
        with patch('torch._inductor.virtualized.V') as mock_v:
            mock_v.graph.sizevars.size_hints.side_effect = [(128, 64), (64, 256)]
              
            # Use symbolic objects instead of ints
            symbolic_m = MagicMock()
            symbolic_n = MagicMock()
            symbolic_k = MagicMock()
              
            m, n, k = collector._get_size_hints(mat1, mat2, symbolic_m, symbolic_n, symbolic_k)
            assert m == 128
            assert n == 256
            assert k == 64

    def test_get_size_hints_with_exception(self):
        """Test _get_size_hints when size_hints fails."""
        collector = MatmulDatasetCollector()
        mat1 = MagicMock()
        mat2 = MagicMock()
          
        with patch('torch._inductor.virtualized.V') as mock_v:
            mock_v.graph.sizevars.size_hints.side_effect = AttributeError("No graph")
              
            # Use symbolic objects instead of ints
            symbolic_m = MagicMock()
            symbolic_n = MagicMock()
            symbolic_k = MagicMock()
              
            m, n, k = collector._get_size_hints(mat1, mat2, symbolic_m, symbolic_n, symbolic_k)
            assert m == 1  # fallback value
            assert n == 1  # fallback value
            assert k == 1  # fallback value


class TestMatmulDatasetCollectorFeedbackHandler:
    """Test the feedback handler functionality."""

    def test_feedback_handler_unsupported_operation(self, caplog):
        """Test feedback handler with unsupported operation."""
        collector = MatmulDatasetCollector()
        
        with caplog.at_level(logging.DEBUG):
            collector._feedback_handler(
                timings={},
                name="unsupported_op",
                input_nodes=[],
                choices=None,
                profiled_time=0.1
            )
            
            assert "Skipping operation: unsupported_op" in caplog.text

    def test_feedback_handler_mm_operation(self):
        """Test feedback handler with mm operation."""
        collector = MatmulDatasetCollector()
          
        # Create mock input nodes
        mat1_mock = MagicMock()
        mat1_mock.layout.size = [128, 64]
        mat1_mock.layout.dtype = torch.float16
          
        mat2_mock = MagicMock()
        mat2_mock.layout.size = [64, 256]
        mat2_mock.layout.dtype = torch.float16
          
        # Create mock choice
        choice_mock = MagicMock()
        choice_mock.log_info = {
            "tile_shape": "(64,32,128)",
            "GROUP_M": 8,
            "num_stages": 4,
            "num_warps": 8
        }
          
        timings = {choice_mock: 0.001}
          
        # Mock the isinstance check to return True for our mock choice
        with patch('torch._inductor.select_algorithm.TritonTemplateCaller', choice_mock.__class__):
            collector._feedback_handler(
                timings=timings,
                name="mm",
                input_nodes=[mat1_mock, mat2_mock],
                choices=None,
                profiled_time=0.1
            )
              
            # Check that timing was added to dataset
            assert len(collector.dataset.hardware) > 0

    def test_feedback_handler_addmm_operation(self):
        """Test feedback handler with addmm operation."""
        collector = MatmulDatasetCollector()
        
        # Create mock input nodes for addmm: bias, mat1, mat2
        bias_mock = MagicMock()
        mat1_mock = MagicMock()
        mat1_mock.layout.size = [128, 64]
        mat1_mock.layout.dtype = torch.float32
        
        mat2_mock = MagicMock()
        mat2_mock.layout.size = [64, 256]
        mat2_mock.layout.dtype = torch.float32
        
        # Create mock choice
        choice_mock = MagicMock()
        choice_mock.log_info = {
            "tile_shape": "(128,64,64)",
            "GROUP_M": 4,
            "num_stages": 2,
            "num_warps": 4
        }
        
        timings = {choice_mock: 0.002}
        
        # Mock the isinstance check to return True for our mock choice
        with patch('torch._inductor.select_algorithm.TritonTemplateCaller', choice_mock.__class__):
            collector._feedback_handler(
                timings=timings,
                name="addmm",
                input_nodes=[bias_mock, mat1_mock, mat2_mock],
                choices=None,
                profiled_time=0.1
            )
            
            # Check that timing was added to dataset
            assert len(collector.dataset.hardware) > 0

    def test_feedback_handler_non_triton_choice(self):
        """Test feedback handler with non-TritonTemplateCaller choice."""
        collector = MatmulDatasetCollector()
        
        mat1_mock = MagicMock()
        mat1_mock.layout.size = [128, 64]
        mat1_mock.layout.dtype = torch.float16
        
        mat2_mock = MagicMock()
        mat2_mock.layout.size = [64, 256]
        mat2_mock.layout.dtype = torch.float16
        
        # Create a non-TritonTemplateCaller choice
        choice_mock = MagicMock()
        timings = {choice_mock: 0.001}
        
        original_len = len(collector.dataset.hardware)
        
        # The original feedback handler should skip non-TritonTemplateCaller choices
        collector._feedback_handler(
            timings=timings,
            name="mm",
            input_nodes=[mat1_mock, mat2_mock],
            choices=None,
            profiled_time=0.1
        )
        
        # Should not add anything to dataset since choice is not TritonTemplateCaller
        assert len(collector.dataset.hardware) == original_len


class TestMatmulDatasetCollectorFileOperations:
    """Test file save/load operations."""

    def test_save_to_file(self):
        """Test saving dataset to file."""
        collector = MatmulDatasetCollector()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            collector.save_to_file(temp_path)
            assert os.path.exists(temp_path)
            
            # Verify file content
            with open(temp_path, 'r') as f:
                content = f.read()
                assert content  # Should not be empty
        finally:
            os.unlink(temp_path)

    def test_load_from_file_success(self, caplog):
        """Test loading dataset from file successfully."""
        collector = MatmulDatasetCollector()
        
        # Create a temporary file with valid dataset content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_file.write(collector.dataset.serialize())
            temp_path = temp_file.name
        
        try:
            with caplog.at_level(logging.INFO):
                collector.load_from_file(temp_path)
                assert f"Loaded data from {temp_path}" in caplog.text
        finally:
            os.unlink(temp_path)

    def test_load_from_file_failure(self, caplog):
        """Test loading dataset from file with invalid content."""
        collector = MatmulDatasetCollector()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_file.write("invalid json content")
            temp_path = temp_file.name
        
        try:
            collector.load_from_file(temp_path)
            assert f"Failed to load data from {temp_path}" in caplog.text
        finally:
            os.unlink(temp_path)

    def test_save_table_to_file(self):
        """Test saving table to file."""
        collector = MatmulDatasetCollector()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            collector.save_table_to_file(temp_path)
            assert os.path.exists(temp_path)
            
            # Verify file content
            with open(temp_path, 'r') as f:
                content = f.read()
                assert content  # Should not be empty
        finally:
            os.unlink(temp_path)


class TestMatmulDatasetCollectorLogNormal:
    """Test log normal size generation."""

    def test_round_small_dimension(self):
        """Test _round_small_dimension method."""
        collector = MatmulDatasetCollector()
        
        # Test dimensions < 2048
        assert collector._round_small_dimension(1) == 8  # rounds to 8 minimum
        assert collector._round_small_dimension(12) == 16  # rounds to nearest 8
        assert collector._round_small_dimension(64) == 64  # already multiple of 8
        assert collector._round_small_dimension(1000) == 1000  # already multiple of 8
        
        # Test dimensions >= 2048
        assert collector._round_small_dimension(2048) == 2048  # no rounding
        assert collector._round_small_dimension(3000) == 3000  # no rounding

    def test_generate_log_normal_sizes(self):
        """Test _generate_log_normal_sizes method."""
        collector = MatmulDatasetCollector(
            mode=CollectionMode.LOG_NORMAL,
            num_shapes=10,
            seed=42
        )
        
        sizes = collector._generate_log_normal_sizes()
        
        assert len(sizes) == 10
        for m, k, n in sizes:
            assert isinstance(m, int)
            assert isinstance(k, int)
            assert isinstance(n, int)
            assert m > 0
            assert k > 0
            assert n > 0
            # Small dimensions should be multiples of 8
            if m < 2048:
                assert m % 8 == 0
            if k < 2048:
                assert k % 8 == 0
            if n < 2048:
                assert n % 8 == 0


class TestMatmulDatasetCollectorShapeGeneration:
    """Test shape and dtype generation."""

    def test_generate_shapes_random_mode(self):
        """Test shape generation in RANDOM mode."""
        collector = MatmulDatasetCollector(
            mode=CollectionMode.RANDOM,
            num_shapes=5,
            operations=["mm", "addmm"],
            dtypes=[torch.float32],
            seed=42
        )
        
        with patch('diode.utils.dataset_utils.generate_matrix_sizes') as mock_generate:
            mock_generate.return_value = [(64, 32, 128), (128, 64, 256), (256, 128, 512), (512, 256, 1024), (1024, 512, 2048)]
            
            shapes_and_dtypes = collector._generate_shapes_and_dtypes()
            
            assert len(shapes_and_dtypes) == 5
            for shape, dtype, op_name in shapes_and_dtypes:
                assert isinstance(shape, tuple)
                assert len(shape) == 3
                assert dtype in collector.dtypes
                assert op_name in collector.operations

    def test_generate_shapes_log_normal_mode(self):
        """Test shape generation in LOG_NORMAL mode."""
        collector = MatmulDatasetCollector(
            mode=CollectionMode.LOG_NORMAL,
            num_shapes=3,
            operations=["mm"],
            dtypes=[torch.float16],
            seed=42
        )
        
        shapes_and_dtypes = collector._generate_shapes_and_dtypes()
        
        assert len(shapes_and_dtypes) == 3
        for shape, dtype, op_name in shapes_and_dtypes:
            assert isinstance(shape, tuple)
            assert len(shape) == 3
            assert dtype == torch.float16
            assert op_name == "mm"

    def test_generate_shapes_operation_shape_set_mode(self):
        """Test shape generation in OPERATION_SHAPE_SET mode."""
        # Create mock shapes
        shape1 = MMShape(
            B=1, M=128, N=256, K=64,
            M_dtype=torch.float16, K_dtype=torch.float16,
            out_dtype=torch.float16,
            out_size=(1, 128, 256),
            out_stride=(32768, 256, 1)
        )
        shape2 = MMShape(
            B=1, M=64, N=128, K=32,
            M_dtype=torch.float32, K_dtype=torch.float32,
            out_dtype=torch.float32,
            out_size=(1, 64, 128),
            out_stride=(8192, 128, 1)
        )
        
        # Create mock OperationShapeSet
        mock_op_shape_set = MagicMock()
        mock_op_shape_set.operations = {"mm": MagicMock()}
        mock_op_shape_set.get_shapes_for_operation.return_value = [shape1, shape2]
        
        collector = MatmulDatasetCollector(
            mode=CollectionMode.OPERATION_SHAPE_SET,
            operation_shape_set=mock_op_shape_set,
            operations=["mm"],
            seed=42
        )
        
        shapes_and_dtypes = collector._generate_shapes_and_dtypes()
        
        assert len(shapes_and_dtypes) == 2  # Two unique shapes
        for shape, dtype, op_name in shapes_and_dtypes:
            assert isinstance(shape, tuple)
            assert len(shape) == 3
            assert dtype in [torch.float16, torch.float32]
            assert op_name == "mm"

    def test_generate_shapes_operation_shape_set_missing_operation(self, caplog):
        """Test shape generation when operation is missing from OperationShapeSet."""
        mock_op_shape_set = MagicMock()
        mock_op_shape_set.operations = {}  # Empty operations
        
        collector = MatmulDatasetCollector(
            mode=CollectionMode.OPERATION_SHAPE_SET,
            operation_shape_set=mock_op_shape_set,
            operations=["missing_op"],
            seed=42
        )
        
        shapes_and_dtypes = collector._generate_shapes_and_dtypes()
        
        assert len(shapes_and_dtypes) == 0
        assert "Operation 'missing_op' not found in OperationShapeSet" in caplog.text

    def test_generate_shapes_operation_shape_set_none(self, caplog):
        """Test shape generation when OperationShapeSet is None."""
        # Since the init raises ValueError, we need to create the collector differently
        # First create with valid params, then set to None
        mock_op_shape_set = MagicMock()
        collector = MatmulDatasetCollector(
            mode=CollectionMode.OPERATION_SHAPE_SET,
            operation_shape_set=mock_op_shape_set,
            operations=["mm"]
        )
        
        # Now set to None to test the error condition
        collector.operation_shape_set = None
        
        shapes_and_dtypes = collector._generate_shapes_and_dtypes()
        
        assert len(shapes_and_dtypes) == 0
        assert "OperationShapeSet is None but mode is OPERATION_SHAPE_SET" in caplog.text


class TestMatmulDatasetCollectorMatrixOperations:
    """Test matrix multiplication operations."""

    def test_run_matrix_multiplication_mm(self):
        """Test running mm operation."""
        collector = MatmulDatasetCollector()
        
        with patch('torch.randn') as mock_randn:
            # Mock tensors
            mock_a = MagicMock()
            mock_a.shape = (128, 64)
            mock_b = MagicMock()
            mock_b.shape = (64, 256)
            mock_randn.side_effect = [mock_a, mock_b]
            
            with patch('torch.compile') as mock_compile:
                mock_compiled_fn = MagicMock()
                mock_compile.return_value = mock_compiled_fn
                
                collector._run_matrix_multiplication(
                    size=(128, 64, 256),
                    dtype=torch.float32,
                    op_name="mm",
                    device="cpu",
                    search_mode="default"
                )
                
                mock_compile.assert_called_once()
                mock_compiled_fn.assert_called_once_with(mock_a, mock_b)

    def test_run_matrix_multiplication_addmm(self):
        """Test running addmm operation."""
        collector = MatmulDatasetCollector()
        
        with patch('torch.randn') as mock_randn:
            # Mock tensors
            mock_a = MagicMock()
            mock_b = MagicMock()
            mock_c = MagicMock()
            mock_randn.side_effect = [mock_a, mock_b, mock_c]
            
            with patch('torch.compile') as mock_compile:
                mock_compiled_fn = MagicMock()
                mock_compile.return_value = mock_compiled_fn
                
                collector._run_matrix_multiplication(
                    size=(128, 64, 256),
                    dtype=torch.float32,
                    op_name="addmm",
                    device="cpu",
                    search_mode="default"
                )
                
                mock_compile.assert_called_once()
                mock_compiled_fn.assert_called_once_with(mock_c, mock_a, mock_b)

    def test_run_matrix_multiplication_bmm(self):
        """Test running bmm operation."""
        collector = MatmulDatasetCollector()
        
        with patch('torch.randn') as mock_randn:
            # Mock tensors
            mock_a = MagicMock()
            mock_b = MagicMock()
            mock_randn.side_effect = [mock_a, mock_b]
            
            with patch('torch.compile') as mock_compile:
                mock_compiled_fn = MagicMock()
                mock_compile.return_value = mock_compiled_fn
                
                collector._run_matrix_multiplication(
                    size=(128, 64, 256),
                    dtype=torch.float32,
                    op_name="bmm",
                    device="cpu",
                    search_mode="default"
                )
                
                mock_compile.assert_called_once()
                mock_compiled_fn.assert_called_once_with(mock_a, mock_b)

    def test_run_matrix_multiplication_unsupported_op(self, caplog):
        """Test running unsupported operation."""
        collector = MatmulDatasetCollector()
        
        collector._run_matrix_multiplication(
            size=(128, 64, 256),
            dtype=torch.float32,
            op_name="unsupported_op",
            device="cpu",
            search_mode="default"
        )
        
        assert "Unsupported operation: unsupported_op" in caplog.text

    def test_run_matrix_multiplication_with_error(self, caplog):
        """Test handling errors in matrix multiplication."""
        collector = MatmulDatasetCollector()
        
        with patch('torch.randn', side_effect=RuntimeError("CUDA out of memory")):
            with pytest.raises(RuntimeError):
                collector._run_matrix_multiplication(
                    size=(128, 64, 256),
                    dtype=torch.float32,
                    op_name="mm",
                    device="cuda",
                    search_mode="default"
                )
            
            assert "Error creating tensors for mm" in caplog.text

    def test_run_matrix_multiplication_shape_assertion_error(self):
        """Test shape assertion error in mm operation."""
        collector = MatmulDatasetCollector()
        
        with patch('torch.randn') as mock_randn:
            # Mock tensors with wrong shapes
            mock_a = MagicMock()
            mock_a.shape = (100, 50)  # Wrong shape
            mock_b = MagicMock()  
            mock_b.shape = (64, 256)
            mock_randn.side_effect = [mock_a, mock_b]
            
            with pytest.raises(AssertionError, match="Tensor a shape mismatch"):
                collector._run_matrix_multiplication(
                    size=(128, 64, 256),
                    dtype=torch.float32,
                    op_name="mm",
                    device="cpu",
                    search_mode="default"
                )


class TestMatmulDatasetCollectorDataCollection:
    """Test the collect_data method."""

    def test_collect_data_default_device(self):
        """Test collect_data with default device selection."""
        collector = MatmulDatasetCollector(
            mode=CollectionMode.RANDOM,
            num_shapes=1,
            operations=["mm"],
            dtypes=[torch.float32]
        )
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch.object(collector, '_generate_shapes_and_dtypes') as mock_generate:
                mock_generate.return_value = [((64, 32, 128), torch.float32, "mm")]
                
                with patch.object(collector, 'start_collection'):
                    with patch.object(collector, 'stop_collection'):
                        with patch.object(collector, '_run_matrix_multiplication'):
                            with patch('torch.set_grad_enabled'):
                                with patch('torch._dynamo.reset'):
                                    collector.collect_data(device=None)

    def test_collect_data_exhaustive_search_space(self, caplog):
        """Test collect_data with exhaustive search space."""
        collector = MatmulDatasetCollector(
            mode=CollectionMode.RANDOM,
            num_shapes=1,
            operations=["mm"],
            dtypes=[torch.float32]
        )
        
        with patch.object(collector, '_generate_shapes_and_dtypes') as mock_generate:
            mock_generate.return_value = [((64, 32, 128), torch.float32, "mm")]
            
            with patch.object(collector, 'start_collection'):
                with patch.object(collector, 'stop_collection'):
                    with patch.object(collector, '_run_matrix_multiplication'):
                        with patch('torch.set_grad_enabled'):
                            with patch('torch._dynamo.reset'):
                                with patch('time.time') as mock_time:
                                    mock_time.side_effect = [0.0, 1.0] + [1.0] * 10  # Extra values to prevent StopIteration
                                    with caplog.at_level(logging.INFO):
                                        collector.collect_data(
                                            device="cpu",
                                            search_space="EXHAUSTIVE"
                                        )
                                        
                                        assert "Set search space to EXHAUSTIVE" in caplog.text

    def test_collect_data_default_search_space(self, caplog):
        """Test collect_data with default search space."""
        collector = MatmulDatasetCollector(
            mode=CollectionMode.RANDOM,
            num_shapes=1,
            operations=["mm"],
            dtypes=[torch.float32]
        )
        
        with patch.object(collector, '_generate_shapes_and_dtypes') as mock_generate:
            mock_generate.return_value = [((64, 32, 128), torch.float32, "mm")]
            
            with patch.object(collector, 'start_collection'):
                with patch.object(collector, 'stop_collection'):
                    with patch.object(collector, '_run_matrix_multiplication'):
                        with patch('torch.set_grad_enabled'):
                            with patch('torch._dynamo.reset'):
                                with patch('time.time') as mock_time:
                                    mock_time.side_effect = [0.0, 1.0] + [1.0] * 10  # Extra values to prevent StopIteration
                                    with caplog.at_level(logging.INFO):
                                        collector.collect_data(
                                            device="cpu",
                                            search_space="DEFAULT"
                                        )
                                        
                                        assert "Set search space to DEFAULT" in caplog.text

    def test_collect_data_with_exception_handling(self):
        """Test collect_data with exception in finally block."""
        collector = MatmulDatasetCollector(
            mode=CollectionMode.RANDOM,
            num_shapes=1,
            operations=["mm"],
            dtypes=[torch.float32]
        )
        
        with patch.object(collector, '_generate_shapes_and_dtypes') as mock_generate:
            mock_generate.return_value = [((64, 32, 128), torch.float32, "mm")]
            
            with patch.object(collector, 'start_collection'):
                with patch.object(collector, 'stop_collection') as mock_stop:
                    with patch.object(collector, '_run_matrix_multiplication', side_effect=RuntimeError("Test error")):
                        with patch('torch.set_grad_enabled'):
                            with patch('torch._dynamo.reset'):
                                with pytest.raises(RuntimeError):
                                    collector.collect_data(device="cpu")
                                
                                # stop_collection should still be called in finally block
                                mock_stop.assert_called_once()

    def test_collect_data_logging_progress(self, caplog):
        """Test collect_data progress logging."""
        collector = MatmulDatasetCollector(
            mode=CollectionMode.RANDOM,
            num_shapes=2,
            operations=["mm", "addmm"],
            dtypes=[torch.float32]
        )
        
        with patch.object(collector, '_generate_shapes_and_dtypes') as mock_generate:
            mock_generate.return_value = [
                ((64, 32, 128), torch.float32, "mm"),
                ((128, 64, 256), torch.float32, "addmm")
            ]
            
            with patch.object(collector, 'start_collection'):
                with patch.object(collector, 'stop_collection'):
                    with patch.object(collector, '_run_matrix_multiplication'):
                        with patch('torch.set_grad_enabled'):
                            with patch('torch._dynamo.reset'):
                                with patch('time.time') as mock_time:
                                    mock_time.side_effect = [0.0, 10.0] + [10.0] * 10  # Extra values to prevent StopIteration
                                    with caplog.at_level(logging.INFO):
                                        collector.collect_data(device="cpu")
                                        
                                        # Check progress logging
                                        assert "[1/2] Running mm with size" in caplog.text
                                        assert "[2/2] Running addmm with size" in caplog.text
                                        assert "Collection completed in" in caplog.text  # Just check that completion message appears


class TestMatmulDatasetCollectorGettersAndToTable:
    """Test getter methods and table conversion."""

    def test_get_dataset(self):
        """Test get_dataset method."""
        collector = MatmulDatasetCollector()
        dataset = collector.get_dataset()
        assert dataset is collector.dataset

    def test_to_table(self):
        """Test to_table method."""
        collector = MatmulDatasetCollector()
        
        with patch.object(collector.dataset, 'to_table') as mock_to_table:
            mock_table = MagicMock()
            mock_to_table.return_value = mock_table
            
            table = collector.to_table()
            
            assert table is mock_table
            mock_to_table.assert_called_once()
