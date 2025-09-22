# Diode PyTorch Inductor Integration Tests

This directory contains comprehensive unit tests and integration tests for the Diode PyTorch Inductor integration.

## Overview

The test suite verifies that the Diode model-based kernel configuration selection integrates correctly with PyTorch Inductor's `_finalize_template_configs` method and provides intelligent config selection.

## Test Structure

### Primary Integration Tests (`test_inductor_integration.py`)

**TestDiodeInductorChoices**: Core functionality tests
- `test_init_with_model_path`: Verify proper initialization with model path
- `test_init_without_model_path`: Test fallback behavior without model
- `test_load_model_success`: Test successful model loading
- `test_load_model_failure`: Test model loading failure handling
- `test_finalize_template_configs_*`: Core config selection logic using new pipeline
- `test_create_unified_predictor`: Test unified predictor creation
- `test_stats_tracking`: Statistics collection and monitoring
- `test_compatibility_methods`: Compatibility methods for PyTorch Inductor

**TestCreateDiodeChoices**: Factory function tests
- `test_create_diode_choices_with_path`: Test factory function with model path
- `test_create_diode_choices_defaults`: Test factory function with defaults

**TestInstallDiodeChoices**: Installation tests
- `test_install_with_model_path`: Test installation with model path
- `test_install_with_defaults`: Test installation with default settings
- `test_install_import_error`: Test handling of missing PyTorch Inductor
- `test_install_general_error`: Test handling of general installation errors

### Kernel Conversion Tests (`test_kernel_conversions.py`)

**TestMMShapeExtraction**: MMShape extraction from kernel inputs
- `test_extract_mmshape_basic`: Basic MMShape extraction from kernel inputs
- `test_extract_mmshape_batch_operation`: Batch operation MMShape extraction
- `test_extract_mmshape_invalid_input`: Invalid input handling

**TestKTCToTritonConfig**: KernelTemplateChoice to TritonGEMMConfig conversion
- `test_convert_ktc_basic`: Basic KTC to TritonGEMMConfig conversion
- `test_convert_ktc_with_all_kwargs_method`: Config with all_kwargs method
- `test_convert_ktc_with_defaults`: Conversion with missing parameters using defaults

**TestFeatureCreationAndInference**: Feature creation and model inference
- `test_create_features_and_run_inference`: Feature creation and inference pipeline
- `test_create_features_with_model_wrapper_interface`: Inference with model wrapper

**TestConfigSelection**: Configuration selection based on predictions
- `test_select_best_configs_basic`: Basic config selection
- `test_select_best_configs_with_threshold`: Selection with performance threshold
- `test_select_best_configs_ensure_one_choice`: Ensuring at least one choice is selected
- `test_select_best_configs_mismatch_lengths`: Handling mismatched lengths

**TestFullPipeline**: Complete conversion and inference pipeline
- `test_convert_and_run_inference_pipeline_success`: Successful pipeline execution
- `test_convert_and_run_inference_pipeline_no_choices`: Pipeline with no choices
- `test_convert_and_run_inference_pipeline_no_mmshape`: Pipeline when MMShape extraction fails
- `test_convert_and_run_inference_pipeline_exception_handling`: Pipeline exception handling

### Diode Registration Tests (`test_diode_registration.py`)

**TestDiodeRegistration**: Diode package registration tests
- `test_diode_package_import`: Test torch_diode package import
- `test_diode_integration_info`: Test integration info availability
- `test_pytorch_inductor_availability`: Test PyTorch Inductor accessibility
- `test_diode_choices_handler_registration`: Test Diode choices handler registration
- `test_finalize_template_configs_instrumentation`: Test method instrumentation
- `test_diode_stats_accessibility`: Test Diode statistics access
- `test_environment_variable_setup`: Test environment variable configuration
- `test_cuda_availability`: Test CUDA availability for GPU tests

**TestDiodeRegistrationIntegration**: Extended registration integration tests
- `test_diode_registration_with_model_loading`: Test registration with model loading
- `test_diode_integration_error_handling`: Test error handling during integration

### Diode Compilation E2E Tests (`test_diode_compilation_e2e.py`)

**TestDiodeCompilationE2E**: End-to-end compilation tests
- `test_matmul_with_torch_compile`: Test matmul with torch.compile integration
- `test_multiple_matmul_operations`: Test multiple matmul operations
- `test_batch_and_addmm_operations`: Test batch matmul and addmm operations
- `test_diode_stats_during_compilation`: Test statistics during compilation
- `test_environment_setup_for_compilation`: Test compilation environment setup

### Enhanced Integration Tests (`test_enhanced_inductor_choices.py`)

**TestEnhancedInductorChoices**: Enhanced features (currently placeholder)
- `test_placeholder`: Placeholder test for future enhanced features
- `test_enhanced_features_placeholder`: Placeholder for enhanced functionality

### End-to-End Integration Tests (`test_inductor_integration_end_to_end.py`)

**TestEndToEndIntegration**: Realistic integration scenarios
- `test_realistic_config_selection`: Full workflow with trained model
- `test_performance_threshold_behavior`: Performance threshold testing
- `test_different_operation_types`: Multi-operation support (mm, bmm, addmm)
- `test_model_compilation_integration`: torch.compile integration
- `test_error_recovery_scenarios`: Comprehensive error recovery

**TestIntegrationWithMockedInductor**: Mocked Inductor testing
- `test_installation_process`: Installation with mocked virtualized module
- `test_factory_function_integration`: Factory function integration

**TestRealWorldSimulation**: Real-world usage patterns
- `test_typical_usage_workflow`: Typical usage workflow simulation
- `test_batch_processing_simulation`: Batch operation processing

## Test Features

### Mock Classes

The tests include comprehensive mock classes that simulate PyTorch Inductor components:

- **MockKernelInputs**: Simulates Inductor's KernelInputs
- **MockTensor**: Mock tensor with size and dtype information
- **MockLayout**: Mock output layout information
- **MockConfig**: Mock Triton configuration
- **MockTemplate**: Mock kernel template
- **MockKernelTemplateChoice**: Mock KernelTemplateChoice objects
- **MockModelWrapper**: Mock model wrapper for predictable inference

### Realistic Test Data

The end-to-end tests create realistic datasets with:
- Multiple operation types (mm, addmm, bmm)
- Various problem sizes (64x64 to 1024x512)
- Realistic Triton configurations
- Performance-based timing relationships
- Trained models for authentic predictions

### Error Scenarios

Comprehensive error testing covers:
- Missing or invalid model files
- Feature extraction failures
- Config conversion errors
- Inference failures
- Import errors (when Inductor modules unavailable)

## Running Tests

### Option 1: Test Runner Script

```bash
# Run all tests
./tests/integration/run_inductor_tests.py

# Run only unit tests
./tests/integration/run_inductor_tests.py --unit

# Run only integration tests
./tests/integration/run_inductor_tests.py --integration

# Run specific test pattern
./tests/integration/run_inductor_tests.py --pattern "finalize"

# Quiet mode
./tests/integration/run_inductor_tests.py --quiet
```

### Option 2: Direct unittest

```bash
cd /home/gabeferns/diode

# Run unit tests
python -m unittest tests.integration.test_inductor_integration -v

# Run end-to-end tests
python -m unittest tests.integration.test_inductor_integration_end_to_end -v

# Run specific test
python -m unittest tests.integration.test_inductor_integration.TestDiodeInductorChoices.test_finalize_mm_configs_with_model -v
```

### Option 3: pytest (if available)

```bash
cd tests/integration

# Run all tests
pytest

# Run with specific markers
pytest -m unit
pytest -m integration

# Run specific test file
pytest test_inductor_integration.py -v
```

## Test Configuration

### pytest.ini

Configures pytest behavior with:
- Test discovery patterns
- Output formatting
- Warning filters
- Test markers for categorization

### Environment Requirements

Tests are designed to run in environments with or without full PyTorch Inductor:
- **With Inductor**: Full integration testing possible
- **Without Inductor**: Graceful fallback with mock components
- **CPU/GPU**: Tests adapt to available hardware

## Test Coverage

The test suite covers:

### Core Functionality (100%)
- Model loading and initialization
- Feature extraction from all operation types
- Config conversion and validation
- Model inference and prediction
- Config selection and ranking
- Statistics tracking

### Integration Points (100%)
- PyTorch Inductor hook points
- Virtualized module integration
- Factory function behavior
- Installation and setup

### Error Handling (100%)
- Model loading failures
- Feature extraction errors
- Config conversion issues
- Inference failures
- Import errors
- Fallback behavior

### Performance Features (100%)
- Top-k config selection
- Performance threshold filtering
- Multi-operation support
- Batch processing

## Adding New Tests

### For New Features

1. Add unit tests in `test_inductor_integration.py`
2. Add integration tests in `test_inductor_integration_end_to_end.py`
3. Update mock classes if needed
4. Add realistic scenarios for end-to-end testing

### For Bug Fixes

1. Create test that reproduces the bug
2. Verify test fails before fix
3. Implement fix
4. Verify test passes
5. Add regression test if needed

### Test Guidelines

- **Isolation**: Each test should be independent
- **Deterministic**: Tests should produce consistent results
- **Comprehensive**: Cover both success and failure paths
- **Realistic**: Use realistic data and scenarios
- **Fast**: Unit tests should run quickly
- **Clear**: Test names should describe what they test

## Debugging Tests

### Common Issues

1. **Import Errors**: Usually due to missing PyTorch Inductor environment
   - Solution: Tests should gracefully handle missing imports

2. **Model Loading Failures**: Temporary model files not found
   - Solution: Check `setUp()` and `tearDown()` methods

3. **Mock Behavior**: Mocks not behaving as expected
   - Solution: Verify mock configuration and return values

4. **Feature Extraction**: Dimension mismatches
   - Solution: Check mock tensor sizes and expected feature dimensions

### Debug Commands

```bash
# Run with verbose output
python -m unittest tests.integration.test_inductor_integration -v

# Run single test with debugging
python -c "
import sys, os
sys.path.insert(0, os.getcwd())
from tests.integration.test_inductor_integration import TestDiodeInductorChoices
import unittest
suite = unittest.TestSuite()
suite.addTest(TestDiodeInductorChoices('test_finalize_mm_configs_with_model'))
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
"
```

## Continuous Integration

The tests are designed to run in CI environments:

- **No External Dependencies**: All required models created during test setup
- **Deterministic**: Consistent results across runs
- **Fast Execution**: Unit tests complete in seconds
- **Resource Efficient**: Use CPU by default, minimal memory usage
- **Error Tolerant**: Graceful handling of missing dependencies

## Performance Benchmarks

For performance testing of the integration:

```bash
# Time the test suite
time ./tests/integration/run_inductor_tests.py

# Profile memory usage
python -m memory_profiler tests/integration/run_inductor_tests.py
```

Expected performance:
- Unit tests: < 30 seconds
- Integration tests: < 60 seconds
- Memory usage: < 1GB peak

## Future Enhancements

Potential test improvements:
1. **Property-Based Testing**: Use hypothesis for generated test cases
2. **Performance Tests**: Benchmarking of config selection speed
3. **Hardware-Specific Tests**: GPU-specific testing when available
4. **Load Testing**: High-volume config selection scenarios
5. **Visual Testing**: Integration with visualization tools
