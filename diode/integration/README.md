# Diode PyTorch Inductor Integration

This directory contains the integration between Diode models and PyTorch Inductor's kernel configuration selection system.

## Overview

The integration allows trained Diode models to be used for intelligent kernel configuration selection in PyTorch's `torch.compile` operations. Instead of relying solely on heuristics, the system can use machine learning models to predict the performance of different kernel configurations and select the best ones.

## Files

### `inductor_integration.py`
The main integration module containing:

- **`DiodeInductorChoices`**: Extended `InductorChoices` class that overrides `_finalize_mm_configs` to use model-based selection
- **`install_diode_choices()`**: Helper function to install Diode choices as the default choice handler
- **`create_diode_choices()`**: Factory function to create DiodeInductorChoices instances

### `usage_example.py`
Example usage demonstrating how to:
- Set up the integration
- Use it with matrix multiplication operations
- Check integration status
- Manually create and use DiodeInductorChoices instances

### `__init__.py`
Package initialization file for the integration module.

## How It Works

### 1. Integration Hook
The integration works by extending PyTorch Inductor's `InductorChoices` class and overriding the `_finalize_mm_configs` method. This method is called during kernel compilation to select which configurations to use.

### 2. Feature Extraction
When `_finalize_mm_configs` is called, the system:
- Extracts problem features from the `KernelInputs` (matrix dimensions, data types, etc.)
- Converts `KernelTemplateChoice` objects to `TritonGEMMConfig` objects
- Extracts configuration features from each config

### 3. Model Inference
For each configuration:
- Problem and config features are fed to the trained Diode model
- The model predicts the log execution time
- Configurations are ranked by predicted performance

### 4. Selection
The system selects the top-k configurations within a performance threshold of the best prediction, ensuring both performance and diversity.

## Usage

### Basic Setup

```python
from diode.integration.inductor_integration import install_diode_choices

# Install Diode choices globally
install_diode_choices(
    model_path="path/to/your/model.pt",
    device="cuda",
    top_k_configs=3,
    performance_threshold=1.1
)

# Now all torch.compile operations will use the Diode model
@torch.compile(mode="max-autotune")
def my_matmul(a, b):
    return torch.mm(a, b)
```

### Manual Usage

```python
from diode.integration.inductor_integration import create_diode_choices
from torch._inductor.virtualized import V

# Create a DiodeInductorChoices instance
choices = create_diode_choices(
    model_path="path/to/your/model.pt",
    device="cuda",
    top_k_configs=5
)

# Temporarily install it
old_handler = V.get_choices_handler()
V.set_choices_handler(choices)

# Use torch.compile operations...

# Restore original handler
V.set_choices_handler(old_handler)
```

## Configuration Options

### `DiodeInductorChoices` Parameters

- **`model_path`**: Path to the trained Diode model (optional, will auto-detect if None)
- **`device`**: Device to run the model on ("cuda" or "cpu")
- **`top_k_configs`**: Maximum number of configurations to return (default: 3)
- **`enable_fallback`**: Whether to fall back to default behavior if model fails (default: True)
- **`performance_threshold`**: Ratio threshold for including configs (1.0 = only best, 1.1 = within 10% of best)

## Model Requirements

The integration expects trained Diode models that:
1. Are compatible with the `ModelWrapper` interface
2. Accept problem and configuration features as input
3. Output predicted log execution times
4. Are trained on similar operation types (mm, addmm, bmm)

## Error Handling and Fallback

The integration includes robust error handling:
- **Model Loading Errors**: Falls back to default behavior if model can't be loaded
- **Feature Extraction Errors**: Uses default selection if features can't be extracted
- **Inference Errors**: Falls back to default if model prediction fails
- **Statistics Tracking**: Tracks usage statistics for monitoring

## Performance Considerations

- **Model Compilation**: The integration uses `torch.compile` on the model for faster inference
- **Caching**: Model wrapper handles internal caching for efficiency
- **Batch Processing**: Processes each configuration individually (could be optimized for batch processing)

## Integration with Choices.py

The integration specifically targets the `_finalize_mm_configs` method in PyTorch Inductor's `choices.py`. This method is called after template-specific heuristics generate initial configuration choices, allowing the Diode model to:

1. **Filter configurations**: Remove poor-performing options
2. **Rank configurations**: Order by predicted performance
3. **Select top-k**: Choose the best configurations within a threshold
4. **Maintain diversity**: Ensure multiple good options are available for further selection

## Statistics and Monitoring

The `DiodeInductorChoices` class tracks statistics:
- `total_calls`: Total number of times `_finalize_mm_configs` was called
- `model_selections`: Number of times the model was successfully used
- `fallback_*`: Various fallback reasons (no model, feature extraction failure, etc.)
- `configs_filtered`: Number of configurations filtered out

Access statistics with:
```python
stats = choices.get_stats()
print(stats)
```

## Development and Testing

For development without a full PyTorch Inductor environment, the integration gracefully handles missing imports and provides mock types. This allows for:
- Unit testing of the integration logic
- Development in environments without Inductor
- Debugging and validation of the feature extraction pipeline

## Future Enhancements

Potential improvements include:
- **Batch Inference**: Process multiple configurations in a single model call
- **Config Caching**: Cache predictions for similar configurations
- **Online Learning**: Update model predictions based on actual performance
- **Multi-operation Support**: Extend beyond matrix multiplication operations
- **Adaptive Thresholding**: Dynamic performance thresholds based on problem characteristics