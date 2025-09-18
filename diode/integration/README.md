# Diode PyTorch Integration

This directory contains the integration system between Diode models and PyTorch interfaces, providing a framework for intelligent kernel selection using trained models.

## Architecture

### Core Classes

#### `BaseIntegration` (`base_integration.py`)
Abstract base class implementing the general integration pattern:
1. Register dummy functions to test interface availability
2. Load actual models for successful registrations
3. Enable configs that engage the models

#### `ModelPointer` (`base_integration.py`)
Represents a pointer to a trained model with metadata including path, purpose, version, and dependencies.

#### `IntegrationRegistry` (`base_integration.py`)
Registry for managing multiple integrations with execution ordering and status tracking.

#### `MatmulIntegration` (`matmul_integration.py`)
Concrete implementation of `BaseIntegration` for matrix multiplication kernel selection. Manages model loading and registration with PyTorch Inductor's choices system.

#### `DiodeInductorChoices` (`inductor_integration.py`)
Extended `InductorChoices` class that overrides `_finalize_mm_configs` to use model-based selection instead of heuristics.

## Integration Process

The system follows a multi-step integration process:

1. **Discovery**: `discover_and_register_integrations()` finds available integration modules
2. **Registration**: Integrations register with the global registry with execution order
3. **Interface Testing**: Dummy functions test if PyTorch interfaces are available
4. **Model Loading**: Load actual models only for available interfaces
5. **Config Enabling**: Enable PyTorch configs that engage the loaded models

## Usage

### Automatic Integration

```python
from diode.integration import integrate_all, discover_and_register_integrations

# Discover and register all available integrations
discover_and_register_integrations()

# Execute all integrations
results = integrate_all()
```

### Manual Integration

```python
from diode.integration.matmul_integration import create_matmul_integration
from diode.integration import register_integration, integrate_all

# Create and register specific integration
integration = create_matmul_integration(enable_fallback=True)
register_integration(integration, execute_order=1)

# Execute integrations
results = integrate_all()
```

### Direct Usage (Legacy)

```python
from diode.integration.inductor_integration import install_diode_choices

# Install Diode choices globally
install_diode_choices(
    model_path="path/to/model.pt",
    device="cuda",
    top_k_configs=3,
    performance_threshold=1.1
)
```

## Configuration

### Integration Parameters
- **`enable_fallback`**: Whether to fall back to default behavior on failures
- **`execute_order`**: Execution order for multiple integrations

### Model Selection Parameters
- **`top_k_configs`**: Maximum configurations to return (default: 3)
- **`performance_threshold`**: Ratio threshold for including configs (1.1 = within 10% of best)
- **`device`**: Device for model inference ("cuda" or "cpu")

## Model Requirements

Models must:
- Be compatible with `ModelWrapper` interface
- Accept problem and configuration features as input
- Output predicted log execution times
- Be trained on similar operation types (mm, addmm, bmm)

## Status and Monitoring

### Integration Status
```python
from diode.integration import get_integration_status

status = get_integration_status()
```

### Usage Statistics
```python
# Get statistics from DiodeInductorChoices instance
stats = choices.get_stats()
```

## Error Handling

The system includes robust fallback mechanisms:
- Interface unavailability detection
- Model loading failure handling
- Feature extraction error recovery
- Prediction failure fallback

## Development

### Adding New Integrations

1. Create integration module inheriting from `BaseIntegration`
2. Implement required abstract methods
3. Add factory function following naming convention
4. Add module name to `discover_and_register_integrations()`

### Testing Without Full Environment

The integration gracefully handles missing PyTorch Inductor imports, allowing development and testing in minimal environments.
