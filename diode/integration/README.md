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

## Model Registration Tutorial

This section provides a step-by-step guide for registering new models in the torch-diode system.

### Overview

Model registration in torch-diode involves three main components:
1. **Model Registry**: Central catalog of available models
2. **Integration System**: Automatic discovery and registration
3. **Model Files**: The actual trained model files

### Step 1: Organize Your Model Files

Place your trained model files in the `trained_models/` directory following this structure:

```
trained_models/
├── <purpose_category>/
│   ├── v1_model.pt
│   ├── v2_model.pt
│   └── ...
└── <other_model_file>.pt
```

**Example**: For matmul kernel runtime prediction:
```
trained_models/
├── matmul_kernel_runtime_prediction/
│   ├── v1_model.pt          # ← Your V1 model here
│   └── v2_model.pt          # ← Future V2 model
└── matmul_model_exhaustive.pt  # ← Alternative model
```

### Step 2: Register Models in the Model Registry

Edit `/diode/model_registry.py` to add your models to the `_initialize_default_models()` method:

```python
def _initialize_default_models(self) -> None:
    """Initialize the registry with default model configurations."""

    # Your new model category
    your_models = [
        ModelPointer(
            model_name="v1_model.pt",                    # Exact filename
            relative_path="your_model_category",         # Subdirectory in trained_models/
            model_purpose="your_model_purpose",          # Logical purpose/category
            interface_name="torch._inductor.choices",    # PyTorch interface to integrate with
            description="Your model description",        # Human-readable description
            version="1.0",                              # Model version
            dependencies=["torch._inductor", "torch._inductor.choices"],  # Required imports
        ),
        # Add more model variants as needed...
    ]

    for model in your_models:
        self.register_model(model)
```

**Key Parameters Explained**:
- `model_name`: Exact filename of your model file
- `relative_path`: Path relative to `trained_models/` directory (use "." for root)
- `model_purpose`: Logical category for grouping related models
- `interface_name`: Which PyTorch interface this model integrates with
- `dependencies`: Python modules required for this model to work

### Step 3: Create or Update Integration Module

If you're adding a new model category, create a new integration module in `/diode/integration/`:

```python
# /diode/integration/your_integration.py

from typing import Any, List, Optional

from .base_integration import BaseIntegration, ModelPointer

class YourIntegration(BaseIntegration):
    """Integration for your model type."""

    def __init__(self, model_pointers: Optional[List[ModelPointer]] = None, enable_fallback: bool = True, **kwargs):
        if model_pointers is None:
            model_pointers = [
                ModelPointer(
                    model_name="v1_model.pt",
                    relative_path="your_model_category",
                    model_purpose="your_model_purpose",
                    interface_name="torch._inductor.choices",
                    description="Your model description",
                    version="1.0",
                    dependencies=["torch._inductor", "torch._inductor.choices"],
                ),
            ]

        super().__init__(
            name="your_model_prediction",
            interface_name="torch._inductor.choices",
            model_pointers=model_pointers,
            enable_fallback=enable_fallback,
            **kwargs,
        )

    def create_dummy_function(self) -> Any:
        """Create dummy function to test interface availability."""
        # Implementation depends on your target interface
        pass

    def load_model(self, model_pointer: ModelPointer) -> Any:
        """Load your model from a model pointer."""
        # Custom loading logic for your model format
        pass

    def register_model(self, model: Any, model_pointer: ModelPointer) -> bool:
        """Register loaded model with target interface."""
        # Integration logic with PyTorch interface
        pass

    def enable_configs(self) -> bool:
        """Enable PyTorch configs for your models."""
        # Configuration setup
        pass

def create_your_integration(enable_fallback: bool = True) -> YourIntegration:
    """Factory function for your integration."""
    return YourIntegration(enable_fallback=enable_fallback)
```

### Step 4: Register Integration in Discovery System

Add your integration to the discovery system in `/diode/integration/base_integration.py`:

```python
def discover_and_register_integrations() -> Dict[str, bool]:
    # Known integration modules - add your module here
    known_integrations = [
        "matmul_integration",
        "your_integration",  # ← Add your integration module name
        # Add more as needed...
    ]
```

### Step 5: Handle Different Model Formats

Different model formats require different loading approaches:

#### A. Standard Checkpoint Format (Recommended)
Save models using the structured checkpoint format:

```python
# When saving your model
checkpoint_data = {
    "problem_feature_dim": 7,
    "config_feature_dim": 5,
    "hidden_layer_widths": [256, 256, 256],
    "model_state_dict": model.state_dict(),
    # ... other metadata
}
torch.save(checkpoint_data, "your_model.pt")
```

#### B. Raw State Dict Format (Like v1_model.pt)
For models saved as raw state dicts, handle them specially:

```python
def load_model(self, model_pointer: ModelPointer) -> Any:
    if model_pointer.model_name == "v1_model.pt":
        # Handle raw state dict format
        model = YourModelClass(
            param1=default_value1,
            param2=default_value2,
        )
        state_dict = torch.load(str(model_path), map_location=device)
        model.load_state_dict(state_dict)
        return model
    else:
        # Handle structured checkpoint format
        model = YourModelClass.load(str(model_path), device=device)
        return model
```

### Step 6: Test Your Registration

Create integration tests to verify your registration works:

```python
# tests/integration/test_your_model_registration.py

import unittest
from pathlib import Path

from diode.integration.base_integration import ModelPointer
from diode.model_registry import get_model_registry, register_model

class TestYourModelRegistration(unittest.TestCase):
    def test_model_exists(self):
        """Test that your model file exists."""
        model_path = Path(__file__).parent.parent.parent / "trained_models" / "your_model_category" / "v1_model.pt"
        self.assertTrue(model_path.exists())

    def test_model_registration(self):
        """Test registering your model."""
        registry = get_model_registry()

        model_pointer = ModelPointer(
            model_name="v1_model.pt",
            relative_path="your_model_category",
            model_purpose="your_model_purpose",
            interface_name="torch._inductor.choices",
            description="Your model description",
            version="1.0",
        )

        register_model(model_pointer)

        retrieved_model = registry.get_model("your_model_purpose", "v1_model.pt")
        self.assertIsNotNone(retrieved_model)
```

### Step 7: Verify Registration

Test that your models are properly registered:

```bash
# Run the model manifest script
cd devops/
python get_model_manifest.py

# Should show your models in the output:
# {
#   "total_models": 2,
#   "models_by_purpose": {
#     "your_model_purpose": [
#       {
#         "name": "v1_model.pt",
#         "relative_path": "your_model_category",
#         "size_mb": 1.23,
#         "version": "1.0"
#       }
#     ]
#   }
# }
```

### Common Issues and Solutions

#### Issue: "Model not found" errors
**Solution**: Verify your `relative_path` and `model_name` match the actual file location.

#### Issue: "Unknown model type in checkpoint" errors
**Solution**: Your model is saved in raw state_dict format. Handle it specially in your `load_model()` method.

#### Issue: "Interface not available" in integration
**Solution**: The PyTorch interface may not be available in your environment. This is normal for development/testing environments.

#### Issue: Models appear in manifest but integrations fail
**Solution**: Integration failures are expected in environments without full PyTorch inductor setup. The important part is that models are detected and registered.

### Best Practices

1. **Use structured checkpoint format** for new models (includes metadata)
2. **Test registration** with integration tests
3. **Document model requirements** in the model pointer description
4. **Handle both checkpoint formats** if you have mixed model types
5. **Use meaningful purpose categories** to group related models
6. **Version your models** appropriately
7. **Keep model files small** when possible for faster loading

### Example: Complete V1 Model Registration

Here's how the V1 matmul model was registered as a complete example:

```python
# 1. Model file placed at: trained_models/matmul_kernel_runtime_prediction/v1_model.pt

# 2. Registered in model_registry.py:
ModelPointer(
    model_name="v1_model.pt",
    relative_path="matmul_kernel_runtime_prediction",
    model_purpose="matmul_kernel_runtime_prediction",
    interface_name="torch._inductor.choices",
    description="Matrix multiplication kernel runtime prediction model v1",
    version="1.0",
    dependencies=["torch._inductor", "torch._inductor.choices"],
),

# 3. Integration updated in matmul_integration.py to handle raw state_dict format
# 4. Tests created in tests/integration/test_matmul_model_registration.py
# 5. Verified with: python devops/get_model_manifest.py
```

This complete process ensures your models are properly discovered, registered, and available to the torch-diode system.
