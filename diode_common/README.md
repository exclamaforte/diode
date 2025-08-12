# diode_common

This package provides common functionality shared between the `diode` and `diode-models` packages.

## Overview

`diode_common` contains shared code that is used by both the main `diode` package and the `diode-models` package. This approach eliminates code duplication and ensures that both packages use the same core functionality.

## Components

### ModelWrapper

The primary component is the `ModelWrapper` class, which provides functionality for:

- Loading trained models from files
- Compiling models using torch.compile
- Running inference on models

### Utility Functions

- `load_model_config`: Loads model configuration from JSON files

## Usage

This package is not meant to be used directly. Instead, it is included as a dependency in both the `diode` and `diode-models` packages.

### In diode

```python
from diode_common.model_wrapper import ModelWrapper as CommonModelWrapper

class ModelWrapper(CommonModelWrapper):
    # diode-specific extensions
    ...
```

### In diode-models

```python
from diode_common.model_wrapper import ModelWrapper as CommonModelWrapper

class ModelWrapper(CommonModelWrapper):
    # diode-models-specific extensions
    ...
```

## Development

When making changes to shared functionality:

1. Update the code in `diode_common`
2. Test with both `diode` and `diode-models` to ensure compatibility
3. Update version numbers as needed

## Installation

This package is automatically included when installing either `diode` or `diode-models`. It is not intended to be installed separately.
