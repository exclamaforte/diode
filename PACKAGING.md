# Diode Packaging Setup

This project supports two different package distributions:

1. **`torch-diode`** - Full package with automatic PyTorch Inductor integration
2. **`torch-diode-lib`** - Library-only package without automatic integration

## Package Configurations

### torch-diode (Main Package)
- **Package Name**: `torch-diode`
- **Auto-registration**: ✅ Yes - automatically registers with PyTorch Inductor on import
- **Use Case**: Drop-in replacement for production environments where you want automatic integration
- **Installation**: `pip install torch-diode`

### torch-diode-lib (Library Package)
- **Package Name**: `torch-diode-lib`
- **Auto-registration**: ❌ No - manual integration required
- **Use Case**: Library environments where you want manual control over when/how integration happens
- **Installation**: `pip install torch-diode-lib`

## File Inclusion

Both packages include:
- All Python source code from the `diode/` directory
- **Model binaries**: `matmul_model.pt` (and any other `.pt` files in `diode/data/`)
- Documentation and examples

The model files are accessible at runtime via:
```python
import diode
import pkg_resources
model_path = pkg_resources.resource_filename('diode', 'data/matmul_model.pt')
```

## Building the Packages

### Using Make (Recommended)

```bash
# Build both packages
make build-all

# Build individual packages
make build-torch-diode        # Main package with auto-registration
make build-torch-diode-lib    # Library package without auto-registration

# Clean build artifacts
make clean

# List built packages
make list-packages

# Test installations
make test-install
```

### Manual Build Process

1. **torch-diode** (main package):
   ```bash
   python -m build
   ```

2. **torch-diode-lib** (library package):
   ```bash
   # Create temporary build directory
   mkdir -p /tmp/torch-diode-lib-build
   cp -r diode/ workflows/ README.md LICENSE /tmp/torch-diode-lib-build/
   cp pyproject-lib.toml /tmp/torch-diode-lib-build/pyproject.toml

   # Replace __init__.py with library version (no auto-registration)
   cp diode/__init___lib.py /tmp/torch-diode-lib-build/diode/__init__.py

   # Build the package
   cd /tmp/torch-diode-lib-build && python -m build

   # Copy results back
   cp /tmp/torch-diode-lib-build/dist/* dist/
   ```

## Usage Differences

### torch-diode (Auto-registration)

```python
import diode  # Automatically registers with PyTorch Inductor
import torch

# Your code here - Diode choices are automatically available
x = torch.randn(100, 100)
y = torch.randn(100, 100)
result = torch.mm(x, y)  # May use Diode heuristics automatically
```

### torch-diode-lib (Manual registration)

```python
import diode
from diode.integration.inductor_integration import install_diode_choices

# Manual registration when needed
install_diode_choices(enable_fallback=True)

import torch
# Now Diode choices are available
x = torch.randn(100, 100)
y = torch.randn(100, 100)
result = torch.mm(x, y)  # May use Diode heuristics
```

## Configuration Files

- **`pyproject.toml`**: Configuration for `torch-diode` (main package)
- **`pyproject-lib.toml`**: Configuration for `torch-diode-lib` (library package)
- **`diode/__init__.py`**: Main package init with auto-registration
- **`diode/__init___lib.py`**: Library package init without auto-registration

## Key Differences

| Feature | torch-diode | torch-diode-lib |
|---------|-------------|-----------------|
| Auto-registration | ✅ Yes | ❌ No |
| Manual control | Limited | ✅ Full |
| Import behavior | Registers on import | No registration |
| Production ready | ✅ Yes | ✅ Yes |
| Library friendly | ⚠️ May interfere | ✅ Yes |

## Deployment

Both packages can be deployed to PyPI:

```bash
# Upload to test PyPI
make upload-test

# Upload to production PyPI
make upload
```

## Dependencies

Both packages share the same dependencies:
- `torch>=2.0.0`
- `msgpack>=1.0.0`

## File Structure

```
dist/
├── torch_diode-0.1.0-py3-none-any.whl      # Main package wheel
├── torch_diode-0.1.0.tar.gz               # Main package source
├── torch_diode_lib-0.1.0-py3-none-any.whl # Library package wheel
└── torch_diode_lib-0.1.0.tar.gz           # Library package source
```

## Choosing the Right Package

- **Use `torch-diode`** if you want plug-and-play functionality
- **Use `torch-diode-lib`** if you're building a library or need manual control over integration timing
