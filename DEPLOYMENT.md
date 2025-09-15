# PyPI Deployment Guide

This document explains how to build and deploy the diode packages to PyPI using the deployment scripts.

## Overview

The diode project consists of 4 separate Python packages:

1. **`diode_common`** - Common utilities shared between packages
2. **`diode_models`** - Pre-trained models and model wrappers
3. **`diode_datasets`** - Dataset collections and utilities
4. **`torch-diode`** - Main package that depends on the others (published as `torch-diode`)

## Deployment Scripts

### `deploy_to_pypi.py` (Primary Script)

A comprehensive Python script that handles building and uploading all packages.

**Usage:**
```bash
python3 deploy_to_pypi.py <test|prod> [OPTIONS]
```

**Arguments:**
- `test` - Deploy to TestPyPI (for testing)
- `prod` - Deploy to production PyPI

**Options:**
- `--build-only` - Only build packages, don't upload
- `--skip-build` - Skip building, use existing dist files
- `--help` - Show help message

**Examples:**
```bash
# Build and deploy to TestPyPI
python3 deploy_to_pypi.py test

# Build and deploy to production PyPI
python3 deploy_to_pypi.py prod

# Only build packages for testing
python3 deploy_to_pypi.py test --build-only

# Use existing builds and upload to TestPyPI
python3 deploy_to_pypi.py test --skip-build
```

### `deploy.sh` (Bash Wrapper)

A simpler bash wrapper script for the Python deployment script.

**Usage:**
```bash
./deploy.sh <test|prod> [OPTIONS]
```

**Examples:**
```bash
# Build and deploy to TestPyPI
./deploy.sh test

# Build and deploy to production PyPI
./deploy.sh prod

# Only build packages
./deploy.sh test --build-only
```

## Prerequisites

### Required Tools

Install the required build and upload tools:

```bash
pip install build twine
```

### PyPI Credentials

You need to configure PyPI credentials before uploading. There are two ways to do this:

#### Option 1: Environment Variables

**For TestPyPI:**
```bash
export TESTPYPI_USERNAME="your_testpypi_username"
export TESTPYPI_PASSWORD="your_testpypi_password_or_token"
```

**For Production PyPI:**
```bash
export PYPI_USERNAME="your_pypi_username"
export PYPI_PASSWORD="your_pypi_password_or_token"
```

#### Option 2: ~/.pypirc Configuration File

Create a `~/.pypirc` file with your credentials:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = your_pypi_username
password = your_pypi_password_or_token

[testpypi]
repository = https://test.pypi.org/legacy/
username = your_testpypi_username
password = your_testpypi_password_or_token
```

**Note:** It's recommended to use API tokens instead of passwords. You can generate tokens from your PyPI account settings.

## Deployment Process

The script follows this process:

1. **Validation** - Checks for required tools and credentials
2. **Package Discovery** - Finds all packages with `pyproject.toml` files
3. **Cleaning** - Removes old build artifacts (`build/`, `dist/`, `*.egg-info/`)
4. **Building** - Builds wheel and source distributions for each package
5. **Uploading** - Uploads packages in dependency order:
   - `diode_common` (base dependency)
   - `diode_models` and `diode_datasets` (parallel)
   - `diode` (main package, depends on others)

## Testing Your Deployment

### TestPyPI

After deploying to TestPyPI, you can test installation:

```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ torch-diode

# Or install with extra dependencies from PyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ torch-diode
```

### Production PyPI

After deploying to production PyPI:

```bash
pip install torch-diode
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies Error**
   - Install required tools: `pip install build twine`

2. **Authentication Failed**
   - Check your PyPI credentials
   - Ensure you're using the correct username/password or API token
   - Verify your `~/.pypirc` file format

3. **Package Already Exists**
   - You can't overwrite existing package versions on PyPI
   - Increment the version numbers in `pyproject.toml` files

4. **Build Failures**
   - Check that all required files are present
   - Ensure `pyproject.toml` files are valid
   - Look for missing dependencies or import errors

### Version Management

Before deploying, make sure to update version numbers in all `pyproject.toml` files:

- `/pyproject.toml` (main diode package)
- `/diode_common/pyproject.toml`
- `/diode_models/pyproject.toml`
- `/diode_datasets/pyproject.toml`

### Dependency Order

The script automatically handles dependency order, but be aware:

- `diode_common` must be uploaded first (other packages depend on it)
- `diode_models` and `diode_datasets` can be uploaded in parallel
- `diode` main package should be uploaded last

## Security Notes

- Never commit PyPI credentials to version control
- Use API tokens instead of passwords when possible
- Keep your `~/.pypirc` file permissions restricted (`chmod 600 ~/.pypirc`)
- Use TestPyPI for testing before deploying to production

## Example Workflow

Here's a typical deployment workflow:

```bash
# 1. Update version numbers in all pyproject.toml files
# 2. Test build locally
./deploy.sh test --build-only

# 3. Deploy to TestPyPI for testing
./deploy.sh test

# 4. Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ torch-diode

# 5. If everything works, deploy to production
./deploy.sh prod
```
