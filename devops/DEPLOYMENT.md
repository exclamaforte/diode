# PyPI Deployment Guide

This document explains how to build and deploy the diode packages to PyPI using the deployment scripts.

## Overview

The diode project consists of 2 separate Python packages:

1. **`torch-diode`** - Full package with all code and models. Self registers into Inductor upon import.
2. **`torch-diode-lib`** - Library package with all code and models. Does not register into Inductor.

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
