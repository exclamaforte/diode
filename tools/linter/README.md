# Linting Setup for Diode Project

This document explains how to set up and use the linting tools for the Diode project.

## Quick Setup

From a fresh conda environment:

```bash
# Create and activate environment
conda create -n diode_dev python=3.10 -y
conda activate diode_dev

# Install lintrunner and initialize all linters
pip install lintrunner
lintrunner init
```

## Easy Commands

### Check for Issues
```bash
# Using lintrunner (recommended)
lintrunner
```

### Apply Automatic Fixes
```bash

# Method 1: Use lintrunner directly
lintrunner --apply-patches

# Method 2: Apply formatting fixes only
lintrunner format
```

## Available Linters

The project includes the following linters:

### Core Python Linters
- **FLAKE8**: Python code style and error checking
- **MYPY**: Static type checking
- **RUFF**: Fast Python linter and formatter
- **PYFMT**: Python code formatting (black, usort, isort)

### Code Quality Linters
- **TYPEIGNORE**: Ensures type: ignore comments are qualified
- **NOQA**: Ensures noqa comments are qualified
- **ROOT_LOGGING**: Prevents use of root logger
- **ERROR_PRONE_ISINSTANCE**: Catches problematic isinstance usage
- **CONTEXT_DECORATOR**: Prevents context managers used as decorators
- **IMPORT_LINTER**: Checks for disallowed imports

### General File Linters
- **NEWLINE**: Ensures files end with newlines
- **SPACES**: Removes trailing whitespace
- **TABS**: Converts tabs to spaces
- **CODESPELL**: Spell checking in code and comments
- **PYPROJECT**: Validates pyproject.toml files

## Command Reference

| Command | Description |
|---------|-------------|
| `lintrunner` | Check all files for issues |
| `lintrunner --apply-patches` | Apply all automatic fixes |
| `lintrunner format` | Apply formatting fixes only |
| `lintrunner --take FLAKE8` | Run only FLAKE8 linter |
| `lintrunner --skip MYPY` | Run all linters except MYPY |

## Configuration Files

- `.lintrunner.toml`: Main linter configuration
- `mypy.ini`: MyPy type checker configuration
- `pyproject.toml`: Ruff and other tool configurations
- `tools/linter/adapters/`: Linter adapter scripts

## Installation Options

### Option 1: Using lintrunner (Recommended)
```bash
pip install lintrunner
lintrunner init
```

### Option 2: Using pip with optional dependencies
```bash
# Install all linting dependencies
pip install -e ".[lint]"

# Or if you already have the package installed
pip install "diode[lint]"
```

## Usage

### Running Individual Linters

You can run individual linter adapters directly:

```bash
# Run flake8 on Python files
python3 tools/linter/adapters/flake8_linter.py diode/**/*.py

# Run mypy with config
python3 tools/linter/adapters/mypy_linter.py --config=mypy.ini diode/**/*.py

# Run ruff
python3 tools/linter/adapters/ruff_linter.py --config=pyproject.toml diode/**/*.py
```

### Running with lintrunner

```bash
# Run all linters
lintrunner

# Run specific linter
lintrunner --take FLAKE8

# Auto-fix issues where possible
lintrunner --apply-patches
```

## Customization

To customize the linter setup:

1. **Add new linters**: Add new `[[linter]]` sections to `.lintrunner.toml`
2. **Modify patterns**: Update `include_patterns` and `exclude_patterns` in the config
3. **Adjust rules**: Modify tool-specific configurations in `pyproject.toml` or `mypy.ini`
4. **Create new adapters**: Add new linter adapter scripts in `tools/linter/adapters/`

## Linter Adapter Scripts

The `tools/linter/adapters/` directory contains Python scripts that wrap various linting tools:

- `pip_init.py`: Handles package installation
- `flake8_linter.py`: Flake8 wrapper
- `mypy_linter.py`: MyPy wrapper
- `grep_linter.py`: Generic grep-based linting
- `ruff_linter.py`: Ruff wrapper
- `pyfmt_linter.py`: Python formatting wrapper
- `codespell_linter.py`: Codespell wrapper
- `newlines_linter.py`: Newline checking
- `pyproject_linter.py`: pyproject.toml validation
- `import_linter.py`: Import restriction checking

Each adapter outputs JSON-formatted lint results that can be consumed by lintrunner or other tools.

## Troubleshooting

If you encounter issues:

1. **Reinstall linters**: `lintrunner init` to reinstall all dependencies
2. **Check environment**: Ensure you're in the correct conda environment
3. **Update lintrunner**: `pip install --upgrade lintrunner`
4. **Manual installation**: Use `python run_linters.py --use-standalone` for debugging
5. **Temporary file errors**: If you see `@/tmp/` errors, try running linters on specific files instead of all files
