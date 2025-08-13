# Contributing to Diode

Thank you for your interest in contributing to Diode! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Community](#community)

## Getting Started

Diode is a framework for developing heuristics that plug into PyTorch's external interfaces. Before contributing, please:

1. Read the [README.md](/home/gabeferns/diode/README.md) to understand the project's goals and features
2. Check out the [examples](/home/gabeferns/diode/examples) to see how the framework is used
3. Review our [Code of Conduct](/home/gabeferns/diode/CODE_OF_CONDUCT.md)

## Development Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch (compatible version)
- Git

### Installation

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/diode.git
   cd diode
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install the project in development mode:
   ```bash
   pip install -e .
   pip install -e ./diode_models
   pip install -e ./diode_datasets
   pip install -e ./diode_common
   ```

5. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt  # If this file exists
   ```

## Project Structure

The project is organized into several packages:

```
diode/
├── diode/                    # Core framework
├── diode_models/            # Pre-trained models
├── diode_datasets/          # Dataset utilities
├── diode_common/            # Shared utilities
├── examples/                # Usage examples
├── tests/                   # Test suite
├── tools/                   # Development tools
└── docs/                    # Documentation
```

### Key Components

- **Data Collection**: Gather data from PyTorch external interfaces
- **Model Training**: Train ML models on collected data
- **Heuristics**: Support for ML models, lookup tables, and custom logic
- **Caching**: Create lookup tables for performance optimization

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports**: Report issues with clear reproduction steps
2. **Feature Requests**: Propose new features or improvements
3. **Code Contributions**: Bug fixes, new features, or optimizations
4. **Documentation**: Improve existing docs or add new documentation
5. **Models**: Contribute pre-trained models for specific hardware
6. **Datasets**: Share datasets for training heuristics

### Before You Start

- Check existing issues and pull requests to avoid duplicates
- For major changes, open an issue first to discuss the approach
- Ensure your contribution aligns with the project's goals

## Code Style

### Python Code Standards

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write clear, descriptive variable and function names
- Add docstrings for public functions and classes

### Linting and Formatting

The project uses several linters and formatters:

```bash
# Run all linters
python fix_linters.py

# Individual tools (if available)
flake8 .
mypy .
ruff check .
```

### Code Organization

- Keep functions focused and single-purpose
- Use appropriate abstractions for hardware-specific code
- Follow the existing patterns for model organization
- Maintain consistency with the existing codebase

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/examples/
```

### Writing Tests

- Write unit tests for new functionality
- Include integration tests for end-to-end workflows
- Test edge cases and error conditions
- Ensure tests are deterministic and can run in isolation

### Test Structure

- Place unit tests in `tests/` mirroring the source structure
- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases

## Submitting Changes

### Pull Request Process

1. **Create a Branch**: Create a feature branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**: Implement your changes following the guidelines above

3. **Test**: Ensure all tests pass and add new tests as needed

4. **Lint**: Run linters and fix any issues

5. **Commit**: Write clear, descriptive commit messages
   ```bash
   git commit -m "Add support for new hardware heuristic"
   ```

6. **Push**: Push your branch to your fork
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Pull Request**: Open a pull request with:
   - Clear title and description
   - Reference to related issues
   - Summary of changes made
   - Any breaking changes noted

### Pull Request Guidelines

- Keep pull requests focused and atomic
- Include tests for new functionality
- Update documentation as needed
- Ensure CI passes before requesting review
- Be responsive to feedback and suggestions

### Commit Message Format

Use clear, descriptive commit messages:

```
Add matmul kernel prediction for AMD GPUs

- Implement AMD-specific timing model
- Add dataset collection for RDNA architecture
- Include validation tests and examples
- Update documentation with AMD support details

Fixes #123
```

## Community

### Getting Help

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and general discussion
- **Examples**: Check the `/examples` directory for usage patterns

### Contributing Models and Datasets

When contributing pre-trained models or datasets:

1. **Models**: Place in `diode_models/diode_models/<heuristic>/<hardware>/`
2. **Datasets**: Add to `diode_datasets/diode_datasets/datasets/`
3. **Documentation**: Include clear documentation about:
   - Hardware compatibility
   - Performance characteristics
   - Training methodology
   - Usage examples

### Hardware Support

We especially welcome contributions for:

- New hardware architectures
- Hardware-specific optimizations
- Performance benchmarks
- Validation datasets

### Recognition

Contributors will be recognized in:

- Release notes for significant contributions
- Documentation acknowledgments
- Community highlights

## Questions?

If you have questions about contributing, please:

1. Check existing documentation and issues
2. Open a GitHub discussion for general questions
3. Open an issue for specific bugs or feature requests

Thank you for contributing to Diode!
