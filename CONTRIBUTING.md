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

1. Read the [README.md](README.md) to understand the project's goals and features
2. Check out the [workflows](workflows/) to see how the framework is used

## Development Setup

### Prerequisites

- Python 3.10 or higher
- PyTorch Nightly or PyTorch 2.9 or later

### Installation

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/diode.git
   cd diode
   ```

4. Install the project in development mode:
   ```bash
   pip install -e ./diode-models
   pip install -e ./diode-datasets
   pip install -e ./diode_common
   pip install -e .
   ```

5. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt  # If this file exists
   ```

## Project Structure

TODO

### Key Components

TODO

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports**: Report issues with clear reproduction steps
2. **Feature Requests**: Propose new features or improvements
3. **Code Contributions**: Bug fixes, new features, or optimizations
4. **Documentation**: Improve existing docs or add new documentation
5. **Models**: Contribute pre-trained models for specific hardware
6. **Datasets**: Share datasets for training heuristics

## Code Style

### Python Code Standards

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write clear, descriptive variable and function names
- Add docstrings for public functions and classes

### Linting and Formatting

Run all linters:

```bash
lintrunner
```

### Code Organization

- Keep functions focused and single-purpose
- Maintain consistency with the existing codebase

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/workflows/
```

### Writing Tests

- Write unit tests for new functionality
- Include integration tests for end-to-end workflows
- Test edge cases and error conditions
- Ensure tests are deterministic and can run in isolation

### Test Structure

- Place unit tests in `tests/` mirroring the source structure
- Use descriptive test names that explain what is being tested

## Continuous Integration (CI)

### CI Setup

The `/ci` directory contains the continuous integration setup for the Diode project:

- `ci.sh`: Main CI script that sets up environments and runs tests
- `requirements-ci.txt`: CI-specific dependencies including testing tools and linters

### Running CI Locally

To run the CI pipeline locally, execute:

```bash
bash ci/ci.sh
```
# Testing and Code Coverage

This document describes how to run tests and generate code coverage reports for the Diode project.

## Prerequisites

Install the test dependencies:

```bash
make install-test-deps
# OR
pip install -e ".[test]"
```

## Running Tests

### All Tests
```bash
# Run all tests
make test
# OR
pytest
```

### Unit Tests Only
```bash
# Run unit tests (excludes integration tests)
make test-unit
# OR
pytest -m "not integration"
```

### Integration Tests Only
```bash
# Run integration tests only
make test-integration
# OR
pytest -m "integration"
```

## Code Coverage

### Running Tests with Coverage

#### All Tests with Coverage
```bash
# Run all tests with coverage
make test-cov
# OR
pytest --cov=diode --cov-branch --cov-report=term-missing --cov-report=html --cov-report=xml
```

#### Unit Tests with Coverage
```bash
# Run unit tests with coverage
make test-cov-unit
# OR
pytest -m "not integration" --cov=diode --cov-branch --cov-report=term-missing --cov-report=html --cov-report=xml
```

#### Integration Tests with Coverage
```bash
# Run integration tests with coverage
make test-cov-integration
# OR
pytest -m "integration" --cov=diode --cov-branch --cov-report=term-missing --cov-report=html --cov-report=xml
```

### Coverage Reports

#### Terminal Report
The terminal report shows coverage statistics directly in your terminal with missing line numbers.

#### HTML Report
An interactive HTML report is generated at `htmlcov/index.html`. Open it in your browser to see detailed coverage information with highlighted source code.

```bash
# Generate HTML report (if .coverage file exists)
make coverage-html
# OR
coverage html

# Run tests with coverage and automatically open HTML report
make test-and-view
```

#### XML Report
An XML report is generated at `coverage.xml` for use with CI/CD systems and code analysis tools.

```bash
# Generate XML report (if .coverage file exists)
make coverage-xml
# OR
coverage xml
```

### Coverage Configuration

Coverage is configured in `pyproject.toml` under the `[tool.coverage.*]` sections:

- **Source**: Only tracks coverage for the `diode` package
- **Branch Coverage**: Enabled to track both statement and branch coverage
- **Omit**: Excludes test files, cache files, and build artifacts
- **Exclude Lines**: Common patterns that shouldn't count against coverage (e.g., `pragma: no cover`, `if __name__ == "__main__":`)
- **Minimum Coverage**: Set to 80% (configurable via `--cov-fail-under`)

### Coverage Targets

The project aims for:
- **Minimum**: 80% overall coverage
- **Target**: 90%+ coverage for core functionality
- **Branch Coverage**: Enabled to ensure both paths of conditions are tested

### Cleaning Coverage Files

```bash
# Clean all coverage-related files
make clean-coverage
```

This removes:
- `htmlcov/` directory
- `coverage.xml` file
- `.coverage` and `.coverage.*` files

## Test Markers

Tests are organized using pytest markers:

- `unit`: Unit tests for individual components
- `integration`: End-to-end integration tests
- `slow`: Tests that take a long time to run
- `mock`: Tests that use mocking extensively
- `inductor`: Tests specifically for PyTorch Inductor integration

Use markers to run specific test categories:

```bash
# Run only unit tests
pytest -m "unit"

# Run only fast tests (exclude slow tests)
pytest -m "not slow"

# Run inductor-specific tests
pytest -m "inductor"
```

## CI/CD Integration

For continuous integration, use:

```bash
# Run tests with coverage and generate XML report for CI tools
pytest --cov=diode --cov-branch --cov-report=xml --cov-fail-under=80
```

The `coverage.xml` file can be consumed by CI systems like:
- GitHub Actions
- GitLab CI
- Jenkins
- SonarQube
- Codecov
- Coveralls

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
- Include validation tests and workflows
- Update documentation with AMD support details

Fixes #123
```

## Community

### Getting Help

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and general discussion
- **Workflows**: Check the `/workflows` directory for usage patterns

### Contributing Models and Datasets

When contributing pre-trained models or datasets:

1. **Models**: Place in `diode-models/diode_models/<heuristic>/<hardware>/`
2. **Datasets**: Add to `diode-datasets/diode_datasets/datasets/`
3. **Documentation**: Include clear documentation about:
   - Hardware compatibility
   - Performance characteristics
   - Training methodology
   - Usage workflows

### Hardware Support

We especially welcome contributions for:

- New hardware architectures
- Hardware-specific optimizations
- Performance benchmarks
- Validation datasets


Thank you for contributing to Diode!
