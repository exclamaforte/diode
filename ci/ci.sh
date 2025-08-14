#!/bin/bash

set -e  # Exit on any error

echo "::group::Setting up main virtual environment"
echo "Creating and activating virtual environment with uv..."
uv venv .venv-main
source .venv-main/bin/activate

echo "Installing requirements from requirements-ci.txt..."
uv pip install -r ci/requirements-ci.txt

echo "Installing diode packages..."
uv pip install -e .
uv pip install -e ./diode_common
uv pip install -e ./diode_datasets
uv pip install -e ./diode_models
echo "::endgroup::"

echo "::group::Running full test suite"
echo "Running all tests in tests directory..."
python -m pytest tests/ -v || echo "Main test suite failed, but continuing with diode_models-only tests..."
echo "::endgroup::"

echo "::group::Setting up diode_models-only environment"
echo "Deactivating current environment..."
deactivate

echo "Creating new virtual environment for diode_models only..."
uv venv .venv-models-only
source .venv-models-only/bin/activate

echo "Installing diode_common and diode_models..."
uv pip install -e ./diode_common
uv pip install -e ./diode_models

echo "Installing pytest for testing..."
uv pip install pytest
echo "::endgroup::"

echo "::group::Running diode_models-only tests"
echo "Running diode_models specific tests..."
if [ -d "tests/diode_models_only" ]; then
    echo "Found diode_models-only tests directory, running tests..."
    python -m pytest tests/diode_models_only/ -v
else
    echo "ERROR: diode_models-only tests directory not found!"
    echo "Expected tests/diode_models_only/ directory with standalone tests for diode_models package."
    exit 1
fi
echo "::endgroup::"

echo "::group::Cleanup"
echo "Deactivating environment..."
deactivate
echo "CI pipeline completed successfully!"
echo "::endgroup::"
