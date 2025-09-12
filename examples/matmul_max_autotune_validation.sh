#!/bin/bash

# Matrix multiplication max-autotune validation script
#
# This script runs the new validate-max-autotune feature using the same parameters
# as the matmul_workflow.sh script. It compares the model's ability to select
# optimal shapes against predefined max-autotune shapes.
#
# Prerequisites:
# - Run matmul_workflow.sh first to generate the model and validation dataset
# - This script will use the same paths and parameters

set -e  # Exit on any error
set -x

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLKIT_PATH="${SCRIPT_DIR}/matmul_toolkit.py"
PYTHON_CMD="${HOME}/.conda/envs/foo2/bin/python"

# Set random seed for reproducibility (same as workflow)
SEED=41

# Define paths (same as workflow script)
DATA_DIR="${SCRIPT_DIR}/data"
VALIDATION_DATASET="${DATA_DIR}/validation"
MODEL_PATH="${DATA_DIR}/matmul_model.pt"

# Define max-autotune solution path
MAX_AUTOTUNE_SOLUTION="${DATA_DIR}/max_autotune.json"

# Create max-autotune solution file if it doesn't exist
if [ ! -f "${MAX_AUTOTUNE_SOLUTION}" ]; then
    echo "Creating example max-autotune solution file..."

    # Create CSV with example max-autotune configurations first
    TEMP_CSV="${DATA_DIR}/max_autotune_configs.csv"

    # Use the CSV creation script to create an example
    "${PYTHON_CMD}" "${SCRIPT_DIR}/create_max_autotune_solution.py" \
        "${TEMP_CSV}" \
        --output "${MAX_AUTOTUNE_SOLUTION}" \
        --create-example \
        --solution-name "example_max_autotune" \
        --validate

    # Now create the actual solution from the CSV
    "${PYTHON_CMD}" "${SCRIPT_DIR}/create_max_autotune_solution.py" \
        "${TEMP_CSV}" \
        --output "${MAX_AUTOTUNE_SOLUTION}" \
        --solution-name "example_max_autotune" \
        --validate

    # Clean up temporary CSV
    rm -f "${TEMP_CSV}"

    echo "✓ Max-autotune solution file created: ${MAX_AUTOTUNE_SOLUTION}"
else
    echo "✓ Max-autotune solution file found: ${MAX_AUTOTUNE_SOLUTION}"
fi

echo "=================================================================="
echo "Matrix Multiplication Max-Autotune Validation"
echo "=================================================================="
echo "Script directory: ${SCRIPT_DIR}"
echo "Python command: ${PYTHON_CMD}"
echo "Toolkit path: ${TOOLKIT_PATH}"
echo "Seed: ${SEED}"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check if validation dataset directory exists
if [ ! -d "${VALIDATION_DATASET}" ]; then
    echo "✗ Error: Validation dataset directory not found at: ${VALIDATION_DATASET}"
    echo "Please run matmul_workflow.sh first to generate the validation dataset."
    exit 1
fi
echo "✓ Found validation dataset directory: ${VALIDATION_DATASET}"

# Check if trained model exists
if [ ! -f "${MODEL_PATH}" ]; then
    echo "✗ Error: Trained model not found at: ${MODEL_PATH}"
    echo "Please run matmul_workflow.sh first to train the model."
    exit 1
fi
echo "✓ Found trained model: ${MODEL_PATH}"

echo ""

# Run max-autotune validation
echo "Running max-autotune validation..."
echo "This will compare the model's config selection ability against max-autotune configs"
echo "for n ∈ {1, 5, 10, 20, 50, 100} and provide comprehensive statistics."
echo ""

# Build the command
CMD=(
    "${PYTHON_CMD}" "${TOOLKIT_PATH}"
    --seed "${SEED}"
    validate-max-autotune
    --model "${MODEL_PATH}"
    --dataset "${VALIDATION_DATASET}"
    --batch-size 64
    --max-autotune-solution "${MAX_AUTOTUNE_SOLUTION}"
)

echo "Command: ${CMD[*]}"
echo ""

# Execute the command
"${CMD[@]}"

echo ""
echo "=================================================================="
echo "🎯 Max-autotune validation completed!"
echo "=================================================================="
echo "Model: ${MODEL_PATH}"
echo "Validation dataset: ${VALIDATION_DATASET}"
echo "Max-autotune solution: ${MAX_AUTOTUNE_SOLUTION}"
