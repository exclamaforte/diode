#!/bin/bash

# Matrix multiplication workflow script that orchestrates data collection, training, and validation.
#
# This script performs the following steps:
# 1. Creates a data directory
# 2. Generates a training set using log normal distribution with exhaustive autotuning (1000 samples, writing every 5)
# 3. Trains a deep model on the collected data
# 4. Creates a validation set from operation_shapeset.json
# 5. Compares the validation set to the trained model

set -e  # Exit on any error
set -x

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLKIT_PATH="${SCRIPT_DIR}/matmul_toolkit.py"
PYTHON_CMD="${HOME}/.conda/envs/foo2/bin/python"

# Set random seed for reproducibility
SEED=51

# Define paths
DATA_DIR="${SCRIPT_DIR}/data"
VALIDATION_DIR="${DATA_DIR}/validation"
TRAIN_DATASET="${DATA_DIR}/seed_${SEED}_train_dataset.msgpack"
VALIDATION_DATASET="${DATA_DIR}/validation/my_validation.msgpack"
MODEL_PATH="${DATA_DIR}/matmul_model.pt"
LOG_DIR="${DATA_DIR}/logs"
NUM_SHAPES=0
NUM_EPOCHS=5000
CHUNK_SIZE=5
SEARCH_SPACE="EXHAUSTIVE"

# Path to operation_shapeset.json
OPERATION_SHAPESET_PATH="${SCRIPT_DIR}/../diode_datasets/diode_datasets/datasets/operation_shapeset.json"

echo "========================================="
echo "Matrix Multiplication Workflow"
echo "========================================="
echo "Script directory: ${SCRIPT_DIR}"
echo "Python command: ${PYTHON_CMD}"
echo "Toolkit path: ${TOOLKIT_PATH}"
echo ""

# Step 1: Create data directory
echo "Step 1: Creating data directory"
echo "Data directory: ${DATA_DIR}"
mkdir -p "${DATA_DIR}"
echo "✓ Data directory created"
echo ""

# Check if operation_shapeset.json exists
if [ ! -f "${OPERATION_SHAPESET_PATH}" ]; then
    echo "✗ Error: operation_shapeset.json not found at: ${OPERATION_SHAPESET_PATH}"
    exit 1
fi
echo "✓ Found operation_shapeset.json at: ${OPERATION_SHAPESET_PATH}"
echo ""

# Step 2: Generate training set using log normal distribution with exhaustive autotuning
echo "Step 2: Generating training dataset with log normal distribution and exhaustive autotuning"
echo "Command: ${PYTHON_CMD} ${TOOLKIT_PATH} --format msgpack --seed ${SEED} collect --output ${TRAIN_DATASET} --num-shapes ${NUM_SHAPES} --log-normal --search-space ${SEARCH_SPACE} --search-mode max-autotune --chunk-size ${CHUNK_SIZE}"
echo ""

${PYTHON_CMD} "${TOOLKIT_PATH}" \
    --format msgpack \
    --seed "${SEED}" \
    collect \
    --output "${TRAIN_DATASET}" \
    --num-shapes ${NUM_SHAPES} \
    --log-normal \
    --search-space ${SEARCH_SPACE} \
    --search-mode max-autotune \
    --chunk-size ${CHUNK_SIZE}

echo ""
echo "✓ Training data collection completed"
echo "Training dataset saved to: ${TRAIN_DATASET}"
echo ""

# Step 3: Train deep model on the collected data
# echo "Step 3: Training deep model on collected data"
# echo "Command: ${PYTHON_CMD} ${TOOLKIT_PATH} --seed ${SEED} train --data-dir ${DATA_DIR} --model ${MODEL_PATH} --model-type deep --batch-size 64 --num-epochs ${NUM_EPOCHS} --learning-rate 0.001 --log-dir ${LOG_DIR}"
# echo ""

# ${PYTHON_CMD} "${TOOLKIT_PATH}" \
#     --seed "${SEED}" \
#     train \
#     --data-dir "${DATA_DIR}" \
#     --model "${MODEL_PATH}" \
#     --model-type deep \
#     --batch-size 64 \
#     --num-epochs ${NUM_EPOCHS} \
#     --learning-rate 0.001 \
#     --log-dir "${LOG_DIR}"

echo ""
echo "✓ Model training completed"
echo "Trained model saved to: ${MODEL_PATH}"
echo ""

echo "Step 4: Creating validation dataset from operation_shapeset.json"
echo "Command: ${PYTHON_CMD} ${TOOLKIT_PATH} --format msgpack --seed ${SEED} create-validation --output ${VALIDATION_DATASET} --shapeset ${OPERATION_SHAPESET_PATH} --operations mm addmm bmm --search-space ${SEARCH_SPACE} --search-mode max-autotune"
echo ""

# ${PYTHON_CMD} "${TOOLKIT_PATH}" \
#     --format msgpack \
#     --seed "${SEED}" \
#     create-validation \
#     --output "${VALIDATION_DATASET}" \
#     --shapeset "${OPERATION_SHAPESET_PATH}" \
#     --chunk-size ${CHUNK_SIZE} \
#     --operations mm addmm bmm \
#     --search-space ${SEARCH_SPACE} \
#     --search-mode max-autotune

echo ""
echo "✓ Validation dataset creation completed"
echo "Validation dataset saved to: ${VALIDATION_DATASET}"
echo ""

# Step 5: Compare validation set to the model
echo "Step 5: Validating model performance on validation dataset"
echo "Command: ${PYTHON_CMD} ${TOOLKIT_PATH} --seed ${SEED} validate-model --model ${MODEL_PATH} --dataset ${VALIDATION_DATASET} --batch-size 64 --top-n-worst 10"
echo ""

${PYTHON_CMD} "${TOOLKIT_PATH}" \
    --seed "${SEED}" \
    validate-model \
    --model "${MODEL_PATH}" \
    --dataset "${VALIDATION_DIR}" \
    --batch-size 64 \
    --top-n-worst 10

echo ""
echo "========================================="
echo "🎉 Workflow completed successfully!"
echo "========================================="
echo "Training dataset: ${TRAIN_DATASET}"
echo "Validation dataset: ${VALIDATION_DIR}"
echo "Trained model: ${MODEL_PATH}"
echo "Logs directory: ${LOG_DIR}"
echo "========================================="
