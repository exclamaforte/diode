# Matrix Multiplication Toolkit

A unified toolkit for matrix multiplication data collection, model training, and evaluation.

## Overview

The `matmul_toolkit.py` script provides a comprehensive interface for working with matrix multiplication operations, including:

1. Collecting timing data for matrix multiplications
2. Training neural network models to predict execution times
3. Evaluating model performance
4. Visualizing results

This toolkit consolidates functionality from multiple separate scripts into a single, unified interface controlled through command-line flags.

## Usage

The toolkit provides multiple modes of operation, each accessible through a different subcommand:

```bash
python matmul_toolkit.py <mode> [options]
```

### Available Modes

- `collect`: Collect matrix multiplication timing data
- `create-validation`: Create a separate validation dataset
- `train`: Train a model on collected data
- `validate-model`: Validate a trained model on a validation dataset
- `collector-example`: Run an example demonstrating the MatmulDatasetCollector
- `collector-basic-example`: Run an example demonstrating the basic MatmulCollector
- `model-example`: Run an example demonstrating model training and evaluation
- `collect-and-train`: Collect data and train a model in one step

### Examples

#### Collecting Data

```bash
# Collect timing data for 100 matrix shapes
python matmul_toolkit.py collect --output matmul_dataset.json --num-shapes 100

# Collect data with power-of-two sizes only
python matmul_toolkit.py collect --output matmul_dataset_pow2.json --num-shapes 50 --power-of-two
```

#### Training a Model

```bash
# Train a deep model on collected data
python matmul_toolkit.py train --dataset matmul_dataset.json --model matmul_model.pt

# Train a base model with custom parameters
python matmul_toolkit.py train --dataset matmul_dataset.json --model matmul_base_model.pt \
    --model-type base --batch-size 128 --num-epochs 200 --learning-rate 0.0005
```

#### Validating a Model

```bash
# Validate a trained model on a separate validation dataset
python matmul_toolkit.py validate-model --model matmul_model.pt --dataset matmul_validation_dataset.json
```

#### End-to-End Collection and Training

```bash
# Collect data and train a model in one step
python matmul_toolkit.py collect-and-train --dataset matmul_dataset.json --model matmul_model.pt

# Skip collection if dataset already exists
python matmul_toolkit.py collect-and-train --dataset matmul_dataset.json --model matmul_model.pt --skip-collection
```

## Key Features

### Data Collection

- Configurable matrix sizes (power-of-two, rectangular, odd sizes)
- Support for different data types (float16, float32)
- Exhaustive or default search space for autotuning
- Separate validation dataset creation

### Model Training

- Two model types: base and deep neural network
- Configurable hyperparameters (learning rate, batch size, etc.)
- Early stopping to prevent overfitting
- Training history visualization

### Model Evaluation

- Performance metrics (MSE, RMSE)
- Analysis of worst predictions
- Detailed configuration information for each prediction

### Collectors

- `MatmulDatasetCollector`: Collects data in a structured dataset format
- `MatmulCollector`: Basic collector for simpler use cases

## Command-Line Options

Each mode has its own set of command-line options. Use the `--help` flag to see the available options for each mode:

```bash
python matmul_toolkit.py <mode> --help
```

## Common Options

- `--seed`: Random seed for reproducibility (default: 42)
- `--device`: Device to run on (default: "cuda" if available, otherwise "cpu")

## Data Collection Options

- `--num-shapes`: Number of matrix shapes to test
- `--min-size`/`--max-size`: Minimum/maximum matrix dimension
- `--power-of-two`: Generate only power-of-two sizes
- `--no-rectangular`: Exclude rectangular matrices
- `--no-odd-sizes`: Exclude odd-sized matrices
- `--search-mode`: Search mode for torch.compile
- `--search-space`: Search space for autotuning (EXHAUSTIVE or DEFAULT)

## Model Training Options

- `--model-type`: Type of model to train ("base" or "deep")
- `--batch-size`: Batch size for training
- `--num-epochs`: Number of epochs to train for
- `--learning-rate`: Learning rate for the optimizer
- `--weight-decay`: Weight decay for the optimizer
- `--patience`: Number of epochs to wait for improvement before early stopping
- `--hidden-dim`: Hidden dimension of the model
- `--num-layers`: Number of layers in the model

## Advanced Usage

### Hardware and Operation Filtering

You can filter data by hardware name or operation name:

```bash
# Train on data from a specific GPU
python matmul_toolkit.py train --dataset matmul_dataset.json --hardware-name "NVIDIA A100"

# Train on data from a specific operation
python matmul_toolkit.py train --dataset matmul_dataset.json --op-name "mm"
```

### Skipping Steps in Collect-and-Train

The `collect-and-train` mode allows skipping specific steps:

```bash
# Skip data collection
python matmul_toolkit.py collect-and-train --skip-collection

# Skip validation dataset creation
python matmul_toolkit.py collect-and-train --skip-validation

# Skip model training
python matmul_toolkit.py collect-and-train --skip-training
```

## Internal Structure

The toolkit is organized into logical sections:

1. Utility Functions: Shared functionality across all modes
2. Data Collection Functions: Functions for collecting matrix multiplication data
3. Model Training and Evaluation Functions: Functions for training and evaluating models
4. Main Function: Command-line interface and mode selection
