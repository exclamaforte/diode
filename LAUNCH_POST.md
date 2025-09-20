# 🚀 Launch: torch-diode - ML-Driven PyTorch Compilation Optimization

## Overview

**torch-diode** is a library that enables programmatic control over performance decisions in PyTorch's `torch.compile`. Instead of relying on fixed heuristics, it allows you to collect data on compilation choices and train machine learning models to make optimal decisions for your specific hardware and workloads.

### Key Features

- **Data Collection**: Gather comprehensive timing data from `torch.compile` operations
- **Model Training**: Train deep neural networks on performance data
- **Max-Autotune Integration**: Compare model predictions against exhaustive search results
- **Hardware Optimization**: Adapt compilation decisions to specific hardware configurations
- **Pre-trained Models**: Leverage community-trained models via `torch-diode` package

### Target Audience

- Hardware vendors optimizing PyTorch for their platforms
- Developers seeking workload-specific compilation tuning
- OSS contributors adding support for emerging hardware
- Researchers studying compilation optimization

## Installation

```bash
# With pre-trained models (auto-registration to torch.compile)
pip install torch-diode

# Library-only version (no auto-registration)
pip install torch-diode-lib

# From source
git clone https://github.com/exclamaforte/diode.git
cd diode && pip install .
```

## Tutorial: Matrix Multiplication Performance Optimization

This tutorial demonstrates collecting data, training a model, and validating performance using the matrix multiplication toolkit.

### Step 1: Data Collection

Collect timing data for various matrix multiplication shapes using log-normal distribution sampling:

```bash
~/.conda/envs/foo2/bin/python matmul_toolkit.py \
    --format msgpack --seed 42 collect \
    --output tutorial_dataset.msgpack \
    --num-shapes 10 \
    --search-space EXHAUSTIVE \
    --search-mode max-autotune \
    --log-normal
```

**Sample Output:**
```
[INFO] [8/10] Running bmm with size (400, 24) x (24, 416) and dtype torch.bfloat16
Autotune Choices Stats:
{"num_choices": 560, "best_kernel": "triton_mm_6255", 
 "best_time": 0.021824000403285027, "best_triton_pos": 0}

Dataset Statistics:
Hardware: NVIDIA H100
  Operation 'mm': 9 problems, 7440 configs
  Operation 'addmm': 1 problems, 1206 configs
```

### Step 2: Model Training

Train a deep neural network on the collected performance data:

```bash
~/.conda/envs/foo2/bin/python matmul_toolkit.py \
    --seed 42 train \
    --data-dir . \
    --model tutorial_model.pt \
    --model-type deep \
    --batch-size 32 \
    --num-epochs 50 \
    --learning-rate 0.001 \
    --file-extensions msgpack
```

*Note: Training can take significant time. Pre-trained model available at `tutorial_model.pt`.*

### Step 3: Validation Dataset Creation

Create a separate validation dataset for model evaluation:

```bash
~/.conda/envs/foo2/bin/python matmul_toolkit.py \
    --format msgpack --seed 43 create-validation \
    --output tutorial_validation.msgpack \
    --num-shapes 5 \
    --search-space EXHAUSTIVE \
    --search-mode max-autotune \
    --log-normal
```

**Sample Output:**
```
Dataset Statistics:
Hardware: NVIDIA H100
  Operation 'mm': 3 problems, 2414 configs
    Problem 1: M=1376, N=80, K=216, dtype=torch.float16, 1039 configs
    Fastest config: block_m=128, block_n=16, block_k=128, time=22.016 ms
    Problem 2: M=2552, N=10616, K=7616, dtype=torch.bfloat16, 1206 configs
    Fastest config: block_m=256, block_n=128, block_k=32, time=739.616 ms
  Operation 'addmm': 2 problems, 2412 configs
```

### Step 4: Model Validation

Evaluate model performance on the validation dataset:

```bash
~/.conda/envs/foo2/bin/python matmul_toolkit.py \
    --seed 42 validate-model \
    --model tutorial_model.pt \
    --dataset tutorial_validation.msgpack \
    --batch-size 32 \
    --top-n-worst 5
```

**Results:**
```
Validation Loss (MSE): 2.849390
Validation RMSE: 1.688013
Validation RMSE (exp): 5.408725

Top 5 worst predictions:
Error 1: Predicted: 0.457ms, Actual: 13.427ms (0.03x ratio)
  Matrix: (2552, 7615) x (7615, 10616)
  Config: block_m=16, block_n=16, block_k=16
Error 2: Predicted: 0.457ms, Actual: 10.772ms (0.04x ratio)
  Matrix: (2552, 7615) x (7615, 10616) 
  Config: block_m=16, block_n=16, block_k=32
```

### Step 5: Max-Autotune Comparison

Compare model predictions against exhaustive autotuning results:

```bash
# Create max-autotune solution reference
echo '{"name": "tutorial_max_autotune", "configs": [
  {"M": 1376, "N": 80, "K": 216, "block_m": 128, "block_n": 16, "block_k": 128},
  {"M": 2552, "N": 10616, "K": 7616, "block_m": 256, "block_n": 128, "block_k": 32}
]}' > tutorial_max_autotune.json

# Run comparison
~/.conda/envs/foo2/bin/python matmul_toolkit.py \
    --seed 42 validate-max-autotune \
    --model tutorial_model.pt \
    --dataset tutorial_validation.msgpack \
    --max-autotune-solution tutorial_max_autotune.json \
    --batch-size 32
```

**Output:**
```
MODEL VS MAX-AUTOTUNE COMPARISON RESULTS
Comprehensive per-op statistics saved to: model_performance_analysis_by_op.json
Max-autotune validation completed successfully
```

## Advanced Usage

### Full Workflow Script

The complete workflow is automated in `/home/gabeferns/diode/workflows/matmul_workflow.sh`:

```bash
#!/bin/bash
# Matrix multiplication workflow - data collection, training, and validation
set -e

PYTHON_CMD="${HOME}/.conda/envs/foo2/bin/python"
SEED=51
DATA_DIR="./data"
TRAIN_DATASET="${DATA_DIR}/seed_${SEED}_train_dataset.msgpack"
MODEL_PATH="${DATA_DIR}/matmul_model.pt"

# Step 1: Collect training data
${PYTHON_CMD} matmul_toolkit.py \
    --format msgpack --seed ${SEED} collect \
    --output ${TRAIN_DATASET} \
    --log-normal \
    --search-space EXHAUSTIVE \
    --search-mode max-autotune

# Step 2: Train model (commented for speed)
# ${PYTHON_CMD} matmul_toolkit.py train --data-dir ${DATA_DIR} --model ${MODEL_PATH}

# Step 3: Create validation dataset
# ${PYTHON_CMD} matmul_toolkit.py create-validation --output ${VALIDATION_DATASET}

# Step 4: Validate model
${PYTHON_CMD} matmul_toolkit.py validate-model \
    --model ${MODEL_PATH} \
    --dataset ${VALIDATION_DIR}
```

### Max-Autotune Validation Script

Enhanced validation using `/home/gabeferns/diode/workflows/matmul_max_autotune_validation.sh`:

```bash
#!/bin/bash
# Compare model predictions against max-autotune for various top-N values
PYTHON_CMD="${HOME}/.conda/envs/foo2/bin/python"
SEED=41
MODEL_PATH="./data/matmul_model.pt"
VALIDATION_DATASET="./data/validation/my_validation_86.msgpack"
MAX_AUTOTUNE_SOLUTION="./data/max_autotune_focused.json"

${PYTHON_CMD} matmul_toolkit.py \
    --seed ${SEED} validate-max-autotune \
    --model ${MODEL_PATH} \
    --dataset ${VALIDATION_DATASET} \
    --max-autotune-solution ${MAX_AUTOTUNE_SOLUTION}
```

## Command Reference

### Data Collection
```bash
# Basic collection
python matmul_toolkit.py collect --output data.msgpack --num-shapes 100

# Log-normal distribution with exhaustive search
python matmul_toolkit.py --format msgpack collect \
    --output data.msgpack --log-normal --search-space EXHAUSTIVE

# From operation shapeset
python matmul_toolkit.py collect-shapeset \
    --shapeset operation_shapeset.json --output shapeset_data.msgpack
```

### Model Training
```bash
# Train on single dataset
python matmul_toolkit.py train --dataset data.msgpack --model model.pt

# Train on directory of files
python matmul_toolkit.py train --data-dir ./data --model model.pt \
    --model-type deep --batch-size 64 --num-epochs 100
```

### Model Validation
```bash
# Basic validation
python matmul_toolkit.py validate-model \
    --model model.pt --dataset validation.msgpack

# Max-autotune comparison
python matmul_toolkit.py validate-max-autotune \
    --model model.pt --dataset validation.msgpack \
    --max-autotune-solution solutions.json
```

## Performance Impact

The trained models enable `torch.compile` to make optimal kernel choices without exhaustive search, providing:

- **Faster Compilation**: Avoid expensive autotuning for common patterns
- **Better Performance**: ML-driven decisions outperform simple heuristics
- **Hardware Adaptation**: Models learn hardware-specific optimization patterns
- **Workload Specialization**: Training on domain-specific data improves accuracy

## Project Structure

```
diode/
├── torch_diode/           # Core library
│   ├── collection/        # Data collection utilities
│   ├── model/            # Model training and inference
│   ├── types/            # Type definitions
│   └── integration/      # PyTorch integration
├── workflows/            # Example workflows and scripts
├── trained_models/       # Pre-trained model storage
└── diode_datasets/       # Dataset definitions and examples
```

## Contributing

- **Data Contribution**: Share performance datasets for community models
- **Hardware Support**: Add support for new hardware platforms
- **Model Improvements**: Contribute enhanced model architectures
- **Benchmarking**: Validate models across different workloads

## Status

⚠️ **Pre-Alpha**: Active development, APIs subject to change

---

*For detailed documentation and examples, see the [workflows README](workflows/README.md) and [project repository](https://github.com/exclamaforte/diode).*