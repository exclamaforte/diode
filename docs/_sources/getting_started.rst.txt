Getting Started with Training Models with Diode
==========================

This comprehensive guide will walk you through the complete process of creating a machine learning model from scratch using the Diode toolkit. You'll learn how to generate a dataset of matrix multiplication performance data and train a model to predict optimal configurations.

Overview
--------

The Diode workflow involves four main steps:

1. **Data Collection**: Generate matrix multiplication performance data using PyTorch's autotuning capabilities
2. **Model Training**: Train a deep learning model on the collected data
3. **Validation Dataset Creation**: Create a separate validation dataset from predefined operation shapes
4. **Model Validation**: Evaluate the trained model's performance on the validation dataset

Prerequisites
-------------

Before starting, ensure you have:

* Access to your target hardware
* PyTorch nightlies
* The Diode toolkit

Step 1: Data Collection
-----------------------

The first step is to generate a training dataset by collecting matrix multiplication performance data. Diode uses PyTorch's feedback saver interface to automatically collect timing information for different matrix multiplication configurations.

Setting Up the Data Collector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``MatmulDatasetCollector`` class provides flexible data collection capabilities:

.. code-block:: python

    from diode.collection.matmul_dataset_collector import MatmulDatasetCollector, CollectionMode

    # Initialize the collector with log-normal distribution mode
    collector = MatmulDatasetCollector(
        hardware_name="your_gpu_name",
        mode=CollectionMode.LOG_NORMAL,
        operations=["mm", "addmm", "bmm"],
        num_shapes=1000,
        seed=50,
    )

Collection Modes
~~~~~~~~~~~~~~~~

Diode supports three collection modes:

1. **LOG_NORMAL**: Uses log-normal distributions to generate realistic matrix sizes based on production workloads
2. **RANDOM**: Generates uniformly random matrix sizes within specified bounds
3. **OPERATION_SHAPE_SET**: Uses predefined shapes from a configuration file

Running Data Collection
~~~~~~~~~~~~~~~~~~~~~~~

Use the matmul_toolkit.py script to collect training data:

.. code-block:: bash

    python matmul_toolkit.py \
        --format msgpack \
        --seed 50 \
        collect \
        --output train_dataset.msgpack \
        --num-shapes 1000 \
        --log-normal \
        --search-space EXHAUSTIVE \
        --search-mode max-autotune \
        --chunk-size 5

Key parameters:

* ``--format msgpack``: Use MessagePack format for efficient serialization
* ``--seed 50``: Set random seed for reproducibility
* ``--num-shapes 1000``: Generate 1000 different matrix configurations
* ``--log-normal``: Use log-normal distribution for realistic sizes
* ``--search-space EXHAUSTIVE``: Use exhaustive search for optimal configurations
* ``--search-mode max-autotune``: Use PyTorch's max-autotune mode
* ``--chunk-size 5``: Write data every 5 operations to prevent data during collection

Understanding the Collection Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The data collection process works by:

1. Generating matrix shapes based on the selected mode
2. Creating random tensors with the specified dimensions and data types
3. Compiling matrix multiplication operations with PyTorch's autotuning
4. Capturing timing data for different Triton GEMM configurations through the feedback saver interface
5. Storing the results in a structured dataset format

Step 2: Model Training
----------------------

Once you have collected training data, train a deep learning model to predict optimal GEMM configurations:

.. code-block:: bash

    python matmul_toolkit.py \
        --seed 50 \
        train \
        --data-dir ./data \
        --model matmul_model.pt \
        --model-type deep \
        --batch-size 64 \
        --num-epochs 1000 \
        --learning-rate 0.001 \
        --log-dir ./logs

Training parameters:

* ``--model-type deep``: Use a deep neural network architecture
* ``--batch-size 64``: Process 64 samples per batch
* ``--num-epochs 1000``: Train for 1000 epochs
* ``--learning-rate 0.001``: Set the learning rate
* ``--log-dir``: Directory to save training logs and metrics

The model learns to predict optimal Triton GEMM configurations based on matrix dimensions, data types, and hardware characteristics.

Model Architecture
~~~~~~~~~~~~~~~~~~

Diode provides two simple neural network architectures for timing prediction. These are not meant to be state-of-the-art models, but rather serve as a starting point for further experimentation and development:

**Standard Model (MatmulTimingModel)**

The standard model uses a feedforward neural network with the following architecture:

.. code-block:: python

    class MatmulTimingModel(nn.Module):
        def __init__(
            self,
            problem_feature_dim: int,
            config_feature_dim: int,
            hidden_dims: List[int] = [256, 512, 256, 128, 64],
            dropout_rate: float = 0.2,
        ):

Architecture components:

* **Input Layer**: Concatenates problem features (matrix dimensions, data types) and configuration features (Triton GEMM parameters)
* **Hidden Layers**: Multiple fully connected layers with ReLU activation, batch normalization, and dropout
* **Output Layer**: Single neuron predicting log execution time
* **Regularization**: Dropout and batch normalization to prevent overfitting

**Deep Model (DeepMatmulTimingModel)**

The deep model uses residual connections for training deeper networks:

.. code-block:: python

    class DeepMatmulTimingModel(nn.Module):
        def __init__(
            self,
            problem_feature_dim: int,
            config_feature_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 10,
            dropout_rate: float = 0.2,
        ):

Key features:

* **Residual Blocks**: Each block contains two linear layers with skip connections
* **Deeper Architecture**: 10+ layers with consistent hidden dimensions
* **Better Gradient Flow**: Residual connections help train deeper networks effectively

**Residual Block Implementation**

.. code-block:: python

    class ResidualBlock(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            identity = x
            out = self.block(x)
            out += identity  # Skip connection
            out = self.relu(out)
            return out

The residual blocks enable training much deeper networks while maintaining stable gradients throughout the network depth.

Step 3: Creating a Validation Dataset
-------------------------------------

Create a separate validation dataset using predefined operation shapes to evaluate model performance:

.. code-block:: bash

    python matmul_toolkit.py \
        --format msgpack \
        --seed 50 \
        create-validation \
        --output validation_dataset.msgpack \
        --shapeset operation_shapeset.json \
        --operations mm addmm bmm \
        --search-space EXHAUSTIVE \
        --search-mode max-autotune

This step:

* Loads predefined matrix shapes from ``operation_shapeset.json``
* Runs autotuning to find optimal configurations for these shapes
* Creates a validation dataset with known ground truth performance data

Step 4: Model Validation
------------------------

Finally, evaluate your trained model against the validation dataset:

.. code-block:: bash

    python matmul_toolkit.py \
        --seed 50 \
        validate-model \
        --model matmul_model.pt \
        --dataset validation_dataset.msgpack \
        --batch-size 64 \
        --top-n-worst 10

This validation step:

* Loads the trained model and validation dataset
* Makes predictions for each validation sample
* Compares predictions against ground truth timing data
* Reports accuracy metrics and identifies the worst-performing predictions

Complete Workflow Script
------------------------

Here's a complete bash script that orchestrates the entire process:

.. code-block:: bash

    #!/bin/bash

    set -e  # Exit on any error

    # Configuration
    SEED=50
    DATA_DIR="./data"
    TRAIN_DATASET="${DATA_DIR}/seed_${SEED}_train_dataset.msgpack"
    VALIDATION_DATASET="${DATA_DIR}/validation/validation_dataset.msgpack"
    MODEL_PATH="${DATA_DIR}/matmul_model.pt"
    LOG_DIR="${DATA_DIR}/logs"
    NUM_SHAPES=1000
    NUM_EPOCHS=1000
    PYTHON_CMD="python"
    TOOLKIT_PATH="matmul_toolkit.py"
    OPERATION_SHAPESET_PATH="operation_shapeset.json"

    echo "Starting Diode workflow..."

    # Step 1: Create data directory
    mkdir -p "${DATA_DIR}"
    mkdir -p "${DATA_DIR}/validation"

    # Step 2: Generate training dataset
    echo "Collecting training data..."
    ${PYTHON_CMD} "${TOOLKIT_PATH}" \
        --format msgpack \
        --seed "${SEED}" \
        collect \
        --output "${TRAIN_DATASET}" \
        --num-shapes ${NUM_SHAPES} \
        --log-normal \
        --search-space EXHAUSTIVE \
        --search-mode max-autotune \
        --chunk-size 5

    # Step 3: Train model
    echo "Training model..."
    ${PYTHON_CMD} "${TOOLKIT_PATH}" \
        --seed "${SEED}" \
        train \
        --data-dir "${DATA_DIR}" \
        --model "${MODEL_PATH}" \
        --model-type deep \
        --batch-size 64 \
        --num-epochs ${NUM_EPOCHS} \
        --learning-rate 0.001 \
        --log-dir "${LOG_DIR}"

    # Step 4: Create validation dataset
    echo "Creating validation dataset..."
    ${PYTHON_CMD} "${TOOLKIT_PATH}" \
        --format msgpack \
        --seed "${SEED}" \
        create-validation \
        --output "${VALIDATION_DATASET}" \
        --shapeset "${OPERATION_SHAPESET_PATH}" \
        --operations mm addmm bmm \
        --search-space EXHAUSTIVE \
        --search-mode max-autotune

    # Step 5: Validate model
    echo "Validating model..."
    ${PYTHON_CMD} "${TOOLKIT_PATH}" \
        --seed "${SEED}" \
        validate-model \
        --model "${MODEL_PATH}" \
        --dataset "${VALIDATION_DATASET}" \
        --batch-size 64 \
        --top-n-worst 10

    echo "Workflow completed successfully!"

Advanced Configuration
----------------------

Custom Collection Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more control over data collection, you can customize the log-normal distribution parameters:

.. code-block:: python

    # Custom parameters for different workload characteristics
    collector = MatmulDatasetCollector(
        mode=CollectionMode.LOG_NORMAL,
        # Larger matrices (shift mean higher)
        log_normal_m_mean=7.0,
        log_normal_n_mean=6.5,
        log_normal_k_mean=6.8,
        # Smaller variance for more consistent sizes
        log_normal_m_std=1.5,
        log_normal_n_std=1.2,
        log_normal_k_std=1.8,
    )

Tips
----------------

1. **Start Small**: Begin with a smaller number of shapes (100-200) to validate your setup
2. **Monitor Memory**: Keep an eye on GPU memory usage during collection
3. **Save Frequently**: Use the ``--chunk-size`` parameter to save data periodically
4. **Reproducibility**: Always set a random seed for consistent results
5. **Hardware Consistency**: Collect training and validation data on the same hardware

Next Steps
----------

After completing this workflow, you can:

* Experiment with different model architectures
* Collect data for specific workloads using OPERATION_SHAPE_SET mode
* Integrate the trained model into your own applications
* Analyze the collected data to understand performance patterns

For more advanced usage, see the API documentation and examples in the repository.
