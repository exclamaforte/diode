# Diode Collection Module

This module provides tools for collecting data from PyTorch's feedback saver interface and storing it in structured types.

## MatmulDatasetCollector

The `MatmulDatasetCollector` class is designed to hook into PyTorch's feedback saver interface to collect matrix multiplication data with timing information and store it in structured types defined in `diode.types.matmul_dataset`.

### Features

- Hooks into PyTorch's feedback saver interface
- Collects data from matrix multiplication operations (`mm` and `addmm`)
- Stores data in structured types (`Dataset`, `TimedConfig`, `TritonGEMMConfig`, `MMShape`, etc.)
- Captures timing information for different configurations
- Supports saving and loading data to/from files
- Can be used as a context manager

### Usage

#### Basic Usage

```python
from torch_diode.collection.matmul_dataset_collector import MatmulDatasetCollector
import torch

# Create a collector instance
collector = MatmulDatasetCollector(hardware_name="my_gpu")

# Start collecting data
collector.start_collection()

# Run some matrix multiplication operations with torch.compile
# This will trigger the feedback saver interface
a = torch.randn(128, 256, device="cuda", dtype=torch.float16)
b = torch.randn(256, 512, device="cuda", dtype=torch.float16)

def mm_fn(x, y):
    return torch.mm(x, y)

compiled_mm = torch.compile(mm_fn, mode="max-autotune")
result = compiled_mm(a, b)

# Stop collecting data
collector.stop_collection()

# Save the collected data to a file
collector.save_to_file("matmul_dataset.json")
```

#### Using as a Context Manager

```python
with MatmulDatasetCollector(hardware_name="my_gpu") as collector:
    # Run matrix multiplication operations
    # ...

# Save the collected data
collector.save_to_file("matmul_dataset.json")
```

#### Loading Data from a File

```python
collector = MatmulDatasetCollector()
collector.load_from_file("matmul_dataset.json")

# Access the collected data
dataset = collector.get_dataset()

# Convert to a table (selecting fastest config for each problem)
table = collector.to_table()
```

### API Reference

#### `__init__(hardware_name="unknown")`

Initialize the MatmulDatasetCollector.

- `hardware_name`: The name of the hardware being used.

#### `start_collection()`

Start collecting data by hooking into the feedback saver interface.

#### `stop_collection()`

Stop collecting data by removing the feedback saver hook.

#### `get_dataset()`

Get the collected data as a `Dataset` object.

#### `to_table()`

Convert the dataset to a table by selecting the fastest configuration for each problem.

#### `save_to_file(file_path)`

Save the collected dataset to a file.

- `file_path`: Path to save the data to.

#### `load_from_file(file_path)`

Load dataset from a file.

- `file_path`: Path to load the data from.

#### `save_table_to_file(file_path)`

Convert the dataset to a table and save it to a file.

- `file_path`: Path to save the table to.

### Example

See the [example script](../../examples/matmul_dataset_collector_example.py) for a complete example of how to use the `MatmulDatasetCollector` class.
