# Diode Collection Module

This module provides tools for collecting data from PyTorch's feedback saver interface and storing it in structured types.

## MatmulCollector

The `MatmulCollector` class is designed to hook into PyTorch's feedback saver interface to collect matrix multiplication data and store it in structured types defined in `diode.types.matmul_types`.

### Features

- Hooks into PyTorch's feedback saver interface
- Collects data from matrix multiplication operations (`mm` and `addmm`)
- Stores data in structured types (`TritonGEMMConfig`, `MMProblem`, `Solution`, etc.)
- Supports saving and loading data to/from files
- Can be used as a context manager

### Usage

#### Basic Usage

```python
from diode.collection.matmul_collector import MatmulCollector
import torch

# Create a collector instance
collector = MatmulCollector(hardware_name="my_gpu")

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
collector.save_to_file("matmul_data.json")
```

#### Using as a Context Manager

```python
with MatmulCollector(hardware_name="my_gpu") as collector:
    # Run matrix multiplication operations
    # ...

# Save the collected data
collector.save_to_file("matmul_data.json")
```

#### Loading Data from a File

```python
collector = MatmulCollector()
collector.load_from_file("matmul_data.json")

# Access the collected data
table = collector.get_table()
```

### API Reference

#### `__init__(hardware_name="unknown")`

Initialize the MatmulCollector.

- `hardware_name`: The name of the hardware being used.

#### `start_collection()`

Start collecting data by hooking into the feedback saver interface.

#### `stop_collection()`

Stop collecting data by removing the feedback saver hook.

#### `get_table()`

Get the collected data as a `Table` object.

#### `save_to_file(file_path)`

Save the collected data to a file.

- `file_path`: Path to save the data to.

#### `load_from_file(file_path)`

Load data from a file.

- `file_path`: Path to load the data from.

### Example

See the [example script](../../examples/matmul_collector_example.py) for a complete example of how to use the `MatmulCollector` class.
