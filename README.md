Warning: code is in pre-Alpha


<img width="718" height="571" alt="diode" src="https://github.com/user-attachments/assets/308cb05a-01d9-4fc4-9c03-7e13ade91475" />

# torch-diode
`torch-diode` is a library for programattically altering the performance-relevant decisions made by `torch.compile`. It makes it easy to gather data on the outcomes of decisions, and then train Machine Learning models on that data. It initially focuses on Matmul Kernel selection, but it will be expanded to other decisions in the future.

## Target Audience:
- Developers looking to adapt the compilation of their model to their specific situation.
- Hardware Vendors looking to optimize `torch.compile` heuristics for their hardware.
- OSS Contributors looking to add support for less popular hardware.

## Features:
- Pre-Trained Models: Profit from community efforts to gather data and train models.
- Data collection: Gather data from torch external interfaces.
- Stable Type Definitions: storing data from the external interfaces.
- Model Training Code: Train ML models on the gathered data and contribute back to the `torch` community.


## Featured Models
- Matmul Kernel Prediction: Predict the runtime of matmul kernels. The results of this model are enabled in `fast-autotune`.

## Option 1: Installation with Pre-Trained Models

If you want to get access to the pre-trained performance models, as well as the libraries, install `torch-diode`:
```
$ pip install torch-diode
```
And then import torch_diode in python:
```
import torch_diode
```

This import has several side-effects, each of which are dependent on the success of the previous step:
1. Attempt to import `torch`.
1. Register dummy models to the relevant `torch.compile` interfaces.
1. For each registration that is successful, it will load the actual model and register it.
1. Enable the configs in `torch.compile` that engage the models.
## Option 2: Installation without Pre-Trained Models

`diode` requires nightly pytorch, or pytorch `2.9` or later.

For developers who don't want these side effects, simply installing `torch-diode-lib` will get access to the library.

```
$ pip install torch-diode-lib
```

The import remains the same:
```
import torch_diode
```

### Option 3: Install from Source
```bash
git clone https://github.com/exclamaforte/diode.git
cd diode
pip install .
```

## Model Organization

### Directory Structure
Models are organized in a structured directory format:
```
trained_models/<model_purpose>/<model_name>.pt
```
## Get Started

[The main entry point is in workflows.](https://github.com/exclamaforte/diode/tree/main/workflows#readme)

### Package Variants
- **torch-diode**: Full package with auto-registration to PyTorch Inductor
- **torch-diode-lib**: Library-only version without auto-registration
