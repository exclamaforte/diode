Warning: code is in pre-Alpha


<img width="718" height="571" alt="diode" src="https://github.com/user-attachments/assets/308cb05a-01d9-4fc4-9c03-7e13ade91475" />

# diode
`diode` is a framework that makes it easy to develop heuristics that plug into the external interfaces of `torch` and `torch.compile`. It allows users to both gather data from torch and train Machine Learning models on the gathered data.

## Target Audience:
- Hardware Vendors looking to optimize `torch` heuristics for their hardware.
- OSS Contributors looking to add support for less popular hardware.
- Developers looking to adapt the compilation of their model to their specific situation.

## Features:
- Pre-Trained Models: Profit from community efforts to gather data and train models.
- Data collection: Gather data from torch external interfaces.
- Stable Type Definitions: storing data from the external interfaces.
- Model Training Code: Train ML models on the gathered data and contribute back to the `torch` community.

## Common Herustic Types
- ML Models: Collect and train on data directly from `torch`.
- Custom Logic: Come with your own ideas/functions.

## Featured Heruistics
- Matmul Kernel Prediction: Predict the runtime of matmul kernel.

## Model Organization

### Directory Structure
Models are organized in a structured directory format:
```
diode_models/diode_models/<heuristic>/<hardware>/model
```

For example:
```
diode_models/diode_models/matmul/nvidia-h100/matmul_nvidia_h100_deep.pt
```

## Get Started

[The main entry point is in examples.](https://github.com/exclamaforte/diode/tree/main/examples#readme)
## Interface Locations
- _inductor/choices.py
- _inductor/config.py

## Install

### Option 1: Install from PyPi (Pending)
```
pip install torch-diode
```

### Option 2: Install from Source
```
git clone https://github.com/exclamaforte/diode.git
cd diode
pip install .
```
