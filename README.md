Warning: code is in pre-Alpha

# diode
`diode` is a framework for defining heuristics in `torch` and `torch.compile`. It makes it easy to develop heuristics that plug into the external interfaces of `torch` and `torch.compile`.

Torch has interfaces that allow users to both gather data and make decisions on and with the compile process. These interfaces can hard to discover and use, and this project aims to systematize these to make them more approachable.

## Target Audience:
- Hardware Vendors looking to optimize `torch` heuristics for their hardware.
- OSS Contributors looking to add support for less popular hardware.
- Developers looking to adapt the compilation of their model to their specific situation.

## Common Herustic Types
- ML Models: Collect and train on data directly from `torch`.
- Lookup Tables: Build a json based lookup table that's editable by external tools like `jq`.
- Custom Logic: Come with your own ideas/functions.

## Features:
- Pre-Trained Models: Profit from community efforts to gather data and train models.
- Data collection: Gather data from torch external interfaces.
- Stable Type Definitions: storing data from the external interfaces.
- Model Training Code: Train ML models on the gathered data and contribute back to the `torch` community.
- Caching code: Create Lookup Tables (LUT) to ensure maximum performance.

## Featured Heruistics
- Matmul Kernel Prediction: Predict the runtime of matmul kernel.

## Model Organization

### Directory Structure
Models are organized in a structured directory format:
```
diode-models/diode_models/<heuristic>/<hardware>/model
```

For example:
```
diode-models/diode_models/matmul/nvidia-h100/matmul_nvidia_h100_deep.pt
```

## Get Started

[The main entry point is in examples.](https://github.com/exclamaforte/diode/tree/main/examples#readme)
## Interface Locations
- _inductor/choices.py
- _inductor/config.py
