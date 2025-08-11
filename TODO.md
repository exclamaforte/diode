- TODO create
- TODO finish set up documentation generation
- TODO set up CI


# diode
`diode` is a framework for defining diodeistics in `torch` and `torch.compile`. It makes it easy to develop diodeistics that plug into the external interfaces of `torch` and `torch.compile`.

Torch has interfaces that allow users to both gather data and make decisions on and with the compile process. These interfaces can hard to discover and use, and this project aims to systematize these to make them more approachable.

## Target Audience:
- Hardware Vendors looking to optimize `torch` diodeistics for their hardware.
- OSS Contributors looking to add support for less popular hardware.
- Developers looking to adapt the compilation of their model to their specific situation.

## Common Herustic Types
- ML Models
- Lookup Tables
- Custom Logic

## Features:
- Data collection: gather data from torch external interfaces.
- Stable Type Definitions: storing data from the external interfaces.
- Model Training Code: train ML models to predict code.
- Caching code: Create and Edit a LUT from these intefraces using standard tools

## Featured Heruistics
- Matmul Kernel Prediction: Predict the runtime of matmul kernel

## Interface Locations
- _inductor/choices.py
- _inductor/config.py
