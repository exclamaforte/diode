# heur
`heur` is a framework for defining heuristics in `torch` and `torch.compile`. It makes it easy to develop heuristics that plug into the external interfaces of `torch` and `torch.compile` in an effort to make these more hackable.

Torch has interfaces that allow users to both gather data and make decisions via registration mechanisms. These interfaces can hard to discover and use, and this project aims to systematize these to make them more approachable.

## Target Audience:
- Hardware Vendors looking to optimize `torch` heuristics for their hardware.
- Developers looking to adapt models to their specific situation.

## Common Herustic Types
- ML Models: Train
- Lookup Tables: Cache

## Features:
- Data collection: gather data from torch external interfaces.
- Stable Type Definitions: storing data from the external interfaces.
- Model Training Code: train ML models to predict code.
- Caching code: Create a LUT from these intefraces.

## Featured Heruistics
- Matmul Kernel Prediction: Predict the runtime of matmul kernel
