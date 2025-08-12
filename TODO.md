- TODO create
- TODO finish set up documentation generation
- TODO set up CI
1. create a trained_models directory that contains trained models
2. run README.md this to create some example models in that directory for h100.
3. Create a model_wrapper code in diode/model that will load matmul_timing_model.pys, torch.compile them, and then run inference on the model.
4. I want to create two pypi packages: diode and diode-models. One contains everything, and the other contains just the models in trained_models along with the new inference code so that they can be run. The model wrapper should be importable via it's file name on the module diode-object
5. create some integration tests that are tests for this functionality


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
