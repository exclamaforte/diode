# heur
`heur` is a framework for defining heuristics in `torch` and `torch.compile`. It makes it easy to develop heuristics that plug into the external interfaces of `torch` and `torch.compile`.

## Target Audience:
- Hardware Vendors looking to optimize `torch` heuristics for their hardware.
- Developers looking to adapt models to their specific situation.

## Common Herustic Types
- ML Models: Train
- Lookup Tables: Cache

## Features:
- Data collection code: gather data from the external interfaces
- Stable Type definitions for input/output of heuristics
- Classes that encapsulate entrypoints for torch
- Model Training Code
