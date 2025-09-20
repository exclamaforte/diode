#!/usr/bin/env python3
"""
Create a max-autotune solution based on the fastest configs from validation dataset.

This script analyzes the validation dataset to find the fastest configs for different
matrix sizes and creates a max-autotune solution file that can be used for validation.
TODO this script isn't done but should be done at some point
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from torch_diode.model.directory_dataset_loader import create_directory_dataloaders
from torch_diode.types.matmul_types import Solution, TritonGEMMConfig

def analyze_validation_data_and_create_solution():
    """Analyze validation data and create a max-autotune solution with the fastest configs."""
    
    validation_dataset_path = "/home/gabeferns/diode/examples/data/validation"
    output_path = "/home/gabeferns/diode/examples/validation_based_max_autotune.json"
    
    print("Loading validation dataset...")
    
    # Create dataloaders to get all validation data
    _, val_dataloader, _ = create_directory_dataloaders(
        data_dir=validation_dataset_path,
        batch_size=1000,  # Large batch size for efficiency
        hardware_name=None,
        op_name=None,
        log_transform=True,
        num_workers=4,
        seed=42,
        file_extensions=["msgpack"],
    )
    
    print(f"Dataset loaded with {len(val_dataloader.dataset)} samples")
    
    # Collect all configs and their runtimes
    all_configs = []
    all_runtimes = []
    
    print("Extracting configs and runtimes...")
    batch_count = 0
    for batch_idx, (_, _, targets) in enumerate(val_dataloader):
        batch_count += 1
        if batch_count % 100 == 0:
            print(f"  Processed {batch_count} batches...")
            
        batch_size_actual = len(targets)
        for i in range(batch_size_actual):
            subset_idx = batch_idx * val_dataloader.batch_size + i
            if subset_idx < len(val_dataloader.dataset):
                actual_dataset_idx = val_dataloader.dataset.indices[subset_idx]
                config = val_dataloader.dataset.dataset.timing_dataset.configs[actual_dataset_idx]
                
                all_configs.append(config)
                all_runtimes.append(float(targets[i].cpu().numpy()))
    
    print(f"Extracted {len(all_configs)} configs and runtimes")
    
    # Group configs by similar characteristics and find the fastest ones
    print("Analyzing configs to find fastest performers...")
    
    # Group by block sizes (this is a simple grouping strategy)
    config_groups = defaultdict(list)
    
    for i, config in enumerate(all_configs):
        # Create a key based on block sizes (you can modify this grouping strategy)
        key = (config.block_m, config.block_n, config.block_k)
        config_groups[key].append((config, all_runtimes[i], i))
    
    print(f"Grouped configs into {len(config_groups)} groups")
    
    # Find the fastest config in each group
    fastest_configs = []
    
    for key, configs_in_group in config_groups.items():
        if len(configs_in_group) > 0:
            # Sort by runtime (log-transformed, so lower is better)
            configs_in_group.sort(key=lambda x: x[1])
            
            fastest_config, fastest_runtime, original_idx = configs_in_group[0]
            fastest_configs.append((fastest_config, fastest_runtime, key))
    
    print(f"Found {len(fastest_configs)} fastest configs")
    
    # Sort by runtime and take top configs
    fastest_configs.sort(key=lambda x: x[1])
    
    # Take top 50 fastest configs (or fewer if we don't have enough)
    num_configs_to_take = min(50, len(fastest_configs))
    selected_configs = [config for config, runtime, key in fastest_configs[:num_configs_to_take]]
    
    print(f"Selected top {len(selected_configs)} configs for max-autotune solution")
    
    # Print some statistics
    print("\nTop 10 fastest configs:")
    for i, (config, runtime, key) in enumerate(fastest_configs[:10]):
        actual_runtime = np.exp(runtime)
        print(f"  {i+1}. {key} - log_runtime: {runtime:.6f}, actual_runtime: {actual_runtime:.6f}")
    
    # Create Solution object
    solution = Solution(config=selected_configs)
    
    # Save to JSON
    print(f"\nSaving solution to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(str(solution))
    
    print(f"Max-autotune solution created with {len(selected_configs)} configs")
    print(f"Saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    analyze_validation_data_and_create_solution()
