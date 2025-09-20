#!/usr/bin/env python3
"""
Script to analyze configs in the validation dataset and compare them to max-autotune solution.
This will help identify config mismatches and find the closest available configs.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from torch_diode.types.matmul_dataset import Dataset as MatmulDataset
from torch_diode.types.matmul_types import Solution, TritonGEMMConfig


def config_distance(config1: TritonGEMMConfig, config2: TritonGEMMConfig) -> float:
    """
    Compute a distance metric between two configs.
    Lower distance means more similar configs.
    """
    # Key numerical parameters to compare
    distance = 0.0
    
    # Block size differences (weighted heavily)
    distance += abs(config1.block_m - config2.block_m) * 0.5
    distance += abs(config1.block_n - config2.block_n) * 0.5
    distance += abs(config1.block_k - config2.block_k) * 0.5
    
    # Group size difference
    distance += abs(config1.group_m - config2.group_m) * 0.3
    
    # Stage and warp differences
    distance += abs(config1.num_stages - config2.num_stages) * 0.2
    distance += abs(config1.num_warps - config2.num_warps) * 0.2
    
    # Boolean differences
    if config1.EVEN_K != config2.EVEN_K:
        distance += 1.0
    if config1.ALLOW_TF32 != config2.ALLOW_TF32:
        distance += 1.0
    if config1.USE_FAST_ACCUM != config2.USE_FAST_ACCUM:
        distance += 1.0
    
    # String differences
    if config1.ACC_TYPE != config2.ACC_TYPE:
        distance += 1.0
    
    return distance


def analyze_dataset_configs(dataset_path: str) -> Dict[str, any]:
    """Analyze all configs in the dataset."""
    print(f"Loading dataset from {dataset_path}...")
    
    if dataset_path.endswith(".msgpack"):
        with open(dataset_path, "rb") as f:
            dataset_data = f.read()
        dataset = MatmulDataset.from_msgpack(dataset_data)
    else:
        with open(dataset_path, "r") as f:
            dataset_json = f.read()
        dataset = MatmulDataset.deserialize(dataset_json)
    
    if dataset is None:
        raise ValueError(f"Failed to load dataset from {dataset_path}")
    
    print(f"Dataset loaded successfully")
    
    # Extract all configs
    all_configs = set()
    config_count = defaultdict(int)
    shape_config_mapping = defaultdict(set)
    
    # Navigate through dataset structure
    for hw_name, hardware in dataset.hardware.items():
        print(f"Processing hardware: {hw_name}")
        
        for op_name, operation in hardware.operation.items():
            print(f"  Processing operation: {op_name}")
            
            for mmshape, solution in operation.solution.items():
                print(f"    Processing shape: {mmshape}")
                
                for timed_config in solution.timed_configs:
                    config = timed_config.config
                    # Add to sets and counts
                    all_configs.add(config)
                    config_count[config] += 1
                    shape_config_mapping[mmshape].add(config)
    
    return {
        "all_configs": all_configs,
        "config_count": config_count,
        "shape_config_mapping": shape_config_mapping,
        "total_configs": len(all_configs),
        "total_shapes": len(shape_config_mapping)
    }


def analyze_max_autotune_solution(solution_path: str) -> List[TritonGEMMConfig]:
    """Load and return max-autotune configs."""
    print(f"Loading max-autotune solution from {solution_path}...")
    
    with open(solution_path, "r", encoding="utf-8") as f:
        solution_json = f.read()
    
    max_autotune_solution = Solution.parse(solution_json)
    if max_autotune_solution is None:
        raise ValueError("Failed to deserialize max-autotune solution")
    
    print(f"Max-autotune solution loaded with {len(max_autotune_solution.config)} configs")
    return max_autotune_solution.config


def find_closest_configs(dataset_configs: Set[TritonGEMMConfig], 
                        max_autotune_configs: List[TritonGEMMConfig]) -> Dict[str, any]:
    """Find the closest dataset configs to each max-autotune config."""
    results = {}
    
    for ma_config in max_autotune_configs:
        print(f"\nFinding closest match for max-autotune config: {ma_config}")
        
        # Compute distances to all dataset configs
        distances = []
        for dataset_config in dataset_configs:
            distance = config_distance(ma_config, dataset_config)
            distances.append((distance, dataset_config))
        
        # Sort by distance (closest first)
        distances.sort(key=lambda x: x[0])
        
        # Store results
        closest_configs = distances[:5]  # Top 5 closest
        results[ma_config.name] = {
            "max_autotune_config": ma_config,
            "closest_matches": closest_configs,
            "exact_match_found": any(d == 0.0 for d, _ in closest_configs)
        }
        
        print(f"  Closest matches:")
        for i, (distance, config) in enumerate(closest_configs):
            exact = " (EXACT MATCH)" if distance == 0.0 else ""
            print(f"    {i+1}. {config}")
    
    return results


def print_dataset_summary(analysis: Dict[str, any]):
    """Print a summary of the dataset configs."""
    print(f"\n{'='*80}")
    print("DATASET CONFIG SUMMARY")
    print(f"{'='*80}")
    
    print(f"Total unique configs: {analysis['total_configs']}")
    print(f"Total shapes: {analysis['total_shapes']}")
    
    print(f"\nAll configs in dataset:")
    for i, config in enumerate(analysis['all_configs'], 1):
        count = analysis['config_count'][config]
        print(f"  {i}. {config} (used {count} times)")


def generate_better_max_autotune_solution(dataset_analysis: Dict[str, any], 
                                        output_path: str):
    """Generate a better max-autotune solution using actual dataset configs."""
    print(f"\n{'='*80}")
    print("GENERATING IMPROVED MAX-AUTOTUNE SOLUTION")
    print(f"{'='*80}")
    
    # Use all available configs from the dataset as potential max-autotune configs
    # This ensures we have exact matches
    available_configs = list(dataset_analysis['all_configs'])
    
    print(f"Creating max-autotune solution with {len(available_configs)} configs from dataset")
    
    # Create the solution
    solution_dict = {
        "config": [],
        "version": 1
    }
    
    for config in available_configs:
        config_dict = {
            "name": config.name,
            "grid": config.grid,
            "block_m": config.block_m,
            "block_n": config.block_n,
            "block_k": config.block_k,
            "group_m": config.group_m,
            "num_stages": config.num_stages,
            "num_warps": config.num_warps,
            "EVEN_K": config.EVEN_K,
            "ALLOW_TF32": config.ALLOW_TF32,
            "USE_FAST_ACCUM": config.USE_FAST_ACCUM,
            "ACC_TYPE": config.ACC_TYPE
        }
        solution_dict["config"].append(config_dict)
    
    # Save to file
    with open(output_path, "w") as f:
        json.dump(solution_dict, f, indent=2)
    
    print(f"Improved max-autotune solution saved to: {output_path}")


def main():
    # Paths
    validation_dataset_path = "/home/gabeferns/diode/workflows/data/validation/my_validation_86.msgpack"
    max_autotune_solution_path = "/home/gabeferns/diode/workflows/max_autotune.json"
    improved_solution_path = "/home/gabeferns/diode/workflows/data/max_autotune_improved.json"
    
    # Check if files exist
    if not os.path.exists(validation_dataset_path):
        print(f"ERROR: Validation dataset not found at {validation_dataset_path}")
        return
    
    if not os.path.exists(max_autotune_solution_path):
        print(f"ERROR: Max-autotune solution not found at {max_autotune_solution_path}")
        return
    
    try:
        # Analyze dataset configs
        print("Step 1: Analyzing dataset configs...")
        dataset_analysis = analyze_dataset_configs(validation_dataset_path)
        print_dataset_summary(dataset_analysis)
        
        # Analyze max-autotune solution
        print("\nStep 2: Analyzing max-autotune solution...")
        max_autotune_configs = analyze_max_autotune_solution(max_autotune_solution_path)
        
        # Find closest matches
        print("\nStep 3: Finding closest matches...")
        closest_matches = find_closest_configs(dataset_analysis['all_configs'], max_autotune_configs)
        
        # Print summary of matches
        print(f"\n{'='*80}")
        print("MATCH SUMMARY")
        print(f"{'='*80}")
        
        exact_matches = 0
        for config_name, result in closest_matches.items():
            if result['exact_match_found']:
                exact_matches += 1
                status = "✓ EXACT MATCH FOUND"
            else:
                best_distance = result['closest_matches'][0][0]
                status = f"✗ No exact match (closest distance: {best_distance:.2f})"
            
            print(f"{config_name}: {status}")
        
        print(f"\nTotal exact matches: {exact_matches}/{len(max_autotune_configs)}")
        
        # Generate improved solution
        print(f"\nStep 4: Generating improved max-autotune solution...")
        generate_better_max_autotune_solution(dataset_analysis, improved_solution_path)
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Dataset configs analyzed: {dataset_analysis['total_configs']}")
        print(f"Max-autotune configs analyzed: {len(max_autotune_configs)}")
        print(f"Exact matches found: {exact_matches}")
        print(f"Improved solution saved to: {improved_solution_path}")
        
        if exact_matches == 0:
            print(f"\n⚠️  WARNING: No exact matches found!")
            print(f"   The current max-autotune solution contains configs that don't exist in the dataset.")
            print(f"   Use the improved solution ({improved_solution_path}) for meaningful validation.")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
