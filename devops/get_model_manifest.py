#!/usr/bin/env python3
"""
Script to generate model manifest for the build system.

This script helps the build system understand which model files should be included
in the distribution packages by querying the model registry.
"""

import json
import sys
from pathlib import Path


def get_model_manifest():
    """Get the model manifest from the diode package."""
    # Add the project root to the path so we can import torch_diode
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from torch_diode.model_registry import generate_model_manifest, get_model_paths_for_build
        
        manifest = generate_model_manifest()
        model_paths = get_model_paths_for_build()
        
        # Add path information to the manifest
        manifest["model_files"] = [str(path) for path in model_paths]
        
        return manifest
        
    except ImportError as e:
        print(f"Error importing diode: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error generating manifest: {e}", file=sys.stderr)
        return None


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate model manifest for torch-diode build system"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path (defaults to stdout)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "paths"],
        default="json",
        help="Output format: json (full manifest) or paths (just file paths)"
    )
    
    args = parser.parse_args()
    
    manifest = get_model_manifest()
    if manifest is None:
        print("Failed to generate model manifest", file=sys.stderr)
        sys.exit(1)
    
    if args.format == "json":
        output_data = json.dumps(manifest, indent=2)
    elif args.format == "paths":
        output_data = "\n".join(manifest.get("model_files", []))
    else:
        raise Exception("Unknown output format: {}".format(args.format))

    
    if args.output:
        args.output.write_text(output_data)
        print(f"Model manifest written to {args.output}")
    else:
        print(output_data)


if __name__ == "__main__":
    main()
