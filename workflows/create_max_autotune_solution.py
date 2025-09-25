#!/usr/bin/env python3
"""
Script to parse CSV files containing TritonGEMMConfig parameters and create a Solution JSON file.

This script reads CSV files with columns for TritonGEMMConfig parameters and creates a
Solution object containing a list of these configurations, saved as a JSON file.

Expected CSV format:
name,grid,block_m,block_n,block_k,group_m,num_stages,num_warps,EVEN_K,ALLOW_TF32,USE_FAST_ACCUM,ACC_TYPE
config_1,1,64,64,32,1,3,4,False,True,False,tl.float32
config_2,1,128,128,32,1,4,8,True,True,False,tl.float32
...
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List

# Add parent directory to path to import torch_diode modules
sys.path.insert(0, str(Path(__file__).parent.parent))
# noqa: E402
from torch_diode.types.matmul_types import Solution, TritonGEMMConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_bool_value(value: str) -> bool:
    """Parse a string value to boolean, handling various formats."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ("true", "t", "1", "yes", "y"):
            return True
        elif value in ("false", "f", "0", "no", "n", ""):
            return False
        else:
            raise ValueError(f"Cannot convert '{value}' to boolean")
    return bool(value)


def parse_csv_to_configs(csv_path: str) -> List[tuple[str, TritonGEMMConfig]]:
    """
    Parse a CSV file to create a list of (operation, TritonGEMMConfig) tuples.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of (operation_name, TritonGEMMConfig) tuples

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV format is invalid or required columns are missing
    """
    configs = []

    logger.info(f"Reading CSV file: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        # Detect delimiter
        sample = csvfile.read(1024)
        csvfile.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")

        reader = csv.DictReader(csvfile, dialect=dialect)

        # Check required columns
        required_columns = [
            "name",
            "grid",
            "block_m",
            "block_n",
            "block_k",
            "group_m",
            "num_stages",
            "num_warps",
        ]

        if not all(col in reader.fieldnames for col in required_columns):
            missing = set(required_columns) - set(reader.fieldnames or [])
            raise ValueError(f"Missing required columns: {missing}")

        logger.info(f"Found columns: {reader.fieldnames}")

        for row_num, row in enumerate(
            reader, start=2
        ):  # Start at 2 because header is row 1
            try:
                # Parse operation field (required now)
                operation = row.get("operation", "default").strip()

                # Parse required fields
                config_data = {
                    "name": row["name"].strip(),
                    "grid": int(row["grid"]),
                    "block_m": int(row["block_m"]),
                    "block_n": int(row["block_n"]),
                    "block_k": int(row["block_k"]),
                    "group_m": int(row["group_m"]),
                    "num_stages": int(row["num_stages"]),
                    "num_warps": int(row["num_warps"]),
                }

                # Parse optional boolean fields with defaults
                if "EVEN_K" in row and row["EVEN_K"].strip():
                    config_data["EVEN_K"] = parse_bool_value(row["EVEN_K"])

                if "ALLOW_TF32" in row and row["ALLOW_TF32"].strip():
                    config_data["ALLOW_TF32"] = parse_bool_value(row["ALLOW_TF32"])

                if "USE_FAST_ACCUM" in row and row["USE_FAST_ACCUM"].strip():
                    config_data["USE_FAST_ACCUM"] = parse_bool_value(
                        row["USE_FAST_ACCUM"]
                    )

                # Parse optional string field with default
                if "ACC_TYPE" in row and row["ACC_TYPE"].strip():
                    config_data["ACC_TYPE"] = row["ACC_TYPE"].strip()

                # Create TritonGEMMConfig object
                config = TritonGEMMConfig(**config_data)
                configs.append((operation, config))

                logger.debug(
                    f"Row {row_num}: Created config {config.name} for operation {operation}"
                )

            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing row {row_num}: {e}")
                logger.error(f"Row data: {row}")
                raise ValueError(f"Failed to parse row {row_num}: {e}") from e

    logger.info(f"Successfully parsed {len(configs)} configurations")
    return configs


def create_solution_from_csv(csv_path: str) -> dict[str, Solution]:
    """
    Create a mapping of operation names to Solution objects from a CSV file of TritonGEMMConfig data.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Dictionary mapping operation names to Solution objects containing the configurations
    """
    configs = parse_csv_to_configs(csv_path)

    # Group configurations by operation
    op_configs = {}
    for operation, config in configs:
        if operation not in op_configs:
            op_configs[operation] = []
        op_configs[operation].append(config)

    # Create Solution objects for each operation
    solutions = {}
    for op_name, op_config_list in op_configs.items():
        solutions[op_name] = Solution(config=op_config_list)
        logger.info(
            f"Created solution for operation '{op_name}' with {len(op_config_list)} configurations"
        )

    return solutions


def save_solutions_to_json(solutions: dict[str, Solution], output_path: str) -> None:
    """
    Save a mapping of operation names to Solution objects to a JSON file.

    Args:
        solutions: Dictionary mapping operation names to Solution objects
        output_path: Path where to save the JSON file
    """
    logger.info(f"Saving solutions to: {output_path}")

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert solutions dict to JSON-serializable format
    solutions_dict = {}
    for op_name, solution in solutions.items():
        # Parse the solution JSON string to get the dict structure
        import json

        solution_dict = json.loads(str(solution))
        solutions_dict[op_name] = solution_dict

    # Serialize and save
    import json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(solutions_dict, f, indent=2)

    logger.info(f"Solutions saved successfully to {output_path}")


def create_example_csv(csv_path: str) -> None:
    """
    Create an example CSV file with sample TritonGEMMConfig data.

    Args:
        csv_path: Path where to create the example CSV file
    """
    logger.info(f"Creating example CSV file: {csv_path}")

    # Sample configurations representing typical max-autotune choices
    sample_configs = [
        {
            "name": "small_config",
            "grid": 1,
            "block_m": 64,
            "block_n": 64,
            "block_k": 32,
            "group_m": 1,
            "num_stages": 3,
            "num_warps": 4,
            "EVEN_K": False,
            "ALLOW_TF32": True,
            "USE_FAST_ACCUM": False,
            "ACC_TYPE": "tl.float32",
        },
        {
            "name": "medium_config",
            "grid": 1,
            "block_m": 128,
            "block_n": 128,
            "block_k": 32,
            "group_m": 1,
            "num_stages": 4,
            "num_warps": 8,
            "EVEN_K": True,
            "ALLOW_TF32": True,
            "USE_FAST_ACCUM": False,
            "ACC_TYPE": "tl.float32",
        },
        {
            "name": "large_config",
            "grid": 1,
            "block_m": 256,
            "block_n": 256,
            "block_k": 64,
            "group_m": 8,
            "num_stages": 3,
            "num_warps": 16,
            "EVEN_K": True,
            "ALLOW_TF32": True,
            "USE_FAST_ACCUM": True,
            "ACC_TYPE": "tl.float32",
        },
        {
            "name": "high_stages_config",
            "grid": 1,
            "block_m": 128,
            "block_n": 64,
            "block_k": 64,
            "group_m": 4,
            "num_stages": 5,
            "num_warps": 8,
            "EVEN_K": False,
            "ALLOW_TF32": True,
            "USE_FAST_ACCUM": False,
            "ACC_TYPE": "tl.float32",
        },
        {
            "name": "wide_config",
            "grid": 1,
            "block_m": 64,
            "block_n": 256,
            "block_k": 32,
            "group_m": 8,
            "num_stages": 4,
            "num_warps": 8,
            "EVEN_K": True,
            "ALLOW_TF32": True,
            "USE_FAST_ACCUM": False,
            "ACC_TYPE": "tl.float32",
        },
    ]

    # Create directory if it doesn't exist
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    fieldnames = [
        "name",
        "grid",
        "block_m",
        "block_n",
        "block_k",
        "group_m",
        "num_stages",
        "num_warps",
        "EVEN_K",
        "ALLOW_TF32",
        "USE_FAST_ACCUM",
        "ACC_TYPE",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_configs)

    logger.info(f"Example CSV created with {len(sample_configs)} sample configurations")


def main():
    """Main function to handle command line arguments and orchestrate the conversion."""
    parser = argparse.ArgumentParser(
        description="Create a Solution JSON file from TritonGEMMConfig CSV data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the CSV file containing TritonGEMMConfig data",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the output JSON file"
    )
    parser.add_argument(
        "--solution-name",
        type=str,
        default="max_autotune",
        help="Name for the solution",
    )
    parser.add_argument(
        "--create-example",
        action="store_true",
        help="Create an example CSV file at the specified path instead of processing it",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the created JSON by loading it back",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        if args.create_example:
            create_example_csv(args.csv_path)
            logger.info(f"Example CSV created at: {args.csv_path}")
            logger.info(
                "You can now edit this file and run the script again without --create-example"
            )
            return 0

        # Check if CSV file exists
        if not Path(args.csv_path).exists():
            logger.error(f"CSV file not found: {args.csv_path}")
            logger.info("Use --create-example to create an example CSV file")
            return 1

        # Create solutions from CSV
        solutions = create_solution_from_csv(args.csv_path)

        # Save to JSON
        save_solutions_to_json(solutions, args.output)

        # Validate if requested
        if args.validate:
            logger.info("Validating created JSON file...")
            import json

            with open(args.output, encoding="utf-8") as f:
                loaded_solutions = json.load(f)

            # Try to parse each operation's solution
            total_configs = 0
            for op_name, solution_dict in loaded_solutions.items():
                try:
                    solution_json = json.dumps(solution_dict)
                    loaded_solution = Solution.parse(solution_json)
                    if loaded_solution is None:
                        logger.error(
                            f"Failed to validate operation '{op_name}': Could not deserialize JSON"
                        )
                        return 1
                    total_configs += len(loaded_solution.config)
                    logger.info(
                        f"Operation '{op_name}': {len(loaded_solution.config)} configs"
                    )
                except Exception as e:
                    logger.error(f"Failed to validate operation '{op_name}': {e}")
                    return 1

            logger.info(
                f"Validation successful: Total {total_configs} configs across {len(loaded_solutions)} operations"
            )

        logger.info("Script completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
