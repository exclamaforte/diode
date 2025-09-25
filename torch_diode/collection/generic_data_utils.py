"""
Generic data collection utility functions that can be used across different domains.
"""

import json
import logging
import os
from typing import List, Optional

import msgpack

logger = logging.getLogger(__name__)


def convert_json_to_msgpack(
    input_files: List[str],
    output_dir: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Convert a list of JSON files to their equivalent MessagePack files.

    Args:
        input_files: List of JSON file paths to convert
        output_dir: Output directory for MessagePack files (default: same directory as input files)
        overwrite: Whether to overwrite existing MessagePack files if they exist
    """

    converted_count = 0
    error_count = 0

    for input_file in input_files:
        try:
            # Validate input file exists
            if not os.path.exists(input_file):
                logger.error(f"Input file does not exist: {input_file}")
                error_count += 1
                continue

            # Determine output file path
            if output_dir:
                # Use specified output directory
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                output_file = os.path.join(output_dir, f"{base_name}.msgpack")
            else:
                # Use same directory as input file
                base_name = os.path.splitext(input_file)[0]
                output_file = f"{base_name}.msgpack"

            # Load JSON data
            logger.info(f"Converting {input_file} to {output_file}")
            with open(input_file, encoding="utf-8") as f:
                data = json.load(f)

            # Save as MessagePack
            with open(output_file, "wb") as f:
                msgpack.pack(data, f)

            logger.info(f"Successfully converted {input_file} to {output_file}")
            converted_count += 1

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {input_file}: {e}")
            error_count += 1
        except Exception as e:
            logger.error(f"Error converting {input_file}: {e}")
            error_count += 1

    # Print summary
    logger.info("Conversion completed:")
    logger.info(f"  Converted: {converted_count} files")
    logger.info(f"  Errors: {error_count} files")
