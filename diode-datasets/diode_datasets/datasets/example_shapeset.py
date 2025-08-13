"""
Example ShapeSet generator.

This module provides a function to generate a small example ShapeSet.
"""

import torch

from diode.types.matmul_types import MMShape, ShapeSet


def generate_example_shapeset():
    """
    Generate a small example ShapeSet with a few common matrix multiplication shapes.

    Returns:
        ShapeSet: A ShapeSet object containing example shapes
    """
    shape_set = ShapeSet()

    # Add some common matrix multiplication shapes

    # Example 1: Small matrix multiplication (batch=1)
    shape1 = MMShape(
        B=1,
        M=128,
        N=128,
        K=128,
        M_dtype=torch.float16,
        K_dtype=torch.float16,
        out_dtype=torch.float16,
        out_size=(1, 128, 128),
        out_stride=(16384, 128, 1)
    )

    # Example 2: Medium matrix multiplication (batch=1)
    shape2 = MMShape(
        B=1,
        M=512,
        N=512,
        K=512,
        M_dtype=torch.float16,
        K_dtype=torch.float16,
        out_dtype=torch.float16,
        out_size=(1, 512, 512),
        out_stride=(262144, 512, 1)
    )

    # Example 3: Large matrix multiplication (batch=1)
    shape3 = MMShape(
        B=1,
        M=1024,
        N=1024,
        K=1024,
        M_dtype=torch.float16,
        K_dtype=torch.float16,
        out_dtype=torch.float16,
        out_size=(1, 1024, 1024),
        out_stride=(1048576, 1024, 1)
    )

    # Example 4: Batched matrix multiplication
    shape4 = MMShape(
        B=32,
        M=128,
        N=128,
        K=128,
        M_dtype=torch.float16,
        K_dtype=torch.float16,
        out_dtype=torch.float16,
        out_size=(32, 128, 128),
        out_stride=(16384, 128, 1)
    )

    # Example 5: Mixed precision matrix multiplication
    shape5 = MMShape(
        B=1,
        M=256,
        N=256,
        K=256,
        M_dtype=torch.float16,
        K_dtype=torch.float16,
        out_dtype=torch.float32,
        out_size=(1, 256, 256),
        out_stride=(65536, 256, 1)
    )

    # Add shapes to the set
    shape_set.add_shape(shape1)
    shape_set.add_shape(shape2)
    shape_set.add_shape(shape3)
    shape_set.add_shape(shape4)
    shape_set.add_shape(shape5)

    return shape_set


def save_example_shapeset(file_path):
    """
    Generate an example ShapeSet and save it to a JSON file.

    Args:
        file_path: Path to save the JSON file
    """
    shape_set = generate_example_shapeset()
    with open(file_path, 'w') as f:
        f.write(shape_set.serialize())

    print(f"Example ShapeSet saved to {file_path}")


if __name__ == "__main__":
    # Example usage
    save_example_shapeset("example_shapeset.json")
