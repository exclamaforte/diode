"""
Example ShapeSet generator.

This module provides a function to generate a small example ShapeSet.
"""

import csv
import torch

from diode.types.matmul_types import MMShape, ShapeSet, OperationShapeSet


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


def create_shapeset_from_csv(csv_file_path):
    """
    Create an OperationShapeSet object from a CSV file containing matrix multiplication shapes.
    
    Args:
        csv_file_path: Path to the CSV file containing shape data
        
    Returns:
        OperationShapeSet: An OperationShapeSet object containing shapes organized by operation name
    """
    operation_shape_set = OperationShapeSet()
    
    with open(csv_file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse CSV values
            name = row['name']
            B = int(row['B'])
            M = int(row['M'])
            K = int(row['K'])
            N = int(row['N'])
            transposed = row['Transposed'].strip().lower() == 'true'
            
            # Set default dtypes (can be customized as needed)
            M_dtype = torch.float16
            K_dtype = torch.float16
            out_dtype = torch.float16
            
            # Calculate output size and stride
            if transposed:
                out_size = (B, N, M)
                out_stride = (N * M, M, 1)
            else:
                out_size = (B, M, N)
                out_stride = (M * N, N, 1)
            
            # Create MMShape object
            shape = MMShape(
                B=B,
                M=M,
                N=N,
                K=K,
                M_dtype=M_dtype,
                K_dtype=K_dtype,
                out_dtype=out_dtype,
                out_size=out_size,
                out_stride=out_stride
            )
            
            # Add shape to the operation shape set using the operation name
            operation_shape_set.add_shape(name, shape)
    
    return operation_shape_set


def save_operation_shapeset_to_json(operation_shapeset, file_path):
    """
    Save an OperationShapeSet to a JSON file.
    
    Args:
        operation_shapeset: The OperationShapeSet object to save
        file_path: Path to save the JSON file
    """
    with open(file_path, 'w') as f:
        f.write(operation_shapeset.serialize())
    
    print(f"OperationShapeSet saved to {file_path}")


if __name__ == "__main__":
    # Example usage
    save_example_shapeset("example_shapeset.json")
    
    # Create OperationShapeSet from shapes.txt
    shapes_csv_path = "/home/gabeferns/diode/diode_datasets/shapes.txt"
    operation_shapeset_from_csv = create_shapeset_from_csv(shapes_csv_path)
    
    # Count total shapes across all operations
    total_shapes = sum(len(shape_set.shapes) for shape_set in operation_shapeset_from_csv.operations.values())
    operation_names = operation_shapeset_from_csv.get_operation_names()
    
    print(f"Created OperationShapeSet with {len(operation_names)} operations and {total_shapes} total shapes from {shapes_csv_path}")
    print(f"Operations: {', '.join(operation_names)}")
    
    # Print shape counts per operation
    for op_name in operation_names:
        shape_count = len(operation_shapeset_from_csv.get_shapes_for_operation(op_name))
        print(f"  {op_name}: {shape_count} shapes")
    
    # Save OperationShapeSet to JSON file
    operation_shapeset_json_path = "operation_shapeset.json"
    save_operation_shapeset_to_json(operation_shapeset_from_csv, operation_shapeset_json_path)
