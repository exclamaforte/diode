"""
Toolkit for matrix multiplication data collection, model training, and evaluation.

This script provides a unified interface for:
1. Collecting matrix multiplication timing data
2. Training neural network models on the collected data
3. Evaluating model performance
4. Visualizing results

All functionality is controlled through command-line flags.
"""

import argparse
import logging
import os
import random
import sys

import torch

# Add the parent directory to the path so we can import the diode module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import utility functions from the refactored modules
from torch_diode.collection.generic_data_utils import convert_json_to_msgpack
from torch_diode.collection.matmul_data_utils import (
    collect_data,
    create_validation_dataset,
    run_collector_example,
)
from torch_diode.model.model_utils import (
    run_model_example,
    train_model,
    train_model_from_directory,
    validate_max_autotune,
    validate_model,
)
from torch_diode.types.matmul_types import OperationShapeSet

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function that parses command-line arguments and runs the appropriate mode.
    """
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="Matrix multiplication toolkit for data collection, model training, and evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add global arguments
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "msgpack"],
        help="File format for saving/loading datasets and tables",
    )

    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run", required=True)

    # Collect data mode
    collect_parser = subparsers.add_parser(
        "collect", help="Collect matrix multiplication timing data"
    )
    collect_parser.add_argument(
        "--output",
        type=str,
        default="matmul_dataset.json",
        help="Path to save the collected data",
    )
    collect_parser.add_argument(
        "--num-shapes", type=int, default=100, help="Number of matrix shapes to test"
    )
    collect_parser.add_argument(
        "--min-size", type=int, default=32, help="Minimum matrix dimension"
    )
    collect_parser.add_argument(
        "--max-size", type=int, default=4096, help="Maximum matrix dimension"
    )
    collect_parser.add_argument(
        "--power-of-two", action="store_true", help="Generate only power-of-two sizes"
    )
    collect_parser.add_argument(
        "--no-rectangular", action="store_true", help="Exclude rectangular matrices"
    )
    collect_parser.add_argument(
        "--no-odd-sizes", action="store_true", help="Exclude odd-sized matrices"
    )
    collect_parser.add_argument(
        "--search-mode",
        type=str,
        default="max-autotune",
        help="Search mode for torch.compile",
    )
    collect_parser.add_argument(
        "--search-space",
        type=str,
        default="EXHAUSTIVE",
        choices=["EXHAUSTIVE", "DEFAULT"],
        help="Search space for autotuning",
    )
    collect_parser.add_argument(
        "--chunk-size",
        type=int,
        help="Number of shapes to collect before writing to a new file (default: 20 for EXHAUSTIVE, 100 for DEFAULT)",
    )
    collect_parser.add_argument(
        "--log-normal",
        action="store_true",
        help="Use log normal distribution for matrix sizes",
    )
    collect_parser.add_argument(
        "--log-normal-m-mean",
        type=float,
        default=6.5725472164323095,
        help="Log normal mean for M dimension",
    )
    collect_parser.add_argument(
        "--log-normal-m-std",
        type=float,
        default=2.556199441605505,
        help="Log normal std for M dimension",
    )
    collect_parser.add_argument(
        "--log-normal-n-mean",
        type=float,
        default=5.913930073563466,
        help="Log normal mean for N dimension",
    )
    collect_parser.add_argument(
        "--log-normal-n-std",
        type=float,
        default=1.66968141897024,
        help="Log normal std for N dimension",
    )
    collect_parser.add_argument(
        "--log-normal-k-mean",
        type=float,
        default=6.204916071423808,
        help="Log normal mean for K dimension",
    )
    collect_parser.add_argument(
        "--log-normal-k-std",
        type=float,
        default=2.1646646856090177,
        help="Log normal std for K dimension",
    )

    # Collect data from shapeset mode
    collect_shapeset_parser = subparsers.add_parser(
        "collect-shapeset",
        help="Collect matrix multiplication timing data from an operation shapeset",
    )
    collect_shapeset_parser.add_argument(
        "--shapeset",
        type=str,
        required=True,
        help="Path to the operation shapeset JSON file",
    )
    collect_shapeset_parser.add_argument(
        "--output",
        type=str,
        default="operation_shapeset_data.msgpack",
        help="Path to save the collected data",
    )
    collect_shapeset_parser.add_argument(
        "--operations",
        type=str,
        nargs="+",
        default=["mm", "addmm", "bmm"],
        help="Operations to collect data for",
    )
    collect_shapeset_parser.add_argument(
        "--search-mode",
        type=str,
        default="max-autotune",
        help="Search mode for torch.compile",
    )
    collect_shapeset_parser.add_argument(
        "--search-space",
        type=str,
        default="EXHAUSTIVE",
        choices=["EXHAUSTIVE", "DEFAULT"],
        help="Search space for autotuning",
    )

    # Create validation dataset mode
    validate_data_parser = subparsers.add_parser(
        "create-validation", help="Create a separate validation dataset"
    )
    validate_data_parser.add_argument(
        "--output",
        type=str,
        default="matmul_validation_dataset.json",
        help="Path to save the validation data",
    )
    validate_data_parser.add_argument(
        "--shapeset",
        type=str,
        help="Path to the operation shapeset JSON file (if provided, will use shapeset instead of random generation)",
    )
    validate_data_parser.add_argument(
        "--operations",
        type=str,
        nargs="+",
        default=["mm", "addmm", "bmm"],
        help="Operations to collect data for (only used with --shapeset)",
    )
    validate_data_parser.add_argument(
        "--num-shapes",
        type=int,
        default=30,
        help="Number of matrix shapes to test (only used without --shapeset)",
    )
    validate_data_parser.add_argument(
        "--min-size",
        type=int,
        default=32,
        help="Minimum matrix dimension (only used without --shapeset)",
    )
    validate_data_parser.add_argument(
        "--max-size",
        type=int,
        default=4096,
        help="Maximum matrix dimension (only used without --shapeset)",
    )
    validate_data_parser.add_argument(
        "--power-of-two",
        action="store_true",
        help="Generate only power-of-two sizes (only used without --shapeset)",
    )
    validate_data_parser.add_argument(
        "--no-rectangular",
        action="store_true",
        help="Exclude rectangular matrices (only used without --shapeset)",
    )
    validate_data_parser.add_argument(
        "--no-odd-sizes",
        action="store_true",
        help="Exclude odd-sized matrices (only used without --shapeset)",
    )
    validate_data_parser.add_argument(
        "--search-mode",
        type=str,
        default="max-autotune",
        help="Search mode for torch.compile",
    )
    validate_data_parser.add_argument(
        "--search-space",
        type=str,
        default="EXHAUSTIVE",
        choices=["EXHAUSTIVE", "DEFAULT"],
        help="Search space for autotuning",
    )
    validate_data_parser.add_argument(
        "--log-normal",
        action="store_true",
        help="Use log normal distribution for matrix sizes (only used without --shapeset)",
    )
    validate_data_parser.add_argument(
        "--log-normal-m-mean",
        type=float,
        default=6.5725472164323095,
        help="Log normal mean for M dimension (only used without --shapeset)",
    )
    validate_data_parser.add_argument(
        "--log-normal-m-std",
        type=float,
        default=2.556199441605505,
        help="Log normal std for M dimension (only used without --shapeset)",
    )
    validate_data_parser.add_argument(
        "--log-normal-n-mean",
        type=float,
        default=5.913930073563466,
        help="Log normal mean for N dimension (only used without --shapeset)",
    )
    validate_data_parser.add_argument(
        "--log-normal-n-std",
        type=float,
        default=1.66968141897024,
        help="Log normal std for N dimension (only used without --shapeset)",
    )
    validate_data_parser.add_argument(
        "--log-normal-k-mean",
        type=float,
        default=6.204916071423808,
        help="Log normal mean for K dimension (only used without --shapeset)",
    )
    validate_data_parser.add_argument(
        "--log-normal-k-std",
        type=float,
        default=2.1646646856090177,
        help="Log normal std for K dimension (only used without --shapeset)",
    )
    validate_data_parser.add_argument(
        "--chunk-size",
        type=int,
        help="Number of operations to collect before writing to a new file (enables chunked collection and resumption)",
    )

    # Train model mode
    train_parser = subparsers.add_parser(
        "train", help="Train a model on collected data"
    )

    # Create mutually exclusive group for dataset input
    dataset_group = train_parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset", type=str, help="Path to a single dataset file"
    )
    dataset_group.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing multiple data files (JSON/MessagePack)",
    )

    train_parser.add_argument(
        "--model",
        type=str,
        default="matmul_model.pt",
        help="Path to save the trained model",
    )
    train_parser.add_argument(
        "--model-type",
        type=str,
        default="deep",
        choices=["base", "deep"],
        help="Type of model to train",
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    train_parser.add_argument(
        "--num-epochs", type=int, default=100, help="Number of epochs to train for"
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for the optimizer",
    )
    train_parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs to wait for improvement before early stopping",
    )
    train_parser.add_argument(
        "--hidden-dim", type=int, default=128, help="Hidden dimension of the model"
    )
    train_parser.add_argument(
        "--num-layers", type=int, default=10, help="Number of layers in the model"
    )
    train_parser.add_argument(
        "--hardware-name", type=str, help="Hardware name to filter by"
    )
    train_parser.add_argument("--op-name", type=str, help="Operation name to filter by")
    train_parser.add_argument(
        "--log-dir", type=str, default="logs", help="Directory to save logs"
    )
    train_parser.add_argument(
        "--file-extensions",
        type=str,
        nargs="+",
        default=["json", "msgpack"],
        help="File extensions to look for in the directory (only used with --data-dir)",
    )

    # Validate model mode
    validate_model_parser = subparsers.add_parser(
        "validate-model",
        help="Validate a trained model on a separate validation dataset",
    )
    validate_model_parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained model"
    )
    validate_model_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the validation dataset file or directory",
    )
    validate_model_parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for validation"
    )
    validate_model_parser.add_argument(
        "--hardware-name", type=str, help="Hardware name to filter by"
    )
    validate_model_parser.add_argument(
        "--op-name", type=str, help="Operation name to filter by"
    )
    validate_model_parser.add_argument(
        "--top-n-worst",
        type=int,
        default=10,
        help="Number of worst predictions to analyze",
    )

    # Validate max-autotune mode
    validate_max_autotune_parser = subparsers.add_parser(
        "validate-max-autotune",
        help="Validate model's ability to select optimal shapes compared to max-autotune",
    )
    validate_max_autotune_parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained model"
    )
    validate_max_autotune_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the validation dataset file or directory",
    )
    validate_max_autotune_parser.add_argument(
        "--max-autotune-solution",
        type=str,
        required=True,
        help="Path to JSON file containing Solution with max-autotune TritonGEMMConfig objects",
    )
    validate_max_autotune_parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for validation"
    )
    validate_max_autotune_parser.add_argument(
        "--hardware-name", type=str, help="Hardware name to filter by"
    )
    validate_max_autotune_parser.add_argument(
        "--op-name", type=str, help="Operation name to filter by"
    )

    # Collector example mode
    collector_example_parser = subparsers.add_parser(
        "collector-example",
        help="Run an example demonstrating the MatmulDatasetCollector",
    )
    collector_example_parser.add_argument(
        "--output-dir", type=str, default=".", help="Directory to save output files"
    )
    collector_example_parser.add_argument(
        "--use-context-manager",
        action="store_true",
        help="Use the collector as a context manager",
    )
    collector_example_parser.add_argument(
        "--num-shapes", type=int, default=4, help="Number of matrix shapes to test"
    )

    # Model example mode
    model_example_parser = subparsers.add_parser(
        "model-example",
        help="Run an example demonstrating model training and evaluation",
    )
    model_example_parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset file"
    )
    model_example_parser.add_argument(
        "--model-type",
        type=str,
        default="deep",
        choices=["base", "deep"],
        help="Type of model to train",
    )
    model_example_parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for the dataloaders"
    )
    model_example_parser.add_argument(
        "--num-epochs", type=int, default=100, help="Number of epochs to train for"
    )
    model_example_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )
    model_example_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for the optimizer",
    )
    model_example_parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs to wait for improvement before early stopping",
    )
    model_example_parser.add_argument(
        "--log-dir", type=str, default="logs", help="Directory to save logs"
    )
    model_example_parser.add_argument(
        "--model-dir", type=str, default="models", help="Directory to save models"
    )
    model_example_parser.add_argument(
        "--hardware-name", type=str, help="Hardware name to filter by"
    )
    model_example_parser.add_argument(
        "--op-name", type=str, help="Operation name to filter by"
    )

    # Collect and train mode (combines collection and training)
    collect_train_parser = subparsers.add_parser(
        "collect-and-train", help="Collect data and train a model in one step"
    )
    collect_train_parser.add_argument(
        "--dataset",
        type=str,
        default="matmul_dataset.json",
        help="Path to save the collected data",
    )
    collect_train_parser.add_argument(
        "--validation-dataset",
        type=str,
        default="matmul_validation_dataset.json",
        help="Path to save the validation dataset",
    )
    collect_train_parser.add_argument(
        "--model",
        type=str,
        default="matmul_model.pt",
        help="Path to save the trained model",
    )
    collect_train_parser.add_argument(
        "--num-shapes", type=int, default=100, help="Number of matrix shapes to test"
    )
    collect_train_parser.add_argument(
        "--validation-shapes",
        type=int,
        default=30,
        help="Number of matrix shapes for validation",
    )
    collect_train_parser.add_argument(
        "--min-size", type=int, default=32, help="Minimum matrix dimension"
    )
    collect_train_parser.add_argument(
        "--max-size", type=int, default=4096, help="Maximum matrix dimension"
    )
    collect_train_parser.add_argument(
        "--power-of-two", action="store_true", help="Generate only power-of-two sizes"
    )
    collect_train_parser.add_argument(
        "--no-rectangular", action="store_true", help="Exclude rectangular matrices"
    )
    collect_train_parser.add_argument(
        "--no-odd-sizes", action="store_true", help="Exclude odd-sized matrices"
    )
    collect_train_parser.add_argument(
        "--search-mode",
        type=str,
        default="max-autotune",
        help="Search mode for torch.compile",
    )
    collect_train_parser.add_argument(
        "--search-space",
        type=str,
        default="EXHAUSTIVE",
        choices=["EXHAUSTIVE", "DEFAULT"],
        help="Search space for autotuning",
    )
    collect_train_parser.add_argument(
        "--model-type",
        type=str,
        default="deep",
        choices=["base", "deep"],
        help="Type of model to train",
    )
    collect_train_parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    collect_train_parser.add_argument(
        "--num-epochs", type=int, default=100, help="Number of epochs to train for"
    )
    collect_train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )
    collect_train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for the optimizer",
    )
    collect_train_parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs to wait for improvement before early stopping",
    )
    collect_train_parser.add_argument(
        "--hidden-dim", type=int, default=128, help="Hidden dimension of the model"
    )
    collect_train_parser.add_argument(
        "--num-layers", type=int, default=10, help="Number of layers in the model"
    )
    collect_train_parser.add_argument(
        "--log-dir", type=str, default="logs", help="Directory to save logs"
    )
    collect_train_parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip data collection and use existing dataset",
    )
    collect_train_parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation dataset creation",
    )
    collect_train_parser.add_argument(
        "--skip-training", action="store_true", help="Skip model training"
    )
    collect_train_parser.add_argument(
        "--log-normal",
        action="store_true",
        help="Use log normal distribution for matrix sizes",
    )
    collect_train_parser.add_argument(
        "--log-normal-m-mean",
        type=float,
        default=6.5725472164323095,
        help="Log normal mean for M dimension",
    )
    collect_train_parser.add_argument(
        "--log-normal-m-std",
        type=float,
        default=2.556199441605505,
        help="Log normal std for M dimension",
    )
    collect_train_parser.add_argument(
        "--log-normal-n-mean",
        type=float,
        default=5.913930073563466,
        help="Log normal mean for N dimension",
    )
    collect_train_parser.add_argument(
        "--log-normal-n-std",
        type=float,
        default=1.66968141897024,
        help="Log normal std for N dimension",
    )
    collect_train_parser.add_argument(
        "--log-normal-k-mean",
        type=float,
        default=6.204916071423808,
        help="Log normal mean for K dimension",
    )
    collect_train_parser.add_argument(
        "--log-normal-k-std",
        type=float,
        default=2.1646646856090177,
        help="Log normal std for K dimension",
    )

    # Convert JSON to MessagePack mode
    convert_parser = subparsers.add_parser(
        "convert-json-to-msgpack", help="Convert JSON files to MessagePack format"
    )
    convert_parser.add_argument(
        "--input-files",
        type=str,
        nargs="+",
        required=True,
        help="List of JSON files to convert to MessagePack",
    )
    convert_parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for MessagePack files (default: same directory as input files)",
    )
    convert_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing MessagePack files if they exist",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Run the appropriate mode
    if args.mode == "collect":
        # Set default chunk size based on search space if not provided
        chunk_size = args.chunk_size
        if chunk_size is None:
            chunk_size = 20 if args.search_space == "EXHAUSTIVE" else 100

        mode = "log_normal" if args.log_normal else "random"
        collect_data(
            output_file=args.output,
            mode=mode,
            num_shapes=args.num_shapes,
            seed=args.seed,
            min_size=args.min_size,
            max_size=args.max_size,
            power_of_two=args.power_of_two,
            search_mode=args.search_mode,
            search_space=args.search_space,
            file_format=args.format,
            chunk_size=chunk_size,
            log_normal_m_mean=args.log_normal_m_mean,
            log_normal_m_std=args.log_normal_m_std,
            log_normal_n_mean=args.log_normal_n_mean,
            log_normal_n_std=args.log_normal_n_std,
            log_normal_k_mean=args.log_normal_k_mean,
            log_normal_k_std=args.log_normal_k_std,
        )

    elif args.mode == "collect-shapeset":
        # Load the OperationShapeSet from the JSON file
        logger.info(f"Loading operation shapeset from: {args.shapeset}")
        with open(args.shapeset) as f:
            shapeset_content = f.read()

        operation_shape_set = OperationShapeSet.deserialize(shapeset_content)
        if operation_shape_set is None:
            logger.error("Failed to deserialize operation shapeset")
            return 1

        # Log information about the loaded shapeset
        logger.info(
            f"Loaded operation shapeset with operations: {operation_shape_set.get_operation_names()}"
        )
        for op_name in operation_shape_set.get_operation_names():
            shapes = operation_shape_set.get_shapes_for_operation(op_name)
            logger.info(f"  {op_name}: {len(shapes)} shapes")

        # Collect data using the operation shapeset
        collect_data(
            output_file=args.output,
            mode="operation_shape_set",
            operations=args.operations,
            operation_shape_set=operation_shape_set,
            search_mode=args.search_mode,
            search_space=args.search_space,
            file_format=args.format,
        )

    elif args.mode == "create-validation":
        if args.shapeset:
            # Load the OperationShapeSet from the JSON file
            logger.info(f"Loading operation shapeset from: {args.shapeset}")
            with open(args.shapeset) as f:
                shapeset_content = f.read()

            operation_shape_set = OperationShapeSet.deserialize(shapeset_content)
            if operation_shape_set is None:
                logger.error("Failed to deserialize operation shapeset")
                return 1

            # Log information about the loaded shapeset
            logger.info(
                f"Loaded operation shapeset with operations: {operation_shape_set.get_operation_names()}"
            )
            for op_name in operation_shape_set.get_operation_names():
                shapes = operation_shape_set.get_shapes_for_operation(op_name)
                logger.info(f"  {op_name}: {len(shapes)} shapes")

            # Use shapeset mode for validation dataset creation
            collect_data(
                output_file=args.output,
                mode="operation_shape_set",
                operations=args.operations,
                operation_shape_set=operation_shape_set,
                search_mode=args.search_mode,
                search_space=args.search_space,
                file_format=args.format,
                chunk_size=args.chunk_size,
            )
        else:
            # Use random/log-normal generation for validation dataset creation
            mode = "log_normal" if args.log_normal else "random"
            create_validation_dataset(
                output_file=args.output,
                mode=mode,
                num_shapes=args.num_shapes,
                seed=args.seed,
                min_size=args.min_size,
                max_size=args.max_size,
                power_of_two=args.power_of_two,
                search_mode=args.search_mode,
                search_space=args.search_space,
                file_format=args.format,
                chunk_size=args.chunk_size,
                log_normal_m_mean=args.log_normal_m_mean,
                log_normal_m_std=args.log_normal_m_std,
                log_normal_n_mean=args.log_normal_n_mean,
                log_normal_n_std=args.log_normal_n_std,
                log_normal_k_mean=args.log_normal_k_mean,
                log_normal_k_std=args.log_normal_k_std,
            )

    elif args.mode == "train":
        if args.dataset:
            # Train from a single dataset file
            train_model(
                dataset_path=args.dataset,
                model_path=args.model,
                model_type=args.model_type,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                patience=args.patience,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                hardware_name=args.hardware_name,
                op_name=args.op_name,
                seed=args.seed,
                device=args.device,
                log_dir=args.log_dir,
            )
        elif args.data_dir:
            # Train from all files in a directory
            train_model_from_directory(
                data_dir=args.data_dir,
                model_path=args.model,
                model_type=args.model_type,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                patience=args.patience,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                hardware_name=args.hardware_name,
                op_name=args.op_name,
                seed=args.seed,
                device=args.device,
                log_dir=args.log_dir,
                file_extensions=args.file_extensions,
            )

    elif args.mode == "validate-model":
        logger.info("Starting model validation...")
        logger.info(f"Model path: {args.model}")
        logger.info(f"Dataset path: {args.dataset}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Device: {args.device}")
        if args.hardware_name:
            logger.info(f"Hardware filter: {args.hardware_name}")
        if args.op_name:
            logger.info(f"Operation filter: {args.op_name}")
        logger.info(f"Will analyze top {args.top_n_worst} worst predictions")

        validate_model(
            model_path=args.model,
            validation_dataset_path=args.dataset,
            batch_size=args.batch_size,
            device=args.device,
            hardware_name=args.hardware_name,
            op_name=args.op_name,
            top_n_worst=args.top_n_worst,
        )

        logger.info("Model validation completed successfully")

    elif args.mode == "validate-max-autotune":
        logger.info("Starting max-autotune validation...")
        logger.info(f"Model path: {args.model}")
        logger.info(f"Dataset path: {args.dataset}")
        logger.info(f"Max-autotune solution path: {args.max_autotune_solution}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Device: {args.device}")
        if args.hardware_name:
            logger.info(f"Hardware filter: {args.hardware_name}")
        if args.op_name:
            logger.info(f"Operation filter: {args.op_name}")

        validate_max_autotune(
            model_path=args.model,
            validation_dataset_path=args.dataset,
            max_autotune_solution_path=args.max_autotune_solution,
            batch_size=args.batch_size,
            device=args.device,
            hardware_name=args.hardware_name,
            op_name=args.op_name,
        )

        logger.info("Max-autotune validation completed successfully")

    elif args.mode == "collector-example":
        run_collector_example(
            output_dir=args.output_dir,
            use_context_manager=args.use_context_manager,
            num_shapes=args.num_shapes,
        )

    elif args.mode == "model-example":
        run_model_example(
            dataset_path=args.dataset,
            model_type=args.model_type,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            hardware_name=args.hardware_name,
            op_name=args.op_name,
            seed=args.seed,
            device=args.device,
        )

    elif args.mode == "collect-and-train":
        # Create directories if they don't exist
        os.makedirs(
            (
                os.path.dirname(os.path.abspath(args.dataset))
                if os.path.dirname(args.dataset)
                else "."
            ),
            exist_ok=True,
        )
        os.makedirs(
            (
                os.path.dirname(os.path.abspath(args.model))
                if os.path.dirname(args.model)
                else "."
            ),
            exist_ok=True,
        )
        os.makedirs(
            (
                os.path.dirname(os.path.abspath(args.validation_dataset))
                if os.path.dirname(args.validation_dataset)
                else "."
            ),
            exist_ok=True,
        )
        os.makedirs(args.log_dir, exist_ok=True)

        # Collect data if not skipped
        if not args.skip_collection:
            mode = "log_normal" if args.log_normal else "random"
            collect_data(
                output_file=args.dataset,
                mode=mode,
                num_shapes=args.num_shapes,
                seed=args.seed,
                min_size=args.min_size,
                max_size=args.max_size,
                power_of_two=args.power_of_two,
                search_mode=args.search_mode,
                search_space=args.search_space,
                log_normal_m_mean=args.log_normal_m_mean,
                log_normal_m_std=args.log_normal_m_std,
                log_normal_n_mean=args.log_normal_n_mean,
                log_normal_n_std=args.log_normal_n_std,
                log_normal_k_mean=args.log_normal_k_mean,
                log_normal_k_std=args.log_normal_k_std,
            )

        # Create validation dataset if not skipped
        if not args.skip_validation:
            mode = "log_normal" if args.log_normal else "random"
            create_validation_dataset(
                output_file=args.validation_dataset,
                mode=mode,
                num_shapes=args.validation_shapes,
                seed=args.seed + 1,  # Use a different seed for validation
                min_size=args.min_size,
                max_size=args.max_size,
                power_of_two=args.power_of_two,
                search_mode=args.search_mode,
                search_space=args.search_space,
                log_normal_m_mean=args.log_normal_m_mean,
                log_normal_m_std=args.log_normal_m_std,
                log_normal_n_mean=args.log_normal_n_mean,
                log_normal_n_std=args.log_normal_n_std,
                log_normal_k_mean=args.log_normal_k_mean,
                log_normal_k_std=args.log_normal_k_std,
            )

        # Train model if not skipped
        if not args.skip_training:
            train_model(
                dataset_path=args.dataset,
                model_path=args.model,
                model_type=args.model_type,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                patience=args.patience,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                seed=args.seed,
                device=args.device,
                log_dir=args.log_dir,
            )

            # Validate model on the validation dataset
            if not args.skip_validation and os.path.exists(args.validation_dataset):
                validate_model(
                    model_path=args.model,
                    validation_dataset_path=args.validation_dataset,
                    batch_size=args.batch_size,
                    device=args.device,
                )

    elif args.mode == "convert-json-to-msgpack":
        convert_json_to_msgpack(
            input_files=args.input_files,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
