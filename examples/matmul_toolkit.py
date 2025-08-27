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
from diode.collection.data_collection_utils import (
    collect_data,
    create_validation_dataset,
    run_collector_example,
)
from diode.model.model_utils import (
    train_model,
    validate_model,
    run_model_example,
)
from diode.types.matmul_types import OperationShapeSet

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

    # Collect data from shapeset mode
    collect_shapeset_parser = subparsers.add_parser(
        "collect-shapeset", help="Collect matrix multiplication timing data from an operation shapeset"
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
        "--num-shapes", type=int, default=30, help="Number of matrix shapes to test"
    )
    validate_data_parser.add_argument(
        "--min-size", type=int, default=32, help="Minimum matrix dimension"
    )
    validate_data_parser.add_argument(
        "--max-size", type=int, default=4096, help="Maximum matrix dimension"
    )
    validate_data_parser.add_argument(
        "--power-of-two", action="store_true", help="Generate only power-of-two sizes"
    )
    validate_data_parser.add_argument(
        "--no-rectangular", action="store_true", help="Exclude rectangular matrices"
    )
    validate_data_parser.add_argument(
        "--no-odd-sizes", action="store_true", help="Exclude odd-sized matrices"
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

    # Train model mode
    train_parser = subparsers.add_parser(
        "train", help="Train a model on collected data"
    )
    train_parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset file"
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

    # Validate model mode
    validate_model_parser = subparsers.add_parser(
        "validate-model",
        help="Validate a trained model on a separate validation dataset",
    )
    validate_model_parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained model"
    )
    validate_model_parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the validation dataset"
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

    # Parse the arguments
    args = parser.parse_args()

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Run the appropriate mode
    if args.mode == "collect":
        collect_data(
            output_file=args.output,
            num_shapes=args.num_shapes,
            seed=args.seed,
            min_size=args.min_size,
            max_size=args.max_size,
            power_of_two=args.power_of_two,
            include_rectangular=not args.no_rectangular,
            include_odd_sizes=not args.no_odd_sizes,
            search_mode=args.search_mode,
            search_space=args.search_space,
            file_format=args.format,
        )

    elif args.mode == "collect-shapeset":
        # Load the OperationShapeSet from the JSON file
        logger.info(f"Loading operation shapeset from: {args.shapeset}")
        with open(args.shapeset, 'r') as f:
            shapeset_content = f.read()
        
        operation_shape_set = OperationShapeSet.deserialize(shapeset_content)
        if operation_shape_set is None:
            logger.error("Failed to deserialize operation shapeset")
            return 1
        
        # Log information about the loaded shapeset
        logger.info(f"Loaded operation shapeset with operations: {operation_shape_set.get_operation_names()}")
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
        create_validation_dataset(
            output_file=args.output,
            num_shapes=args.num_shapes,
            seed=args.seed,
            min_size=args.min_size,
            max_size=args.max_size,
            power_of_two=args.power_of_two,
            include_rectangular=not args.no_rectangular,
            include_odd_sizes=not args.no_odd_sizes,
            search_mode=args.search_mode,
            search_space=args.search_space,
            file_format=args.format,
        )

    elif args.mode == "train":
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

    elif args.mode == "validate-model":
        validate_model(
            model_path=args.model,
            validation_dataset_path=args.dataset,
            batch_size=args.batch_size,
            device=args.device,
            hardware_name=args.hardware_name,
            op_name=args.op_name,
            top_n_worst=args.top_n_worst,
        )

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
            collect_data(
                output_file=args.dataset,
                num_shapes=args.num_shapes,
                seed=args.seed,
                min_size=args.min_size,
                max_size=args.max_size,
                power_of_two=args.power_of_two,
                include_rectangular=not args.no_rectangular,
                include_odd_sizes=not args.no_odd_sizes,
                search_mode=args.search_mode,
                search_space=args.search_space,
            )

        # Create validation dataset if not skipped
        if not args.skip_validation:
            create_validation_dataset(
                output_file=args.validation_dataset,
                num_shapes=args.validation_shapes,
                seed=args.seed + 1,  # Use a different seed for validation
                min_size=args.min_size,
                max_size=args.max_size,
                power_of_two=args.power_of_two,
                include_rectangular=not args.no_rectangular,
                include_odd_sizes=not args.no_odd_sizes,
                search_mode=args.search_mode,
                search_space=args.search_space,
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


if __name__ == "__main__":
    main()
