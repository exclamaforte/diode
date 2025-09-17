"""
Model utility functions for training and evaluation.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from diode.model.directory_dataset_loader import create_directory_dataloaders
from diode.model.matmul_dataset_loader import create_dataloaders
from diode.model.matmul_model_trainer import (
    analyze_worst_predictions,
    MatmulModelTrainer,
    train_model_from_dataset,
)
from diode.model.matmul_timing_model import DeepMatmulTimingModel, MatmulTimingModel

from diode.types.matmul_dataset import Dataset as MatmulDataset
from diode.utils.dataset_utils import print_dataset_statistics
from diode.utils.visualization_utils import plot_training_history

logger = logging.getLogger(__name__)


def validate_max_autotune(
    model_path: str,
    validation_dataset_path: str,
    max_autotune_solution_path: str,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    hardware_name: Optional[str] = None,
    op_name: Optional[str] = None,
) -> None:
    """
    Validate a trained model's ability to select optimal configs compared to max-autotune.

    This function:
    1. Loads validation data and runs all configs through the model
    2. For each n in [1, 5, 10, 20, 50, 100], selects top n configs based on model predictions
    3. Finds the best actual runtime among those top n configs
    4. Compares with the best runtime from predefined max-autotune configs
    5. Reports aggregate statistics

    Args:
        model_path: Path to the trained model
        validation_dataset_path: Path to the validation dataset file or directory
        max_autotune_solution_path: Path to JSON file containing Solution with max-autotune configs
        batch_size: Batch size for validation
        device: Device to validate on
        hardware_name: Optional hardware name to filter by
        op_name: Optional operation name to filter by
    """
    from diode.types.matmul_types import Solution, TritonGEMMConfig

    # Check if model exists
    logger.info("Checking if model file exists...")
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return
    logger.info(f"Model file found: {model_path}")

    # Check if validation dataset path exists
    logger.info("Checking if validation dataset exists...")
    if not os.path.exists(validation_dataset_path):
        logger.error(f"Validation dataset not found at {validation_dataset_path}")
        return
    logger.info(f"Validation dataset found: {validation_dataset_path}")

    # Check if max-autotune solution path exists
    logger.info("Checking if max-autotune solution file exists...")
    if not os.path.exists(max_autotune_solution_path):
        logger.error(f"Max-autotune solution not found at {max_autotune_solution_path}")
        return
    logger.info(f"Max-autotune solution found: {max_autotune_solution_path}")

    # Load max-autotune solution
    logger.info(f"Loading max-autotune solution from {max_autotune_solution_path}")
    try:
        with open(max_autotune_solution_path, "r", encoding="utf-8") as f:
            solution_json = f.read()
        max_autotune_solution = Solution.parse(solution_json)
        if max_autotune_solution is None:
            logger.error("Failed to deserialize max-autotune solution")
            return
        logger.info(
            f"Loaded max-autotune solution with {len(max_autotune_solution.config)} configs"
        )
    except Exception as e:
        logger.error(f"Failed to load max-autotune solution: {e}")
        return

    # Load validation dataset
    if os.path.isdir(validation_dataset_path):
        logger.info(
            f"Loading all validation data files from directory: {validation_dataset_path}"
        )
        try:
            _, val_dataloader, _ = create_directory_dataloaders(
                data_dir=validation_dataset_path,
                batch_size=batch_size,
                hardware_name=hardware_name,
                op_name=op_name,
                log_transform=True,
                num_workers=4,
                seed=42,
                file_extensions=["json", "msgpack"],
            )
        except Exception as e:
            logger.error(
                f"Failed to create dataloaders from directory {validation_dataset_path}: {e}"
            )
            return
    else:
        logger.info(f"Loading validation dataset from {validation_dataset_path}")

        if validation_dataset_path.endswith(".msgpack"):
            with open(validation_dataset_path, "rb") as f:
                dataset_data = f.read()
            dataset = MatmulDataset.from_msgpack(dataset_data)
        else:
            with open(validation_dataset_path, "r") as f:
                dataset_json = f.read()
            dataset = MatmulDataset.deserialize(dataset_json)

        if dataset is None:
            logger.error(
                f"Failed to load validation dataset from {validation_dataset_path}"
            )
            return

        _, val_dataloader, _ = create_dataloaders(
            dataset=dataset,
            batch_size=batch_size,
            hardware_name=hardware_name,
            op_name=op_name,
            log_transform=True,
            num_workers=4,
            seed=42,
        )

    # Get the feature dimensions
    problem_feature_dim = val_dataloader.dataset.dataset.problem_feature_dim
    config_feature_dim = val_dataloader.dataset.dataset.config_feature_dim

    # Load the trained model
    logger.info(f"Loading model weights from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        return

    # Create model
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        logger.info("Loading from checkpoint format")
        problem_feature_dim = checkpoint.get("problem_feature_dim", problem_feature_dim)
        config_feature_dim = checkpoint.get("config_feature_dim", config_feature_dim)
        hidden_dim = checkpoint.get("hidden_dim", 128)
        num_layers = checkpoint.get("num_layers", 10)
        model_type = checkpoint.get("model_type", "deep")

        if model_type == "base":
            model = MatmulTimingModel(
                problem_feature_dim=problem_feature_dim,
                config_feature_dim=config_feature_dim,
            )
        else:
            model = DeepMatmulTimingModel(
                problem_feature_dim=problem_feature_dim,
                config_feature_dim=config_feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = DeepMatmulTimingModel(
            problem_feature_dim=problem_feature_dim,
            config_feature_dim=config_feature_dim,
        )
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Get all validation configs and their actual runtimes
    logger.info("Extracting validation configs and predictions...")
    validation_configs = []  # List of TritonGEMMConfig objects
    actual_runtimes = []  # Corresponding actual runtimes (log-transformed)
    model_predictions = []  # Model predictions for these configs (log-transformed)

    # Extract configs and runtimes from the validation dataset
    with torch.no_grad():
        for batch_idx, (problem_features, config_features, targets) in enumerate(val_dataloader):
            problem_features = problem_features.to(device)
            config_features = config_features.to(device)
            targets = targets.to(device)

            # Get model predictions
            predictions = model(problem_features, config_features)

            # Extract configs from the underlying dataset
            # The val_dataloader.dataset is a Subset, so we need to access the original indices
            batch_size_actual = len(targets)
              
            for i in range(batch_size_actual):
                # Get the actual dataset index from the subset
                subset_idx = batch_idx * val_dataloader.batch_size + i
                if subset_idx < len(val_dataloader.dataset):
                    actual_dataset_idx = val_dataloader.dataset.indices[subset_idx]
                      
                    # Get the config from the original dataset
                    if hasattr(val_dataloader.dataset.dataset, 'timing_dataset'):
                        # For DirectoryMatmulDataset
                        config = val_dataloader.dataset.dataset.timing_dataset.configs[actual_dataset_idx]
                    else:
                        # For MatmulTimingDataset directly
                        config = val_dataloader.dataset.dataset.configs[actual_dataset_idx]
                      
                    validation_configs.append(config)
                    actual_runtimes.append(float(targets[i].cpu().numpy()))
                    model_predictions.append(float(predictions[i].cpu().numpy()))

    logger.info(f"Collected {len(validation_configs)} validation configs")

    # Convert to numpy arrays for easier manipulation
    model_predictions = np.array(model_predictions)
    actual_runtimes = np.array(actual_runtimes)

    # Define n values to test
    n_values = [1, 5, 10, 20, 50, 100]

    # Results storage
    results = {}

    max_autotune_configs = max_autotune_solution.config
    logger.info(f"Max-autotune configs to compare: {len(max_autotune_configs)} configs")
    logger.info("Analyzing model performance for different n values...")

    for n in n_values:
        if n > len(validation_configs):
            logger.warning(
                f"n={n} is larger than available configs ({len(validation_configs)}), skipping"
            )
            continue

        # Get top n configs based on model predictions (lowest predicted runtime = best)
        # Since we're working with log-transformed values, lower is still better
        top_n_indices = np.argsort(model_predictions)[:n]
        top_n_actual_runtimes = actual_runtimes[top_n_indices]

        # Find the minimum actual runtime among the top n model selections
        best_model_runtime = np.min(top_n_actual_runtimes)
        best_model_config_idx = top_n_indices[np.argmin(top_n_actual_runtimes)]
        best_model_config = validation_configs[best_model_config_idx]

        # Find max-autotune configs that exist in validation set
        max_autotune_runtimes = []
        max_autotune_found_configs = []

        def configs_match(config1, config2):
            """
            Compare two TritonGEMMConfig objects based on their kernel parameters only,
            ignoring fields like name and version that don't affect performance.
            """
            return (
                config1.block_m == config2.block_m and
                config1.block_n == config2.block_n and
                config1.block_k == config2.block_k and
                config1.group_m == config2.group_m and
                config1.num_stages == config2.num_stages and
                config1.num_warps == config2.num_warps and
                config1.EVEN_K == config2.EVEN_K and
                config1.ALLOW_TF32 == config2.ALLOW_TF32 and
                config1.USE_FAST_ACCUM == config2.USE_FAST_ACCUM and
                config1.ACC_TYPE == config2.ACC_TYPE
            )

        for ma_config in max_autotune_configs:
            # Find this max-autotune config in validation set
            for i, val_config in enumerate(validation_configs):
                if configs_match(val_config, ma_config):
                    max_autotune_runtimes.append(actual_runtimes[i])
                    max_autotune_found_configs.append(ma_config)
                    break

        if not max_autotune_runtimes:
            logger.warning(f"No max-autotune configs found in validation set for n={n}")
            results[n] = {
                "best_model_runtime": best_model_runtime,
                "best_model_config": best_model_config,
                "max_autotune_runtime": None,
                "improvement": None,
                "improvement_percent": None,
                "max_autotune_configs_found": 0,
            }
            continue

        # Find the minimum actual runtime among max-autotune configs
        best_max_autotune_runtime = np.min(max_autotune_runtimes)
        best_max_autotune_idx = np.argmin(max_autotune_runtimes)
        best_max_autotune_config = max_autotune_found_configs[best_max_autotune_idx]

        # Calculate improvement (since lower runtime is better, positive improvement means model is better)
        improvement = best_max_autotune_runtime - best_model_runtime
        improvement_percent = (
            (improvement / abs(best_max_autotune_runtime)) * 100
            if best_max_autotune_runtime != 0
            else 0
        )

        results[n] = {
            "best_model_runtime": best_model_runtime,
            "best_model_config": best_model_config,
            "max_autotune_runtime": best_max_autotune_runtime,
            "max_autotune_config": best_max_autotune_config,
            "improvement": improvement,
            "improvement_percent": improvement_percent,
            "max_autotune_configs_found": len(max_autotune_runtimes),
        }

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("MAX-AUTOTUNE VALIDATION RESULTS")
    logger.info("=" * 80)
    logger.info(
        f"Note: All runtimes are log-transformed. Lower values = better performance."
    )

    for n in sorted(results.keys()):
        result = results[n]
        logger.info(f"\nTop-{n} Analysis:")
        logger.info(f"  Best model selection:")
        logger.info(f"    Config: {result['best_model_config'].name}")
        logger.info(
            f"    Block sizes: {result['best_model_config'].block_m}x{result['best_model_config'].block_n}x{result['best_model_config'].block_k}"
        )
        logger.info(
            f"    Warps/Stages: {result['best_model_config'].num_warps}/{result['best_model_config'].num_stages}"
        )
        logger.info(f"    Runtime (log): {result['best_model_runtime']:.6f}")
        logger.info(f"    Runtime (actual): {np.exp(result['best_model_runtime']):.6f}")

        if result["max_autotune_runtime"] is not None:
            logger.info(f"  Best max-autotune:")
            logger.info(f"    Config: {result['max_autotune_config'].name}")
            logger.info(
                f"    Block sizes: {result['max_autotune_config'].block_m}x{result['max_autotune_config'].block_n}x{result['max_autotune_config'].block_k}"
            )
            logger.info(
                f"    Warps/Stages: {result['max_autotune_config'].num_warps}/{result['max_autotune_config'].num_stages}"
            )
            logger.info(f"    Runtime (log): {result['max_autotune_runtime']:.6f}")
            logger.info(
                f"    Runtime (actual): {np.exp(result['max_autotune_runtime']):.6f}"
            )
            logger.info(f"  Performance:")
            if result["improvement"] > 0:
                logger.info(
                    f"    Model is BETTER by {result['improvement']:.6f} log units ({result['improvement_percent']:.2f}%)"
                )
            elif result["improvement"] < 0:
                logger.info(
                    f"    Model is WORSE by {abs(result['improvement']):.6f} log units ({abs(result['improvement_percent']):.2f}%)"
                )
            else:
                logger.info(f"    Model performance is EQUAL")
            logger.info(
                f"  Max-autotune configs found in validation: {result['max_autotune_configs_found']}"
            )
        else:
            logger.info(f"  No max-autotune configs found in validation set")

    # Aggregate statistics
    valid_results = [r for r in results.values() if r["improvement"] is not None]
    if valid_results:
        improvements = [r["improvement"] for r in valid_results]
        improvement_percents = [r["improvement_percent"] for r in valid_results]

        logger.info(f"\n" + "=" * 80)
        logger.info("AGGREGATE STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Average improvement (log units): {np.mean(improvements):.6f}")
        logger.info(f"Average improvement %: {np.mean(improvement_percents):.2f}%")
        logger.info(
            f"Best improvement (log units): {np.max(improvements):.6f} ({np.max(improvement_percents):.2f}%)"
        )
        logger.info(
            f"Worst improvement (log units): {np.min(improvements):.6f} ({np.min(improvement_percents):.2f}%)"
        )
        logger.info(f"Std dev improvement: {np.std(improvements):.6f}")

        better_count = sum(1 for imp in improvements if imp > 0)
        worse_count = sum(1 for imp in improvements if imp < 0)
        equal_count = sum(1 for imp in improvements if imp == 0)

        logger.info(
            f"Model performs better: {better_count}/{len(valid_results)} cases ({100*better_count/len(valid_results):.1f}%)"
        )
        logger.info(
            f"Model performs worse: {worse_count}/{len(valid_results)} cases ({100*worse_count/len(valid_results):.1f}%)"
        )
        logger.info(
            f"Model performs equal: {equal_count}/{len(valid_results)} cases ({100*equal_count/len(valid_results):.1f}%)"
        )

        # Convert log improvements to actual runtime ratios for more intuitive understanding
        logger.info(f"\nActual Runtime Improvements:")
        actual_runtime_ratios = [
            np.exp(-imp) for imp in improvements
        ]  # negative because lower log runtime is better
        avg_ratio = np.mean(actual_runtime_ratios)
        best_ratio = np.max(actual_runtime_ratios)
        worst_ratio = np.min(actual_runtime_ratios)

        logger.info(f"Average speedup ratio: {avg_ratio:.3f}x")
        logger.info(f"Best speedup ratio: {best_ratio:.3f}x")
        logger.info(f"Worst speedup ratio: {worst_ratio:.3f}x")

        if avg_ratio > 1:
            logger.info(
                f"On average, model selections are {avg_ratio:.3f}x faster than max-autotune"
            )
        elif avg_ratio < 1:
            logger.info(
                f"On average, model selections are {1/avg_ratio:.3f}x slower than max-autotune"
            )
        else:
            logger.info("On average, model performance equals max-autotune")

    logger.info("=" * 80)


def train_model(
    dataset_path: str,
    model_path: str,
    model_type: str = "deep",
    batch_size: int = 64,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 20,
    hidden_dim: int = 128,
    num_layers: int = 10,
    hardware_name: Optional[str] = None,
    op_name: Optional[str] = None,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_dir: str = "logs",
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    Train a model on the collected data.

    Args:
        dataset_path: Path to the dataset file
        model_path: Path to save the trained model
        model_type: Type of model to train ("base" or "deep")
        batch_size: Batch size for training
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        patience: Number of epochs to wait for improvement before early stopping
        hidden_dim: Hidden dimension of the model
        num_layers: Number of layers in the model
        hardware_name: Optional hardware name to filter by
        op_name: Optional operation name to filter by
        seed: Random seed for reproducibility
        device: Device to train on
        log_dir: Directory to save logs

    Returns:
        Tuple of (trained model, training history)
    """
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Create directories if they don't exist
    os.makedirs(
        (
            os.path.dirname(os.path.abspath(model_path))
            if os.path.dirname(model_path)
            else "."
        ),
        exist_ok=True,
    )
    os.makedirs(log_dir, exist_ok=True)

    # Load the dataset
    logger.info(f"Loading dataset from {dataset_path}")
    if dataset_path.endswith(".msgpack"):
        with open(dataset_path, "rb") as f:
            dataset_data = f.read()
        dataset = MatmulDataset.from_msgpack(dataset_data)
    else:
        with open(dataset_path, "r") as f:
            dataset_json = f.read()
        dataset = MatmulDataset.deserialize(dataset_json)
    if dataset is None:
        logger.error(f"Failed to load dataset from {dataset_path}")
        return None, {}

    # Create dataloaders
    logger.info("Creating dataloaders")
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        hardware_name=hardware_name,
        op_name=op_name,
        log_transform=True,
        num_workers=4,
        seed=seed,
        debug=True,  # Enable debug mode to check data quality
    )

    # Get the feature dimensions
    problem_feature_dim = train_dataloader.dataset.dataset.problem_feature_dim
    config_feature_dim = train_dataloader.dataset.dataset.config_feature_dim

    # Create the model
    logger.info(
        f"Creating {model_type} model with {problem_feature_dim} problem features and {config_feature_dim} config features"
    )
    if model_type == "base":
        model = MatmulTimingModel(
            problem_feature_dim=problem_feature_dim,
            config_feature_dim=config_feature_dim,
        )
    else:  # "deep"
        model = DeepMatmulTimingModel(
            problem_feature_dim=problem_feature_dim,
            config_feature_dim=config_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    # Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {num_params} parameters")

    # Create the trainer
    trainer = MatmulModelTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
    )

    # Train the model
    logger.info(f"Training model for {num_epochs} epochs")
    history = trainer.train(
        num_epochs=num_epochs,
        patience=patience,
        checkpoint_path=model_path,
        verbose=True,
    )

    # Plot the training history
    history_plot_path = os.path.join(log_dir, f"matmul_timing_{model_type}_history.png")
    plot_training_history(history, history_plot_path)

    # Evaluate the model on the test set
    test_loss = trainer._evaluate(test_dataloader, "Test")
    rmse = torch.sqrt(torch.tensor(test_loss))

    logger.info(f"Test Loss (MSE): {test_loss:.6f}")
    logger.info(f"Test RMSE: {rmse:.6f}")
    logger.info(f"Test RMSE (exp): {torch.exp(torch.tensor(rmse)):.6f}")

    return model, history


def validate_model(
    model_path: str,
    validation_dataset_path: str,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    hardware_name: Optional[str] = None,
    op_name: Optional[str] = None,
    top_n_worst: int = 10,
) -> None:
    """
    Validate a trained model on a separate validation dataset or directory of datasets.

    Args:
        model_path: Path to the trained model
        validation_dataset_path: Path to the validation dataset file or directory containing dataset files
        batch_size: Batch size for validation
        device: Device to validate on
        hardware_name: Optional hardware name to filter by
        op_name: Optional operation name to filter by
        top_n_worst: Number of worst predictions to analyze
    """
    # Check if model exists
    logger.info("Checking if model file exists...")
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return
    logger.info(f"Model file found: {model_path}")

    # Check if validation dataset path exists
    logger.info("Checking if validation dataset exists...")
    if not os.path.exists(validation_dataset_path):
        logger.error(f"Validation dataset not found at {validation_dataset_path}")
        return
    logger.info(f"Validation dataset found: {validation_dataset_path}")

    # Check if validation_dataset_path is a directory or a file
    if os.path.isdir(validation_dataset_path):
        # Load from directory
        logger.info(
            f"Loading all validation data files from directory: {validation_dataset_path}"
        )
        try:
            _, val_dataloader, _ = create_directory_dataloaders(
                data_dir=validation_dataset_path,
                batch_size=batch_size,
                hardware_name=hardware_name,
                op_name=op_name,
                log_transform=True,
                num_workers=4,
                seed=42,  # Use a fixed seed for reproducibility
                file_extensions=["json", "msgpack"],
            )
        except Exception as e:
            logger.error(
                f"Failed to create dataloaders from directory {validation_dataset_path}: {e}"
            )
            return
    else:
        # Load from single file
        logger.info(f"Loading validation dataset from {validation_dataset_path}")
        if validation_dataset_path.endswith(".msgpack"):
            with open(validation_dataset_path, "rb") as f:
                dataset_data = f.read()
            dataset = MatmulDataset.from_msgpack(dataset_data)
        else:
            with open(validation_dataset_path, "r") as f:
                dataset_json = f.read()
            dataset = MatmulDataset.deserialize(dataset_json)

        if dataset is None:
            logger.error(
                f"Failed to load validation dataset from {validation_dataset_path}"
            )
            return

        # Create dataloaders (we only need the validation dataloader)
        logger.info("Creating validation dataloader")
        _, val_dataloader, _ = create_dataloaders(
            dataset=dataset,
            batch_size=batch_size,
            hardware_name=hardware_name,
            op_name=op_name,
        )

    # Get the feature dimensions
    problem_feature_dim = val_dataloader.dataset.dataset.problem_feature_dim
    config_feature_dim = val_dataloader.dataset.dataset.config_feature_dim

    # Load the trained model weights
    logger.info(f"Loading model weights from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        return

    # Check if the model was saved as a complete checkpoint or just state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        logger.info("Loading from checkpoint format")
        # Extract model parameters from checkpoint
        problem_feature_dim = checkpoint.get("problem_feature_dim", problem_feature_dim)
        config_feature_dim = checkpoint.get("config_feature_dim", config_feature_dim)
        hidden_dim = checkpoint.get("hidden_dim", 128)
        num_layers = checkpoint.get("num_layers", 10)
        model_type = checkpoint.get("model_type", "deep")

        # Recreate the model with the correct architecture
        if model_type == "base":
            model = MatmulTimingModel(
                problem_feature_dim=problem_feature_dim,
                config_feature_dim=config_feature_dim,
            )
        else:  # "deep"
            model = DeepMatmulTimingModel(
                problem_feature_dim=problem_feature_dim,
                config_feature_dim=config_feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )

        # Load the state dict
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Direct state dict loading
        # Assume it's a deep model if we don't know
        model = DeepMatmulTimingModel(
            problem_feature_dim=problem_feature_dim,
            config_feature_dim=config_feature_dim,
        )
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Create a trainer just for evaluation
    trainer = MatmulModelTrainer(
        model=model,
        train_dataloader=None,
        val_dataloader=val_dataloader,
        test_dataloader=None,
        device=device,
    )

    # Evaluate the model on the validation dataset
    val_loss = trainer._evaluate(val_dataloader, "Validation")
    rmse = torch.sqrt(torch.tensor(val_loss))

    logger.info(f"Validation Loss (MSE): {val_loss:.6f}")
    logger.info(f"Validation RMSE: {rmse:.6f}")
    logger.info(f"Validation RMSE (exp): {torch.exp(rmse):.6f}")

    # Analyze the worst predictions
    if top_n_worst > 0:
        logger.info(f"Analyzing worst {top_n_worst} predictions...")
        analyze_worst_predictions(model, val_dataloader, device, top_n=top_n_worst)


def run_model_example(
    dataset_path: str,
    model_type: str = "deep",
    batch_size: int = 64,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 20,
    log_dir: str = "logs",
    model_dir: str = "models",
    hardware_name: Optional[str] = None,
    op_name: Optional[str] = None,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Run an example demonstrating how to train and use a matrix multiplication timing prediction model.

    Args:
        dataset_path: Path to the dataset file
        model_type: Type of model to train ("base" or "deep")
        batch_size: Batch size for the dataloaders
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        patience: Number of epochs to wait for improvement before early stopping
        log_dir: Directory to save logs
        model_dir: Directory to save models
        hardware_name: Optional hardware name to filter by
        op_name: Optional operation name to filter by
        seed: Random seed for reproducibility
        device: Device to train on
    """
    # Create the directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Load the dataset
    logger.info(f"Loading dataset from {dataset_path}")
    if dataset_path.endswith(".msgpack"):
        with open(dataset_path, "rb") as f:
            dataset_data = f.read()
        dataset = MatmulDataset.from_msgpack(dataset_data)
    else:
        with open(dataset_path, "r") as f:
            dataset_json = f.read()
        dataset = MatmulDataset.deserialize(dataset_json)
    if dataset is None:
        logger.error(f"Failed to load dataset from {dataset_path}")
        return

    # Print dataset statistics
    print_dataset_statistics(dataset, hardware_name, op_name)

    # Train the model
    logger.info(f"Training {model_type} model")
    checkpoint_path = os.path.join(model_dir, f"matmul_timing_{model_type}_model.pt")

    # Import the config class
    from diode.model.matmul_model_config import MatmulModelConfig

    # Create a config with the specified parameters
    config = MatmulModelConfig(
        model_type=model_type,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        hardware_name=hardware_name,
        op_name=op_name,
        log_transform=True,
        seed=seed,
        device=device,
    )

    model, history, _ = train_model_from_dataset(
        dataset=dataset,
        config=config,
        log_dir=log_dir,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )

    # Plot the training history
    history_plot_path = os.path.join(log_dir, f"matmul_timing_{model_type}_history.png")
    plot_training_history(history, history_plot_path)

    # Evaluate the model on the test set
    logger.info("Making predictions on the test set")
    _, _, test_dataloader = create_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        hardware_name=hardware_name,
        op_name=op_name,
        log_transform=True,
        num_workers=4,
        seed=seed,
    )

    # Move the model to the device
    model = model.to(device)
    model.eval()

    # Initialize variables
    total_loss = 0.0
    criterion = torch.nn.MSELoss()

    # Evaluate on the test set
    with torch.no_grad():
        for problem_features, config_features, targets in test_dataloader:
            # Move the data to the device
            problem_features = problem_features.to(device)
            config_features = config_features.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(problem_features, config_features)

            # Calculate the loss
            loss = criterion(outputs, targets)

            # Update the total loss
            total_loss += loss.item()

    # Calculate the average loss
    avg_loss = total_loss / len(test_dataloader)

    # Calculate the RMSE
    rmse = torch.sqrt(torch.tensor(avg_loss))

    # Print the results
    print("\nModel Evaluation:")
    print("----------------")
    print(f"Test Loss (MSE): {avg_loss:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test RMSE (exp): {torch.exp(rmse):.6f}")

    logger.info("Example completed")


def train_model_from_directory(
    data_dir: str,
    model_path: str,
    model_type: str = "deep",
    batch_size: int = 64,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 20,
    hidden_dim: int = 128,
    num_layers: int = 10,
    hardware_name: Optional[str] = None,
    op_name: Optional[str] = None,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_dir: str = "logs",
    file_extensions: Optional[List[str]] = None,
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    Train a model on all data files found in a directory.

    This function automatically discovers and loads all JSON and MessagePack files
    from the specified directory, combines them into a single dataset, and trains
    a model on the combined data.

    Args:
        data_dir: Directory containing the data files (JSON and/or MessagePack)
        model_path: Path to save the trained model
        model_type: Type of model to train ("base" or "deep")
        batch_size: Batch size for training
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        patience: Number of epochs to wait for improvement before early stopping
        hidden_dim: Hidden dimension of the model
        num_layers: Number of layers in the model
        hardware_name: Optional hardware name to filter by
        op_name: Optional operation name to filter by
        seed: Random seed for reproducibility
        device: Device to train on
        log_dir: Directory to save logs
        file_extensions: List of file extensions to look for (default: ['json', 'msgpack'])

    Returns:
        Tuple of (trained model, training history)
    """
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Create directories if they don't exist
    os.makedirs(
        (
            os.path.dirname(os.path.abspath(model_path))
            if os.path.dirname(model_path)
            else "."
        ),
        exist_ok=True,
    )
    os.makedirs(log_dir, exist_ok=True)

    # Create dataloaders from directory
    logger.info(f"Loading all data files from directory: {data_dir}")
    train_dataloader, val_dataloader, test_dataloader = create_directory_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        hardware_name=hardware_name,
        op_name=op_name,
        log_transform=True,
        num_workers=4,
        seed=seed,
        file_extensions=file_extensions,
    )

    # Get the feature dimensions
    problem_feature_dim = train_dataloader.dataset.dataset.problem_feature_dim
    config_feature_dim = train_dataloader.dataset.dataset.config_feature_dim

    # Create the model
    logger.info(
        f"Creating {model_type} model with {problem_feature_dim} problem features and {config_feature_dim} config features"
    )
    if model_type == "base":
        model = MatmulTimingModel(
            problem_feature_dim=problem_feature_dim,
            config_feature_dim=config_feature_dim,
        )
    else:  # "deep"
        model = DeepMatmulTimingModel(
            problem_feature_dim=problem_feature_dim,
            config_feature_dim=config_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    # Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {num_params} parameters")

    # Create the trainer
    trainer = MatmulModelTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
    )

    # Train the model
    logger.info(f"Training model for {num_epochs} epochs")
    history = trainer.train(
        num_epochs=num_epochs,
        patience=patience,
        checkpoint_path=model_path,
        verbose=True,
    )

    # Plot the training history
    history_plot_path = os.path.join(log_dir, f"matmul_timing_{model_type}_history.png")
    plot_training_history(history, history_plot_path)

    # Evaluate the model on the test set
    test_loss = trainer._evaluate(test_dataloader, "Test")
    rmse = torch.sqrt(torch.tensor(test_loss))

    logger.info(f"Test Loss (MSE): {test_loss:.6f}")
    logger.info(f"Test RMSE: {rmse:.6f}")
    logger.info(f"Test RMSE (exp): {torch.exp(torch.tensor(rmse)):.6f}")

    return model, history
