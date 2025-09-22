"""
Model utility functions for training and evaluation.
"""

import json
import logging
import os
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm
from torch_diode.model.directory_dataset_loader import create_directory_dataloaders
from torch_diode.model.matmul_dataset_loader import create_dataloaders
from torch_diode.model.matmul_model_trainer import (
    analyze_worst_predictions,
    MatmulModelTrainer,
    train_model_from_dataset,
)
from torch_diode.model.matmul_timing_model import (
    DeepMatmulTimingModel,
    MatmulTimingModel,
)

from torch_diode.types.matmul_dataset import Dataset as MatmulDataset
from torch_diode.utils.dataset_utils import print_dataset_statistics
from torch_diode.utils.feature_extraction import (
    extract_config_features,
    extract_problem_features,
)
from torch_diode.utils.debug_config import type_assert
from torch_diode.utils.visualization_utils import plot_training_history

logger = logging.getLogger(__name__)
import json, numpy as np, os, tqdm
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from torch_diode.model.matmul_dataset_loader import (  # if needed by loaders
    MatmulTimingDataset,
)
from torch_diode.types.matmul_dataset import Dataset as MatmulDataset

from torch_diode.types.matmul_types import MMShape, Solution, TritonGEMMConfig


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
    Validate a trained model's ability to select optimal configs vs max-autotune,
    computing stats per op (mm, addmm, bmm) and an overall rollup.
    """

    logger.info("Loading validation dataset and model...")

    # Basic file checks
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return
    if not os.path.exists(validation_dataset_path):
        logger.error(f"Validation dataset not found at {validation_dataset_path}")
        return
    if not os.path.exists(max_autotune_solution_path):
        logger.error(f"Max-autotune solution not found at {max_autotune_solution_path}")
        return

    # ---- Load max-autotune solution(s), accept both shapes:
    # 1) {"config":[...]}                         (single Solution, all ops)
    # 2) {"mm":{"config":[...]}, "addmm":{...}}   (per-op Solutions)
    try:
        with open(max_autotune_solution_path, "r", encoding="utf-8") as f:
            ma_obj = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load max-autotune solution JSON: {e}")
        return

    def _parse_cfg_list(lst) -> List[TritonGEMMConfig]:
        out = []
        for cfg in lst:
            if isinstance(cfg, TritonGEMMConfig):
                out.append(cfg)
            else:
                out.append(TritonGEMMConfig(**cfg))
        return out

    # Normalize into {op_name: List[TritonGEMMConfig]}
    max_autotune_cfgs_by_op: Dict[str, List[TritonGEMMConfig]] = {}
    if "config" in ma_obj and isinstance(ma_obj["config"], list):
        # single-solution file: fill later once we know which ops are present
        single_solution_cfgs = _parse_cfg_list(ma_obj["config"])
        _single_solution_holder = single_solution_cfgs
    else:
        # dict keyed by op
        for k, v in ma_obj.items():
            if isinstance(v, dict) and "config" in v and isinstance(v["config"], list):
                max_autotune_cfgs_by_op[k] = _parse_cfg_list(v["config"])
        _single_solution_holder = None  # not needed

    # ---- Load validation dataset + dataloader (reuse your utility)
    if os.path.isdir(validation_dataset_path):
        from torch_diode.model.directory_dataset_loader import (
            create_directory_dataloaders,
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
                use_lazy=False,
            )
        except Exception as e:
            logger.error(f"Failed to create dataloaders from directory: {e}")
            return
    else:
        # File-based fallback
        if validation_dataset_path.endswith(".msgpack"):
            with open(validation_dataset_path, "rb") as f:
                dataset_data = f.read()
            dataset = MatmulDataset.from_msgpack(dataset_data)
        else:
            with open(validation_dataset_path, "r") as f:
                dataset_json = f.read()
            dataset = MatmulDataset.deserialize(dataset_json)

        if dataset is None:
            logger.error("Failed to load validation dataset")
            return

        # Use create_dataloaders for single file loading
        _, val_dataloader, _ = create_dataloaders(
            dataset=dataset,
            batch_size=batch_size,
            hardware_name=hardware_name,
            op_name=op_name,
            log_transform=True,
            num_workers=4,
            seed=42,
        )

    # ---- Load model
    checkpoint = torch.load(model_path, map_location=device)
    problem_feature_dim = val_dataloader.dataset.dataset.problem_feature_dim
    config_feature_dim = val_dataloader.dataset.dataset.config_feature_dim

    # Supports two shapes of checkpoint (dict with "model_state_dict" or raw sd)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
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
                hidden_dim=checkpoint.get("hidden_dim", 128),
                num_layers=checkpoint.get("num_layers", 10),
            )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = DeepMatmulTimingModel(
            problem_feature_dim=problem_feature_dim,
            config_feature_dim=config_feature_dim,
        )
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()

    # ---- Pull underlying flattened tensors
    timing_dataset = val_dataloader.dataset.dataset
    if hasattr(timing_dataset, "timing_dataset"):  # DirectoryMatmulDataset wrapper
        timing_dataset = timing_dataset.timing_dataset

    problem_features = timing_dataset.problem_features
    timings = timing_dataset.timings
    configs = timing_dataset.configs

    # ---- Build lookups back to original dataset to recover BOTH shape and op
    logger.info("Extracting MMShape + op labels from the original dataset...")
    mmshape_lookup: Dict[int, MMShape] = {}
    oplabel_lookup: Dict[int, str] = {}
    global_idx = 0

    # Try to traverse original hierarchical dataset to assign (shape, op) per row
    if hasattr(val_dataloader.dataset, "dataset") and hasattr(
        val_dataloader.dataset.dataset, "dataset"
    ):
        original_ds = val_dataloader.dataset.dataset.dataset  # MatmulDataset-like
        # original_ds.hardware: OrderedDict[str, DatasetHardware]
        for hw_name, hw in getattr(original_ds, "hardware", {}).items():
            if hardware_name and hw_name != hardware_name:
                continue
            # operations: OrderedDict[str, DatasetOperation]
            operations = getattr(hw, "operation", {})
            for op_k, op_node in operations.items():
                if op_name and op_k != op_name:
                    continue
                # op_node.solution: OrderedDict[MMShape, DatasetSolution]
                solutions = getattr(op_node, "solution", {})
                for mmshape, dsol in solutions.items():
                    # each DatasetSolution has a list of TimedConfig entries
                    tc_list = getattr(dsol, "timed_configs", [])
                    for _tc in tc_list:
                        mmshape_lookup[global_idx] = mmshape
                        oplabel_lookup[global_idx] = op_k
                        global_idx += 1

    # Fallback if we failed to rebuild indices (best-effort)
    if not mmshape_lookup:
        logger.info("Falling back: reconstructing MMShapes from features; op guessed.")
        for i, _ in enumerate(configs):
            pf = problem_features[i].tolist()
            B, M, N, K = int(pf[0]), int(pf[1]), int(pf[2]), int(pf[3])
            mmshape_lookup[i] = MMShape(
                B=B,
                M=M,
                N=N,
                K=K,
                M_dtype=torch.float32,
                K_dtype=torch.float32,
                out_dtype=torch.float32,
                out_size=(B, M, N),
                out_stride=(M * N, N, 1),
            )
            # If a single op was requested, use it; otherwise default to "mm"
            oplabel_lookup[i] = op_name or "mm"

    # If the max-autotune file was the single-solution form, now that we know ops present, fan out:
    if _single_solution_holder is not None:
        present_ops = sorted(set(oplabel_lookup.values()))
        for op_k in present_ops:
            max_autotune_cfgs_by_op.setdefault(op_k, list(_single_solution_holder))

    # ---- Build per-op maps
    logger.info("Building per-op (MMShape, TritonGEMMConfig) -> runtime maps...")
    by_op_mymap: Dict[str, Dict[Tuple[MMShape, TritonGEMMConfig], float]] = {}
    by_op_shapes: Dict[str, List[MMShape]] = {}
    for i, cfg in tqdm.tqdm(
        list(enumerate(configs)), desc="Indexing examples", total=len(configs)
    ):
        mmshape = mmshape_lookup[i]
        opk = oplabel_lookup[i]
        if timing_dataset.log_transform:
            actual_runtime = float(np.exp(timings[i].item()))
        else:
            actual_runtime = float(timings[i].item())
        by_op_mymap.setdefault(opk, {})[(mmshape, cfg)] = actual_runtime
        if opk not in by_op_shapes:
            by_op_shapes[opk] = []
        if mmshape not in by_op_shapes[opk]:
            by_op_shapes[opk].append(mmshape)

    # ---- Build shape->(configs, runtimes) per op
    logger.info("Creating shape-to-configs mappings per op...")
    shape_to_configs_by_op: Dict[
        str, Dict[MMShape, Tuple[List[TritonGEMMConfig], List[float]]]
    ] = {}

    def _process_shape_for_map(opk: str, mmshape: MMShape):
        sc, sr = [], []
        for (s_key, c_key), rt in by_op_mymap[opk].items():
            if s_key == mmshape:
                sc.append(c_key)
                sr.append(rt)
        return mmshape, (sc, sr)

    for opk, shapes in by_op_shapes.items():
        shape_to_configs_by_op[opk] = {}
        shapes_list = list(shapes)
        with ThreadPoolExecutor(
            max_workers=min(len(shapes_list), os.cpu_count() or 1)
        ) as ex:
            futures = {
                ex.submit(_process_shape_for_map, opk, s): s for s in shapes_list
            }
            for fut in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc=f"shape->configs[{opk}]"
            ):
                s, pair = fut.result()
                shape_to_configs_by_op[opk][s] = pair

    # ---- Compute best max-autotune times per shape, per op
    logger.info("Computing max-autotune best results per op...")
    ma_best_by_op: Dict[str, Dict[MMShape, float]] = {}
    for opk, shapes in by_op_shapes.items():
        ma_cfgs = set(max_autotune_cfgs_by_op.get(opk, []))
        if not ma_cfgs:
            logger.warning(f"No max-autotune configs for op '{opk}'. Skipping.")
            ma_best_by_op[opk] = {}
            continue
        shape_best = {}
        for s in shapes:
            sc, sr = shape_to_configs_by_op[opk].get(s, ([], []))
            best = float("inf")
            found = False
            for i, c in enumerate(sc):
                if c in ma_cfgs and sr[i] < best:
                    best = sr[i]
                    found = True
            if found:
                shape_best[s] = best
        ma_best_by_op[opk] = shape_best
        logger.info(f"[{opk}] Found max-autotune results for {len(shape_best)} shapes")

    # ---- Model predictions per op/shape
    logger.info("Computing model predictions per op...")
    shape_predictions_by_op: Dict[str, Dict[MMShape, np.ndarray]] = {}

    for opk, shapes in by_op_shapes.items():
        shape_predictions_by_op[opk] = {}
        for s in tqdm.tqdm(shapes, desc=f"predict[{opk}]"):
            sc, sr = shape_to_configs_by_op[opk].get(s, ([], []))
            if not sc:
                continue
            with torch.no_grad():
                pf = extract_problem_features(s, return_tensors=False)
                problem_batch = torch.tensor(
                    [pf] * len(sc), dtype=torch.float32, device=device
                )
                cfg_feats = [
                    extract_config_features(c, return_tensors=False) for c in sc
                ]
                config_batch = torch.tensor(
                    cfg_feats, dtype=torch.float32, device=device
                )
                preds = model(problem_batch, config_batch).cpu().numpy().flatten()
            shape_predictions_by_op[opk][s] = preds

    # ---- Top-N analysis per op
    n_values = [1, 5, 10, 20, 50, 100, 500]
    model_results_by_op: Dict[str, Dict[int, Dict[MMShape, float]]] = {
        opk: {n: {} for n in n_values} for opk in by_op_shapes.keys()
    }

    for opk, shapes in by_op_shapes.items():
        for n in tqdm.tqdm(n_values, desc=f"top-N[{opk}]"):
            for s in shapes:
                preds = shape_predictions_by_op[opk].get(s)
                if preds is None:
                    continue
                sc, sr = shape_to_configs_by_op[opk].get(s, ([], []))
                if not sc:
                    continue
                idx = np.argsort(preds)[:n]
                top_rts = [sr[i] for i in idx]
                if top_rts:
                    model_results_by_op[opk][n][s] = float(min(top_rts))

    # ---- Stats helper (reuses your original logic)
    def _compute_stats_for(opk: str, n: int):
        ma_map = ma_best_by_op.get(opk, {})
        mdl_map = model_results_by_op.get(opk, {}).get(n, {})
        common = set(ma_map.keys()) & set(mdl_map.keys())
        if not common:
            return None

        model_better = model_worse = model_equal = 0
        total_model = total_ma = 0.0
        improvements, degradations, ratios, shape_details = [], [], [], []

        best_impr = float("-inf")
        worst_impr = float("inf")
        best_deg = float("inf")
        worst_deg = float("-inf")

        for s in common:
            mt = mdl_map[s]
            at = ma_map[s]
            total_model += mt
            total_ma += at
            if mt > 0:
                r = at / mt
                ratios.append(r)
                detail = {
                    "shape": str(s),
                    "model_time": float(mt),
                    "ma_time": float(at),
                    "speedup_ratio": float(r),
                    "performance": "",
                }
                if r > 1.0:
                    model_better += 1
                    improvements.append(r)
                    detail["performance"] = "better"
                    best_impr = max(best_impr, r)
                elif r < 1.0:
                    model_worse += 1
                    degradations.append(r)
                    detail["performance"] = "worse"
                    worst_deg = max(worst_deg, r)
                    best_deg = min(best_deg, r)
                else:
                    model_equal += 1
                    detail["performance"] = "equal"
                worst_impr = min(worst_impr, r)
                shape_details.append(detail)

        if not ratios:
            return None

        avg_ratio = float(np.mean(ratios))
        median_ratio = float(np.median(ratios))
        std_ratio = float(np.std(ratios))
        pct_better = 100.0 * model_better / len(common)
        pct_worse = 100.0 * model_worse / len(common)
        pct_equal = 100.0 * model_equal / len(common)
        time_saved_pct = (
            100.0 * (total_ma - total_model) / total_ma if total_ma > 0 else 0.0
        )

        percentiles = [10, 25, 50, 75, 90, 95, 99]
        ratio_percentiles = {
            f"p{p}": float(np.percentile(ratios, p)) for p in percentiles
        }
        avg_impr = float(np.mean(improvements)) if improvements else 0.0
        avg_deg = float(np.mean(degradations)) if degradations else 0.0

        stats = {
            "op": opk,
            "top_n": n,
            "shapes_analyzed": len(common),
            "performance_breakdown": {
                "model_better": {
                    "count": model_better,
                    "percentage": pct_better,
                    "avg_speedup": avg_impr if improvements else 0.0,
                    "best_speedup": (
                        float(best_impr) if best_impr != float("-inf") else 0.0
                    ),
                },
                "model_worse": {
                    "count": model_worse,
                    "percentage": pct_worse,
                    "avg_slowdown": avg_deg if degradations else 0.0,
                    "best_slowdown": (
                        float(best_deg) if best_deg != float("inf") else 0.0
                    ),
                    "worst_slowdown": (
                        float(worst_deg) if worst_deg != float("-inf") else 0.0
                    ),
                },
                "model_equal": {
                    "count": model_equal,
                    "percentage": pct_equal,
                },
            },
            "overall_metrics": {
                "avg_speedup_ratio": avg_ratio,
                "median_speedup_ratio": median_ratio,
                "std_speedup_ratio": std_ratio,
                "total_time_saved_pct": float(time_saved_pct),
                "best_case": float(best_impr) if best_impr != float("-inf") else 0.0,
                "worst_case": float(worst_impr) if worst_impr != float("inf") else 0.0,
            },
            "distribution": ratio_percentiles,
            "raw_times": {
                "total_model_time": float(total_model),
                "total_ma_time": float(total_ma),
            },
            "shape_details": shape_details,
        }
        return stats

    # ---- Aggregate stats per op and overall
    logger.info("\n" + "=" * 80)
    logger.info("MODEL VS MAX-AUTOTUNE COMPARISON RESULTS (per op)")
    logger.info("=" * 80)

    all_statistics = {"per_op": {}}
    ops_in_data = sorted(by_op_shapes.keys())

    for opk in ops_in_data:
        all_statistics["per_op"][opk] = {}
        for n in n_values:
            stats = _compute_stats_for(opk, n)
            if stats is None:
                logger.warning(f"[{opk}] No common shapes for top-{n}")
                continue
            all_statistics["per_op"][opk][f"top_{n}"] = stats

            # brief log summary
            logger.info(
                f"\n[{opk}] Top-{n}: shapes={stats['shapes_analyzed']}, "
                f"avg_ratio={stats['overall_metrics']['avg_speedup_ratio']:.3f}x, "
                f"median={stats['overall_metrics']['median_speedup_ratio']:.3f}x, "
                f"time_saved={stats['overall_metrics']['total_time_saved_pct']:.2f}%"
            )

    # ---- Overall rollup (union of shapes per op; micro-average by summing times)
    overall = {}
    for n in n_values:
        # Pool all shapes but keep their own op-specific ma/model picks
        total_model = total_ma = 0.0
        ratios = []
        improvements, degradations = [], []
        model_better = model_worse = model_equal = 0
        best_impr = float("-inf")
        worst_impr = float("inf")
        best_deg = float("inf")
        worst_deg = float("-inf")
        shape_details = []

        for opk in ops_in_data:
            ma_map = ma_best_by_op.get(opk, {})
            mdl_map = model_results_by_op.get(opk, {}).get(n, {})
            common = set(ma_map.keys()) & set(mdl_map.keys())
            for s in common:
                mt, at = mdl_map[s], ma_map[s]
                total_model += mt
                total_ma += at
                if mt > 0:
                    r = at / mt
                    ratios.append(r)
                    if r > 1.0:
                        model_better += 1
                        improvements.append(r)
                        best_impr = max(best_impr, r)
                    elif r < 1.0:
                        model_worse += 1
                        degradations.append(r)
                        worst_deg = max(worst_deg, r)
                        best_deg = min(best_deg, r)
                    else:
                        model_equal += 1
                    worst_impr = min(worst_impr, r)
                    shape_details.append(
                        {
                            "op": opk,
                            "shape": str(s),
                            "model_time": float(mt),
                            "ma_time": float(at),
                            "speedup_ratio": float(r),
                            "performance": (
                                "better"
                                if r > 1.0
                                else ("worse" if r < 1.0 else "equal")
                            ),
                        }
                    )

        if ratios:
            avg_ratio = float(np.mean(ratios))
            median_ratio = float(np.median(ratios))
            std_ratio = float(np.std(ratios))
            time_saved_pct = (
                100.0 * (total_ma - total_model) / total_ma if total_ma > 0 else 0.0
            )
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            ratio_percentiles = {
                f"p{p}": float(np.percentile(ratios, p)) for p in percentiles
            }
            overall[f"top_{n}"] = {
                "top_n": n,
                "shapes_analyzed": len(shape_details),
                "performance_breakdown": {
                    "model_better": {
                        "count": model_better,
                        "percentage": 100.0 * model_better / max(1, len(shape_details)),
                        "avg_speedup": (
                            float(np.mean(improvements)) if improvements else 0.0
                        ),
                        "best_speedup": (
                            float(best_impr) if best_impr != float("-inf") else 0.0
                        ),
                    },
                    "model_worse": {
                        "count": model_worse,
                        "percentage": 100.0 * model_worse / max(1, len(shape_details)),
                        "avg_slowdown": (
                            float(np.mean(degradations)) if degradations else 0.0
                        ),
                        "best_slowdown": (
                            float(best_deg) if best_deg != float("inf") else 0.0
                        ),
                        "worst_slowdown": (
                            float(worst_deg) if worst_deg != float("-inf") else 0.0
                        ),
                    },
                    "model_equal": {
                        "count": model_equal,
                        "percentage": 100.0 * model_equal / max(1, len(shape_details)),
                    },
                },
                "overall_metrics": {
                    "avg_speedup_ratio": avg_ratio,
                    "median_speedup_ratio": median_ratio,
                    "std_speedup_ratio": std_ratio,
                    "total_time_saved_pct": float(time_saved_pct),
                    "best_case": (
                        float(best_impr) if best_impr != float("-inf") else 0.0
                    ),
                    "worst_case": (
                        float(worst_impr) if worst_impr != float("inf") else 0.0
                    ),
                },
                "distribution": ratio_percentiles,
                "raw_times": {
                    "total_model_time": float(total_model),
                    "total_ma_time": float(total_ma),
                },
                "shape_details": shape_details,
            }

    all_statistics["overall"] = overall
    all_statistics["metadata"] = {
        "model_path": model_path,
        "validation_dataset_path": validation_dataset_path,
        "timestamp": datetime.now().isoformat(),
        "hardware_name": hardware_name or "all",
        "op_name_filter": op_name or "none",
        "batch_size": batch_size,
        "device": device,
        "ops_present": list(ops_in_data),
        "notes": "Per-op stats computed independently; overall is micro-averaged by time.",
    }

    # ---- Write JSON
    out_path = "model_performance_analysis_by_op.json"
    try:
        with open(out_path, "w") as f:
            json.dump(all_statistics, f, indent=2)
        logger.info(f"Comprehensive per-op statistics saved to: {out_path}")
    except Exception as e:
        logger.error(f"Failed to save statistics to JSON: {e}")


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
    try:
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
    except (FileNotFoundError, OSError) as e:
        logger.error(f"Failed to load dataset from {dataset_path}: {e}")
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
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
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
    from torch_diode.model.matmul_model_config import MatmulModelConfig

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

    # Check if model training was successful (dataset not empty)
    if model is None:
        logger.warning("Model training failed or dataset was empty. Exiting example.")
        return

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

    # Check if test dataloader was created successfully
    if test_dataloader is None:
        logger.warning("Test dataloader is empty, skipping test evaluation.")
        logger.info("Example completed")
        return

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
