"""
PyTorch data loader that finds and loads all data files from a directory.

This module provides dataset classes that can automatically discover and load
all JSON and MessagePack files from a directory, combining them into a single
dataset for training (eager merge) or lazily streaming per-file (lazy mode).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from torch_diode.model.matmul_dataset_loader import MatmulTimingDataset
from torch_diode.types.matmul_dataset import Dataset as MatmulDataset
from torch_diode.utils.debug_config import type_assert
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ----------------------------- Utilities -----------------------------


def _is_cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _pin_memory_default() -> bool:
    # Pin only if we actually have CUDA to benefit from page-locked host I/O.
    return _is_cuda_available()


def _safe_getattr_or_dict(d: Union[dict, object], attr: str, default):
    if isinstance(d, dict):
        return d.get(attr, default)
    return getattr(d, attr, default)


def _safe_setattr_or_dict(d: Union[dict, object], attr: str, value) -> None:
    if isinstance(d, dict):
        d[attr] = value
    else:
        setattr(d, attr, value)


def _get_operations(hw_obj: Union[dict, object]) -> Dict:
    return _safe_getattr_or_dict(hw_obj, "operation", {})


def _set_operations(hw_obj: Union[dict, object], ops: Dict) -> None:
    _safe_setattr_or_dict(hw_obj, "operation", ops)


def _get_solutions(op_obj: Union[dict, object]) -> Dict:
    return _safe_getattr_or_dict(op_obj, "solution", {})


def _set_solutions(op_obj: Union[dict, object], sols: Dict) -> None:
    _safe_setattr_or_dict(op_obj, "solution", sols)


def _median_time_us(sol: Union[dict, object]) -> float:
    # Be liberal: many schemas nest stats differently; default to +inf.
    try:
        if isinstance(sol, dict):
            stats = sol.get("stats", {})
            return float(stats.get("median_us", float("inf")))
        stats = getattr(sol, "stats", None)
        if stats is None:
            return float("inf")
        return float(getattr(stats, "median_us", float("inf")))
    except Exception:
        return float("inf")


# -------------------- DirectoryMatmulDataset (eager) --------------------


class DirectoryMatmulDataset(Dataset):
    """
    Eager, merged PyTorch Dataset that loads all JSON/MessagePack files from a directory.

    Automatically discovers files, loads them (in parallel), filters early by
    hardware/op, merges with an explicit collision policy, then wraps a
    MatmulTimingDataset for feature extraction.
    """

    def __init__(
        self,
        data_dir: str,
        hardware_name: Optional[str] = None,
        op_name: Optional[str] = None,
        log_transform: bool = True,
        file_extensions: Optional[List[str]] = None,
        max_io_workers: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Directory containing the data files.
            hardware_name: Optional hardware name to filter by (early filter).
            op_name: Optional operation name to filter by (early filter).
            log_transform: Whether to apply log transform to timing values.
            file_extensions: Extensions to include (default: ['json','msgpack']).
            max_io_workers: Max threads for I/O (default: min(32, max(4, os.cpu_count()))).
        """
        type_assert(isinstance(data_dir, str), f"data_dir must be str, got {type(data_dir)}")
        type_assert(hardware_name is None or isinstance(hardware_name, str), f"hardware_name must be str or None, got {type(hardware_name)}")
        type_assert(op_name is None or isinstance(op_name, str), f"op_name must be str or None, got {type(op_name)}")
        type_assert(isinstance(log_transform, bool), f"log_transform must be bool, got {type(log_transform)}")
        type_assert(file_extensions is None or isinstance(file_extensions, list), f"file_extensions must be list or None, got {type(file_extensions)}")
        type_assert(max_io_workers is None or isinstance(max_io_workers, int), f"max_io_workers must be int or None, got {type(max_io_workers)}")
        
        self.data_dir = data_dir
        self.hardware_name = hardware_name
        self.op_name = op_name
        self.log_transform = log_transform

        if file_extensions is None:
            file_extensions = ["json", "msgpack"]
        self.file_extensions = [e.lower().lstrip(".") for e in file_extensions]

        self.max_io_workers = max_io_workers or min(32, max(4, os.cpu_count() or 4))

        # A) recursive, de-duped, sorted discovery
        self.data_files = self._find_data_files()
        logger.info("Found %d data files in %s", len(self.data_files), data_dir)

        if not self.data_files:
            raise ValueError(f"No data files found in directory: {data_dir}")

        # B) parallel load with early filtering; then merge with explicit policy
        self.combined_dataset = self._load_and_combine_datasets()

        # Build timing dataset
        self.timing_dataset = MatmulTimingDataset(
            dataset=self.combined_dataset,
            hardware_name=hardware_name,
            op_name=op_name,
            log_transform=log_transform,
        )

        logger.info(
            "Loaded %d samples from %d files (eager merge)",
            len(self.timing_dataset),
            len(self.data_files),
        )

    # --------- A)  ---------

    def _find_data_files(self) -> List[str]:
        """
        Recursive file discovery with pathlib
        """
        root = Path(self.data_dir)
        exts = set(self.file_extensions)
        files: List[str] = []

        # Get all paths for progress tracking
        all_paths = list(root.rglob("*"))

        for p in tqdm(all_paths, desc="Discovering data files", unit="files"):
            if not p.is_file():
                continue
            suf = p.suffix.lower().lstrip(".")
            if suf in exts:
                files.append(str(p))
        files = sorted(set(files))  # de-dupe & stable ordering
        return files

    # --------- B) Parallel load with early filtering & robust errors ---------

    def _load_and_combine_datasets(self) -> MatmulDataset:
        files = self.data_files
        results: List[MatmulDataset] = []
        errors = 0

        def load_one(fp: str) -> Optional[MatmulDataset]:
            try:
                ds = self._load_single_file(fp)
                if ds is None:
                    return None

                # Early filtering by hardware/op to keep memory small
                if self.hardware_name is not None:
                    ds.hardware = {
                        k: v for k, v in ds.hardware.items() if k == self.hardware_name
                    }

                if self.op_name is not None:
                    for hw in list(ds.hardware):
                        hw_obj = ds.hardware[hw]
                        ops = _get_operations(hw_obj)
                        kept = {k: v for k, v in ops.items() if k == self.op_name}
                        _set_operations(hw_obj, kept)
                        if not kept:
                            ds.hardware.pop(hw, None)

                if not ds.hardware:
                    return None
                return ds
            except Exception:
                logger.exception("Failed to load %s", fp)
                return None

        with ThreadPoolExecutor(max_workers=self.max_io_workers) as ex:
            futs = {ex.submit(load_one, f): f for f in files}
            for fut in tqdm(
                as_completed(futs),
                total=len(files),
                desc="Loading datasets",
                unit="files",
            ):
                ds = fut.result()
                if ds is None:
                    errors += 1
                    continue
                results.append(ds)

        if not results:
            raise ValueError("No valid datasets after filtering/errors.")

        if errors:
            logger.warning(
                "%d files failed to load or were empty after filtering.", errors
            )

        # C) Merge with explicit collision policy (keep fastest median_us)
        merged = results[0]
        for ds in results[1:]:
            merged = self._merge_datasets(merged, ds)
        return merged

    def _load_single_file(self, file_path: str) -> Optional[MatmulDataset]:
        try:
            if file_path.lower().endswith(".msgpack"):
                with open(file_path, "rb") as f:
                    data = f.read()
                return MatmulDataset.from_msgpack(data)
            elif file_path.lower().endswith(".json"):
                with open(file_path, "r") as f:
                    data = f.read()
                return MatmulDataset.deserialize(data)
            else:
                logger.warning("Unsupported file extension: %s", file_path)
                return None
        except Exception:
            logger.exception("Failed to load %s", file_path)
            return None

    # --------- C) Safer merge with collision policy ---------

    def _merge_datasets(self, a: MatmulDataset, b: MatmulDataset) -> MatmulDataset:
        out = MatmulDataset(hardware={})

        # copy a
        for hw, hw_data in a.hardware.items():
            out.hardware[hw] = hw_data

        # merge b into out
        for hw, hw_data in b.hardware.items():
            if hw not in out.hardware:
                out.hardware[hw] = hw_data
                continue

            dst_hw = out.hardware[hw]
            dst_ops = _get_operations(dst_hw)
            src_ops = _get_operations(hw_data)

            for op_name, src_op in src_ops.items():
                if op_name not in dst_ops:
                    dst_ops[op_name] = src_op
                    continue

                dst_op = dst_ops[op_name]
                dst_solutions = _get_solutions(dst_op)
                src_solutions = _get_solutions(src_op)

                for sol_id, src_sol in src_solutions.items():
                    if sol_id not in dst_solutions:
                        dst_solutions[sol_id] = src_sol
                        continue

                    # Collision policy: keep the faster (lower median_us)
                    dst_sol = dst_solutions[sol_id]
                    if _median_time_us(src_sol) < _median_time_us(dst_sol):
                        dst_solutions[sol_id] = src_sol

                # write back if dict-style
                _set_solutions(dst_op, dst_solutions)

            _set_operations(dst_hw, dst_ops)

        return out

    # --------- Dataset interface delegates to underlying timing dataset ---------

    def __len__(self) -> int:
        return len(self.timing_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.timing_dataset[idx]

    @property
    def problem_feature_dim(self) -> int:
        return self.timing_dataset.problem_feature_dim

    @property
    def config_feature_dim(self) -> int:
        return self.timing_dataset.config_feature_dim

    @property
    def configs(self):
        return self.timing_dataset.configs

    def get_file_info(self) -> List[Tuple[str, int]]:
        # Unknown per-file counts after merge; preserve API (-1 sentinel).
        return [(Path(p).name, -1) for p in self.data_files]


# -------------------- E) LazyDirectoryMatmulDataset (lazy) --------------------


class LazyDirectoryMatmulDataset(Dataset):
    """
    Lazy dataset that avoids building a single monolithic MatmulDataset.

    It discovers files, computes per-file sample counts once, and serves items
    by instantiating per-file MatmulTimingDataset objects on demand (tiny LRU).
    """

    @dataclass
    class _PerFile:
        path: str
        count: int  # number of timing samples in this file after filtering

    def __init__(
        self,
        data_dir: str,
        hardware_name: Optional[str] = None,
        op_name: Optional[str] = None,
        log_transform: bool = True,
        file_extensions: Optional[List[str]] = None,
        cache_capacity: int = 8,
        max_io_workers: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.hardware_name = hardware_name
        self.op_name = op_name
        self.log_transform = log_transform

        if file_extensions is None:
            file_extensions = ["json", "msgpack"]
        self.file_extensions = [e.lower().lstrip(".") for e in file_extensions]

        self.max_io_workers = max_io_workers or min(32, max(4, os.cpu_count() or 4))
        self.cache_capacity = max(1, int(cache_capacity))

        # discovery
        self.data_files = self._find_data_files()
        if not self.data_files:
            raise ValueError(f"No data files found in directory: {data_dir}")

        # Build per-file counts in parallel (I/O), then cumulative index
        self._files: List[LazyDirectoryMatmulDataset._PerFile] = []
        self._cumulative: List[int] = []  # prefix sums of counts
        self._build_index()

        # LRU cache: file path -> MatmulTimingDataset
        from collections import OrderedDict

        self._cache: "OrderedDict[str, MatmulTimingDataset]" = OrderedDict()

        total = self._cumulative[-1] if self._cumulative else 0
        logger.info(
            "Lazy dataset ready: %d total samples from %d files (cache=%d)",
            total,
            len(self._files),
            self.cache_capacity,
        )

        if total == 0:
            raise ValueError("No samples after filtering; dataset is empty.")

    # Discovery (same as eager)
    def _find_data_files(self) -> List[str]:
        root = Path(self.data_dir)
        exts = set(self.file_extensions)
        files: List[str] = []
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            suf = p.suffix.lower().lstrip(".")
            if suf in exts:
                files.append(str(p))
        files = sorted(set(files))
        return files

    def _load_single_file_raw(self, file_path: str) -> Optional[MatmulDataset]:
        try:
            if file_path.lower().endswith(".msgpack"):
                with open(file_path, "rb") as f:
                    data = f.read()
                return MatmulDataset.from_msgpack(data)
            elif file_path.lower().endswith(".json"):
                with open(file_path, "r") as f:
                    data = f.read()
                return MatmulDataset.deserialize(data)
            else:
                logger.warning("Unsupported file extension: %s", file_path)
                return None
        except Exception:
            logger.exception("Failed to load %s", file_path)
            return None

    def _filter_in_place(self, ds: MatmulDataset) -> None:
        if self.hardware_name is not None:
            ds.hardware = {
                k: v for k, v in ds.hardware.items() if k == self.hardware_name
            }
        if self.op_name is not None:
            for hw in list(ds.hardware):
                hw_obj = ds.hardware[hw]
                ops = _get_operations(hw_obj)
                kept = {k: v for k, v in ops.items() if k == self.op_name}
                _set_operations(hw_obj, kept)
                if not kept:
                    ds.hardware.pop(hw, None)

    def _timing_len_for_file(self, fp: str) -> int:
        ds = self._load_single_file_raw(fp)
        if ds is None:
            return 0
        self._filter_in_place(ds)
        if not ds.hardware:
            return 0
        tds = MatmulTimingDataset(
            dataset=ds,
            hardware_name=self.hardware_name,
            op_name=self.op_name,
            log_transform=self.log_transform,
        )
        return len(tds)

    def _build_index(self) -> None:
        counts: Dict[str, int] = {}

        def count_one(fp: str) -> Tuple[str, int]:
            try:
                return fp, self._timing_len_for_file(fp)
            except Exception:
                logger.exception("Failed counting samples in %s", fp)
                return fp, 0

        with ThreadPoolExecutor(max_workers=self.max_io_workers) as ex:
            futs = {ex.submit(count_one, f): f for f in self.data_files}
            for fut in tqdm(
                as_completed(futs),
                total=len(self.data_files),
                desc="Indexing datasets",
                unit="files",
            ):
                fp, c = fut.result()
                if c > 0:
                    counts[fp] = c

        if not counts:
            # fall back to sequential to get better error logs
            for fp in self.data_files:
                c = self._timing_len_for_file(fp)
                if c > 0:
                    counts[fp] = c

        total = 0
        for fp in sorted(counts.keys()):
            c = counts[fp]
            self._files.append(self._PerFile(path=fp, count=c))
            total += c
            self._cumulative.append(total)

    # LRU cache helpers
    def _get_cached_timing_dataset(self, fp: str) -> MatmulTimingDataset:
        # simple LRU using OrderedDict
        tds = self._cache.get(fp)
        if tds is not None:
            self._cache.move_to_end(fp)
            return tds

        ds = self._load_single_file_raw(fp)
        if ds is None:
            raise RuntimeError(f"Failed to (re)load dataset for {fp}")
        self._filter_in_place(ds)
        if not ds.hardware:
            raise RuntimeError(f"No data after filtering for {fp}")

        tds = MatmulTimingDataset(
            dataset=ds,
            hardware_name=self.hardware_name,
            op_name=self.op_name,
            log_transform=self.log_transform,
        )
        self._cache[fp] = tds
        if len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)  # evict LRU
        return tds

    # Dataset API
    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        # find file via prefix sums
        import bisect

        file_idx = bisect.bisect_right(self._cumulative, idx)
        prev_cum = 0 if file_idx == 0 else self._cumulative[file_idx - 1]
        local_idx = idx - prev_cum

        pf = self._files[file_idx]
        tds = self._get_cached_timing_dataset(pf.path)
        return tds[local_idx]

    @property
    def problem_feature_dim(self) -> int:
        # Load first cached file to query dims
        tds = self._get_cached_timing_dataset(self._files[0].path)
        return tds.problem_feature_dim

    @property
    def config_feature_dim(self) -> int:
        tds = self._get_cached_timing_dataset(self._files[0].path)
        return tds.config_feature_dim

    @property
    def configs(self):
        tds = self._get_cached_timing_dataset(self._files[0].path)
        return tds.configs

    def get_file_info(self) -> List[Tuple[str, int]]:
        return [(Path(pf.path).name, pf.count) for pf in self._files]


# -------------------- D) Deterministic/distributed DataLoaders --------------------


def create_directory_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    hardware_name: Optional[str] = None,
    op_name: Optional[str] = None,
    log_transform: bool = True,
    num_workers: int = 4,
    seed: int = 42,
    file_extensions: Optional[List[str]] = None,
    *,
    use_lazy: bool = False,
    # distributed opts (backward-compatible: optional and default off)
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    cache_capacity: int = 8,  # for lazy mode
    max_io_workers: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders from all files in a directory.

    Args mirror the original, with extra, optional:
      - use_lazy: if True, use LazyDirectoryMatmulDataset (E)
      - distributed, rank, world_size: for multi-GPU training (D)
      - cache_capacity: LRU size for lazy mode (E)
      - max_io_workers: thread pool size for file I/O (B)
    """
    if not (0.0 < train_ratio < 1.0) or not (0.0 <= val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must be in (0,1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0.")

    # Deterministic split generator
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Build dataset (eager or lazy)
    if use_lazy:
        full_dataset: Dataset = LazyDirectoryMatmulDataset(
            data_dir=data_dir,
            hardware_name=hardware_name,
            op_name=op_name,
            log_transform=log_transform,
            file_extensions=file_extensions,
            cache_capacity=cache_capacity,
            max_io_workers=max_io_workers,
        )
    else:
        full_dataset = DirectoryMatmulDataset(
            data_dir=data_dir,
            hardware_name=hardware_name,
            op_name=op_name,
            log_transform=log_transform,
            file_extensions=file_extensions,
            max_io_workers=max_io_workers,
        )

    n = len(full_dataset)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    test_size = n - train_size - val_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError(
            f"Split sizes invalid (train={train_size}, val={val_size}, test={test_size}) for dataset of size {n}"
        )

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=gen
    )

    # Distributed samplers
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=seed
        )
        test_sampler = DistributedSampler(
            test_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=seed
        )
        shuffle_train = False
    else:
        train_sampler = val_sampler = test_sampler = None
        shuffle_train = True

    def _worker_init_fn(worker_id: int):
        worker_seed = seed + worker_id
        import random

        import numpy as np

        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    dl_common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=_pin_memory_default(),
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_fn,
    )
    # prefetch_factor only valid when num_workers > 0
    if num_workers > 0:
        dl_common["prefetch_factor"] = 4

    train_dataloader = DataLoader(
        train_ds, shuffle=shuffle_train, sampler=train_sampler, **dl_common
    )
    val_dataloader = DataLoader(val_ds, shuffle=False, sampler=val_sampler, **dl_common)
    test_dataloader = DataLoader(
        test_ds, shuffle=False, sampler=test_sampler, **dl_common
    )

    # Brief stats
    if hasattr(full_dataset, "get_file_info"):
        try:
            files_info = getattr(full_dataset, "get_file_info")()
            # Handle both real file info list and Mock objects
            if hasattr(files_info, "__iter__") and not isinstance(files_info, str):
                try:
                    file_names = [f for (f, _c) in files_info]
                    logger.info(
                        "Created dataloaders with train=%d, val=%d, test=%d samples; files: %s",
                        train_size,
                        val_size,
                        test_size,
                        file_names,
                    )
                except (TypeError, ValueError):
                    logger.info(
                        "Created dataloaders with train=%d, val=%d, test=%d samples",
                        train_size,
                        val_size,
                        test_size,
                    )
            else:
                logger.info(
                    "Created dataloaders with train=%d, val=%d, test=%d samples",
                    train_size,
                    val_size,
                    test_size,
                )
        except Exception:
            logger.info(
                "Created dataloaders with train=%d, val=%d, test=%d samples",
                train_size,
                val_size,
                test_size,
            )

    return train_dataloader, val_dataloader, test_dataloader
