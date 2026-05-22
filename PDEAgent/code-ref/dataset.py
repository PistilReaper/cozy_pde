"""Dataset utilities for Burgers neural-operator tasks.

Task 1 has only 100 complete trajectories locally. The recommended chunked
rollout experiment therefore trains on many local windows:

    input  = u[t : t + 10]
    target = u[t + 10 : t + 10 + target_horizon]

where ``target_horizon`` is normally one chunk, but can be several chunks for
multi-step/scheduled-sampling training. Validation keeps full trajectories so
metrics are always computed by rolling out from the official first 10 frames to
the full 190-frame horizon.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split


@dataclass
class Normalizer:
    """Scalar normalisation statistics for velocity fields."""

    mean: float
    std: float

    def encode(self, x: Tensor) -> Tensor:
        """Normalise a tensor using stored scalar statistics."""
        return (x - self.mean) / max(self.std, 1.0e-8)

    def decode(self, x: Tensor) -> Tensor:
        """Map a normalised tensor back to physical units."""
        return x * max(self.std, 1.0e-8) + self.mean

    def as_dict(self) -> Dict[str, float]:
        """Return JSON-serialisable statistics."""
        return {"mean": float(self.mean), "std": float(self.std)}


class BurgersDataset(Dataset):
    """HDF5 Burgers trajectory dataset returning full supervised samples."""

    def __init__(
        self,
        hdf5_path: str,
        task: str = "task1",
        t_in: int = 10,
        t_out: int = 190,
        normalizer: Optional[Normalizer] = None,
        compute_normalizer: bool = True,
        augment_shift: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        self.hdf5_path = hdf5_path
        self.task = task
        self.t_in = int(t_in)
        self.t_out = int(t_out)
        self.augment_shift = bool(augment_shift)
        self.rng = np.random.default_rng(seed)

        with h5py.File(hdf5_path, "r") as f:
            if "tensor" not in f:
                raise KeyError(f"{hdf5_path} does not contain dataset 'tensor'")
            data_np = f["tensor"][()].astype(np.float32)
            self.nu: Optional[Tensor] = torch.tensor(f["nu"][()].astype(np.float32), dtype=torch.float32) if "nu" in f else None
        if data_np.ndim != 3:
            raise ValueError(f"tensor must have shape [N,T,Nx], got {data_np.shape}")
        if data_np.shape[1] < self.t_in:
            raise ValueError(f"dataset has only {data_np.shape[1]} time steps, need {self.t_in}")

        self.data = torch.from_numpy(data_np)
        if normalizer is None and compute_normalizer:
            normalizer = Normalizer(float(self.data.mean().item()), float(self.data.std().item() + 1.0e-8))
        self.normalizer = normalizer or Normalizer(0.0, 1.0)
        self.data = self.normalizer.encode(self.data)

        if self.nu is not None:
            log_nu = torch.log(torch.clamp(self.nu, min=1.0e-12))
            self.nu_mean = float(log_nu.mean().item())
            self.nu_std = float(log_nu.std().item() + 1.0e-8)
        else:
            self.nu_mean = 0.0
            self.nu_std = 1.0

    @property
    def mean(self) -> float:
        return self.normalizer.mean

    @property
    def std(self) -> float:
        return self.normalizer.std

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def _maybe_shift(self, u: Tensor) -> Tensor:
        if self.augment_shift:
            return torch.roll(u, shifts=int(self.rng.integers(0, u.shape[-1])), dims=-1)
        return u

    def _nu_embedding(self, idx: int) -> Optional[Tensor]:
        if self.nu is None:
            return None
        nu_emb = (torch.log(torch.clamp(self.nu[idx], min=1.0e-12)) - self.nu_mean) / self.nu_std
        return nu_emb.view(1)

    def __getitem__(self, idx: int) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        u = self._maybe_shift(self.data[idx])
        x = u[: self.t_in]
        y = u[self.t_in : self.t_in + self.t_out]
        if y.shape[0] < self.t_out:
            y = torch.empty(0, u.shape[-1], dtype=u.dtype)
        nu = self._nu_embedding(idx)
        return (x, y, nu) if nu is not None else (x, y)

    def denormalize(self, u: Tensor) -> Tensor:
        return self.normalizer.decode(u)


class WindowedBurgersDataset(Dataset):
    """Sliding-window local dynamics dataset for chunked rollout training."""

    def __init__(
        self,
        base: BurgersDataset,
        indices: Sequence[int],
        t_in: int = 10,
        chunk_size: int = 10,
        stride: int = 1,
        target_horizon: Optional[int] = None,
    ) -> None:
        super().__init__()
        if chunk_size <= 0 or stride <= 0:
            raise ValueError("chunk_size and stride must be positive")
        self.base = base
        self.indices = list(indices)
        self.t_in = int(t_in)
        self.chunk_size = int(chunk_size)
        self.target_horizon = int(target_horizon or chunk_size)
        self.stride = int(stride)
        if self.target_horizon <= 0:
            raise ValueError("target_horizon must be positive")
        total_t = int(base.data.shape[1])
        max_start = total_t - self.t_in - self.target_horizon
        if max_start < 0:
            raise ValueError(
                f"trajectory length {total_t} too short for t_in={t_in}, target_horizon={self.target_horizon}"
            )
        self.windows: List[Tuple[int, int]] = [
            (idx, s) for idx in self.indices for s in range(0, max_start + 1, self.stride)
        ]

    @property
    def normalizer(self) -> Normalizer:
        return self.base.normalizer

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        idx, start = self.windows[i]
        u = self.base._maybe_shift(self.base.data[idx])
        x = u[start : start + self.t_in]
        y = u[start + self.t_in : start + self.t_in + self.target_horizon]
        nu = self.base._nu_embedding(idx)
        return (x, y, nu) if nu is not None else (x, y)


def _get_base_dataset(ds: Dataset) -> BurgersDataset:
    if isinstance(ds, BurgersDataset):
        return ds
    if isinstance(ds, WindowedBurgersDataset):
        return ds.base
    if isinstance(ds, Subset):
        return _get_base_dataset(ds.dataset)  # type: ignore[arg-type]
    if isinstance(ds, ConcatDataset):
        return _get_base_dataset(ds.datasets[0])  # type: ignore[index]
    raise TypeError(f"Cannot extract BurgersDataset from {type(ds)}")


def get_dataset_stats(ds: Dataset) -> Normalizer:
    """Return normalisation statistics from a dataset/subset/concat dataset."""
    return _get_base_dataset(ds).normalizer


def _loader_kwargs(batch_size: int, num_workers: int) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def get_dataloaders(
    data_dir: str,
    task: str = "task1",
    batch_size: int = 16,
    num_workers: int = 0,
    val_fraction: float = 0.2,
    seed: int = 42,
    augment_shift: bool = False,
    model_type: str = "direct",
    chunk_size: int = 10,
    window_stride: int = 1,
    t_in: int = 10,
    t_out: int = 190,
    train_target_horizon: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    For Task 1/chunked training, the train loader returns short-window targets
    while validation returns full 190-step targets for honest rollout metrics.
    """
    if task == "task1":
        path = os.path.join(data_dir, "task1_val.hdf5")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Task1 data not found: {path}")
        raw = BurgersDataset(path, task=task, t_in=t_in, t_out=t_out, compute_normalizer=True, augment_shift=False, seed=seed)
        n_val = max(1, int(round(len(raw) * val_fraction)))
        n_train = len(raw) - n_val
        train_split, val_split = random_split(
            range(len(raw)), [n_train, n_val], generator=torch.Generator().manual_seed(seed)
        )
        train_indices, val_indices = list(train_split), list(val_split)
        train_tensor = raw.data[train_indices]
        normalizer = Normalizer(float(train_tensor.mean().item()), float(train_tensor.std().item() + 1.0e-8))
        train_base = BurgersDataset(
            path,
            task=task,
            t_in=t_in,
            t_out=t_out,
            normalizer=normalizer,
            compute_normalizer=False,
            augment_shift=augment_shift,
            seed=seed,
        )
        val_base = BurgersDataset(
            path,
            task=task,
            t_in=t_in,
            t_out=t_out,
            normalizer=normalizer,
            compute_normalizer=False,
            augment_shift=False,
            seed=seed,
        )
        if model_type.lower() in {"chunk", "chunked", "rollout"}:
            train_set: Dataset = WindowedBurgersDataset(
                train_base,
                train_indices,
                t_in=t_in,
                chunk_size=chunk_size,
                stride=window_stride,
                target_horizon=train_target_horizon or chunk_size,
            )
        else:
            train_set = Subset(train_base, train_indices)
        val_set: Dataset = Subset(val_base, val_indices)
    elif task == "task2":
        files = [
            os.path.join(data_dir, name)
            for name in ("task2_part0_train.h5", "task2_part1_train.h5", "task2_part2_train.h5")
        ]
        files = [p for p in files if os.path.exists(p)]
        if not files:
            raise FileNotFoundError(f"No Task2 training files found in {data_dir}")
        tmp = [BurgersDataset(p, task=task, t_in=t_in, t_out=t_out, compute_normalizer=True) for p in files]
        all_data = torch.cat([d.normalizer.decode(d.data) for d in tmp], dim=0)
        norm = Normalizer(float(all_data.mean().item()), float(all_data.std().item() + 1.0e-8))
        train_set = ConcatDataset(
            [
                BurgersDataset(
                    p,
                    task=task,
                    t_in=t_in,
                    t_out=t_out,
                    normalizer=norm,
                    compute_normalizer=False,
                    augment_shift=augment_shift,
                )
                for p in files
            ]
        )
        val_path = os.path.join(data_dir, "task2_val.h5")
        val_set = BurgersDataset(val_path, task=task, t_in=t_in, t_out=t_out, normalizer=norm, compute_normalizer=False)
    else:
        raise ValueError(f"unknown task: {task}")
    kwargs = _loader_kwargs(batch_size, num_workers)
    return (
        DataLoader(train_set, shuffle=True, drop_last=False, **kwargs),
        DataLoader(val_set, shuffle=False, drop_last=False, **kwargs),
    )


def get_test_loader(
    data_dir: str,
    task: str = "task1",
    batch_size: int = 16,
    normalizer: Optional[Normalizer] = None,
    num_workers: int = 0,
    t_in: int = 10,
) -> Tuple[DataLoader, BurgersDataset]:
    """Create a test dataloader and underlying dataset."""
    filename = "task1_test.hdf5" if task == "task1" else "task2_test.h5"
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test file not found: {path}")
    base = BurgersDataset(
        path,
        task=task,
        t_in=t_in,
        t_out=190,
        normalizer=normalizer,
        compute_normalizer=normalizer is None,
        augment_shift=False,
    )
    return DataLoader(base, shuffle=False, drop_last=False, **_loader_kwargs(batch_size, num_workers)), base
