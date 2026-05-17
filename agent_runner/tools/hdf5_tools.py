from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..safety import WorkspaceSafety
from . import failure, success


def _dataset_stats(dataset: h5py.Dataset) -> dict[str, Any]:
    array = np.asarray(dataset[...])
    stats: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "contains_nan": bool(np.isnan(array).any()) if np.issubdtype(array.dtype, np.floating) else False,
        "contains_inf": bool(np.isinf(array).any()) if np.issubdtype(array.dtype, np.floating) else False,
    }
    if array.size and np.issubdtype(array.dtype, np.number):
        stats.update(
            {
                "min": float(np.nanmin(array)),
                "max": float(np.nanmax(array)),
                "mean": float(np.nanmean(array)),
                "std": float(np.nanstd(array)),
            }
        )
    return stats


def inspect_hdf5(*, path: str, safety: WorkspaceSafety) -> dict:
    check = safety.validate_read_path(path)
    if not check.ok:
        return failure("inspect_hdf5", check.error or "read check failed", path=path)
    assert check.resolved_path is not None
    if not check.resolved_path.exists():
        return failure("inspect_hdf5", "File does not exist", path=str(check.resolved_path))

    datasets: dict[str, dict[str, Any]] = {}
    try:
        with h5py.File(check.resolved_path, "r") as handle:
            keys: list[str] = []

            def collect(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    keys.append(name)
                    datasets[name] = _dataset_stats(obj)

            handle.visititems(collect)
    except OSError as exc:
        return failure("inspect_hdf5", f"Failed to open HDF5: {exc}", path=str(check.resolved_path))

    return success(
        "inspect_hdf5",
        f"Inspected {len(datasets)} datasets in {check.resolved_path.name}",
        file=str(check.resolved_path),
        keys=keys,
        datasets=datasets,
    )

