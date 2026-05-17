from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

CANDIDATE_KEYS = (
    "input",
    "inputs",
    "u",
    "tensor",
    "data",
    "x",
    "test",
    "solution",
    "pred",
)


class DataNotAvailableError(FileNotFoundError):
    """Raised when an expected HDF5 file is not available."""


def require_file(path: str | Path) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise DataNotAvailableError(f"Required file does not exist: {file_path}")
    if not file_path.is_file():
        raise DataNotAvailableError(f"Expected a file but found something else: {file_path}")
    return file_path


def list_hdf5_datasets(path: str | Path) -> List[Dict[str, Any]]:
    file_path = require_file(path)
    datasets: List[Dict[str, Any]] = []
    with h5py.File(file_path, "r") as handle:
        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets.append(
                    {
                        "name": name,
                        "shape": tuple(int(v) for v in obj.shape),
                        "dtype": str(obj.dtype),
                    }
                )

        handle.visititems(visitor)
    return datasets


def infer_dataset_key(path: str | Path) -> str:
    datasets = list_hdf5_datasets(path)
    if not datasets:
        raise ValueError(f"No datasets found in HDF5 file: {path}")

    for candidate in CANDIDATE_KEYS:
        for item in datasets:
            if item["name"].split("/")[-1] == candidate:
                return item["name"]

    for item in datasets:
        shape = item["shape"]
        if len(shape) == 3 and shape[-1] == 256:
            return item["name"]

    return datasets[0]["name"]


def load_array_from_hdf5(
    path: str | Path,
    dataset_key: Optional[str] = None,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, str]:
    file_path = require_file(path)
    use_key = dataset_key or infer_dataset_key(file_path)
    with h5py.File(file_path, "r") as handle:
        if use_key not in handle:
            raise KeyError(f"Dataset key '{use_key}' not found in {file_path}")
        array = np.asarray(handle[use_key], dtype=dtype)
    return array, use_key


def summarize_array(array: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(array)
    return {
        "shape": tuple(int(v) for v in arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
    }


def build_persistence_prediction(
    test_input: np.ndarray,
    total_steps: int = 200,
    context_steps: int = 10,
) -> np.ndarray:
    x = np.asarray(test_input, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array (N, T, C), got shape {x.shape}")
    if x.shape[1] < context_steps:
        raise ValueError(
            f"Input time dimension {x.shape[1]} is smaller than context_steps={context_steps}"
        )
    if x.shape[2] != 256:
        raise ValueError(f"Expected feature dimension 256, got {x.shape[2]}")

    pred = np.empty((x.shape[0], total_steps, 256), dtype=np.float32)
    pred[:, :context_steps, :] = x[:, :context_steps, :]
    anchor = x[:, context_steps - 1 : context_steps, :]
    pred[:, context_steps:, :] = anchor
    return pred


def write_prediction_hdf5(
    path: str | Path,
    prediction: np.ndarray,
    dataset_key: str = "pred",
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as handle:
        handle.create_dataset(dataset_key, data=np.asarray(prediction, dtype=np.float32))
    return out_path


def write_time_csv(path: str | Path, train_time: float, inference_time: float) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "train_time,inference_time\n"
        f"{float(train_time):.6f},{float(inference_time):.6f}\n",
        encoding="utf-8",
    )
    return out_path


def save_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path
