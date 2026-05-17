from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from data_utils import build_persistence_prediction, load_array_from_hdf5, write_prediction_hdf5, write_time_csv


DEFAULTS: Dict[str, Any] = {
    "total_steps": 200,
    "context_steps": 10,
    "output_key": "pred",
}


def predict_from_hdf5(test_hdf5: str, output_path: str, dataset_key: str | None = None) -> str:
    array, _ = load_array_from_hdf5(test_hdf5, dataset_key)
    pred = build_persistence_prediction(array, total_steps=DEFAULTS["total_steps"], context_steps=DEFAULTS["context_steps"])
    write_prediction_hdf5(output_path, pred, dataset_key=DEFAULTS["output_key"])
    return output_path


def write_zero_time_csv(path: str) -> str:
    write_time_csv(path, train_time=0.0, inference_time=0.0)
    return path
