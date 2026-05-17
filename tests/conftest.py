from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    for relative in [
        "data",
        "checkpoints",
        "baselines",
        "runs/scratch",
        "internal_logs",
        "llm_logs",
        "submission/code",
    ]:
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def fake_test_hdf5(workspace: Path) -> Path:
    path = workspace / "data" / "test.hdf5"
    array = np.linspace(0.0, 1.0, num=2 * 200 * 256, dtype=np.float32).reshape(2, 200, 256)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=array)
    return path


@pytest.fixture()
def valid_submission_bundle(workspace: Path, fake_test_hdf5: Path) -> Path:
    submission_dir = workspace / "submission"
    with h5py.File(fake_test_hdf5, "r") as source:
        test_tensor = source["tensor"][:]

    pred = test_tensor.copy()
    pred[:, 10:, :] = pred[:, 10:, :] + 0.1
    with h5py.File(submission_dir / "pred.hdf5", "w") as handle:
        handle.create_dataset("pred", data=pred)

    (submission_dir / "time.csv").write_text(
        "train_time,inference_time\n12.5,0.8\n",
        encoding="utf-8",
    )
    (submission_dir / "logs.log").write_text(
        json.dumps(
            {
                "timestamp": "2026-05-16T12:00:00+00:00",
                "elapsed_seconds": 1.23,
                "response": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (submission_dir / "code" / "generated.py").write_text("print('generated')\n", encoding="utf-8")
    return submission_dir
