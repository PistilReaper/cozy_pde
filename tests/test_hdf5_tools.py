from __future__ import annotations

import h5py
import numpy as np

from agent_runner.safety import WorkspaceSafety
from agent_runner.tools.hdf5_tools import inspect_hdf5


def test_inspect_hdf5_reports_keys_shape_dtype_and_stats(workspace):
    path = workspace / "data" / "sample.hdf5"
    data = np.arange(2 * 12 * 8, dtype=np.float32).reshape(2, 12, 8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=data)

    result = inspect_hdf5(path="data/sample.hdf5", safety=WorkspaceSafety(workspace))

    assert result["ok"] is True
    assert result["data"]["keys"] == ["tensor"]
    dataset_info = result["data"]["datasets"]["tensor"]
    assert dataset_info["shape"] == [2, 12, 8]
    assert dataset_info["dtype"] == "float32"
    assert dataset_info["contains_nan"] is False
    assert dataset_info["contains_inf"] is False
    assert dataset_info["max"] == float(data.max())
