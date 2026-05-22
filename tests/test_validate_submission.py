from __future__ import annotations

import h5py
import numpy as np

from agent_runner.tools.validate_tools import validate_submission


def test_validate_submission_accepts_valid_fake_bundle(valid_submission_bundle, fake_test_hdf5):
    result = validate_submission(
        submission_dir=valid_submission_bundle,
        test_hdf5=fake_test_hdf5,
    )

    assert result["ok"] is True
    assert result["data"]["pred_shape"] == [2, 200, 256]


def test_validate_submission_rejects_wrong_prediction_shape(valid_submission_bundle, fake_test_hdf5):
    pred_path = valid_submission_bundle / "pred.hdf5"
    with h5py.File(pred_path, "w") as handle:
        handle.create_dataset("pred", data=np.zeros((2, 199, 256), dtype=np.float32))

    result = validate_submission(
        submission_dir=valid_submission_bundle,
        test_hdf5=fake_test_hdf5,
    )

    assert result["ok"] is False
    assert "shape" in result["error"].lower()


def test_validate_submission_prefers_tensor_dataset_over_coordinate_metadata(valid_submission_bundle, fake_test_hdf5):
    structured_test_hdf5 = valid_submission_bundle.parent / "structured_test.hdf5"
    with h5py.File(fake_test_hdf5, "r") as source:
        tensor = source["tensor"][:]

    with h5py.File(structured_test_hdf5, "w") as handle:
        handle.create_dataset("t-coordinate", data=np.linspace(0.0, 1.0, num=200, dtype=np.float32))
        handle.create_dataset("tensor", data=tensor)
        handle.create_dataset("x-coordinate", data=np.linspace(-1.0, 1.0, num=256, dtype=np.float32))

    result = validate_submission(
        submission_dir=valid_submission_bundle,
        test_hdf5=structured_test_hdf5,
    )

    assert result["ok"] is True
    assert result["data"]["pred_shape"] == [2, 200, 256]
