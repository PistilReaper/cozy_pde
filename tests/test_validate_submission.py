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
