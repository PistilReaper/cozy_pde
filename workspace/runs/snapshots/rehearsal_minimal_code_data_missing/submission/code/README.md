# Rehearsal Code Notes

This code was generated in autonomous_rehearsal mode with no available HDF5 data in `workspace/data`, no checkpoints, and no baselines.

Generated components:
- `data_utils.py`: HDF5 dataset discovery/loading and prediction writing helpers.
- `model.py`: tiny temporal convolution model plus rollout helpers.
- `train.py`: bounded smoke trainer. It exits with a data-missing status if the requested HDF5 file is absent.
- `infer.py`: bounded inference script. It writes `(N, 200, 256)` predictions when a real test HDF5 is available and preserves the first 10 steps.
- `validate_submission.py`: small helper API for local prediction generation.

Compliance notes:
- No external data is downloaded.
- No numerical solver is called.
- Task 2 configuration explicitly forbids reuse of Task 1 artifacts.
- With no official HDF5 in this rehearsal, smoke training/inference is intentionally skipped rather than using fabricated data.
