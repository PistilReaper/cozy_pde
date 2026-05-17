# Autonomous Rehearsal Report

## Mode
- Current mode: `autonomous_rehearsal`
- Target tasks: Task 1 and Task 2
- Per-task smoke training budget: 600 seconds

## Required observations
- `data/`: inspected and contains 0 entries.
- `checkpoints/`: inspected and contains 0 entries.
- `baselines/`: inspected and contains 0 entries.
- `docs/`: no `docs` directory is present in this workspace; the only available documentation was the user-provided summary.
- Existing old `submission/task1_logs.log` and `submission/task2_logs.log` were deliberately not read, per rehearsal rule.

## Decision
Because no official HDF5 file is available in `workspace/data`, I followed the required data-missing path:
- did not fabricate HDF5 data;
- did not run smoke training;
- did not run smoke inference;
- did not call any numerical solver;
- did not download external data.

## Generated code in `submission/code`
All code files were generated through `write_file`:
- `data_utils.py`: HDF5 discovery/loading, persistence prediction, prediction HDF5 writing, time CSV writing.
- `model.py`: tiny temporal convolution model and rollout helpers.
- `train.py`: bounded smoke trainer with data-missing exit path and finite-loss checks.
- `infer.py`: bounded inference script producing `(N, 200, 256)` predictions from real test HDF5 and preserving the first 10 steps.
- `validate_submission.py`: lightweight helper API.
- `config_task1.yaml`: rehearsal config; notes that Task 1 may use official checkpoint in formal run if available.
- `config_task2.yaml`: rehearsal config; explicitly trains from scratch and forbids Task 1 artifact reuse.
- `requirements.txt`: minimal dependencies.
- `README.md`: compliance notes.

## Minimal validation performed
- Python syntax check: `python -m py_compile submission/code/*.py` succeeded.
- Import and shape sanity check succeeded:
  - persistence output shape `(2, 200, 256)`;
  - first 10 steps exactly match input;
  - tiny model output shape `(2, 256)`.

## Snapshot
Created stable snapshot:
- `runs/snapshots/rehearsal_minimal_code_data_missing`

## Experiment log
### Experiment 1: infrastructure-only code skeleton
- Hypothesis: With no data, the correct rehearsal goal is to verify generated code syntax and core shape invariants while explicitly skipping training/inference.
- Code modification: Generated minimal Task 1/2 training and inference infrastructure in `submission/code`.
- Result: Syntax, imports, and shape checks passed.
- Conclusion: Rehearsal infrastructure is ready for a future run with official HDF5 files; training and submission validation were correctly not attempted due to missing data.
