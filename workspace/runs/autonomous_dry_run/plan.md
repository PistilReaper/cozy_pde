# Autonomous Dry-Run Plan

Mode: `autonomous_dry_run`
Target tasks: `task1`, `task2`

## Dry-run observations

- Attempted to read `docs/TEST_PLAN.md` and `docs/pde_agent_implementation_plan.md`; these files were not present at the expected workspace-relative paths in this dry-run environment, so I relied on the user-provided docs summary.
- `workspace/data` contains 0 entries.
- `workspace/checkpoints` contains 0 entries.
- `workspace/baselines` contains 0 entries.
- No HDF5 files were available to inspect.
- No training, inference, prediction generation, submission code writing, external download, or numerical solver invocation was performed.

## Formal autonomous run constraints

- `workspace/submission/code` must be generated only by agent `write_file` calls.
- No extra training data may be used.
- No numerical solver may be called to generate data.
- Task 1 may use official PDEBench checkpoints if present.
- Task 2 must be trained from scratch and must not use Task 1 checkpoints or Task 1 data.
- Task 1 and Task 2 predictions must have shape `(N, 200, 256)`.
- The first 10 time steps of each prediction must match the test input.
- Inference must finish within 2 minutes.
- Every experiment must record hypothesis, code modification, result, and conclusion.
- Prioritize compliance, stability, verifiability, and fast inference.
- First version should complete Task 1 and Task 2 only; do not implement Task 3.

## Formal run next-step plan

### 1. Discover files

- Re-list `workspace/data`, `workspace/checkpoints`, and `workspace/baselines`.
- If HDF5 files exist, inspect them before writing model code:
  - keys
  - dataset shapes
  - dtypes
  - min/max/mean/std
  - NaN/Inf status
  - train/test split availability
- If baselines exist, read their key files and infer official data conventions.
- If checkpoints exist, identify which are official and whether they are Task 1-compatible.

### 2. Generate minimal submission code

Use `write_file` to create only the necessary files under `workspace/submission/code`, likely:

- `model.py`: small stable neural operator or convolutional temporal model.
- `data_utils.py`: HDF5 loading, normalization, shape handling, first-10-step preservation.
- `train.py`: task-aware training entrypoint.
- `infer.py`: task-aware inference entrypoint generating `(N, 200, 256)` predictions.
- `validate_local.py`: lightweight syntax, smoke, and HDF5 prediction checks.
- `config_task1.yaml` and `config_task2.yaml`: separate configs to avoid leakage.

### 3. Minimal validation before training

After each code generation/modification:

1. Run syntax/import validation.
2. Run tiny HDF5 loading test if data exists.
3. Run one-batch smoke train only in the formal run.
4. Run tiny inference into a temporary run path only in the formal run.
5. Verify prediction shape and first-10-step identity before full inference.

### 4. Task 1 formal strategy

- Prefer official PDEBench checkpoint fine-tuning if available.
- If no checkpoint is available, train a small conservative model on Task 1 official training data only.
- Keep inference vectorized and batch-based for under-2-minute runtime.
- Always force `pred[:, :10, :] = test_input[:, :10, :]` immediately before writing predictions.

### 5. Task 2 formal strategy

- Train from scratch only.
- Do not load Task 1 checkpoints, Task 1 normalizers, or Task 1 data.
- Use separate config and output directories.
- Use the same validation discipline as Task 1.
- Prefer a small model over an unstable or slow architecture.

### 6. Experiment logging discipline

For each formal experiment, write a structured record containing:

- hypothesis
- code/config change
- command or action
- result metrics or failure
- conclusion and next action

Ensure final task logs are valid JSONL with ISO timestamps, `elapsed_seconds`, and either `response` or `tool_calls` per line.

### 7. Finalization checklist

- Generate `task1_pred.hdf5`, `task1_time.csv`, `task1_logs.log`.
- Generate `task2_pred.hdf5`, `task2_time.csv`, `task2_logs.log`.
- Validate both predictions:
  - file exists
  - shape `(N, 200, 256)`
  - finite values
  - first 10 steps match test input
  - reasonable dtype
- Generate `submission.json` and methodology files if required.
- Run submission validator.
- Package only after validation passes.

## Dry-run conclusion

The dry-run completed discovery within the allowed action set. Since no data, checkpoints, or baselines were visible, the formal autonomous run should start with file discovery and HDF5 inspection, then generate a minimal compliant training/inference pipeline.
