# CozyPDE Competition Brief

CozyPDE targets the PDE neural-operator competition as an autonomous research Agent. The runner is responsible for task understanding, code generation, experiment execution, debugging, validation, log export, and packaging. Final code and modeling decisions must remain traceable in Agent logs.

## Global rules

- Do not call numerical PDE solvers for prediction, pseudo-labels, training labels, or extra data generation.
- Do not generate extra trajectories.
- Do not use external datasets.
- Do not use external pretrained weights, except official Task 1 PDEBench checkpoints when solving Task 1.
- Keep formal task-session wall-clock under 12 hours.
- Keep inference under 120 seconds per submitted task.

## Task summary

### Task 1

- Fixed-viscosity 1D Burgers.
- Input window: first 10 time steps.
- Output shape: `(N, 200, 256)`.
- Output steps `0:10` must match the test input within `1e-3`.
- Official Task 1 checkpoints under `workspace/checkpoints/task1_official/` may be used.

### Task 2

- Multi-viscosity 1D Burgers.
- Input window: first 10 time steps.
- Output shape: `(N, 200, 256)`.
- Output steps `0:10` must match the test input within `1e-3`.
- Train from scratch.
- Do not use Task 1 data, checkpoints, fine-tuned weights, or external pretrained weights.
- Test-time viscosity is unavailable.

### Task 3

- Kuramoto-Sivashinsky multi-parameter prediction.
- Input window: first 20 time steps.
- Output shape: `(1000, 400, 256)`.
- Output steps `0:20` must match the test input within `1e-3`.
- Train from scratch using only official KS data.
- Test-time `lambda2` is unavailable.
- Logs should explain KS dynamics, unknown-parameter handling, and model-selection rationale.

## Formal session policy

- CozyPDE should run one formal autonomous session per task.
- Independent sessions produce independent `task{N}_logs.log` timelines.
- Multi-task formal sessions are supported only by explicit override and should be avoided.
- Human operators may copy and package outputs mechanically, but should not manually edit generated code or predictions.

## Required workspace layout

```text
workspace/
  runs/
    task1/
    task2/
    task3/
  llm_logs/
    task1_all_llm_calls.jsonl
    task2_all_llm_calls.jsonl
    task3_all_llm_calls.jsonl
  internal_logs/
    task1_tool_calls.jsonl
    task2_tool_calls.jsonl
    task3_tool_calls.jsonl
  submission/
    task1_pred.hdf5
    task1_time.csv
    task1_logs.log
    task2_pred.hdf5
    task2_time.csv
    task2_logs.log
    task3_pred.hdf5
    task3_time.csv
    task3_logs.log
    code/
      task1/
      task2/
      task3/
```

## Final packaging

- Formal task sessions should only generate task outputs and task-specific code.
- `package_final` validates existing artifacts and creates `submission.zip` without rerunning modeling.
- Preserve Agent-generated provenance when exporting logs, writing metadata, and packaging.
