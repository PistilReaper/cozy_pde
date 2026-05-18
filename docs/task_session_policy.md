# Task Session Policy

CozyPDE should treat each formal competition task as its own Agent session by default.

## Default policy

- Run `--mode autonomous` with exactly one task.
- Export logs for that same task immediately after the session.
- Keep each task's LLM timeline independent unless a multi-task session is explicitly allowed.

## Why this matters

- Independent sessions produce independent `task{N}_logs.log` timelines.
- Shared sessions require the full shared log to be exported to every task touched in that session.
- Separate sessions reduce the risk of submission-log mismatches and cross-task provenance confusion.

## Allowed mechanical operator steps

- Start the Agent separately for Task 1, Task 2, and Task 3.
- Run `export_task_logs` after each formal session.
- Run `package_final` after all requested task outputs exist.
- Copy, archive, and zip artifacts mechanically.

## Forbidden operator behavior

- Do not manually edit generated code under `workspace/submission/code/`.
- Do not manually rewrite prediction arrays.
- Do not alter Agent logs in a way that breaks provenance.

## Expected formal workflow

```bash
python -m agent_runner.main --mode autonomous --config config.yaml --tasks task1
python -m agent_runner.main --mode export_task_logs --config config.yaml --tasks task1

python -m agent_runner.main --mode autonomous --config config.yaml --tasks task2
python -m agent_runner.main --mode export_task_logs --config config.yaml --tasks task2

python -m agent_runner.main --mode autonomous --config config.yaml --tasks task3
python -m agent_runner.main --mode export_task_logs --config config.yaml --tasks task3

python -m agent_runner.main --mode package_final --config config.yaml --tasks task1,task2,task3
```
