# Phase 1 Acceptance Hardening Report

Date: 2026-05-17

## Command note

The current CLI requires `--mode`, so the requested commands were run with the equivalent form:

- `.venv/bin/python -m agent_runner.main --config agent_runner/config.example.yaml --mode preflight`
- `.venv/bin/python -m agent_runner.main --config agent_runner/config.example.yaml --mode live_api_check`
- `.venv/bin/python -m agent_runner.main --config agent_runner/config.example.yaml --mode research_api_check`
- `.venv/bin/python -m agent_runner.main --config agent_runner/config.example.yaml --mode autonomous_dry_run --max-steps 6`

## Results

- `pytest -q`: PASS, `35 passed in 0.19s`
- `preflight`: PASS
  - workspace directories writable
  - Python `3.10.12`
  - `openai` SDK importable
  - router/profile config uses `responses`
  - `LLM_API_KEY` detected
  - `validate_jsonl_logs` runnable
  - warning only: `torch_cuda: CUDA unavailable`
- `live_api_check`: FAIL due upstream provider instability
  - first live Responses call succeeded with a normal `response` object
  - failure happened during the tool-calling round with upstream `502 Bad Gateway`
  - Cloudflare response timestamp: `2026-05-17T09:34:46Z`
- `research_api_check`: FAIL due the same upstream `502 Bad Gateway`
  - Cloudflare response timestamp: `2026-05-17T09:38:19Z`
- `autonomous_dry_run --max-steps 6`: FAIL due the same upstream `502 Bad Gateway`
  - router call succeeded first
  - failure happened on the next planner Responses call
  - Cloudflare response timestamp: `2026-05-17T09:41:19Z`

## Acceptance summary

- Whether all live calls used Responses: Yes for all recorded live calls before the upstream failures.
  - Evidence: `agent_runner/responses_client.py` calls `client.responses.create(...)`
  - Evidence: recorded live log entries use `raw_response.object == "response"`
- Whether function calling worked: Not confirmed live in this run.
  - `live_api_check` reached the live simple-response phase, but the run failed with upstream `502` before `echo_tool` / `write_file` tool execution completed
  - Local non-live coverage remains green, including `tests/test_responses_loop_tool_call.py`
- Whether `web_search` worked for arXiv/GitHub: Not confirmed live in this run.
  - `research_api_check` failed with upstream `502` before a hosted research-tool result was returned
- Whether skill catalog loaded: Yes
  - Installed default bundles exist under `skills/`
  - `test_installed_default_skills_exist` passes
  - `_load_skill_catalog(config)` returned a non-empty catalog for the four enabled skills
- Whether logs passed `validate_jsonl_logs` and `validate_responses_logs`: Mixed
  - `workspace/runs/scratch/preflight.jsonl` passed `validate_jsonl_logs`
  - the current `workspace/llm_logs/all_llm_calls.jsonl` passed `validate_jsonl_logs`
  - the current `workspace/llm_logs/all_llm_calls.jsonl` failed `validate_responses_logs` under the new hardening rule because existing files under `workspace/submission/code/` are not traceable from that log
  - legacy `workspace/submission/task1_logs.log` and `workspace/submission/task2_logs.log` passed `validate_jsonl_logs` but failed `validate_responses_logs` because they are not Responses-format logs and are missing required fields such as `profile`

## Code hardening delivered

- Added minimal installed skill bundles:
  - `skills/pdebench/SKILL.md`
  - `skills/neural_operator_research/SKILL.md`
  - `skills/hdf5_validation/SKILL.md`
  - `skills/rollout_training/SKILL.md`
- Strengthened `validate_responses_logs` to:
  - aggregate `write_file` calls across the full JSONL log
  - require every `workspace/submission/code/**` file to be traceable
  - require exact `write_file` content equality for traced files
  - fail on untraced or content-mismatched files
- Added/verified tests:
  - `test_validate_responses_logs_fails_for_untraced_code_file`
  - `test_validate_responses_logs_fails_for_write_file_without_content`
  - `test_validate_responses_logs_accepts_traced_code_file`
  - `test_installed_default_skills_exist`
  - additional coverage: `test_validate_responses_logs_fails_for_content_mismatch`
