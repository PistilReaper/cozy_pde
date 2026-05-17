from __future__ import annotations

import json

from agent_runner.config import RunnerConfig
from agent_runner.main import _scan_for_secret_leaks, run_final_check


def test_final_check_warns_on_missing_artifacts_without_strict(workspace, capsys):
    (workspace / "submission" / "code" / "placeholder.py").write_text("print('ok')\n", encoding="utf-8")
    (workspace / "llm_logs" / "all_llm_calls.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-05-16T12:00:00+00:00",
                "elapsed_seconds": 0.1,
                "response": "hello",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = RunnerConfig.from_workspace(workspace)
    exit_code = run_final_check(config, strict=False)
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "WARN" in captured


def test_final_check_fails_on_missing_artifacts_with_strict(workspace):
    (workspace / "submission" / "code" / "placeholder.py").write_text("print('ok')\n", encoding="utf-8")
    (workspace / "llm_logs" / "all_llm_calls.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-05-16T12:00:00+00:00",
                "elapsed_seconds": 0.1,
                "response": "hello",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = RunnerConfig.from_workspace(workspace)
    exit_code = run_final_check(config, strict=True)

    assert exit_code == 1


def test_secret_leak_scan_ignores_task_specific_plain_text(workspace):
    log_path = workspace / "submission" / "task1_logs.log"
    log_path.write_text("task-specific planning note\n", encoding="utf-8")

    hits = _scan_for_secret_leaks([workspace / "submission"])

    assert hits == []
