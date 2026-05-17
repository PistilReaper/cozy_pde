from __future__ import annotations

import json

from agent_runner.main import export_task_logs


def test_export_task_logs_copies_full_llm_session_and_validates(workspace):
    llm_log = workspace / "llm_logs" / "all_llm_calls.jsonl"
    llm_log.write_text(
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

    result = export_task_logs(workspace=workspace, tasks=["task1", "task2"])

    assert result["ok"] is True
    assert (workspace / "submission" / "task1_logs.log").exists()
    assert (workspace / "submission" / "task2_logs.log").exists()
