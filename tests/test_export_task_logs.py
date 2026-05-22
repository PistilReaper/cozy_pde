from __future__ import annotations

import json

from cozy_pde_v3.log_export import export_task_logs
from cozy_pde_v3.validation.logs import validate_task_log_jsonl


def test_export_task_logs_copies_single_task_session_and_validates(workspace):
    llm_log = workspace / "llm_logs" / "task1_all_llm_calls.jsonl"
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

    result = export_task_logs(workspace=workspace, tasks=["task1"])

    assert result["ok"] is True
    assert result["data"]["source_log"] == str(llm_log)
    assert result["data"]["exported"] == [str(workspace / "submission" / "task1_logs.log")]
    assert (workspace / "submission" / "task1_logs.log").read_text(encoding="utf-8") == llm_log.read_text(
        encoding="utf-8"
    )


def test_export_task_logs_rejects_multi_task_export_without_explicit_override(workspace):
    shared_log = workspace / "llm_logs" / "task1_task2_all_llm_calls.jsonl"
    shared_log.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-16T12:00:00+00:00",
                "elapsed_seconds": 0.1,
                "response": "shared session",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = export_task_logs(workspace=workspace, tasks=["task1", "task2"])

    assert result["ok"] is False
    assert "independent task sessions" in result["error"]


def test_export_task_logs_allows_shared_multi_task_source_when_explicitly_enabled(workspace):
    shared_log = workspace / "llm_logs" / "task1_task2_all_llm_calls.jsonl"
    shared_log.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-16T12:00:00+00:00",
                "elapsed_seconds": 0.1,
                "tool_calls": [{"name": "write_file"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = export_task_logs(workspace=workspace, tasks=["task1", "task2"], allow_multi_task_session=True)

    assert result["ok"] is True
    assert result["data"]["source_log"] == str(shared_log)
    assert (workspace / "submission" / "task1_logs.log").read_text(encoding="utf-8") == shared_log.read_text(
        encoding="utf-8"
    )
    assert (workspace / "submission" / "task2_logs.log").read_text(encoding="utf-8") == shared_log.read_text(
        encoding="utf-8"
    )


def test_validate_task_log_jsonl_requires_timestamp_elapsed_and_response_or_tool_calls(workspace):
    log_path = workspace / "submission" / "task1_logs.log"
    log_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-16T12:00:00+00:00",
                "elapsed_seconds": 0.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_task_log_jsonl(log_path)

    assert result["ok"] is False
    assert "response or tool_calls" in result["error"]
