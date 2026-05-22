from __future__ import annotations

import json

from cozy_pde_v3.log_export import export_task_logs


def _write_jsonl_log(path, *, response: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-22T00:00:00+00:00",
                "elapsed_seconds": 0.1,
                "response": response,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_export_task_logs_supports_task3_with_structured_result(workspace) -> None:
    source_log = workspace / "llm_logs" / "task3_all_llm_calls.jsonl"
    _write_jsonl_log(source_log, response="task3 complete")

    result = export_task_logs(workspace=workspace, tasks=["task3"])

    assert result["ok"] is True
    assert result["data"]["tasks"] == ["task3"]
    assert result["data"]["source_log"] == str(source_log)
    assert result["data"]["task_logs"] == {
        "task3": {
            "source_log": str(source_log),
            "destination": str(workspace / "submission" / "task3_logs.log"),
            "record_count": 1,
        }
    }


def test_export_task_logs_rejects_invalid_jsonl_source(workspace) -> None:
    source_log = workspace / "llm_logs" / "task1_all_llm_calls.jsonl"
    source_log.write_text("not-json\n", encoding="utf-8")

    result = export_task_logs(workspace=workspace, tasks=["task1"])

    assert result["ok"] is False
    assert "valid JSON" in result["error"]
    assert result["data"]["source_log"] == str(source_log)


def test_export_task_logs_does_not_merge_unrelated_shared_session_implicitly(workspace) -> None:
    source_log = workspace / "llm_logs" / "all_llm_calls.jsonl"
    _write_jsonl_log(source_log, response="shared session")

    result = export_task_logs(workspace=workspace, tasks=["task1", "task2"])

    assert result["ok"] is False
    assert "independent task sessions" in result["error"]
    assert not (workspace / "submission" / "task1_logs.log").exists()
    assert not (workspace / "submission" / "task2_logs.log").exists()


def test_export_task_logs_exports_per_task_from_explicit_shared_source(workspace) -> None:
    source_log = workspace / "runs" / "shared" / "llm_logs" / "task1_task2_all_llm_calls.jsonl"
    _write_jsonl_log(source_log, response="shared task session")

    result = export_task_logs(workspace=workspace, tasks=["task1", "task2"], source_log=source_log)

    assert result["ok"] is True
    assert result["data"]["tasks"] == ["task1", "task2"]
    assert result["data"]["task_logs"]["task1"]["record_count"] == 1
    assert result["data"]["task_logs"]["task2"]["record_count"] == 1
    assert (workspace / "submission" / "task1_logs.log").read_text(encoding="utf-8") == source_log.read_text(
        encoding="utf-8"
    )
    assert (workspace / "submission" / "task2_logs.log").read_text(encoding="utf-8") == source_log.read_text(
        encoding="utf-8"
    )
