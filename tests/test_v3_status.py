from __future__ import annotations

import json

from cozy_pde_v3.status import collect_submission_status_v3


def test_collect_submission_status_v3_reads_latest_state_and_finalize_gate(workspace) -> None:
    (workspace / "agent_state.json").write_text(
        json.dumps(
            {
                "current_phase": "validation",
                "latest_error_summary": "task1_time.csv missing",
                "best_artifact_path": "submission/task1_pred.hdf5",
                "shared_code_version": "sha256:v2",
                "supported_tasks": ["task1", "task2"],
            }
        ),
        encoding="utf-8",
    )
    (workspace / "submission" / "finalize_gate.json").write_text(
        json.dumps(
            {
                "overall_ok": False,
                "supported_tasks": ["task1", "task2"],
                "prediction_ok": True,
                "time_csv_ok": False,
                "logs_ok": True,
                "code_manifest_ok": True,
                "methodology_ok": True,
                "shared_code_ok": True,
                "code_provenance_ok": False,
                "failures": ["missing code provenance linkage: submission/code/train.py"],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    status = collect_submission_status_v3(workspace_root=workspace, tasks=["task1"])

    assert status["current_phase"] == "validation"
    assert status["blocker_summary"] == "missing code provenance linkage: submission/code/train.py"
    assert status["latest_error"] == "task1_time.csv missing"
    assert status["best_artifact"] == "submission/task1_pred.hdf5"
    assert status["shared_code_version"] == "sha256:v2"
    assert status["supported_tasks"] == ["task1", "task2"]
    assert status["missing_gates"] == ["code_provenance_ok", "time_csv_ok"]


def test_collect_submission_status_v3_falls_back_to_validation_report_when_state_missing(workspace) -> None:
    (workspace / "submission" / "validation_report.json").write_text(
        json.dumps(
            {
                "finalize_gate": {
                    "overall_ok": True,
                    "supported_tasks": ["task3"],
                    "prediction_ok": True,
                    "time_csv_ok": True,
                    "logs_ok": True,
                    "code_manifest_ok": True,
                    "methodology_ok": True,
                    "shared_code_ok": True,
                    "code_provenance_ok": True,
                    "failures": [],
                    "warnings": [],
                }
            }
        ),
        encoding="utf-8",
    )
    (workspace / "submission" / "submission.zip").write_bytes(b"zip")

    status = collect_submission_status_v3(workspace_root=workspace, tasks=["task3"])

    assert status["current_phase"] == "ready_to_submit"
    assert status["blocker_summary"] == ""
    assert status["latest_error"] == ""
    assert status["best_artifact"] == "submission/submission.zip"
    assert status["supported_tasks"] == ["task3"]
    assert status["missing_gates"] == []
