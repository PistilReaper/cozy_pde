from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_finalize_gate(submission_dir: Path) -> dict[str, Any]:
    finalize_gate = _read_json(submission_dir / "finalize_gate.json")
    if finalize_gate:
        return finalize_gate
    validation_report = _read_json(submission_dir / "validation_report.json")
    nested_gate = validation_report.get("finalize_gate")
    return nested_gate if isinstance(nested_gate, dict) else {}


def _best_artifact(submission_dir: Path, tasks: list[str], state: dict[str, Any]) -> str:
    best_artifact = str(state.get("best_artifact_path", "")).strip()
    if best_artifact:
        return best_artifact
    zip_path = submission_dir / "submission.zip"
    if zip_path.exists():
        return "submission/submission.zip"
    for task in tasks:
        pred_path = submission_dir / f"{task}_pred.hdf5"
        if pred_path.exists():
            return f"submission/{pred_path.name}"
    return ""


def _shared_code_version(submission_dir: Path, state: dict[str, Any]) -> str:
    state_version = str(state.get("shared_code_version", "")).strip()
    if state_version:
        return state_version
    union_payload = _read_json(submission_dir / "shared_code_union.json")
    versions = union_payload.get("shared_code_versions", [])
    if isinstance(versions, list) and versions:
        last = versions[-1]
        if isinstance(last, dict):
            return str(last.get("version", "")).strip()
    submission_payload = _read_json(submission_dir / "submission.json")
    nested_union = submission_payload.get("shared_code_union", {})
    nested_versions = nested_union.get("shared_code_versions", []) if isinstance(nested_union, dict) else []
    if isinstance(nested_versions, list) and nested_versions:
        last = nested_versions[-1]
        if isinstance(last, dict):
            return str(last.get("version", "")).strip()
    return ""


def _missing_gates(finalize_gate: dict[str, Any]) -> list[str]:
    missing = [
        key
        for key, value in finalize_gate.items()
        if key.endswith("_ok") and key != "overall_ok" and value is False
    ]
    return sorted(missing)


def collect_submission_status_v3(
    *,
    workspace_root: str | Path,
    tasks: list[str],
) -> dict[str, Any]:
    workspace_path = Path(workspace_root)
    submission_dir = workspace_path / "submission"
    state = _read_json(workspace_path / "agent_state.json")
    finalize_gate = _load_finalize_gate(submission_dir)

    overall_ok = bool(finalize_gate.get("overall_ok", False))
    current_phase = str(state.get("current_phase", "")).strip()
    if not current_phase:
        current_phase = "ready_to_submit" if overall_ok and (submission_dir / "submission.zip").exists() else "validation"

    failures = [str(item) for item in finalize_gate.get("failures", []) if str(item)]
    latest_error = str(state.get("latest_error_summary", "")).strip()
    blocker_summary = failures[0] if failures else ""
    if not latest_error:
        latest_error = blocker_summary

    supported_tasks = state.get("supported_tasks")
    if not isinstance(supported_tasks, list) or not supported_tasks:
        supported_tasks = finalize_gate.get("supported_tasks")
    if not isinstance(supported_tasks, list) or not supported_tasks:
        supported_tasks = list(tasks)

    return {
        "current_phase": current_phase,
        "blocker_summary": blocker_summary,
        "latest_error": latest_error,
        "best_artifact": _best_artifact(submission_dir, list(tasks), state),
        "shared_code_version": _shared_code_version(submission_dir, state),
        "supported_tasks": [str(task) for task in supported_tasks],
        "missing_gates": _missing_gates(finalize_gate),
        "finalize_gate": finalize_gate,
    }
