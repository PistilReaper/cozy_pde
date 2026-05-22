from __future__ import annotations

from pathlib import Path
from typing import Any

from cozy_pde_v3.task_specs import TASK_IDS
from cozy_pde_v3.validation.logs import validate_task_log_jsonl


def _failure(message: str, **data: Any) -> dict[str, Any]:
    payload = {"ok": False, "error": message}
    if data:
        payload["data"] = data
    return payload


def _success(summary: str, **data: Any) -> dict[str, Any]:
    payload = {"ok": True, "summary": summary}
    if data:
        payload["data"] = data
    return payload


def _resolve_export_source_log(
    *,
    workspace: Path,
    tasks: list[str],
    allow_multi_task_session: bool,
) -> Path | None:
    llm_logs_dir = workspace / "llm_logs"
    if len(tasks) == 1:
        task = tasks[0]
        task_log = llm_logs_dir / f"{task}_all_llm_calls.jsonl"
        return task_log

    if not allow_multi_task_session:
        return None

    joined_name = "_".join(tasks)
    shared_task_log = llm_logs_dir / f"{joined_name}_all_llm_calls.jsonl"
    if shared_task_log.exists():
        return shared_task_log
    fallback_shared_log = llm_logs_dir / "all_llm_calls.jsonl"
    if fallback_shared_log.exists():
        return fallback_shared_log
    return shared_task_log


def export_task_logs(
    workspace: str | Path,
    tasks: list[str],
    source_log: str | Path | None = None,
    allow_multi_task_session: bool = False,
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    selected_tasks = [str(task).strip() for task in tasks if str(task).strip()]
    if not selected_tasks:
        return _failure("at least one task is required")
    unknown_tasks = [task for task in selected_tasks if task not in TASK_IDS]
    if unknown_tasks:
        return _failure("unknown task ids", tasks=selected_tasks, unknown_tasks=unknown_tasks)

    resolved_source = Path(source_log) if source_log is not None else _resolve_export_source_log(
        workspace=workspace_path,
        tasks=selected_tasks,
        allow_multi_task_session=allow_multi_task_session,
    )
    if resolved_source is None:
        return _failure(
            "Formal autonomous runs should use independent task sessions. Export one task at a time, or pass allow_multi_task_session=True for an intentional shared session.",
            tasks=selected_tasks,
        )

    source_validation = validate_task_log_jsonl(resolved_source)
    if not source_validation["ok"]:
        return _failure(source_validation["error"], source_log=str(resolved_source))

    content = resolved_source.read_text(encoding="utf-8")
    exported: list[str] = []
    task_logs: dict[str, dict[str, Any]] = {}
    for task in selected_tasks:
        destination = workspace_path / "submission" / f"{task}_logs.log"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        validation = validate_task_log_jsonl(destination)
        if not validation["ok"]:
            return _failure(validation["error"], source_log=str(resolved_source), destination=str(destination))
        exported.append(str(destination))
        task_logs[task] = {
            "source_log": str(resolved_source),
            "destination": str(destination),
            "record_count": int(validation.get("data", {}).get("record_count", 0)),
        }

    return _success(
        f"Exported logs for {', '.join(selected_tasks)}",
        tasks=selected_tasks,
        source_log=str(resolved_source),
        exported=exported,
        task_logs=task_logs,
    )
