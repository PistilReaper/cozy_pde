from __future__ import annotations

import json
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import Any

from cozy_pde_v3.memory_store import MemoryStore
from cozy_pde_v3.methodology import (
    STRUCTURED_METHODOLOGY_SOURCES,
    build_methodology_record_bundle,
    write_deterministic_methodology_pdf,
)
from cozy_pde_v3.validation.provenance import build_shared_code_union
from cozy_pde_v3.validation.submission import validate_submission_bundle_v3


def _success(summary: str, **data: Any) -> dict[str, Any]:
    payload = {"ok": True, "summary": summary}
    if data:
        payload["data"] = data
    return payload


def _failure(message: str, **data: Any) -> dict[str, Any]:
    payload = {"ok": False, "error": message}
    if data:
        payload["data"] = data
    return payload


def _sha256_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _code_files(submission_dir: Path) -> list[Path]:
    code_dir = submission_dir / "code"
    if not code_dir.exists():
        return []
    return [path for path in sorted(code_dir.rglob("*")) if path.is_file()]


def _normalize_task_context(task_context: object, tasks: list[str]) -> str:
    text = str(task_context or "").strip()
    if text in tasks:
        return text
    for task in tasks:
        if task and task in text:
            return task
    return tasks[0] if tasks else ""


def _load_memory_records(workspace_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidate_paths = [workspace_root / "internal_logs" / "memory.db", workspace_root / "memory.db"]
    internal_logs = workspace_root / "internal_logs"
    if internal_logs.exists():
        candidate_paths.extend(sorted(path for path in internal_logs.glob("*.db") if path.is_file()))
    seen: set[Path] = set()
    for db_path in candidate_paths:
        if db_path in seen or not db_path.exists():
            continue
        seen.add(db_path)
        store = MemoryStore(db_path)
        return store.list_code_snapshots(), store.list_patch_records()
    return [], []


def _build_generated_code_manifest_entries(
    *,
    workspace_root: Path,
    tasks: list[str],
) -> list[dict[str, Any]]:
    submission_dir = workspace_root / "submission"
    code_files = _code_files(submission_dir)
    snapshots, patches = _load_memory_records(workspace_root)
    if not code_files:
        return []
    if not snapshots and not patches:
        raise RuntimeError("memory.db provenance records are required to generate code_manifest.json")

    snapshots_by_version = {
        str(snapshot.get("code_version", "")).strip(): snapshot
        for snapshot in snapshots
        if str(snapshot.get("code_version", "")).strip()
    }
    latest_snapshot = snapshots[-1] if snapshots else {}
    latest_version = str(latest_snapshot.get("code_version", "")).strip()
    latest_created_at = str(latest_snapshot.get("created_at", "")).strip()
    patch_by_path: dict[str, dict[str, object]] = {}
    for patch in patches:
        changed_files = patch.get("changed_files", [])
        if not isinstance(changed_files, list):
            continue
        for changed_file in changed_files:
            path = str(changed_file).strip()
            if path:
                patch_by_path[path] = patch

    entries: list[dict[str, Any]] = []
    for file_path in code_files:
        relative_path = str(file_path.relative_to(workspace_root)).replace("\\", "/")
        patch = patch_by_path.get(relative_path, {})
        code_version = str(patch.get("new_code_version", "")).strip() or latest_version
        originating_task = _normalize_task_context(patch.get("task_context", ""), tasks)
        snapshot = snapshots_by_version.get(code_version, latest_snapshot)
        timestamp = str(snapshot.get("created_at", "")).strip() or latest_created_at or "1970-01-01T00:00:00Z"
        llm_call_ids = sorted({str(call_id).strip() for call_id in patch.get("llm_call_ids", []) if str(call_id).strip()})
        patch_id = str(patch.get("patch_id", "")).strip() or f"snapshot:{code_version}"
        entries.append(
            {
                "path": relative_path,
                "sha256": _sha256_file(file_path),
                "size": file_path.stat().st_size,
                "code_version": code_version,
                "originating_task": originating_task,
                "llm_call_ids": llm_call_ids,
                "patch_id": patch_id,
                "step_id": patch_id,
                "task_id": originating_task,
                "timestamp": timestamp,
            }
        )
    return entries


def _normalize_code_manifest_entries(
    *,
    entries: object,
    workspace_root: Path,
) -> tuple[bool, list[dict[str, Any]] | None, str]:
    if not isinstance(entries, list):
        return False, None, "code_manifest.json must contain a list of file metadata"

    normalized_entries: list[dict[str, Any]] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            return False, None, "code manifest entry must be an object"
        path = str(raw_entry.get("path", "")).strip()
        code_version = str(raw_entry.get("code_version", "")).strip()
        originating_task = str(raw_entry.get("originating_task", "")).strip()
        patch_id = str(raw_entry.get("patch_id", "")).strip()
        sha_value = str(raw_entry.get("sha256", "")).strip()
        llm_call_ids = sorted(
            {str(call_id).strip() for call_id in raw_entry.get("llm_call_ids", []) if str(call_id).strip()}
        )
        if not path:
            return False, None, "code manifest entry missing path"
        if not code_version:
            return False, None, f"code manifest entry missing code_version: {path}"
        if not originating_task:
            return False, None, f"code manifest entry missing originating_task: {path}"
        if not patch_id:
            return False, None, f"code manifest entry missing patch_id: {path}"
        if not llm_call_ids:
            return False, None, f"code manifest entry missing llm_call_ids: {path}"

        file_path = workspace_root / path
        if not file_path.exists():
            return False, None, f"code manifest entry missing file: {path}"
        actual_sha = _sha256_file(file_path)
        if sha_value and sha_value != actual_sha:
            return False, None, f"code manifest sha256 mismatch: {path}"

        normalized_entries.append(
            {
                "path": path,
                "sha256": actual_sha,
                "size": file_path.stat().st_size,
                "code_version": code_version,
                "originating_task": originating_task,
                "llm_call_ids": llm_call_ids,
                "patch_id": patch_id,
                "step_id": str(raw_entry.get("step_id", patch_id)).strip() or patch_id,
                "task_id": str(raw_entry.get("task_id", originating_task)).strip() or originating_task,
                "timestamp": str(raw_entry.get("timestamp", "1970-01-01T00:00:00Z")).strip() or "1970-01-01T00:00:00Z",
            }
        )
    normalized_entries.sort(key=lambda entry: str(entry["path"]))
    return True, normalized_entries, ""


def build_code_manifest_entries(
    *,
    workspace_root: str | Path,
    tasks: list[str],
) -> list[dict[str, Any]]:
    return _build_generated_code_manifest_entries(workspace_root=Path(workspace_root), tasks=list(tasks))


def build_shared_code_union_for_workspace(
    *,
    workspace_root: str | Path,
    tasks: list[str],
    code_manifest_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    workspace_path = Path(workspace_root)
    snapshots, patches = _load_memory_records(workspace_path)
    snapshots_by_version = {
        str(snapshot.get("code_version", "")).strip(): snapshot
        for snapshot in snapshots
        if str(snapshot.get("code_version", "")).strip()
    }
    records: list[dict[str, Any]] = []
    for snapshot in snapshots:
        version = str(snapshot.get("code_version", "")).strip()
        if not version:
            continue
        record: dict[str, Any] = {
            "version": version,
            "validated_tasks": list(snapshot.get("supported_tasks", [])),
        }
        raw_parent = snapshot.get("parent_version")
        parent = str(raw_parent).strip() if raw_parent is not None else ""
        if parent:
            record["parent"] = parent
        records.append(record)

    for patch in patches:
        version = str(patch.get("new_code_version", "")).strip()
        if not version:
            continue
        snapshot = snapshots_by_version.get(version, {})
        validation_results = patch.get("validation_results", {})
        validated_tasks = []
        if isinstance(validation_results, dict):
            raw_validated = validation_results.get("validated_tasks", [])
            if isinstance(raw_validated, list):
                validated_tasks = [str(task).strip() for task in raw_validated if str(task).strip()]
        if not validated_tasks:
            validated_tasks = [str(task).strip() for task in snapshot.get("supported_tasks", []) if str(task).strip()]
        records.append(
            {
                "version": version,
                "created_during": _normalize_task_context(patch.get("task_context", ""), list(tasks)),
                "parent": str(patch.get("base_code_version", "")).strip(),
                "changed_files": [str(path).strip() for path in patch.get("changed_files", []) if str(path).strip()],
                "validated_tasks": validated_tasks,
                "llm_call_ids": [str(call_id).strip() for call_id in patch.get("llm_call_ids", []) if str(call_id).strip()],
            }
        )
    return build_shared_code_union(records)


def _resolve_code_manifest_entries(
    *,
    workspace_root: Path,
    submission_dir: Path,
    tasks: list[str],
    explicit_entries: list[dict[str, Any]] | None,
) -> tuple[bool, list[dict[str, Any]] | None, str]:
    if explicit_entries is not None:
        return _normalize_code_manifest_entries(entries=explicit_entries, workspace_root=workspace_root)

    code_manifest_path = submission_dir / "code_manifest.json"
    if code_manifest_path.exists():
        try:
            payload = json.loads(code_manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return False, None, f"failed to read code_manifest.json: {exc}"
        ok, normalized, error = _normalize_code_manifest_entries(entries=payload, workspace_root=workspace_root)
        if not ok:
            return False, None, f"invalid code_manifest.json: {error}"
        return True, normalized, ""

    try:
        generated_entries = build_code_manifest_entries(workspace_root=workspace_root, tasks=tasks)
    except RuntimeError as exc:
        return False, None, str(exc)
    return _normalize_code_manifest_entries(entries=generated_entries, workspace_root=workspace_root)


def _submission_members(submission_dir: Path, tasks: list[str]) -> list[tuple[Path, str]]:
    members: list[tuple[Path, str]] = []
    for name in (
        "submission.json",
        "methodology.pdf",
        "methodology_records.json",
        "code_manifest.json",
        "shared_code_union.json",
        "finalize_gate.json",
        "validation_report.json",
        "manifest.json",
    ):
        candidate = submission_dir / name
        if candidate.exists():
            members.append((candidate, candidate.name))
    for task in tasks:
        for suffix in ("pred.hdf5", "time.csv", "logs.log"):
            candidate = submission_dir / f"{task}_{suffix}"
            if candidate.exists():
                members.append((candidate, candidate.name))
    for file_path in _code_files(submission_dir):
        relative = file_path.relative_to(submission_dir / "code")
        members.append((file_path, str(Path("shared_code") / relative).replace("\\", "/")))
    deduped: dict[str, tuple[Path, str]] = {}
    for source_path, archive_path in members:
        deduped[archive_path] = (source_path, archive_path)
    return [deduped[key] for key in sorted(deduped)]


def _write_manifest(submission_dir: Path, tasks: list[str]) -> Path:
    entries = []
    for source_path, archive_path in _submission_members(submission_dir, tasks):
        if archive_path == "manifest.json":
            continue
        entries.append(
            {
                "path": str(source_path.relative_to(submission_dir)).replace("\\", "/"),
                "archive_path": archive_path,
                "size": source_path.stat().st_size,
                "sha256": _sha256_file(source_path),
            }
        )
    manifest_path = submission_dir / "manifest.json"
    manifest_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _write_submission_json(
    *,
    submission_dir: Path,
    tasks: list[str],
    finalize_gate: dict[str, Any],
    shared_code_union: dict[str, Any],
    methodology_sources: list[str],
) -> Path:
    payload = {
        "tasks": list(tasks),
        "supported_tasks": list(finalize_gate.get("supported_tasks", tasks)),
        "finalize_gate_overall_ok": bool(finalize_gate.get("overall_ok", False)),
        "shared_code_union": shared_code_union,
        "methodology_sources": list(methodology_sources),
        "archive_layout": {
            "shared_code_root": "shared_code",
        },
    }
    path = submission_dir / "submission.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def package_submission_v3(
    *,
    submission_dir: str | Path,
    tasks: list[str],
    test_data_roots: list[str | Path],
    strict: bool = True,
    code_manifest_entries: list[dict[str, Any]] | None = None,
    methodology_sources: list[str] | None = None,
) -> dict[str, Any]:
    submission_path = Path(submission_dir)
    workspace_root = submission_path.parent
    ok, resolved_code_manifest_entries, code_manifest_error = _resolve_code_manifest_entries(
        workspace_root=workspace_root,
        submission_dir=submission_path,
        tasks=tasks,
        explicit_entries=code_manifest_entries,
    )
    if not ok or resolved_code_manifest_entries is None:
        return _failure(code_manifest_error or "failed to resolve code_manifest.json")

    resolved_methodology_sources = sorted(
        {
            str(source).strip()
            for source in (methodology_sources or STRUCTURED_METHODOLOGY_SOURCES)
            if str(source).strip()
        }
    )
    shared_code_union = build_shared_code_union_for_workspace(
        workspace_root=workspace_root,
        tasks=tasks,
        code_manifest_entries=resolved_code_manifest_entries,
    )
    provisional_record_bundle = build_methodology_record_bundle(
        tasks=tasks,
        finalize_gate={"overall_ok": False, "failures": [], "warnings": [], "supported_tasks": list(tasks)},
        code_manifest_entries=resolved_code_manifest_entries,
        shared_code_union=shared_code_union,
    )
    provisional_methodology_result = write_deterministic_methodology_pdf(
        workspace_root=workspace_root,
        tasks=tasks,
        record_bundle=provisional_record_bundle,
    )
    if not provisional_methodology_result["ok"]:
        return _failure(provisional_methodology_result["error"], methodology=provisional_methodology_result)

    validation_result = validate_submission_bundle_v3(
        workspace_root=workspace_root,
        tasks=tasks,
        strict=strict,
        code_manifest_entries=resolved_code_manifest_entries,
        methodology_sources=resolved_methodology_sources,
    )
    if not validation_result["ok"]:
        return _failure(validation_result["error"], validation=validation_result)

    finalize_gate = validation_result.get("data", {}).get("finalize_gate", {})
    record_bundle = build_methodology_record_bundle(
        tasks=tasks,
        finalize_gate=finalize_gate,
        code_manifest_entries=resolved_code_manifest_entries,
        shared_code_union=shared_code_union,
    )
    methodology_result = write_deterministic_methodology_pdf(
        workspace_root=workspace_root,
        tasks=tasks,
        record_bundle=record_bundle,
    )
    if not methodology_result["ok"]:
        return _failure(methodology_result["error"], methodology=methodology_result)

    code_manifest_path = submission_path / "code_manifest.json"
    code_manifest_path.write_text(
        json.dumps(resolved_code_manifest_entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shared_code_union_path = submission_path / "shared_code_union.json"
    shared_code_union_path.write_text(json.dumps(shared_code_union, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    submission_json_path = _write_submission_json(
        submission_dir=submission_path,
        tasks=tasks,
        finalize_gate=finalize_gate,
        shared_code_union=shared_code_union,
        methodology_sources=resolved_methodology_sources,
    )
    finalize_gate_path = submission_path / "finalize_gate.json"
    finalize_gate_path.write_text(json.dumps(finalize_gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation_report_path = submission_path / "validation_report.json"
    validation_report_path.write_text(json.dumps(validation_result.get("data", {}), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_path = _write_manifest(submission_path, tasks)
    zip_path = submission_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source_path, archive_path in _submission_members(submission_path, tasks):
            if source_path == zip_path:
                continue
            archive.write(source_path, arcname=archive_path)

    return _success(
        f"Packaged submission for {', '.join(tasks)}",
        zip_path=str(zip_path),
        manifest_path=str(manifest_path),
        code_manifest_path=str(code_manifest_path),
        shared_code_union_path=str(shared_code_union_path),
        submission_json=str(submission_json_path),
        finalize_gate_path=str(finalize_gate_path),
        validation_report_path=str(validation_report_path),
        finalize_gate=finalize_gate,
    )
