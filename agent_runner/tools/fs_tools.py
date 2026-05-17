from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ..safety import WorkspaceSafety
from . import failure, success


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _code_manifest_path(safety: WorkspaceSafety) -> Path:
    return safety.workspace_root / "submission" / "code_manifest.json"


def _update_code_manifest(
    *,
    safety: WorkspaceSafety,
    resolved_path: Path,
    payload: bytes,
    runner_context: dict | None,
) -> None:
    code_root = safety.workspace_root / "submission" / "code"
    if not safety._is_relative_to(resolved_path, code_root):
        return

    manifest_path = _code_manifest_path(safety)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    else:
        existing = []
    if not isinstance(existing, list):
        existing = []

    context = runner_context or {}
    existing.append(
        {
            "path": str(resolved_path.relative_to(safety.workspace_root)),
            "sha256": _sha256_bytes(payload),
            "size": len(payload),
            "step_id": context.get("step_id", "unknown"),
            "task_id": context.get("task_id", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    manifest_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def read_file(*, path: str, safety: WorkspaceSafety, max_chars: int = 20000) -> dict:
    check = safety.validate_read_path(path)
    if not check.ok:
        return failure("read_file", check.error or "read check failed", path=path)
    assert check.resolved_path is not None
    if not check.resolved_path.exists():
        return failure("read_file", "File does not exist", path=str(check.resolved_path))

    try:
        content = check.resolved_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return failure("read_file", "File is not valid UTF-8 text", path=str(check.resolved_path))

    truncated = content[:max_chars]
    return success(
        "read_file",
        f"Read {len(truncated)} characters from {check.resolved_path.name}",
        path=str(check.resolved_path),
        exists=True,
        size_bytes=check.resolved_path.stat().st_size,
        content=truncated,
        truncated=len(truncated) < len(content),
    )


def write_file(*, path: str, content: str, safety: WorkspaceSafety, runner_context: dict | None = None) -> dict:
    check = safety.validate_write_path(path)
    if not check.ok:
        return failure("write_file", check.error or "write check failed", path=path)
    assert check.resolved_path is not None
    check.resolved_path.parent.mkdir(parents=True, exist_ok=True)
    payload = content.encode("utf-8")
    check.resolved_path.write_bytes(payload)
    _update_code_manifest(
        safety=safety,
        resolved_path=check.resolved_path,
        payload=payload,
        runner_context=runner_context,
    )
    return success(
        "write_file",
        f"Wrote {len(payload)} bytes to {check.resolved_path.name}",
        path=str(check.resolved_path),
        size_bytes=len(payload),
        sha256=_sha256_bytes(payload),
        first_200_chars=content[:200],
        last_200_chars=content[-200:],
    )


def list_files(*, path: str, safety: WorkspaceSafety, recursive: bool = False, max_entries: int = 200) -> dict:
    check = safety.validate_read_path(path)
    if not check.ok:
        return failure("list_files", check.error or "list check failed", path=path)
    assert check.resolved_path is not None
    if not check.resolved_path.exists():
        return failure("list_files", "Path does not exist", path=str(check.resolved_path))
    if not check.resolved_path.is_dir():
        return failure("list_files", "Path is not a directory", path=str(check.resolved_path))

    iterator = check.resolved_path.rglob("*") if recursive else check.resolved_path.glob("*")
    entries = []
    for index, entry in enumerate(sorted(iterator)):
        if index >= max_entries:
            break
        stats = entry.stat()
        entries.append(
            {
                "path": str(entry),
                "type": "directory" if entry.is_dir() else "file",
                "size_bytes": stats.st_size,
                "modified_at": datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc).isoformat(),
            }
        )

    return success(
        "list_files",
        f"Listed {len(entries)} entries under {check.resolved_path.name}",
        path=str(check.resolved_path),
        entries=entries,
        recursive=recursive,
    )
