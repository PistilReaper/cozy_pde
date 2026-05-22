from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


def _normalize_text_content(content: str | bytes) -> str:
    if isinstance(content, bytes):
        return content.decode("utf-8")
    return content


def _canonical_entries(
    content: Mapping[str, str | bytes] | list[tuple[str, str | bytes]]
) -> list[tuple[str, str]]:
    if isinstance(content, Mapping):
        items = list(content.items())
    else:
        items = list(content)
    return sorted(
        (str(path), _normalize_text_content(value))
        for path, value in items
    )


def stable_shared_code_hash(
    content: Mapping[str, str | bytes] | list[tuple[str, str | bytes]]
) -> str:
    canonical_payload = json.dumps(
        _canonical_entries(content),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical_payload.encode('utf-8')).hexdigest()}"


def directory_api_contract_hash(payload: object) -> str:
    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical_payload.encode('utf-8')).hexdigest()}"


def read_shared_code_directory_content(code_dir: str | Path) -> dict[str, str]:
    root = Path(code_dir)
    if not root.exists():
        return {}

    content: dict[str, str] = {}
    for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = str(file_path.relative_to(root)).replace("\\", "/")
        content[f"submission/code/{relative}"] = file_path.read_text(encoding="utf-8")
    return content


@dataclass(slots=True)
class CodeSnapshot:
    code_version: str
    parent_version: str | None
    content_hash: str
    api_contract_hash: str
    supported_tasks: list[str] = field(default_factory=list)
    task_support_matrix: dict[str, object] = field(default_factory=dict)
    created_by_run_id: str = ""
    created_at: str = ""


@dataclass(slots=True)
class CodePatchRecord:
    patch_id: str
    base_code_version: str
    new_code_version: str
    task_context: str = ""
    changed_files: list[str] = field(default_factory=list)
    change_intent: str = ""
    backward_compatibility_claim: str = ""
    affected_interfaces: list[str] = field(default_factory=list)
    llm_call_ids: list[str] = field(default_factory=list)
    validation_results: dict[str, object] = field(default_factory=dict)


def snapshot_shared_code_directory(
    *,
    code_dir: str | Path,
    api_contract_payload: object,
    parent_version: str | None,
    supported_tasks: list[str],
    task_support_matrix: dict[str, object],
    created_by_run_id: str,
    created_at: str,
) -> CodeSnapshot:
    content = read_shared_code_directory_content(code_dir)
    content_hash = stable_shared_code_hash(content)
    api_contract_hash = directory_api_contract_hash(api_contract_payload)
    return CodeSnapshot(
        code_version=stable_shared_code_hash(
            [
                ("content_hash", content_hash),
                ("api_contract_hash", api_contract_hash),
            ]
        ),
        parent_version=parent_version,
        content_hash=content_hash,
        api_contract_hash=api_contract_hash,
        supported_tasks=sorted({task for task in supported_tasks if task}),
        task_support_matrix=dict(task_support_matrix),
        created_by_run_id=created_by_run_id,
        created_at=created_at,
    )
