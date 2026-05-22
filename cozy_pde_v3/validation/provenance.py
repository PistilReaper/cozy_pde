from __future__ import annotations

import re
from typing import Any

_TASK_SEGMENT_PATTERN = re.compile(r"^task\d+$")


def _stable_task_order(task: str) -> tuple[int, str]:
    suffix = task[4:]
    if task.startswith("task") and suffix.isdigit():
        return int(suffix), task
    return 10**9, task


def _stable_version_order(version: str) -> tuple[int, str]:
    suffix = version[1:]
    if version.startswith("v") and suffix.isdigit():
        return int(suffix), version
    return 10**9, version


def _normalized_segments(path: str) -> list[str]:
    normalized = path.replace("\\", "/").strip()
    return [segment for segment in normalized.split("/") if segment and segment != "."]


def _sorted_unique_strings(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def detect_task_specific_code_forks(paths: list[str]) -> list[str]:
    violations: list[str] = []
    for path in sorted(paths):
        normalized = path.replace("\\", "/")
        segments = _normalized_segments(normalized)
        for index, segment in enumerate(segments[:-2]):
            if segment == "code" and _TASK_SEGMENT_PATTERN.fullmatch(segments[index + 1]):
                violations.append(normalized)
                break
    return violations


def build_shared_code_union(
    records: list[dict[str, Any]] | None = None,
    *,
    snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    source_records = snapshots if snapshots is not None else records or []
    merged_by_version: dict[str, dict[str, Any]] = {}
    task_sets: dict[str, set[str]] = {}
    call_id_sets: dict[str, set[str]] = {}

    for record in source_records:
        version = str(record.get("version", "")).strip()
        if not version:
            continue
        merged = merged_by_version.setdefault(version, {"version": version})

        created_during = str(record.get("created_during", "")).strip()
        if created_during:
            merged["created_during"] = created_during

        parent = str(record.get("parent", "")).strip()
        if parent:
            merged["parent"] = parent

        for field_name in ("files", "changed_files"):
            raw_values = record.get(field_name, [])
            values = [str(value).strip() for value in raw_values if str(value).strip()]
            if not values:
                continue
            existing_values = [str(value) for value in merged.get(field_name, [])]
            merged[field_name] = _sorted_unique_strings(existing_values + values)

        task_sets.setdefault(version, set()).update(
            str(task).strip() for task in record.get("validated_tasks", []) if str(task).strip()
        )
        call_id_sets.setdefault(version, set()).update(
            str(call_id).strip() for call_id in record.get("llm_call_ids", []) if str(call_id).strip()
        )

    shared_code_versions: list[dict[str, Any]] = []
    for version in sorted(merged_by_version, key=_stable_version_order):
        merged = dict(merged_by_version[version])
        merged["validated_tasks"] = sorted(task_sets.get(version, ()), key=_stable_task_order)
        merged["llm_call_ids"] = sorted(call_id_sets.get(version, ()))
        shared_code_versions.append(merged)
    return {"shared_code_versions": shared_code_versions}
