from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STRUCTURED_METHODOLOGY_SOURCES = [
    "agent_state_snapshots",
    "decision_records",
    "experiment_cards",
    "validation_reports",
    "artifact_metadata",
    "final_package_snapshot",
    "code_snapshots",
    "code_patch_records",
]


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


def _format_task_labels(tasks: list[str]) -> str:
    labels = [f"Task {task.removeprefix('task')}" for task in tasks]
    if not labels:
        return "No Tasks"
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _group_patch_records(code_manifest_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entry in code_manifest_entries:
        patch_id = str(entry.get("patch_id", "")).strip()
        code_version = str(entry.get("code_version", "")).strip()
        originating_task = str(entry.get("originating_task", "")).strip()
        if not patch_id or not code_version:
            continue
        key = (patch_id, code_version, originating_task)
        record = grouped.setdefault(
            key,
            {
                "patch_id": patch_id,
                "code_version": code_version,
                "originating_task": originating_task,
                "changed_files": [],
                "llm_call_ids": [],
            },
        )
        changed_files = [str(path) for path in record["changed_files"]]
        changed_files.append(str(entry.get("path", "")).strip())
        record["changed_files"] = sorted({path for path in changed_files if path})
        llm_call_ids = [str(call_id) for call_id in record["llm_call_ids"]]
        llm_call_ids.extend(str(call_id).strip() for call_id in entry.get("llm_call_ids", []) if str(call_id).strip())
        record["llm_call_ids"] = sorted({call_id for call_id in llm_call_ids if call_id})
    return list(grouped.values())


def build_methodology_record_bundle(
    *,
    tasks: list[str],
    finalize_gate: dict[str, Any],
    code_manifest_entries: list[dict[str, Any]],
    shared_code_union: dict[str, Any],
) -> dict[str, Any]:
    code_snapshots = []
    for record in shared_code_union.get("shared_code_versions", []):
        if not isinstance(record, dict):
            continue
        code_snapshots.append(
            {
                "code_version": str(record.get("version", "")).strip(),
                "parent_version": str(record.get("parent", "")).strip() or None,
                "validated_tasks": list(record.get("validated_tasks", [])),
            }
        )

    return {
        "agent_state_snapshots": [],
        "decision_records": [],
        "experiment_cards": [],
        "validation_reports": {
            "finalize_gate": finalize_gate,
        },
        "artifact_metadata": {
            "tasks": list(tasks),
            "code_manifest_entry_count": len(code_manifest_entries),
            "shared_code_version_count": len(shared_code_union.get("shared_code_versions", [])),
        },
        "final_package_snapshot": {
            "shared_code_union": shared_code_union,
            "code_manifest_paths": [str(entry.get("path", "")).strip() for entry in code_manifest_entries],
        },
        "code_snapshots": code_snapshots,
        "code_patch_records": _group_patch_records(code_manifest_entries),
    }


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_methodology_text(*, tasks: list[str], record_bundle: dict[str, Any]) -> str:
    finalize_gate = record_bundle.get("validation_reports", {}).get("finalize_gate", {})
    failures = [str(item) for item in finalize_gate.get("failures", []) if str(item)]
    shared_versions = record_bundle.get("final_package_snapshot", {}).get("shared_code_union", {}).get(
        "shared_code_versions",
        [],
    )
    shared_version_labels = [str(record.get("version", "")).strip() for record in shared_versions if str(record.get("version", "")).strip()]
    lines = [
        "Cozy PDE Deterministic Methodology",
        "",
        f"Scope: {_format_task_labels(tasks)}",
        "Generation mode: structured-record mechanical export",
        f"Structured sources: {', '.join(STRUCTURED_METHODOLOGY_SOURCES)}",
        f"Code manifest entries: {record_bundle.get('artifact_metadata', {}).get('code_manifest_entry_count', 0)}",
        f"Shared code versions: {len(shared_version_labels)}",
        f"Shared code lineage: {', '.join(shared_version_labels) if shared_version_labels else 'none'}",
        f"Finalize gate overall_ok: {bool(finalize_gate.get('overall_ok', False))}",
        f"Finalize gate failures: {len(failures)}",
    ]
    lines.extend(f"- {failure}" for failure in failures)
    lines.append("")
    lines.append("No LLM-authored scientific claims were added by this export.")
    return "\n".join(lines).strip()


def _build_simple_pdf_bytes(text: str) -> bytes:
    lines = text.splitlines() or [""]
    content_lines = ["BT", "/F1 10 Tf", "72 760 Td", "12 TL"]
    for line in lines:
        content_lines.append(f"({_pdf_escape(line)}) Tj")
        content_lines.append("T*")
    content_lines.append("ET")
    content_stream = "\n".join(content_lines).encode("latin-1", "replace")

    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Count 1 /Kids [3 0 R] >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        b"5 0 obj << /Length " + str(len(content_stream)).encode("ascii") + b" >> stream\n" + content_stream + b"\nendstream endobj\n",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf)


def write_deterministic_methodology_pdf(
    *,
    workspace_root: str | Path,
    tasks: list[str],
    record_bundle: dict[str, Any],
) -> dict[str, Any]:
    workspace_path = Path(workspace_root)
    submission_dir = workspace_path / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    missing_sources = [source for source in STRUCTURED_METHODOLOGY_SOURCES if source not in record_bundle]
    if missing_sources:
        return _failure("methodology record bundle missing structured sources", missing_sources=missing_sources)

    records_path = submission_dir / "methodology_records.json"
    pdf_path = submission_dir / "methodology.pdf"
    records_path.write_text(json.dumps(record_bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pdf_path.write_bytes(_build_simple_pdf_bytes(_build_methodology_text(tasks=tasks, record_bundle=record_bundle)))
    return _success(
        f"Wrote deterministic methodology for {', '.join(tasks)}",
        methodology_path=str(pdf_path),
        records_path=str(records_path),
        methodology_sources=list(STRUCTURED_METHODOLOGY_SOURCES),
    )
