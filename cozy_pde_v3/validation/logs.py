from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def validate_jsonl_logs(path: str | Path) -> dict[str, Any]:
    return validate_task_log_jsonl(path)


def _failure(path: Path, message: str, **data: Any) -> dict[str, Any]:
    payload = {"ok": False, "error": message, "path": str(path)}
    if data:
        payload["data"] = data
    return payload


def validate_task_log_jsonl(path: str | Path) -> dict[str, Any]:
    log_path = Path(path)
    if not log_path.exists():
        return _failure(log_path, f"{log_path} does not exist")

    text = log_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        return _failure(log_path, f"{log_path} is empty")

    record_count = 0
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            return _failure(log_path, f"line {line_number} is not valid JSON: {exc.msg}")
        if not isinstance(record, dict):
            return _failure(log_path, f"line {line_number} must contain a JSON object")
        if "timestamp" not in record:
            return _failure(log_path, f"line {line_number} missing timestamp")
        if "elapsed_seconds" not in record:
            return _failure(log_path, f"line {line_number} missing elapsed_seconds")
        if "response" not in record and "tool_calls" not in record:
            return _failure(log_path, f"line {line_number} must include response or tool_calls")
        record_count += 1

    return {
        "ok": True,
        "path": str(log_path),
        "data": {
            "record_count": record_count,
        },
    }
