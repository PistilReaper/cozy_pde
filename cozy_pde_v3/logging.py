from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_llm_log_entry(
    *,
    elapsed_seconds: float,
    provider: str,
    model: str,
    profile: str,
    phase: str,
    raw_response: dict[str, Any],
    standard_output_items: list[dict[str, Any]],
    task_id: str,
    run_id: str,
    step_id: str,
) -> dict[str, Any]:
    response_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in standard_output_items:
        item_type = str(item.get("type", ""))
        if item_type == "function_call":
            tool_calls.append(
                {
                    "name": str(item.get("name", "")),
                    "call_id": str(item.get("call_id", "")),
                    "arguments": item.get("arguments"),
                }
            )
            continue
        if item_type != "message":
            continue
        for chunk in item.get("content", []):
            if chunk.get("type") == "output_text":
                response_chunks.append(str(chunk.get("text", "")))

    payload: dict[str, Any] = {
        "timestamp": utc_timestamp(),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "provider": provider,
        "model": model,
        "profile": profile,
        "phase": phase,
        "task_id": task_id,
        "run_id": run_id,
        "step_id": step_id,
        "raw_response": raw_response,
        "standard_output_items": standard_output_items,
    }
    if tool_calls:
        payload["tool_calls"] = tool_calls
    if response_chunks or not tool_calls:
        payload["response"] = "\n".join(chunk for chunk in response_chunks if chunk).strip()
    return payload
