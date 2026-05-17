from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _to_jsonable(value.dict())
    return str(value)


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def write(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = _to_jsonable(payload)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
        return normalized


class LLMCallLogger(JsonlLogger):
    def log_call(
        self,
        *,
        step_id: str,
        task_id: str,
        model: str,
        profile: str,
        phase: str,
        elapsed_seconds: float,
        raw_response: Any,
        response: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        hosted_tool_calls: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "timestamp": _timestamp(),
            "elapsed_seconds": elapsed_seconds,
            "model": model,
            "profile": profile,
            "phase": phase,
            "raw_response": _to_jsonable(raw_response),
            "step_id": step_id,
            "task_id": task_id,
        }
        if response is not None:
            payload["response"] = response
        elif not tool_calls:
            payload["response"] = ""
        if tool_calls:
            payload["tool_calls"] = _to_jsonable(tool_calls)
        if hosted_tool_calls:
            payload["hosted_tool_calls"] = _to_jsonable(hosted_tool_calls)
        return self.write(payload)


class ToolCallLogger(JsonlLogger):
    def log_call(
        self,
        *,
        tool_name: str,
        elapsed_seconds: float,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        return self.write(
            {
                "timestamp": _timestamp(),
                "elapsed_seconds": elapsed_seconds,
                "tool_name": tool_name,
                "arguments": _to_jsonable(arguments),
                "result": _to_jsonable(result),
            }
        )
