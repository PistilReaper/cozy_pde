from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def _as_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _as_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_as_dict(item) for item in value]
    if isinstance(value, tuple):
        return [_as_dict(item) for item in value]
    if hasattr(value, "model_dump"):
        return _as_dict(value.model_dump())
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _as_dict(vars(value))
    return value


def system_text(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text", "text": text}],
    }


def user_text(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }


def assistant_text(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def function_call_output(call_id: str, output: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(output, ensure_ascii=False),
    }


@dataclass(slots=True)
class ResponsesFunctionCall:
    name: str
    arguments: dict[str, Any]
    call_id: str
    raw_arguments: Any | None = field(default=None, compare=False)


def _output_items(response: Any) -> list[Any]:
    if isinstance(response, dict):
        return list(response.get("output", []))
    return list(getattr(response, "output", []) or [])


def extract_output_text(response: Any) -> str:
    if isinstance(response, dict) and response.get("output_text"):
        return str(response["output_text"])
    if hasattr(response, "output_text") and getattr(response, "output_text"):
        return str(getattr(response, "output_text"))

    chunks: list[str] = []
    for item in _output_items(response):
        item_data = _as_dict(item)
        if item_data.get("type") != "message":
            continue
        for content_item in item_data.get("content", []):
            if content_item.get("type") == "output_text":
                text = content_item.get("text")
                if text:
                    chunks.append(str(text))
    return "\n".join(chunks)


def extract_function_calls(response: Any) -> list[ResponsesFunctionCall]:
    calls: list[ResponsesFunctionCall] = []
    for item in _output_items(response):
        item_data = _as_dict(item)
        if item_data.get("type") != "function_call":
            continue
        raw_arguments = item_data.get("arguments")
        arguments: dict[str, Any]
        if isinstance(raw_arguments, dict):
            arguments = raw_arguments
        elif isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError:
                parsed = {}
            arguments = parsed if isinstance(parsed, dict) else {"value": parsed}
        else:
            arguments = {}
        calls.append(
            ResponsesFunctionCall(
                name=str(item_data.get("name", "")),
                arguments=arguments,
                call_id=str(item_data.get("call_id") or item_data.get("id") or ""),
                raw_arguments=raw_arguments,
            )
        )
    return calls


def extract_hosted_tool_calls(response: Any) -> list[dict[str, Any]]:
    hosted: list[dict[str, Any]] = []
    for item in _output_items(response):
        item_data = _as_dict(item)
        item_type = item_data.get("type")
        if item_type in {"message", "function_call", "function_call_output"}:
            continue
        hosted.append(item_data)
    return hosted


def response_to_ledger_items(response: Any) -> list[dict[str, Any]]:
    return [_as_dict(item) for item in _output_items(response)]
