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


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _parse_action_json(text: str) -> Any | None:
    stripped = _strip_code_fence(text)
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _action_calls_from_payload(payload: Any) -> list[ResponsesFunctionCall]:
    if not isinstance(payload, dict):
        return []

    actions = payload.get("actions")
    if isinstance(actions, list):
        calls: list[ResponsesFunctionCall] = []
        for index, action in enumerate(actions, start=1):
            if not isinstance(action, dict):
                continue
            tool_name = action.get("tool_name")
            arguments = action.get("arguments")
            if isinstance(tool_name, str) and isinstance(arguments, dict):
                calls.append(
                    ResponsesFunctionCall(
                        name=tool_name,
                        arguments=arguments,
                        call_id=str(action.get("call_id") or f"json_action_{index}"),
                        raw_arguments=arguments,
                    )
                )
        return calls

    tool_name = payload.get("tool_name")
    arguments = payload.get("arguments")
    action_type = payload.get("type")
    if action_type == "action" and isinstance(tool_name, str) and isinstance(arguments, dict):
        return [
            ResponsesFunctionCall(
                name=tool_name,
                arguments=arguments,
                call_id=str(payload.get("call_id") or "json_action_1"),
                raw_arguments=arguments,
            )
        ]
    if isinstance(tool_name, str) and isinstance(arguments, dict):
        return [
            ResponsesFunctionCall(
                name=tool_name,
                arguments=arguments,
                call_id=str(payload.get("call_id") or "json_action_1"),
                raw_arguments=arguments,
            )
        ]
    return []


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
    if calls:
        return calls
    payload = _parse_action_json(extract_output_text(response))
    return _action_calls_from_payload(payload)


def extract_final_output_text(response: Any) -> str | None:
    payload = _parse_action_json(extract_output_text(response))
    if not isinstance(payload, dict):
        return None
    if payload.get("type") == "final" and isinstance(payload.get("message"), str):
        return str(payload["message"])
    if isinstance(payload.get("final"), str):
        return str(payload["final"])
    return None


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
