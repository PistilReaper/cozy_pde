from __future__ import annotations

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


def normalize_raw_response(raw_response: Any) -> dict[str, Any]:
    normalized = _as_dict(raw_response)
    if isinstance(normalized, dict):
        return normalized
    return {"value": normalized}


def provider_output_items(raw_response: Any) -> list[dict[str, Any]]:
    normalized = normalize_raw_response(raw_response)
    output = normalized.get("output", [])
    if not isinstance(output, list):
        return []
    return [_as_dict(item) for item in output if isinstance(_as_dict(item), dict)]


def standard_output_items(raw_response: Any) -> list[dict[str, Any]]:
    return provider_output_items(raw_response)


def usage_payload(raw_response: Any) -> dict[str, Any]:
    normalized = normalize_raw_response(raw_response)
    usage = normalized.get("usage", {})
    return usage if isinstance(usage, dict) else {}
