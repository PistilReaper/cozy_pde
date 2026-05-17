from __future__ import annotations

from typing import Any


def success(tool: str, summary: str, **data: Any) -> dict[str, Any]:
    return {
        "ok": True,
        "tool": tool,
        "summary": summary,
        "data": data,
    }


def failure(tool: str, error: str, **data: Any) -> dict[str, Any]:
    return {
        "ok": False,
        "tool": tool,
        "error": error,
        "data": data,
    }

