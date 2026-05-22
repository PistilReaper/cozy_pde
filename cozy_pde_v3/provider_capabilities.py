from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cozy_pde_v3.config import V3Config


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _provider_report(
    *,
    payload: dict[str, Any],
    primary: bool,
    fallback: bool,
) -> dict[str, Any]:
    report = dict(payload)
    report["model_id"] = str(payload["model_id"])
    report["base_url_hash"] = _hash_payload(str(payload["base_url"]))
    report["primary"] = primary
    report["fallback"] = fallback
    return report


def write_provider_capability_report(
    path: str | Path,
    *,
    config_payload: Any,
    tool_schemas: list[dict[str, Any]],
    proxy_payload: Any,
    adapter_version: str,
    sdk_version: str,
    checked_at: str,
    expires_at: str,
    forced_failover: dict[str, Any],
    primary: dict[str, Any],
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary_report = _provider_report(payload=primary, primary=True, fallback=False)
    fallback_report = _provider_report(payload=fallback, primary=False, fallback=True) if fallback is not None else None
    report: dict[str, Any] = {
        "config_hash": _hash_payload(config_payload),
        "tool_schema_hash": _hash_payload(tool_schemas),
        "proxy_version_hash": _hash_payload(proxy_payload),
        "adapter_version": adapter_version,
        "sdk_version": sdk_version,
        "checked_at": checked_at,
        "expires_at": expires_at,
        "forced_failover": dict(forced_failover),
        "formal_ready": bool(primary_report["formal_ready"]),
        "primary": primary_report,
    }
    if fallback_report is not None:
        report["fallback"] = fallback_report

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def load_provider_capability_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def default_provider_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "echo_tool",
            "description": "Echoes a short string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        }
    ]


def verify_provider_capability_report(
    report: dict[str, Any],
    *,
    config: V3Config,
    tool_schemas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    expected_config_hash = _hash_payload(config.payload())
    expected_tool_schema_hash = _hash_payload(tool_schemas or default_provider_tool_schemas())
    expected_proxy_hash = _hash_payload(config.proxy.payload())

    if str(report.get("config_hash", "")) != expected_config_hash:
        return {"ok": False, "error": "provider report config_hash mismatch"}
    if str(report.get("tool_schema_hash", "")) != expected_tool_schema_hash:
        return {"ok": False, "error": "provider report tool_schema_hash mismatch"}
    if str(report.get("proxy_version_hash", "")) != expected_proxy_hash:
        return {"ok": False, "error": "provider report proxy_version_hash mismatch"}

    expires_at = str(report.get("expires_at", "")).strip()
    if expires_at:
        expires_at_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        if expires_at_dt <= datetime.now(timezone.utc):
            return {"ok": False, "error": "provider report has expired"}

    primary = report.get("primary") or {}
    if not bool(primary.get("formal_ready", False)):
        return {"ok": False, "error": "primary provider is not formal-ready"}

    if config.provider.require_fallback:
        fallback = report.get("fallback") or {}
        if not fallback:
            return {"ok": False, "error": "provider report is missing required fallback section"}
        if not bool(fallback.get("formal_ready", False)):
            return {"ok": False, "error": "required fallback provider is not formal-ready"}

    return {
        "ok": True,
        "data": {
            "formal_ready": bool(report.get("formal_ready", False)),
            "primary_ready": bool(primary.get("formal_ready", False)),
            "fallback_ready": bool((report.get("fallback") or {}).get("formal_ready", False)),
        },
    }
