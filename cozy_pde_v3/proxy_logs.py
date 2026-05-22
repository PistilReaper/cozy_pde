from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


REDACTED = "[REDACTED]"
USER_REDACTED = "[USER]"

_GENERIC_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api(?:[_ -]?key))\b(?:\s*([=:])\s*|\s+)([^\s,;]+)"
)
_ENV_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(OPENAI_API_KEY|DEEPSEEK_API_KEY)\s*=\s*([^\s,;]+)"
)
_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]+|hf_[A-Za-z0-9_-]+|ghp_[A-Za-z0-9_-]+|github_pat_[A-Za-z0-9_-]+)\b"
)
_HOME_PATH_RE = re.compile(r"(/home/)([^/\s]+)")


def llm_log_path(log_dir: str | Path, now: datetime | None = None) -> Path:
    timestamp = now or datetime.now()
    return Path(log_dir) / f"llm-{timestamp.strftime('%Y%m%d')}.jsonl"


def redact_secret_text(text: str) -> str:
    text = _ENV_SECRET_ASSIGNMENT_RE.sub(r"\1=" + REDACTED, text)

    def _replace_assignment(match: re.Match[str]) -> str:
        key = match.group(1)
        separator = match.group(2)
        if separator == "=":
            return f"{key}={REDACTED}"
        if separator:
            return f"{key}{separator} {REDACTED}"
        return f"{key} {REDACTED}"

    text = _GENERIC_SECRET_ASSIGNMENT_RE.sub(_replace_assignment, text)
    text = _TOKEN_RE.sub(REDACTED, text)
    text = _HOME_PATH_RE.sub(r"\1" + USER_REDACTED, text)
    return text


def redact_proxy_entry(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and key.lower() == "authorization":
                redacted[key] = REDACTED
            else:
                redacted[key] = redact_proxy_entry(item)
        return redacted
    if isinstance(value, list):
        return [redact_proxy_entry(item) for item in value]
    if isinstance(value, str):
        return redact_secret_text(value)
    return value


def write_redacted_proxy_entry(log_dir: str | Path, entry: dict[str, Any]) -> Path:
    log_file = llm_log_path(log_dir)
    redacted_entry = redact_proxy_entry(entry)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(redacted_entry, ensure_ascii=False) + "\n")
    return log_file


def merge_provider_logs(primary_dir: str | Path, fallback_dir: str | Path | None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for source_dir in [primary_dir, fallback_dir]:
        if source_dir is None:
            continue
        for log_file in sorted(Path(source_dir).glob("llm-*.jsonl")):
            for line in log_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entries.append(json.loads(line))
    entries.sort(key=lambda entry: str(entry.get("timestamp", "")))
    return entries
