from __future__ import annotations

import json
from pathlib import Path

from cozy_pde_v3.proxy_logs import merge_provider_logs, redact_proxy_entry, write_redacted_proxy_entry


def test_redact_proxy_entry_redacts_secrets_but_keeps_raw_response_shape() -> None:
    entry = {
        "provider": "primary",
        "request_headers": {"Authorization": "Bearer sk-secret"},
        "request_body": {
            "input": [{"role": "user", "content": "OPENAI_API_KEY=sk-live-key"}],
            "tool_call": {
                "name": "write_file",
                "arguments": '{"path":"/home/alice/work/app.py","api_key":"ghp_secret"}',
            },
        },
        "response_body": {
            "output": [
                {
                    "type": "function_call",
                    "name": "write_file",
                    "call_id": "call-1",
                    "arguments": '{"path":"submission/code/app.py","content":"print(\\"ok\\")"}',
                }
            ]
        },
    }

    redacted = redact_proxy_entry(entry)
    payload = json.dumps(redacted, ensure_ascii=False)

    assert redacted["provider"] == "primary"
    assert redacted["request_headers"]["Authorization"] == "[REDACTED]"
    assert "sk-live-key" not in payload
    assert "ghp_secret" not in payload
    assert "/home/alice/" not in payload
    assert "/home/[USER]/work/app.py" in payload
    assert redacted["response_body"]["output"][0]["type"] == "function_call"
    assert redacted["response_body"]["output"][0]["call_id"] == "call-1"


def test_write_redacted_proxy_entry_persists_jsonl_log(tmp_path: Path) -> None:
    entry = {
        "timestamp": "2026-05-22T00:00:00+00:00",
        "provider": "fallback",
        "response": "normal output",
        "request_body": {"message": "DEEPSEEK_API_KEY=hf-secret"},
    }

    log_path = write_redacted_proxy_entry(tmp_path, entry)
    lines = log_path.read_text(encoding="utf-8").splitlines()
    written = json.loads(lines[0])

    assert written["provider"] == "fallback"
    assert written["response"] == "normal output"
    assert "hf-secret" not in json.dumps(written, ensure_ascii=False)


def test_merge_provider_logs_sorts_across_primary_and_fallback_dirs(tmp_path: Path) -> None:
    primary_dir = tmp_path / "primary"
    fallback_dir = tmp_path / "fallback"
    primary_dir.mkdir()
    fallback_dir.mkdir()
    (primary_dir / "llm-20260522.jsonl").write_text(
        json.dumps({"timestamp": "2026-05-22T00:00:02+00:00", "provider": "primary", "id": 3}) + "\n"
        + json.dumps({"timestamp": "2026-05-22T00:00:00+00:00", "provider": "primary", "id": 1})
        + "\n",
        encoding="utf-8",
    )
    (fallback_dir / "llm-20260522.jsonl").write_text(
        json.dumps({"timestamp": "2026-05-22T00:00:01+00:00", "provider": "fallback", "id": 2}) + "\n",
        encoding="utf-8",
    )

    merged = merge_provider_logs(primary_dir, fallback_dir)

    assert [entry["id"] for entry in merged] == [1, 2, 3]
    assert [entry["provider"] for entry in merged] == ["primary", "fallback", "primary"]
