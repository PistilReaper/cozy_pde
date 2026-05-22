from __future__ import annotations

import json
from pathlib import Path

from cozy_pde_v3.proxy_logs import merge_provider_logs, redact_proxy_entry
from scripts.proxy import write_llm_log


def test_redact_proxy_entry_redacts_headers_tokens_env_vars_and_home_usernames() -> None:
    entry = {
        "request_headers": {
            "Authorization": "Bearer sk-live-secret-token",
            "X-Test": "api_key=plain-secret api-key: second-secret api key third-secret",
        },
        "request_body": {
            "env": "OPENAI_API_KEY=sk-openai-secret\nDEEPSEEK_API_KEY=hf_deepseek_secret",
            "message": "Use sk-inline-secret, hf_inline_secret, ghp_inline_secret, github_pat_inline_secret here.",
            "path": "/home/alice/work/project",
        },
    }

    redacted = redact_proxy_entry(entry)
    payload = json.dumps(redacted, ensure_ascii=False)

    assert redacted["request_headers"]["Authorization"] == "[REDACTED]"
    assert "plain-secret" not in payload
    assert "second-secret" not in payload
    assert "third-secret" not in payload
    assert "sk-openai-secret" not in payload
    assert "hf_deepseek_secret" not in payload
    assert "sk-inline-secret" not in payload
    assert "hf_inline_secret" not in payload
    assert "ghp_inline_secret" not in payload
    assert "github_pat_inline_secret" not in payload
    assert "/home/alice/" not in payload
    assert "/home/[USER]/work/project" in payload
    assert "api_key=[REDACTED]" in payload
    assert "api-key: [REDACTED]" in payload
    assert "api key [REDACTED]" in payload
    assert "OPENAI_API_KEY=[REDACTED]" in payload
    assert "DEEPSEEK_API_KEY=[REDACTED]" in payload


def test_redact_proxy_entry_preserves_provenance_relevant_payload_structure() -> None:
    entry = {
        "provider": "openai",
        "target": "https://api.openai.com/v1/responses",
        "request_body": {
            "messages": [
                {"role": "user", "content": "Summarize the plan."},
                {
                    "role": "assistant",
                    "content": "I will call a tool, then return `print('hello')`.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": (
                                    '{"path":"/home/bob/work/app.py","contents":"print(\\"hello\\")",'
                                    '"api_key":"ghp_tool_secret"}'
                                ),
                            },
                        }
                    ],
                },
            ]
        },
        "response_body": {
            "output_text": "Here is normal assistant output.",
            "code_example": 'if prefix == "sk":\n    print("not a secret by itself")',
        },
    }

    redacted = redact_proxy_entry(entry)

    assert redacted["provider"] == "openai"
    assert redacted["target"] == "https://api.openai.com/v1/responses"
    assert redacted["request_body"]["messages"][0]["content"] == "Summarize the plan."
    assert "I will call a tool" in redacted["request_body"]["messages"][1]["content"]
    assert "`print('hello')`" in redacted["request_body"]["messages"][1]["content"]
    tool_call = redacted["request_body"]["messages"][1]["tool_calls"][0]
    assert tool_call["id"] == "call_1"
    assert tool_call["function"]["name"] == "write_file"
    assert '"contents"' in tool_call["function"]["arguments"]
    assert "hello" in tool_call["function"]["arguments"]
    assert "/home/[USER]/work/app.py" in tool_call["function"]["arguments"]
    assert "ghp_tool_secret" not in tool_call["function"]["arguments"]
    assert redacted["response_body"]["output_text"] == "Here is normal assistant output."
    assert 'if prefix == "sk":' in redacted["response_body"]["code_example"]


def test_write_llm_log_writes_redacted_structured_entry(tmp_path: Path) -> None:
    write_llm_log(
        str(tmp_path),
        {
            "timestamp": "2026-05-21T12:00:00+00:00",
            "provider": "deepseek",
            "target": "https://api.deepseek.com/v1/chat/completions",
            "request_id": "req-123",
            "request_body": {
                "messages": [{"role": "user", "content": "Use OPENAI_API_KEY=sk-secret"}],
            },
            "response_body": {
                "choices": [{"message": {"content": "Normal assistant text remains."}}],
            },
            "response": "Normal assistant text remains.",
        },
    )

    log_files = list(tmp_path.glob("llm-*.jsonl"))
    assert len(log_files) == 1

    written = json.loads(log_files[0].read_text(encoding="utf-8").splitlines()[0])

    assert written["provider"] == "deepseek"
    assert written["target"] == "https://api.deepseek.com/v1/chat/completions"
    assert written["request_id"] == "req-123"
    assert written["response"] == "Normal assistant text remains."
    assert written["response_body"]["choices"][0]["message"]["content"] == "Normal assistant text remains."
    assert "sk-secret" not in json.dumps(written, ensure_ascii=False)
    assert "OPENAI_API_KEY=[REDACTED]" in json.dumps(written, ensure_ascii=False)


def test_merge_provider_logs_returns_timestamp_sorted_entries(tmp_path: Path) -> None:
    primary_dir = tmp_path / "aixj"
    fallback_dir = tmp_path / "deepseek"
    primary_dir.mkdir()
    fallback_dir.mkdir()

    (primary_dir / "llm-20260521.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"timestamp": "2026-05-21T12:00:02+00:00", "provider": "aixj", "response": "third"}),
                json.dumps({"timestamp": "2026-05-21T12:00:00+00:00", "provider": "aixj", "response": "first"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (fallback_dir / "llm-20260521.jsonl").write_text(
        json.dumps({"timestamp": "2026-05-21T12:00:01+00:00", "provider": "deepseek", "response": "second"}) + "\n",
        encoding="utf-8",
    )

    merged = merge_provider_logs(primary_dir, fallback_dir)

    assert [entry["response"] for entry in merged] == ["first", "second", "third"]
    assert [entry["provider"] for entry in merged] == ["aixj", "deepseek", "aixj"]
