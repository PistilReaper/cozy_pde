from __future__ import annotations

import json

from agent_runner.tools.validate_tools import validate_responses_logs


def _write_log(path, payloads):
    path.write_text(
        "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in payloads),
        encoding="utf-8",
    )


def _base_payload(tool_calls):
    return {
        "timestamp": "2026-05-17T00:00:00+00:00",
        "elapsed_seconds": 0.1,
        "model": "gpt-5.4",
        "profile": "coder",
        "phase": "implementation",
        "raw_response": {"id": "resp_1"},
        "tool_calls": tool_calls,
    }


def test_validate_responses_logs_fails_for_untraced_code_file(workspace):
    code_file = workspace / "submission" / "code" / "generated.py"
    code_file.write_text("print('from disk')\n", encoding="utf-8")
    log_path = workspace / "llm_logs" / "all_llm_calls.jsonl"
    _write_log(
        log_path,
        [
            _base_payload(
                [
                    {
                        "name": "write_file",
                        "call_id": "call_1",
                        "arguments": {
                            "path": "submission/code/other.py",
                            "content": "print('other')\n",
                        },
                    }
                ]
            )
        ],
    )

    result = validate_responses_logs(log_path, workspace_root=workspace)

    assert result["ok"] is False
    assert "untraced" in result["error"].lower()
    assert result["data"]["untraced_files"] == ["submission/code/generated.py"]


def test_validate_responses_logs_fails_for_write_file_without_content(workspace):
    log_path = workspace / "llm_logs" / "all_llm_calls.jsonl"
    _write_log(
        log_path,
        [
            _base_payload(
                [
                    {
                        "name": "write_file",
                        "call_id": "call_1",
                        "arguments": {
                            "path": "submission/code/generated.py",
                        },
                    }
                ]
            )
        ],
    )

    result = validate_responses_logs(log_path, workspace_root=workspace)

    assert result["ok"] is False
    assert "missing content" in result["error"].lower()


def test_validate_responses_logs_accepts_traced_code_file(workspace):
    code_file = workspace / "submission" / "code" / "generated.py"
    code_file.write_text("print('traced')\n", encoding="utf-8")
    log_path = workspace / "llm_logs" / "all_llm_calls.jsonl"
    _write_log(
        log_path,
        [
            _base_payload(
                [
                    {
                        "name": "write_file",
                        "call_id": "call_1",
                        "arguments": {
                            "path": "submission/code/generated.py",
                            "content": "print('traced')\n",
                        },
                    }
                ]
            )
        ],
    )

    result = validate_responses_logs(log_path, workspace_root=workspace)

    assert result["ok"] is True
    assert result["data"]["traced_write_paths"] == ["submission/code/generated.py"]


def test_validate_responses_logs_fails_for_content_mismatch(workspace):
    code_file = workspace / "submission" / "code" / "generated.py"
    code_file.write_text("print('from disk')\n", encoding="utf-8")
    log_path = workspace / "llm_logs" / "all_llm_calls.jsonl"
    _write_log(
        log_path,
        [
            _base_payload(
                [
                    {
                        "name": "write_file",
                        "call_id": "call_1",
                        "arguments": {
                            "path": "submission/code/generated.py",
                            "content": "print('from log')\n",
                        },
                    }
                ]
            )
        ],
    )

    result = validate_responses_logs(log_path, workspace_root=workspace)

    assert result["ok"] is False
    assert "mismatch" in result["error"].lower()
    assert result["data"]["content_mismatch_files"] == ["submission/code/generated.py"]


def test_validate_responses_logs_normalizes_workspace_prefixed_write_paths(workspace):
    code_file = workspace / "submission" / "code" / "generated.py"
    code_file.write_text("print('traced')\n", encoding="utf-8")
    log_path = workspace / "llm_logs" / "all_llm_calls.jsonl"
    _write_log(
        log_path,
        [
            _base_payload(
                [
                    {
                        "name": "write_file",
                        "call_id": "call_1",
                        "arguments": {
                            "path": "workspace/submission/code/generated.py",
                            "content": "print('traced')\n",
                        },
                    }
                ]
            )
        ],
    )

    result = validate_responses_logs(log_path, workspace_root=workspace)

    assert result["ok"] is True
    assert result["data"]["traced_write_paths"] == ["submission/code/generated.py"]


def test_validate_responses_logs_uses_final_logged_content_per_path(workspace):
    code_file = workspace / "submission" / "code" / "generated.py"
    code_file.write_text("print('final')\n", encoding="utf-8")
    log_path = workspace / "llm_logs" / "all_llm_calls.jsonl"
    _write_log(
        log_path,
        [
            _base_payload(
                [
                    {
                        "name": "write_file",
                        "call_id": "call_1",
                        "arguments": {
                            "path": "submission/code/generated.py",
                            "content": "print('first')\n",
                        },
                    }
                ]
            ),
            _base_payload(
                [
                    {
                        "name": "write_file",
                        "call_id": "call_2",
                        "arguments": {
                            "path": "submission/code/generated.py",
                            "content": "print('final')\n",
                        },
                    }
                ]
            ),
        ],
    )

    result = validate_responses_logs(log_path, workspace_root=workspace)

    assert result["ok"] is True
    assert result["data"]["traced_write_paths"] == ["submission/code/generated.py"]
