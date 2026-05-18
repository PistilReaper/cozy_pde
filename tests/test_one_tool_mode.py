from __future__ import annotations

import json

from agent_runner.config import RunnerConfig
from agent_runner.main import execute_agent_loop
from agent_runner.logger import ToolCallLogger
from agent_runner.responses_items import system_text, user_text
from agent_runner.tool_registry import build_tool_registry
from agent_runner.tools.validate_tools import validate_responses_logs


class GatewayError(RuntimeError):
    status_code = 502


def _router_response(profile: str = "coder", phase: str = "implementation") -> dict:
    return {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(
                            {
                                "profile": profile,
                                "phase": phase,
                                "enable_hosted_tools": False,
                                "reason": "test route",
                            }
                        ),
                    }
                ],
            }
        ]
    }


def _message_response(text: str) -> dict:
    return {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ]
    }


def _final_response(message: str = "RUNNER_FINALIZED") -> dict:
    return _message_response(json.dumps({"type": "final", "message": message}))


class SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        next_item = self.responses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


def test_loop_executes_single_tool_call(workspace):
    config = RunnerConfig.from_workspace(workspace)
    client = SequenceClient(
        [
            _router_response(),
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "write_file",
                        "arguments": json.dumps(
                            {
                                "path": "submission/code/hello.py",
                                "content": "print('hello one tool')\n",
                            }
                        ),
                    }
                ]
            },
            _router_response(),
            _final_response(),
        ]
    )

    ok, ledger, last_text = execute_agent_loop(
        config=config,
        initial_items=[system_text("You are a runner."), user_text("Create hello.py and finish.")],
        task_id="single_tool_mode",
        max_steps=3,
        client=client,
    )

    assert ok is True
    assert last_text == "RUNNER_FINALIZED"
    assert (workspace / "submission" / "code" / "hello.py").exists()
    assert sum(1 for item in ledger if item["type"] == "function_call_output") == 1


def test_loop_rejects_multiple_tool_calls_without_execution(workspace):
    config = RunnerConfig.from_workspace(workspace)
    client = SequenceClient(
        [
            _router_response(),
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "write_file",
                        "arguments": json.dumps(
                            {
                                "path": "submission/code/first.py",
                                "content": "print('first')\n",
                            }
                        ),
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "write_file",
                        "arguments": json.dumps(
                            {
                                "path": "submission/code/second.py",
                                "content": "print('second')\n",
                            }
                        ),
                    },
                ]
            },
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_3",
                        "name": "write_file",
                        "arguments": json.dumps(
                            {
                                "path": "submission/code/final.py",
                                "content": "print('final')\n",
                            }
                        ),
                    }
                ]
            },
            _router_response(),
            _final_response(),
        ]
    )

    ok, ledger, _ = execute_agent_loop(
        config=config,
        initial_items=[system_text("You are a runner."), user_text("Create one file and finish.")],
        task_id="multi_tool_violation",
        max_steps=3,
        client=client,
    )

    assert ok is True
    assert not (workspace / "submission" / "code" / "first.py").exists()
    assert not (workspace / "submission" / "code" / "second.py").exists()
    assert (workspace / "submission" / "code" / "final.py").exists()
    assert sum(1 for item in ledger if item["type"] == "function_call_output") == 1


def test_multi_tool_violation_retry_prompt(workspace):
    config = RunnerConfig.from_workspace(workspace)
    client = SequenceClient(
        [
            _router_response(),
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "write_file",
                        "arguments": json.dumps({"path": "submission/code/a.py", "content": "print('a')\n"}),
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "write_file",
                        "arguments": json.dumps({"path": "submission/code/b.py", "content": "print('b')\n"}),
                    },
                ]
            },
            _final_response(),
        ]
    )

    execute_agent_loop(
        config=config,
        initial_items=[system_text("You are a runner."), user_text("Create one file and finish.")],
        task_id="retry_prompt",
        max_steps=2,
        client=client,
    )

    retry_call = client.calls[2]
    text_chunks = [
        chunk["text"]
        for item in retry_call["input_items"]
        for chunk in item.get("content", [])
        if "text" in chunk
    ]
    assert any("exactly one action" in text.lower() or "json action" in text.lower() for text in text_chunks)


def test_provider_502_with_tools_reports_gateway_error_without_executing_tools(workspace):
    config = RunnerConfig.from_workspace(workspace)
    client = SequenceClient(
        [
            _router_response(),
            GatewayError("502 gateway error"),
            GatewayError("502 gateway error"),
        ]
    )

    ok, _, last_text = execute_agent_loop(
        config=config,
        initial_items=[system_text("You are a runner."), user_text("Create one file and finish.")],
        task_id="provider_gateway",
        max_steps=1,
        client=client,
    )

    assert ok is False
    assert "provider_multi_tool_or_gateway_error" in last_text
    assert not (workspace / "submission" / "code" / "hello.py").exists()
    tool_log_lines = (workspace / "internal_logs" / "tool_calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert all('"tool_name": "write_file"' not in line for line in tool_log_lines)


def test_composite_tool_still_counts_as_single_tool_call(workspace):
    config = RunnerConfig.from_workspace(workspace)
    registry = build_tool_registry(
        config,
        ToolCallLogger(workspace / "internal_logs" / "tool_calls.jsonl"),
    )
    client = SequenceClient(
        [
            _router_response(profile="json_judge", phase="finalization"),
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "validate_full_submission",
                        "arguments": json.dumps(
                            {
                                "submission_dir": "submission",
                                "responses_log_path": "llm_logs/all_llm_calls.jsonl",
                            }
                        ),
                    }
                ]
            },
            _router_response(profile="json_judge", phase="finalization"),
            _final_response(),
        ]
    )

    ok, ledger, _ = execute_agent_loop(
        config=config,
        initial_items=[system_text("You are a runner."), user_text("Validate and finish.")],
        task_id="composite_tool",
        max_steps=3,
        client=client,
        registry=registry,
    )

    assert ok is True
    assert sum(1 for item in ledger if item["type"] == "function_call_output") == 1
    lines = (workspace / "internal_logs" / "tool_calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert sum('"tool_name": "validate_full_submission"' in line for line in lines) == 1
    assert sum('"tool_name": "validate_submission"' in line for line in lines) == 0
    assert sum('"tool_name": "validate_responses_logs"' in line for line in lines) == 0


def test_write_file_provenance_still_passes_single_tool_mode(workspace):
    config = RunnerConfig.from_workspace(workspace)
    client = SequenceClient(
        [
            _router_response(),
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "write_file",
                        "arguments": json.dumps(
                            {
                                "path": "submission/code/provenance.py",
                                "content": "print('provenance')\n",
                            }
                        ),
                    }
                ]
            },
            _router_response(),
            _final_response(),
        ]
    )

    ok, _, _ = execute_agent_loop(
        config=config,
        initial_items=[system_text("You are a runner."), user_text("Create provenance.py and finish.")],
        task_id="single_tool_provenance",
        max_steps=3,
        client=client,
    )

    assert ok is True
    assert validate_responses_logs(config.llm_log_path, workspace_root=workspace)["ok"] is True
