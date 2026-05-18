from __future__ import annotations

import json

from agent_runner.config import RunnerConfig
from agent_runner.main import execute_agent_loop, run_provider_health_check
from agent_runner.responses_items import system_text, user_text
from agent_runner.tools.validate_tools import validate_jsonl_logs, validate_responses_logs


class FakeLoopResponsesClient:
    def __init__(self):
        self.model_calls = []
        self.main_call_count = 0

    def create(self, **kwargs):
        self.model_calls.append(kwargs)
        profile_name = kwargs["profile"].name
        if profile_name == "router":
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
                                        "profile": "coder",
                                        "phase": "implementation",
                                        "enable_hosted_tools": False,
                                        "reason": "Need local file write access.",
                                    }
                                ),
                            }
                        ],
                    }
                ]
            }

        self.main_call_count += 1
        if self.main_call_count == 1:
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
                                        "type": "action",
                                        "tool_name": "write_file",
                                        "arguments": {
                                            "path": "submission/code/hello.py",
                                            "content": "print('hello from responses')\n",
                                        },
                                    }
                                ),
                            }
                        ],
                    }
                ]
            }
        return {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({"type": "final", "message": "RUNNER_FINALIZED"}),
                        }
                    ],
                }
            ]
        }


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


def test_execute_agent_loop_records_function_call_outputs_and_valid_logs(workspace):
    config = RunnerConfig.from_workspace(workspace)
    client = FakeLoopResponsesClient()

    ok, ledger, last_text = execute_agent_loop(
        config=config,
        initial_items=[system_text("You are a runner."), user_text("Create hello.py and finish.")],
        task_id="autonomous",
        max_steps=3,
        client=client,
    )

    assert ok is True
    assert last_text == "RUNNER_FINALIZED"
    assert (workspace / "submission" / "code" / "hello.py").exists()
    assert any(item["type"] == "function_call_output" for item in ledger)
    assert validate_jsonl_logs(config.llm_log_path)["ok"] is True
    assert validate_responses_logs(config.llm_log_path, workspace_root=workspace)["ok"] is True


def test_execute_agent_loop_retries_concatenated_action_and_final_without_false_success(workspace):
    config = RunnerConfig.from_workspace(workspace)
    client = SequenceClient(
        [
            _router_response(),
            _message_response(
                '{"type":"action","tool_name":"write_file","arguments":{"path":"submission/code/bad.py","content":"print(1)\\n"}}'
                '{"type":"final","message":"RUNNER_FINALIZED"}'
            ),
            _message_response(json.dumps({"type": "final", "message": "RUNNER_FINALIZED"})),
        ]
    )

    ok, ledger, last_text = execute_agent_loop(
        config=config,
        initial_items=[system_text("You are a runner."), user_text("Finish only when valid.")],
        task_id="concatenated_action_final",
        max_steps=2,
        client=client,
    )

    assert ok is True
    assert last_text == "RUNNER_FINALIZED"
    assert not (workspace / "submission" / "code" / "bad.py").exists()
    assert not any(item["type"] == "function_call_output" for item in ledger)
    assert any(
        "not a valid action json object" in chunk["text"].lower()
        for item in ledger
        for chunk in item.get("content", [])
        if item["type"] == "message" and item.get("role") == "user" and "text" in chunk
    )


def test_execute_agent_loop_rejects_completion_token_only_text(workspace):
    config = RunnerConfig.from_workspace(workspace)
    client = SequenceClient(
        [
            _router_response(),
            _message_response("RUNNER_FINALIZED"),
            _message_response(json.dumps({"type": "final", "message": "RUNNER_FINALIZED"})),
        ]
    )

    ok, ledger, last_text = execute_agent_loop(
        config=config,
        initial_items=[system_text("You are a runner."), user_text("Finish only with valid final JSON.")],
        task_id="token_only_final_text",
        max_steps=2,
        client=client,
    )

    assert ok is True
    assert last_text == "RUNNER_FINALIZED"
    assert not any(item["type"] == "function_call_output" for item in ledger)
    assert any(
        "not a valid action json object" in chunk["text"].lower()
        for item in ledger
        for chunk in item.get("content", [])
        if item["type"] == "message" and item.get("role") == "user" and "text" in chunk
    )


def test_provider_health_check_requires_actual_write_file_execution(workspace, monkeypatch):
    config = RunnerConfig.from_workspace(workspace, project_root="/home/cty/cozy_pde")
    config.endpoint.api_key = "test-key"

    class FakeHealthClient:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            profile_name = kwargs["profile"].name
            if self.calls == 1 and profile_name == "coder":
                return _message_response("Confirmed.")
            if self.calls == 2 and profile_name == "coder":
                return _message_response(
                    '{"type":"action","tool_name":"echo_tool","arguments":{"text":"hello-tool"}}'
                    '{"type":"final","message":"PROVIDER_HEALTH_CHECK_COMPLETE"}'
                )
            if self.calls == 3 and profile_name == "coder":
                return _message_response(json.dumps({"type": "final", "message": "PROVIDER_HEALTH_CHECK_COMPLETE"}))
            if self.calls == 4 and profile_name == "coder":
                return _message_response(
                    '{"type":"action","tool_name":"write_file","arguments":{"path":"runs/scratch/provider_health_check.txt","content":"ok\\n"}}'
                    '{"type":"final","message":"PROVIDER_HEALTH_CHECK_COMPLETE"}'
                )
            if self.calls == 5 and profile_name == "coder":
                return _message_response(json.dumps({"type": "final", "message": "PROVIDER_HEALTH_CHECK_COMPLETE"}))
            raise AssertionError(f"Unexpected call {self.calls} for profile {profile_name}")

    monkeypatch.setattr("agent_runner.main.JsonActionClient", FakeHealthClient)

    exit_code = run_provider_health_check(config)

    assert exit_code == 1
    assert not (workspace / "runs" / "scratch" / "provider_health_check.txt").exists()
    assert (workspace / "internal_logs" / "tool_calls.jsonl").read_text(encoding="utf-8") == ""
