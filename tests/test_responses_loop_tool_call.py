from __future__ import annotations

import json

from agent_runner.config import RunnerConfig
from agent_runner.main import execute_agent_loop
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
