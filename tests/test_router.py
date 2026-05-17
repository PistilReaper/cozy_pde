from __future__ import annotations

import json

from agent_runner.config import RunnerConfig
from agent_runner.logger import LLMCallLogger
from agent_runner.router import Router


class FakeResponsesClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


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


def test_router_uses_llm_json_when_valid(workspace):
    config = RunnerConfig.from_workspace(workspace)
    client = FakeResponsesClient(
        [
            _message_response(
                json.dumps(
                    {
                        "profile": "coder",
                        "phase": "implementation",
                        "enable_hosted_tools": False,
                        "reason": "Need to patch local code.",
                    }
                )
            )
        ]
    )
    router = Router(
        client=client,
        config=config,
        llm_logger=LLMCallLogger(workspace / "llm_logs" / "all_llm_calls.jsonl"),
    )

    decision = router.choose(
        summary="Need to modify generated training code after a shape validation failure.",
        task_id="autonomous",
        step_id="step-001",
        phase_hint="implementation",
    )

    assert decision.profile == "coder"
    assert decision.phase == "implementation"
    assert decision.enable_hosted_tools is False
    assert client.calls[0]["profile"].name == "router"


def test_router_falls_back_to_deterministic_rule_on_invalid_json(workspace):
    config = RunnerConfig.from_workspace(workspace)
    client = FakeResponsesClient([_message_response("not-json")])
    router = Router(
        client=client,
        config=config,
        llm_logger=LLMCallLogger(workspace / "llm_logs" / "all_llm_calls.jsonl"),
    )

    decision = router.choose(
        summary="Need to validate final JSON bundle before packaging.",
        task_id="autonomous",
        step_id="step-002",
        phase_hint="validation",
    )

    assert decision.profile == "json_judge"
    assert decision.phase == "validation"
    assert decision.enable_hosted_tools is False
