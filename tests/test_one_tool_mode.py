from __future__ import annotations

import json
from pathlib import Path

from cozy_pde_v3.agent_loop import run_formal_task_session
from cozy_pde_v3.task_specs import DEFAULT_TASK_SPECS


class FakeResponsesClient:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        if not self.responses:
            raise AssertionError("no queued fake response")
        return dict(self.responses.pop(0))


def _provider_report_payload() -> dict[str, object]:
    return {
        "formal_ready": True,
        "primary": {
            "provider": "primary",
            "model_id": "gpt-5.4",
            "formal_ready": True,
        },
        "forced_failover": {"required": False},
    }


def _turn(response_id: str, output_items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "provider": "primary",
        "model": "gpt-5.4",
        "raw_response": {"id": response_id, "model": "gpt-5.4", "output": output_items},
        "standard_output_items": output_items,
        "provider_output_items": output_items,
        "usage": {"total_tokens": 10},
    }


def _config_stub(workspace: Path) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        workspace_root=workspace,
        task_specs=DEFAULT_TASK_SPECS,
    )


def test_formal_run_requests_non_parallel_tool_calls_from_responses_client(workspace: Path) -> None:
    provider_report_path = workspace / "provider_report.json"
    provider_report_path.write_text(json.dumps(_provider_report_payload()), encoding="utf-8")
    client = FakeResponsesClient(
        [
            _turn(
                "resp_done",
                [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "FORMAL_DONE"}],
                    }
                ],
            )
        ]
    )

    result = run_formal_task_session(
        config=_config_stub(workspace),
        task="task1",
        provider_report_path=provider_report_path,
        responses_client=client,
    )

    assert result["ok"] is True
    assert client.calls
    assert all(call["parallel_tool_calls"] is False for call in client.calls)


def test_formal_run_rejects_multi_tool_turns_without_executing_side_effects(workspace: Path) -> None:
    provider_report_path = workspace / "provider_report.json"
    provider_report_path.write_text(json.dumps(_provider_report_payload()), encoding="utf-8")
    client = FakeResponsesClient(
        [
            _turn(
                "resp_multi",
                [
                    {
                        "type": "function_call",
                        "name": "write_file",
                        "call_id": "call_1",
                        "arguments": json.dumps(
                            {
                                "path": "submission/code/alpha.py",
                                "content": "print('alpha')\n",
                            }
                        ),
                    },
                    {
                        "type": "function_call",
                        "name": "write_file",
                        "call_id": "call_2",
                        "arguments": json.dumps(
                            {
                                "path": "submission/code/beta.py",
                                "content": "print('beta')\n",
                            }
                        ),
                    },
                ],
            ),
            _turn(
                "resp_single",
                [
                    {
                        "type": "function_call",
                        "name": "write_file",
                        "call_id": "call_3",
                        "arguments": json.dumps(
                            {
                                "path": "submission/code/final.py",
                                "content": "print('final')\n",
                            }
                        ),
                    }
                ],
            ),
            _turn(
                "resp_done",
                [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "FORMAL_DONE"}],
                    }
                ],
            ),
        ]
    )

    result = run_formal_task_session(
        config=_config_stub(workspace),
        task="task1",
        provider_report_path=provider_report_path,
        responses_client=client,
    )

    assert result["ok"] is True
    assert not (workspace / "submission" / "code" / "alpha.py").exists()
    assert not (workspace / "submission" / "code" / "beta.py").exists()
    assert (workspace / "submission" / "code" / "final.py").read_text(encoding="utf-8") == "print('final')\n"
    retry_input = client.calls[1]["input"]
    assert any(
        item.get("role") == "user"
        and "at most one function call" in json.dumps(item.get("content", []), ensure_ascii=False)
        for item in retry_input
    )
