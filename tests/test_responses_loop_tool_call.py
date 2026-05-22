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


def test_formal_run_passes_function_call_output_into_the_next_turn(workspace: Path) -> None:
    provider_report_path = workspace / "provider_report.json"
    provider_report_path.write_text(json.dumps(_provider_report_payload()), encoding="utf-8")
    client = FakeResponsesClient(
        [
            _turn(
                "resp_write",
                [
                    {
                        "type": "function_call",
                        "name": "write_file",
                        "call_id": "call_1",
                        "arguments": json.dumps(
                            {
                                "path": "submission/code/hello.py",
                                "content": "print('hello from loop')\n",
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
    assert (workspace / "submission" / "code" / "hello.py").read_text(encoding="utf-8") == "print('hello from loop')\n"
    second_call_input = client.calls[1]["input"]
    assert any(item.get("type") == "function_call_output" for item in second_call_input)


def test_formal_run_records_shared_code_provenance_for_write_file_tool_calls(workspace: Path) -> None:
    provider_report_path = workspace / "provider_report.json"
    provider_report_path.write_text(json.dumps(_provider_report_payload()), encoding="utf-8")
    client = FakeResponsesClient(
        [
            _turn(
                "resp_write",
                [
                    {
                        "type": "function_call",
                        "name": "write_file",
                        "call_id": "call_1",
                        "arguments": json.dumps(
                            {
                                "path": "submission/code/model.py",
                                "content": "print('shared model')\n",
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
    manifest_entries = json.loads((workspace / "submission" / "code_manifest.json").read_text(encoding="utf-8"))
    assert manifest_entries == [
        {
            "path": "submission/code/model.py",
            "sha256": manifest_entries[0]["sha256"],
            "size": len("print('shared model')\n".encode("utf-8")),
            "code_version": manifest_entries[0]["code_version"],
            "originating_task": "task1",
            "patch_id": manifest_entries[0]["patch_id"],
            "step_id": manifest_entries[0]["step_id"],
            "task_id": "task1",
            "timestamp": manifest_entries[0]["timestamp"],
            "llm_call_ids": ["resp_write"],
        }
    ]
    assert manifest_entries[0]["code_version"].startswith("sha256:")
    assert manifest_entries[0]["patch_id"] == "patch-call_1"

    log_lines = [
        json.loads(line)
        for line in (workspace / "llm_logs" / "all_llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(line.get("tool_calls") for line in log_lines)
    assert any(line.get("response") == "FORMAL_DONE" for line in log_lines)
