from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cozy_pde_v3.agent_loop import run_formal_task_session
from cozy_pde_v3.memory_store import MemoryStore
from cozy_pde_v3.task_specs import DEFAULT_TASK_SPECS
from cozy_pde_v3.validation.logs import load_jsonl_records, validate_jsonl_logs


class FakeResponsesClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("no queued fake response")
        return self._responses.pop(0)


def _config_stub(workspace: Path) -> Any:
    return SimpleNamespace(
        workspace_root=workspace,
        task_specs=DEFAULT_TASK_SPECS,
    )


def _provider_report_payload(*, formal_ready: bool) -> dict[str, Any]:
    return {
        "formal_ready": formal_ready,
        "primary": {
            "provider": "primary",
            "model_id": "gpt-5.4",
            "formal_ready": formal_ready,
        },
        "forced_failover": {"required": False},
    }


def _write_provider_report(workspace: Path, *, formal_ready: bool) -> Path:
    path = workspace / "provider_report.json"
    path.write_text(
        json.dumps(_provider_report_payload(formal_ready=formal_ready), ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _turn(
    *,
    response_id: str,
    output_items: list[dict[str, Any]],
    provider: str = "primary",
    model: str = "gpt-5.4",
) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "raw_response": {"id": response_id, "model": model, "output": output_items},
        "standard_output_items": output_items,
        "provider_output_items": output_items,
        "usage": {"total_tokens": 10},
    }


def test_formal_run_raises_when_provider_capability_report_is_missing(workspace: Path) -> None:
    missing_report = workspace / "missing_provider_report.json"

    with pytest.raises(FileNotFoundError, match="provider capability report does not exist"):
        run_formal_task_session(
            config=_config_stub(workspace),
            task="task1",
            provider_report_path=missing_report,
            responses_client=None,
        )


def test_formal_run_consumes_provider_readiness_and_blocks_when_not_formal_ready(
    workspace: Path,
) -> None:
    provider_report_path = _write_provider_report(workspace, formal_ready=False)

    result = run_formal_task_session(
        config=_config_stub(workspace),
        task="task1",
        provider_report_path=provider_report_path,
        responses_client=None,
    )

    assert result["ok"] is False
    assert "formal-ready" in result["error"]
    assert result["state"]["current_phase"] == "capability_readiness"
    decisions = MemoryStore(result["memory_db_path"]).list_decision_records()
    assert decisions[0]["selected_phase"] == "capability_readiness"
    assert decisions[0]["outcome"] == "blocked"


def test_formal_run_rejects_multi_task_input(workspace: Path) -> None:
    result = run_formal_task_session(
        config=_config_stub(workspace),
        task="task1,task2",
        provider_report_path=workspace / "provider_report.json",
        responses_client=None,
    )

    assert result["ok"] is False
    assert "exactly one task" in result["error"]


def test_formal_run_executes_local_function_tool_and_continues_with_function_call_output(
    workspace: Path,
) -> None:
    provider_report_path = _write_provider_report(workspace, formal_ready=True)
    client = FakeResponsesClient(
        [
            _turn(
                response_id="resp_tool",
                output_items=[
                    {
                        "type": "function_call",
                        "name": "write_file",
                        "call_id": "call_write_1",
                        "arguments": json.dumps(
                            {
                                "path": "submission/code/model.py",
                                "content": "print('hello from integration')\n",
                            }
                        ),
                    }
                ],
            ),
            _turn(
                response_id="resp_final",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "FORMAL_RUN_COMPLETE",
                            }
                        ],
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
    assert result["final_text"] == "FORMAL_RUN_COMPLETE"
    assert result["state"]["last_tool_call_id"] == "call_write_1"
    assert result["state"]["last_llm_call_id"] == "resp_final"
    assert result["state"]["shared_code_version"].startswith("sha256:")
    assert (workspace / "submission" / "code" / "model.py").read_text(encoding="utf-8") == "print('hello from integration')\n"

    assert len(client.calls) == 2
    assert all(tool["type"] == "function" for tool in client.calls[0]["tools"])
    continuation_items = client.calls[1]["input"]
    function_outputs = [item for item in continuation_items if item.get("type") == "function_call_output"]
    assert len(function_outputs) == 1
    assert function_outputs[0]["call_id"] == "call_write_1"
    function_output_payload = json.loads(function_outputs[0]["output"])
    assert function_output_payload["ok"] is True
    assert function_output_payload["path"] == "submission/code/model.py"

    llm_log_path = Path(result["llm_log_path"])
    assert validate_jsonl_logs(llm_log_path)["ok"] is True
    llm_records = load_jsonl_records(llm_log_path)
    assert len(llm_records) == 2
    assert llm_records[0]["tool_calls"][0]["name"] == "write_file"
    assert llm_records[0]["tool_calls"][0]["call_id"] == "call_write_1"
    assert llm_records[1]["response"] == "FORMAL_RUN_COMPLETE"

    store = MemoryStore(result["memory_db_path"])
    snapshots = store.list_code_snapshots()
    patches = store.list_patch_records()
    decisions = store.list_decision_records()
    assert len(snapshots) == 2
    assert len(patches) == 1
    assert len(decisions) >= 3
    assert patches[0]["llm_call_ids"] == ["resp_tool"]
    assert patches[0]["changed_files"] == ["submission/code/model.py"]
    assert decisions[-1]["outcome"] == "finalized"
