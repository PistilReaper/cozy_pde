from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from cozy_pde_v3 import cli as cli_module
from cozy_pde_v3.cli import main
from cozy_pde_v3.agent_loop import (
    run_formal_task_session,
    should_allow_finalize,
    should_start_formal_run,
)
from cozy_pde_v3.memory_store import MemoryStore
from cozy_pde_v3.task_specs import DEFAULT_TASK_SPECS
from cozy_pde_v3.validation.logs import validate_jsonl_logs


class FakeResponsesClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("no queued fake response")
        return self.responses.pop(0)


def _provider_report_payload(*, formal_ready: bool = True) -> dict[str, Any]:
    return {
        "formal_ready": formal_ready,
        "primary": {
            "provider": "primary",
            "model_id": "gpt-5.4",
            "formal_ready": formal_ready,
        },
        "forced_failover": {"required": False},
    }


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


def _config_stub(workspace) -> Any:
    return SimpleNamespace(
        workspace_root=workspace,
        task_specs=DEFAULT_TASK_SPECS,
    )


@pytest.mark.parametrize(
    ("primary_ready", "fallback_ready", "require_fallback", "expected"),
    [
        (False, False, False, False),
        (True, False, False, True),
        (True, False, True, False),
        (True, True, True, True),
    ],
)
def test_should_start_formal_run_respects_primary_and_fallback_requirements(
    primary_ready: bool,
    fallback_ready: bool,
    require_fallback: bool,
    expected: bool,
) -> None:
    assert should_start_formal_run(
        primary_ready=primary_ready,
        fallback_ready=fallback_ready,
        require_fallback=require_fallback,
    ) is expected


def test_finalize_gate_blocks_when_shared_code_contract_is_not_ready() -> None:
    ok, message = should_allow_finalize(
        {
            "overall_ok": False,
            "shared_code_ok": False,
            "failures": [
                "shared code baseline missing",
                "finalization contract incomplete",
            ],
        }
    )

    assert ok is False
    assert "shared code baseline missing" in message
    assert "finalization contract incomplete" in message


def test_formal_run_rejects_multi_task_input(workspace) -> None:
    result = run_formal_task_session(
        config=_config_stub(workspace),
        task="task1,task2",
        provider_report_path=workspace / "provider_report.json",
        responses_client=None,
    )

    assert result["ok"] is False
    assert "exactly one task" in result["error"]


def test_run_formal_task_session_executes_responses_tool_call_and_records_logs(
    workspace,
) -> None:
    provider_report_path = workspace / "provider_report.json"
    provider_report_path.write_text(
        json.dumps(_provider_report_payload(), ensure_ascii=False),
        encoding="utf-8",
    )
    client = FakeResponsesClient(
        [
            _turn(
                response_id="resp_1",
                output_items=[
                    {
                        "type": "function_call",
                        "name": "write_file",
                        "call_id": "call_1",
                        "arguments": json.dumps(
                            {
                                "path": "submission/code/model.py",
                                "content": "print('hello from v3')\n",
                            }
                        ),
                    }
                ],
            ),
            _turn(
                response_id="resp_2",
                output_items=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "FORMAL_RUN_COMPLETE"}],
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
    assert result["state"]["shared_code_version"].startswith("sha256:")
    assert result["state"]["last_llm_call_id"] == "resp_2"
    assert (workspace / "submission" / "code" / "model.py").read_text(encoding="utf-8") == "print('hello from v3')\n"
    assert (workspace / "submission" / "code_manifest.json").exists()
    assert validate_jsonl_logs(workspace / "llm_logs" / "all_llm_calls.jsonl")["ok"] is True

    store = MemoryStore(result["memory_db_path"])
    assert len(store.list_code_snapshots()) == 2
    assert len(store.list_patch_records()) == 1
    assert len(store.list_decision_records()) >= 3
    assert store.list_patch_records()[0]["llm_call_ids"] == ["resp_1"]
    first_call = client.calls[0]
    assert first_call["metadata"]["phase"] == "implementation"
    assert any(tool["name"] == "write_file" for tool in first_call["tools"])


def test_run_formal_task_session_rejects_patch_without_compatibility_evidence_for_later_task(
    workspace,
) -> None:
    provider_report_path = workspace / "provider_report.json"
    provider_report_path.write_text(
        json.dumps(_provider_report_payload(), ensure_ascii=False),
        encoding="utf-8",
    )
    code_dir = workspace / "submission" / "code"
    (code_dir / "shared.py").write_text("print('baseline')\n", encoding="utf-8")

    memory_db_path = workspace / "internal_logs" / "memory.db"
    seeded_result = run_formal_task_session(
        config=_config_stub(workspace),
        task="task1",
        provider_report_path=provider_report_path,
        responses_client=FakeResponsesClient(
            [
                _turn(
                    response_id="resp_seed",
                    output_items=[
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "BASELINE_ONLY"}],
                        }
                    ],
                )
            ]
        ),
        memory_db_path=memory_db_path,
    )
    assert seeded_result["ok"] is True

    client = FakeResponsesClient(
        [
            _turn(
                response_id="resp_patch",
                output_items=[
                    {
                        "type": "function_call",
                        "name": "write_file",
                        "call_id": "call_patch",
                        "arguments": json.dumps(
                            {
                                "path": "submission/code/shared.py",
                                "content": "print('task2 change')\n",
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
                        "content": [{"type": "output_text", "text": "ATTEMPTED_COMPLETE"}],
                    }
                ],
            ),
        ]
    )

    result = run_formal_task_session(
        config=_config_stub(workspace),
        task="task2",
        provider_report_path=provider_report_path,
        responses_client=client,
        memory_db_path=memory_db_path,
    )

    assert result["ok"] is False
    assert result["state"]["latest_error_type"] == "compatibility_guard_failed"
    assert (workspace / "submission" / "code" / "shared.py").read_text(encoding="utf-8") == "print('baseline')\n"
    store = MemoryStore(memory_db_path)
    assert len(store.list_code_snapshots()) == 1
    assert store.list_patch_records() == []


@pytest.mark.parametrize(
    ("argv", "handler_name"),
    [
        (["run", "--config", "config.yaml", "--task", "task1"], None),
        (["check-provider", "--config", "config.yaml"], None),
        (["check-research", "--config", "config.yaml"], None),
        (["validate", "--config", "config.yaml", "--task", "task1"], "validate_command"),
        (["package", "--config", "config.yaml", "--task", "task1"], "package_command"),
        (["status", "--config", "config.yaml", "--task", "task1"], "status_command"),
    ],
)
def test_cli_main_reaches_v3_commands(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    handler_name: str | None,
) -> None:
    target_handler = handler_name
    if target_handler is None:
        if argv[0] == "run":
            target_handler = "run_command"
        elif argv[0] == "check-provider":
            target_handler = "check_provider_command"
        elif argv[0] == "check-research":
            target_handler = "check_research_command"
    if target_handler is not None:
        monkeypatch.setattr(cli_module, target_handler, lambda *args, **kwargs: 0)
    assert main(argv) == 0
