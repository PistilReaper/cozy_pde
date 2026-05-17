from __future__ import annotations

import json
from datetime import datetime

from agent_runner.logger import LLMCallLogger, ToolCallLogger
from agent_runner.tools.validate_tools import validate_jsonl_logs


def test_llm_logger_writes_valid_jsonl(workspace):
    path = workspace / "llm_logs" / "all_llm_calls.jsonl"
    logger = LLMCallLogger(path)

    logger.log_call(
        step_id="step-001",
        task_id="task1",
        model="gpt-5.5",
        profile="coder",
        phase="implementation",
        elapsed_seconds=0.42,
        response="summary",
        raw_response={"id": "resp_1"},
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["model"] == "gpt-5.5"
    assert payload["profile"] == "coder"
    assert payload["phase"] == "implementation"
    assert payload["response"] == "summary"
    assert payload["raw_response"] == {"id": "resp_1"}
    assert payload["step_id"] == "step-001"
    assert payload["task_id"] == "task1"
    assert datetime.fromisoformat(payload["timestamp"])

    result = validate_jsonl_logs(path)
    assert result["ok"] is True


def test_tool_logger_writes_valid_jsonl(workspace):
    path = workspace / "internal_logs" / "tool_calls.jsonl"
    logger = ToolCallLogger(path)

    logger.log_call(
        tool_name="write_file",
        elapsed_seconds=0.11,
        arguments={"path": "submission/code/hello.py"},
        result={"ok": True},
    )

    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["tool_name"] == "write_file"
    assert payload["arguments"]["path"] == "submission/code/hello.py"
    assert payload["result"] == {"ok": True}
