from __future__ import annotations

from agent_runner.config import RunnerConfig
from agent_runner.logger import ToolCallLogger
from agent_runner.tool_registry import build_tool_registry


def test_registry_rejects_arguments_outside_local_tool_schema(workspace):
    config = RunnerConfig.from_workspace(workspace)
    registry = build_tool_registry(config, ToolCallLogger(workspace / "internal_logs" / "tool_calls.jsonl"))
    registry.set_context(task_id="task", step_id="step-001", phase="implementation")

    result = registry.execute(
        "write_file",
        {
            "path": "submission/code/hello.py",
            "content": "print('hello')\n",
            "unexpected": True,
        },
    )

    assert result["ok"] is False
    assert "unexpected" in result["error"].lower()
