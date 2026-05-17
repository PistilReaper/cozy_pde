from __future__ import annotations

from agent_runner.config import RunnerConfig
from agent_runner.logger import ToolCallLogger
from agent_runner.tool_registry import build_tool_registry


def test_registry_exposes_flat_responses_function_schema(workspace):
    config = RunnerConfig.from_workspace(workspace)
    registry = build_tool_registry(config, ToolCallLogger(workspace / "internal_logs" / "tool_calls.jsonl"))

    schemas = registry.response_function_tools()

    write_file_schema = next(schema for schema in schemas if schema["name"] == "write_file")
    assert write_file_schema["type"] == "function"
    assert "function" not in write_file_schema
    assert write_file_schema["strict"] is True
    assert write_file_schema["parameters"]["type"] == "object"
