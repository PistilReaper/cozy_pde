from __future__ import annotations

from agent_runner.config import RunnerConfig
from agent_runner.logger import ToolCallLogger
from agent_runner.tool_registry import build_tool_registry


def test_research_tools_not_available_in_implementation_phase(workspace):
    config = RunnerConfig.from_workspace(workspace)
    registry = build_tool_registry(
        config,
        ToolCallLogger(workspace / "internal_logs" / "tool_calls.jsonl"),
    )
    registry.set_context(task_id="task1", step_id="step-001", phase="implementation")

    schema_names = {schema["name"] for schema in registry.response_function_tools()}
    assert "search_arxiv" not in schema_names
    assert "search_github" not in schema_names
    assert "web_search" not in schema_names
    assert "fetch_url" not in schema_names
    assert "parse_html" not in schema_names

    result = registry.execute("search_arxiv", {"query": "FNO Burgers"})
    assert result["ok"] is False
    assert "phase" in result["error"].lower()
