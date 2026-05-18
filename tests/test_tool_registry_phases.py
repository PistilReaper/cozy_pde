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


def test_phase_tool_narrowing_reduces_available_tools(workspace):
    config = RunnerConfig.from_workspace(workspace)
    registry = build_tool_registry(
        config,
        ToolCallLogger(workspace / "internal_logs" / "tool_calls.jsonl"),
    )

    registry.set_context(task_id="task-research", step_id="step-001", phase="research")
    research_names = {schema["name"] for schema in registry.response_function_tools()}
    assert research_names == {
        "read_file",
        "search_arxiv",
        "search_github",
        "fetch_url",
        "fetch_pdf",
        "parse_pdf",
        "parse_html",
        "research_cache_write",
        "research_cache_read",
        "research_cache_search",
    }

    registry.set_context(task_id="task-impl", step_id="step-002", phase="implementation")
    implementation_names = {schema["name"] for schema in registry.response_function_tools()}
    assert implementation_names == {
        "read_file",
        "write_file",
        "run_shell",
        "run_python",
        "analyze_log",
        "snapshot",
        "rollback",
    }

    registry.set_context(task_id="task-final", step_id="step-003", phase="finalization")
    finalization_names = {schema["name"] for schema in registry.response_function_tools()}
    assert finalization_names == {
        "generate_methodology_pdf",
        "inspect_hdf5",
        "validate_submission",
        "validate_responses_logs",
        "package_submission",
        "validate_full_submission",
    }
