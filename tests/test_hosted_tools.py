from __future__ import annotations

from agent_runner.config import RunnerConfig
from agent_runner.hosted_tools import build_hosted_tools


def test_research_phase_includes_web_search_and_implementation_does_not(workspace):
    config = RunnerConfig.from_workspace(workspace)

    research_tools = build_hosted_tools(config, phase="research")
    implementation_tools = build_hosted_tools(config, phase="implementation")

    web_search_tool = next(tool for tool in research_tools if tool["type"] == "web_search")
    assert web_search_tool["filters"]["allowed_domains"][:2] == ["arxiv.org", "github.com"]
    assert web_search_tool["return_token_budget"] == 4096
    assert implementation_tools == []
