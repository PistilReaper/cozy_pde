from __future__ import annotations

from agent_runner.config import RunnerConfig
from agent_runner.hosted_tools import build_hosted_tools


def test_hosted_web_search_is_disabled_by_default_and_requires_experimental_flag(workspace):
    config = RunnerConfig.from_workspace(workspace)

    research_tools = build_hosted_tools(config, phase="research")
    assert research_tools == []

    config.responses_tools.enable_web_search = True
    config.responses_tools.experimental_enable_hosted_web_search = True

    research_tools = build_hosted_tools(config, phase="research")
    web_search_tool = next(tool for tool in research_tools if tool["type"] == "web_search")
    assert web_search_tool["filters"]["allowed_domains"][:2] == ["arxiv.org", "github.com"]
    assert web_search_tool["return_token_budget"] == 4096
    assert build_hosted_tools(config, phase="implementation") == []
