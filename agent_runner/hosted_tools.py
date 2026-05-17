from __future__ import annotations

from typing import Any

from .config import RunnerConfig


def build_hosted_tools(config: RunnerConfig, phase: str) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []

    if (
        config.responses_tools.enable_web_search
        and config.responses_tools.experimental_enable_hosted_web_search
        and phase in {"research", "planning"}
    ):
        tool: dict[str, Any] = {"type": "web_search"}
        web_cfg = config.responses_tools.web_search
        allowed_domains = web_cfg.get("allowed_domains")
        if allowed_domains:
            tool["filters"] = {"allowed_domains": allowed_domains}
        if "return_token_budget" in web_cfg:
            tool["return_token_budget"] = web_cfg["return_token_budget"]
        tools.append(tool)

    if config.responses_tools.enable_file_search and phase in {"research", "planning"}:
        vector_store_ids = config.responses_tools.file_search.get("vector_store_ids", [])
        if vector_store_ids:
            tools.append({"type": "file_search", "vector_store_ids": vector_store_ids})

    return tools
