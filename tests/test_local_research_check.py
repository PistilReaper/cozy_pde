from __future__ import annotations

import json
from pathlib import Path

import httpx

from agent_runner.config import RunnerConfig
from agent_runner.main import run_local_research_check


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def test_local_research_check_does_not_require_hosted_web_search(workspace, monkeypatch):
    config = RunnerConfig.from_workspace(workspace)
    config.responses_tools.enable_web_search = False
    config.responses_tools.experimental_enable_hosted_web_search = False
    for key in ["TAVILY_API_KEY", "EXA_API_KEY", "BRAVE_SEARCH_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CSE_ID"]:
        monkeypatch.delenv(key, raising=False)

    arxiv_fixture = (FIXTURE_DIR / "arxiv_search.xml").read_text(encoding="utf-8")
    github_fixture = (FIXTURE_DIR / "github_repo_search.json").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "export.arxiv.org" in url:
            return httpx.Response(200, text=arxiv_fixture, headers={"content-type": "application/atom+xml"}, request=request)
        if "/search/repositories" in url:
            return httpx.Response(200, text=github_fixture, headers={"content-type": "application/json"}, request=request)
        if "/abs/" in url:
            return httpx.Response(200, text="<html><title>arXiv</title><body>abstract</body></html>", headers={"content-type": "text/html"}, request=request)
        raise AssertionError(f"unexpected URL {url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))

    exit_code = run_local_research_check(config, http_client=client)

    assert exit_code == 0
    report_path = workspace / "runs" / "local_research_report.md"
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "hosted web_search" not in report_text.lower()
    assert "arxiv" in report_text.lower()
    assert "github" in report_text.lower()

    cache_path = workspace / "research" / "cache" / "research_sources.jsonl"
    lines = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines()]
    assert any(line["source_type"] == "arxiv" for line in lines)
    assert any(line["source_type"] == "github_repo" for line in lines)
