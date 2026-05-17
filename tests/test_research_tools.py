from __future__ import annotations

import os
from pathlib import Path

import httpx

from agent_runner.config import RunnerConfig
from agent_runner.tools.research_tools import fetch_url, search_arxiv, search_github, web_search


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _client_with_text(expected_substring: str, text: str, content_type: str = "text/plain") -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert expected_substring in str(request.url)
        return httpx.Response(
            200,
            text=text,
            headers={"content-type": content_type},
            request=request,
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_search_arxiv_parses_atom_fixture(workspace):
    config = RunnerConfig.from_workspace(workspace)
    fixture_text = (FIXTURE_DIR / "arxiv_search.xml").read_text(encoding="utf-8")
    client = _client_with_text("export.arxiv.org", fixture_text, content_type="application/atom+xml")

    result = search_arxiv(
        query="Fourier Neural Operator Burgers PDEBench",
        research=config.research,
        http_client=client,
    )

    assert result["ok"] is True
    records = result["data"]["results"]
    assert len(records) == 1
    assert records[0]["source_type"] == "arxiv"
    assert records[0]["arxiv_id"] == "2010.08895v1"
    assert records[0]["title"].startswith("Fourier Neural Operator")
    assert records[0]["authors"] == ["Zongyi Li", "Nikola Kovachki"]
    assert records[0]["pdf_url"] == "https://arxiv.org/pdf/2010.08895v1.pdf"


def test_search_github_parses_rest_fixture(workspace):
    config = RunnerConfig.from_workspace(workspace)
    fixture_text = (FIXTURE_DIR / "github_code_search.json").read_text(encoding="utf-8")
    client = _client_with_text("/search/code", fixture_text, content_type="application/json")

    result = search_github(
        query="neuraloperator Fourier neural operator Burgers",
        kind="code",
        research=config.research,
        http_client=client,
    )

    assert result["ok"] is True
    records = result["data"]["results"]
    assert len(records) == 1
    assert records[0]["source_type"] == "github_file"
    assert records[0]["repo"] == "neuraloperator"
    assert records[0]["owner"] == "neuraloperator"
    assert records[0]["path"] == "neuralop/models/fourier_2d.py"
    assert records[0]["raw_url"] == "https://raw.githubusercontent.com/neuraloperator/neuraloperator/main/neuralop/models/fourier_2d.py"
    assert records[0]["license_hint"] == "MIT"


def test_search_github_retries_without_invalid_token_when_unauthenticated_allowed(workspace):
    config = RunnerConfig.from_workspace(workspace)
    config.research.providers.github.api_key = "invalid-token"
    fixture_text = (FIXTURE_DIR / "github_code_search.json").read_text(encoding="utf-8")
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        assert "/search/code" in str(request.url)
        if request.headers.get("Authorization"):
            return httpx.Response(401, json={"message": "Bad credentials"}, request=request)
        return httpx.Response(
            200,
            text=fixture_text,
            headers={"content-type": "application/json"},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    result = search_github(
        query="neuraloperator Fourier neural operator Burgers",
        kind="code",
        research=config.research,
        http_client=client,
    )

    assert result["ok"] is True
    assert attempts["count"] == 2
    assert result["data"]["results"][0]["repo"] == "neuraloperator"


def test_web_search_skips_missing_provider_keys(workspace, monkeypatch):
    config = RunnerConfig.from_workspace(workspace)
    for key in ["TAVILY_API_KEY", "EXA_API_KEY", "BRAVE_SEARCH_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CSE_ID"]:
        monkeypatch.delenv(key, raising=False)

    result = web_search(
        query="PDEBench Burgers FNO baseline",
        domains=["github.com", "arxiv.org"],
        research=config.research,
    )

    assert result["ok"] is True
    assert result["data"]["results"] == []
    assert result["data"]["provider"] is None
    assert result["data"]["skipped_providers"] == ["tavily", "exa", "brave", "google_cse"]


def test_fetch_url_blocks_checkpoint_extension(workspace):
    config = RunnerConfig.from_workspace(workspace)

    result = fetch_url(
        url="https://github.com/example/repo/releases/download/v1/model.ckpt",
        research=config.research,
    )

    assert result["ok"] is False
    assert "blocked" in result["error"].lower()
    assert ".ckpt" in result["error"]


def test_fetch_url_blocks_hdf5_extension(workspace):
    config = RunnerConfig.from_workspace(workspace)

    result = fetch_url(
        url="https://example.com/data/test.hdf5",
        research=config.research,
    )

    assert result["ok"] is False
    assert "blocked" in result["error"].lower()
    assert ".hdf5" in result["error"]


def test_fetch_url_allows_raw_github_python_file(workspace):
    config = RunnerConfig.from_workspace(workspace)
    client = _client_with_text(
        "raw.githubusercontent.com",
        "print('hello research')\n",
        content_type="text/plain; charset=utf-8",
    )

    result = fetch_url(
        url="https://raw.githubusercontent.com/neuraloperator/neuraloperator/main/train.py",
        research=config.research,
        http_client=client,
    )

    assert result["ok"] is True
    assert result["data"]["content"] == "print('hello research')\n"
    assert result["data"]["content_sha256"]
    assert Path(result["data"]["cache_path"]).exists()
