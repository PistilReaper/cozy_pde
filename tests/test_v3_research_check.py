from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from cozy_pde_v3 import cli as cli_module
from cozy_pde_v3.config import load_config


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _write_config(path: Path, *, allow_network: bool) -> None:
    path.write_text(
        f"""
        workspace_root: ./workspace
        provider:
          wire_api: responses
          primary:
            base_url: https://primary.example.com
            api_key_env: PRIMARY_API_KEY
            model_id: gpt-5.4
        research:
          enabled: true
          allow_network: {'true' if allow_network else 'false'}
          cache_dir: ./workspace/research/cache
        """.strip()
        + "\n",
        encoding="utf-8",
    )


def test_check_research_returns_degraded_but_runnable_when_network_disabled(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, allow_network=False)
    config = load_config(config_path)

    exit_code = cli_module.check_research_command(config)
    captured = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured["ok"] is True
    assert captured["data"]["degraded"] is True
    assert captured["data"]["runnable"] is True
    assert captured["data"]["network_available"] is False
    assert config.research.cache_index_path.exists()


def test_check_research_uses_v3_wrappers_and_writes_cache_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, allow_network=True)
    config = load_config(config_path)

    arxiv_fixture = (FIXTURE_DIR / "arxiv_search.xml").read_text(encoding="utf-8")
    github_fixture = (FIXTURE_DIR / "github_repo_search.json").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "export.arxiv.org" in url:
            return httpx.Response(200, text=arxiv_fixture, headers={"content-type": "application/atom+xml"}, request=request)
        if "/search/repositories" in url:
            return httpx.Response(200, text=github_fixture, headers={"content-type": "application/json"}, request=request)
        if "github.com" in url:
            return httpx.Response(200, text="<html><title>Repo</title><body>repo page</body></html>", headers={"content-type": "text/html"}, request=request)
        raise AssertionError(f"unexpected URL {url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))

    exit_code = cli_module.check_research_command(config, http_client=client)
    captured = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured["ok"] is True
    assert captured["data"]["degraded"] is False
    lines = [json.loads(line) for line in config.research.cache_index_path.read_text(encoding="utf-8").splitlines()]
    assert any(line["source_type"] == "arxiv" for line in lines)
    assert any(line["source_type"] == "github_repo" for line in lines)
