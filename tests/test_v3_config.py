from __future__ import annotations

from pathlib import Path

import pytest

from cozy_pde_v3.config import load_config


def _write_config(path: Path, body: str) -> None:
    path.write_text(body.strip() + "\n", encoding="utf-8")


def test_load_config_expands_v3_responses_only_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    workspace_root = tmp_path / "workspace"
    monkeypatch.setenv("PRIMARY_API_KEY", "primary-secret")
    monkeypatch.setenv("FALLBACK_API_KEY", "fallback-secret")

    _write_config(
        config_path,
        """
        workspace_root: ./workspace
        submission_dir: ./workspace/submission
        shared_code_dir: ./workspace/submission/code/shared
        data_roots:
          primary: ./workspace/data
          evaluation:
            - ./workspace/data
            - ./workspace/baselines
        provider:
          wire_api: responses
          primary:
            provider: openai_compatible
            base_url: https://primary.example.com
            api_key_env: PRIMARY_API_KEY
            model_id: gpt-5.4
          fallback:
            provider: deepseek_compatible
            base_url: https://fallback.example.com
            api_key_env: FALLBACK_API_KEY
            model_id: deepseek-v4-pro
          require_fallback: true
        proxy:
          enabled: true
          primary_log_dir: ./workspace/proxy_logs/primary
          fallback_log_dir: ./workspace/proxy_logs/fallback
          proxy_version: 2026-05-22
        budget:
          max_total_usd: 25.0
          max_steps: 80
        timeout:
          provider_seconds: 35
          formal_run_seconds: 1800
        research:
          enabled: true
          allow_network: false
          cache_dir: ./workspace/research/cache
        artifacts:
          provider_report_path: ./workspace/capabilities/provider_report.json
          package_output_path: ./workspace/submission/submission.zip
        task_policy:
          task_ids: [task1, task2]
          strict_validation: true
        """,
    )

    config = load_config(config_path)

    assert config.config_path == config_path
    assert config.workspace_root == workspace_root.resolve()
    assert config.submission_dir == (workspace_root / "submission").resolve()
    assert config.shared_code_dir == (workspace_root / "submission" / "code" / "shared").resolve()
    assert config.data_roots.primary == (workspace_root / "data").resolve()
    assert config.data_roots.evaluation == [
        (workspace_root / "data").resolve(),
        (workspace_root / "baselines").resolve(),
    ]
    assert config.provider.primary.api_key == "primary-secret"
    assert config.provider.fallback is not None
    assert config.provider.fallback.api_key == "fallback-secret"
    assert config.provider.require_fallback is True
    assert config.proxy.primary_log_dir == (workspace_root / "proxy_logs" / "primary").resolve()
    assert config.proxy.fallback_log_dir == (workspace_root / "proxy_logs" / "fallback").resolve()
    assert config.research.cache_index_path == (workspace_root / "research" / "cache" / "research_sources.jsonl").resolve()
    assert config.research.raw_cache_dir == (workspace_root / "research" / "cache" / "raw").resolve()
    assert config.research.papers_dir == (workspace_root / "research" / "papers").resolve()
    assert config.artifacts.provider_report_path == (workspace_root / "capabilities" / "provider_report.json").resolve()
    assert config.task_policy.task_ids == ["task1", "task2"]


@pytest.mark.parametrize(
    "body",
    [
        """
        workspace_root: ./workspace
        provider:
          wire_api: invalid_wire_api
          primary:
            base_url: https://primary.example.com
            api_key_env: PRIMARY_API_KEY
            model_id: gpt-5.4
        """,
        """
        workspace_root: ./workspace
        provider:
          wire_api: legacy_text_mode
          primary:
            base_url: https://primary.example.com
            api_key_env: PRIMARY_API_KEY
            model_id: gpt-5.4
        """,
        """
        workspace_root: ./workspace
        router:
          wire_api: legacy_router_mode
        provider:
          wire_api: responses
          primary:
            base_url: https://primary.example.com
            api_key_env: PRIMARY_API_KEY
            model_id: gpt-5.4
        """,
    ],
)
def test_load_config_rejects_non_responses_wire_api_config(
    tmp_path: Path,
    body: str,
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, body)

    with pytest.raises(ValueError, match="responses"):
        load_config(config_path)


def test_load_config_requires_fallback_when_configured(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        workspace_root: ./workspace
        provider:
          wire_api: responses
          require_fallback: true
          primary:
            base_url: https://primary.example.com
            api_key_env: PRIMARY_API_KEY
            model_id: gpt-5.4
        """,
    )

    with pytest.raises(ValueError, match="fallback"):
        load_config(config_path)
