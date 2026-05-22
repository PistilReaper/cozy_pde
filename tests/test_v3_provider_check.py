from __future__ import annotations

import json
from pathlib import Path

import pytest

from cozy_pde_v3.config import load_config
from cozy_pde_v3.provider_capabilities import load_provider_capability_report, verify_provider_capability_report
from cozy_pde_v3 import cli as cli_module


def _write_config(path: Path) -> None:
    path.write_text(
        """
        workspace_root: ./workspace
        provider:
          wire_api: responses
          primary:
            provider: primary
            base_url: https://primary.example.com
            api_key_env: PRIMARY_API_KEY
            model_id: gpt-5.4
          fallback:
            provider: fallback
            base_url: https://fallback.example.com
            api_key_env: FALLBACK_API_KEY
            model_id: deepseek-v4-pro
          require_fallback: true
        proxy:
          enabled: true
          primary_log_dir: ./workspace/proxy_logs/primary
          fallback_log_dir: ./workspace/proxy_logs/fallback
          proxy_version: 2026-05-22
        artifacts:
          provider_report_path: ./workspace/capabilities/provider_report.json
        """.strip()
        + "\n",
        encoding="utf-8",
    )


def test_check_provider_writes_report_and_verifies_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    monkeypatch.setenv("PRIMARY_API_KEY", "primary-key")
    monkeypatch.setenv("FALLBACK_API_KEY", "fallback-key")
    _write_config(config_path)
    config = load_config(config_path)

    class FakeResponsesClient:
        def __init__(self, *, primary, fallback=None):
            self.primary = primary
            self.fallback = fallback

        @classmethod
        def from_config(cls, config):
            return cls(primary=config.provider.primary, fallback=config.provider.fallback)

        def probe_capabilities(self, *, tool_schemas, proxy_log_dirs):
            assert tool_schemas
            assert proxy_log_dirs["primary"] == config.proxy.primary_log_dir
            return {
                "primary": {
                    "provider": "primary",
                    "model_id": "gpt-5.4",
                    "base_url": "https://primary.example.com/v1",
                    "text_probe_ok": True,
                    "function_call_ok": True,
                    "function_call_output_ok": True,
                    "strict_schema_ok": True,
                    "proxy_raw_log_ok": True,
                    "formal_ready": True,
                    "forced_failover": False,
                },
                "fallback": {
                    "provider": "fallback",
                    "model_id": "deepseek-v4-pro",
                    "base_url": "https://fallback.example.com/v1",
                    "text_probe_ok": True,
                    "function_call_ok": True,
                    "function_call_output_ok": True,
                    "strict_schema_ok": True,
                    "proxy_raw_log_ok": True,
                    "formal_ready": True,
                    "forced_failover": True,
                },
                "forced_failover": {
                    "supported": True,
                    "selected_provider": "fallback",
                    "probe_id": "forced-failover-1",
                    "observed_model": "deepseek-v4-pro",
                },
                "sdk_version": "2.37.0",
                "adapter_version": "responses-v3",
            }

    monkeypatch.setattr(cli_module, "ResponsesClient", FakeResponsesClient)

    exit_code = cli_module.check_provider_command(config)

    assert exit_code == 0
    report_path = config.artifacts.provider_report_path
    assert report_path.exists()

    report = load_provider_capability_report(report_path)
    assert report["primary"]["strict_schema_ok"] is True
    assert report["fallback"]["function_call_output_ok"] is True
    assert report["forced_failover"]["selected_provider"] == "fallback"
    verification = verify_provider_capability_report(report, config=config)
    assert verification["ok"] is True
    assert verification["data"]["formal_ready"] is True


def test_verify_provider_report_rejects_mismatched_config_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    monkeypatch.setenv("PRIMARY_API_KEY", "primary-key")
    monkeypatch.setenv("FALLBACK_API_KEY", "fallback-key")
    _write_config(config_path)
    config = load_config(config_path)

    report_path = config.artifacts.provider_report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "config_hash": "wrong",
                "tool_schema_hash": "tool-hash",
                "proxy_version_hash": "proxy-hash",
                "adapter_version": "responses-v3",
                "sdk_version": "2.37.0",
                "checked_at": "2026-05-22T00:00:00Z",
                "expires_at": "2026-05-22T01:00:00Z",
                "formal_ready": True,
                "forced_failover": {"supported": True},
                "primary": {
                    "provider": "primary",
                    "model_id": "gpt-5.4",
                    "base_url_hash": "abc",
                    "primary": True,
                    "fallback": False,
                    "forced_failover": False,
                    "formal_ready": True,
                    "text_probe_ok": True,
                    "function_call_ok": True,
                    "function_call_output_ok": True,
                    "strict_schema_ok": True,
                    "proxy_raw_log_ok": True,
                },
                "fallback": {
                    "provider": "fallback",
                    "model_id": "deepseek-v4-pro",
                    "base_url_hash": "def",
                    "primary": False,
                    "fallback": True,
                    "forced_failover": True,
                    "formal_ready": True,
                    "text_probe_ok": True,
                    "function_call_ok": True,
                    "function_call_output_ok": True,
                    "strict_schema_ok": True,
                    "proxy_raw_log_ok": True,
                },
            }
        ),
        encoding="utf-8",
    )

    report = load_provider_capability_report(report_path)
    verification = verify_provider_capability_report(report, config=config)

    assert verification["ok"] is False
    assert "config_hash" in verification["error"]
