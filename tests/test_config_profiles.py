from __future__ import annotations

from pathlib import Path

import pytest

from cozy_pde_v3.cli import build_parser
from cozy_pde_v3.config import V3Config, load_config
from cozy_pde_v3.task_specs import TASK_IDS


def test_load_config_returns_minimal_v3_responses_only_contract(tmp_path: Path) -> None:
    config = load_config(tmp_path / "config.yaml", workspace_root=tmp_path / "workspace")

    assert isinstance(config, V3Config)
    assert config.config_path == tmp_path / "config.yaml"
    assert config.workspace_root == tmp_path / "workspace"
    assert sorted(config.task_specs) == list(TASK_IDS)
    assert not hasattr(config, "router")
    assert not hasattr(config, "endpoint")
    assert not hasattr(config, "llm_profiles")


def test_load_config_does_not_restore_legacy_profile_surface(tmp_path: Path) -> None:
    config_path = tmp_path / "legacy-config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "provider:",
                '  wire_api: "responses"',
                "  primary:",
                '    base_url: "https://primary.example.com"',
                '    api_key_env: "PRIMARY_API_KEY"',
                '    model_id: "gpt-5.4"',
                "router:",
                '  transport: "legacy_router"',
                "llm_profiles:",
                "  coder:",
                '    model: "stale-profile"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path, workspace_root=tmp_path / "workspace")

    assert config.config_path == config_path
    assert config.workspace_root == tmp_path / "workspace"
    assert not hasattr(config, "router")
    assert not hasattr(config, "wire_api")


def test_v3_cli_rejects_legacy_wire_api_flags() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--config",
                "config.yaml",
                "--task",
                "task1",
                "--wire-api",
                "legacy_wire_api",
            ]
        )


def test_v3_cli_check_research_is_not_task_scoped() -> None:
    parser = build_parser()
    args = parser.parse_args(["check-research", "--config", "config.yaml"])

    assert args.command == "check-research"
    assert args.config == "config.yaml"
    assert not hasattr(args, "task")
