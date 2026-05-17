from __future__ import annotations

from pathlib import Path

import pytest

from agent_runner.config import load_config


def _write_config(path: Path, *, router_wire_api: str = "json_action", coder_wire_api: str = "json_action") -> None:
    path.write_text(
        "\n".join(
            [
                "workspace: workspace",
                "",
                "router:",
                '  model: "gpt-5.4"',
                '  reasoning_effort: "medium"',
                '  verbosity: "medium"',
                f'  wire_api: "{router_wire_api}"',
                "  temperature: 0.0",
                "  max_tokens: 2048",
                "",
                "llm_profiles:",
                "  strong_planner:",
                '    model: "gpt-5.5"',
                '    reasoning_effort: "xhigh"',
                '    verbosity: "medium"',
                '    wire_api: "json_action"',
                "  coder:",
                '    model: "gpt-5.4"',
                '    reasoning_effort: "high"',
                '    verbosity: "medium"',
                f'    wire_api: "{coder_wire_api}"',
                "  log_summarizer:",
                '    model: "gpt-5.2"',
                '    reasoning_effort: "medium"',
                '    verbosity: "low"',
                '    wire_api: "json_action"',
                "  json_judge:",
                '    model: "gpt-5.2"',
                '    reasoning_effort: "medium"',
                '    verbosity: "low"',
                '    wire_api: "json_action"',
                "",
                "openai:",
                '  provider: "third_party_openai_compatible"',
                '  base_url: "https://aixj.vip"',
                '  api_key_env: "LLM_API_KEY"',
                "  append_v1: true",
                "  store: false",
                "  streaming: false",
                "",
                "fallback_provider:",
                "  enabled: true",
                '  provider: "deepseek_openai_compatible"',
                '  base_url: "https://api.deepseek.com"',
                '  api_key_env: "DEEPSEEK_API_KEY"',
                "  append_v1: false",
                '  pro_model_env: "DEEPSEEK_PRO_MODEL"',
                '  flash_model_env: "DEEPSEEK_FLASH_MODEL"',
                "",
                "responses_tools:",
                "  enable_web_search: true",
                "  enable_file_search: false",
                "  enable_skills: true",
                "  enable_tool_search: true",
                "  web_search:",
                "    allowed_domains:",
                '      - "arxiv.org"',
                '      - "github.com"',
                "    return_token_budget: 4096",
                "  skills:",
                "    local_skill_dirs:",
                '      - "skills"',
                "    enabled:",
                '      - "pdebench"',
                "",
                "budget:",
                "  max_agent_steps: 50",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_config_reads_router_profiles_and_endpoint_from_dotenv(tmp_path, monkeypatch):
    project_root = tmp_path
    agent_runner_dir = project_root / "agent_runner"
    agent_runner_dir.mkdir(parents=True, exist_ok=True)
    config_path = agent_runner_dir / "config.yaml"
    _write_config(config_path)
    (project_root / ".env").write_text(
        "\n".join(
            [
                'LLM_API_KEY="dummy-from-dotenv"',
                'DEEPSEEK_API_KEY="deepseek-key"',
                'DEEPSEEK_PRO_MODEL="deepseek-v4-pro"',
                'DEEPSEEK_FLASH_MODEL="deepseek-v4-flash"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_PRO_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_FLASH_MODEL", raising=False)

    config = load_config(config_path)

    assert config.workspace_root == project_root / "workspace"
    assert config.endpoint.api_key == "dummy-from-dotenv"
    assert config.router.model == "gpt-5.4"
    assert config.router.wire_api == "json_action"
    assert set(config.llm_profiles) == {"strong_planner", "coder", "log_summarizer", "json_judge"}
    assert config.llm_profiles["coder"].wire_api == "json_action"
    assert config.responses_tools.enable_web_search is True
    assert config.responses_tools.skills["enabled"] == ["pdebench"]
    assert config.fallback_provider.api_key == "deepseek-key"
    assert config.fallback_provider.pro_model == "deepseek-v4-pro"
    assert config.fallback_provider.flash_model == "deepseek-v4-flash"


def test_load_config_rejects_non_json_action_router_wire_api(tmp_path):
    project_root = tmp_path
    agent_runner_dir = project_root / "agent_runner"
    agent_runner_dir.mkdir(parents=True, exist_ok=True)
    config_path = agent_runner_dir / "config.yaml"
    _write_config(config_path, router_wire_api="responses")

    with pytest.raises(ValueError, match="json_action"):
        load_config(config_path)


def test_load_config_rejects_non_json_action_profile_wire_api(tmp_path):
    project_root = tmp_path
    agent_runner_dir = project_root / "agent_runner"
    agent_runner_dir.mkdir(parents=True, exist_ok=True)
    config_path = agent_runner_dir / "config.yaml"
    _write_config(config_path, coder_wire_api="responses")

    with pytest.raises(ValueError, match="json_action"):
        load_config(config_path)
