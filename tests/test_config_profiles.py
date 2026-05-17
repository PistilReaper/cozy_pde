from __future__ import annotations

from pathlib import Path

import pytest

from agent_runner.config import load_config


def _write_config(path: Path, *, router_wire_api: str = "responses", coder_wire_api: str = "responses") -> None:
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
                '    wire_api: "responses"',
                "  coder:",
                '    model: "gpt-5.4"',
                '    reasoning_effort: "high"',
                '    verbosity: "medium"',
                f'    wire_api: "{coder_wire_api}"',
                "  log_summarizer:",
                '    model: "gpt-5.2"',
                '    reasoning_effort: "medium"',
                '    verbosity: "low"',
                '    wire_api: "responses"',
                "  json_judge:",
                '    model: "gpt-5.2"',
                '    reasoning_effort: "medium"',
                '    verbosity: "low"',
                '    wire_api: "responses"',
                "",
                "openai:",
                '  provider: "third_party_openai_compatible"',
                '  base_url: "https://aixj.vip"',
                '  api_key_env: "LLM_API_KEY"',
                "  store: false",
                "  streaming: false",
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
    (project_root / ".env").write_text('LLM_API_KEY="dummy-from-dotenv"\n', encoding="utf-8")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    config = load_config(config_path)

    assert config.workspace_root == project_root / "workspace"
    assert config.endpoint.api_key == "dummy-from-dotenv"
    assert config.router.model == "gpt-5.4"
    assert config.router.wire_api == "responses"
    assert set(config.llm_profiles) == {"strong_planner", "coder", "log_summarizer", "json_judge"}
    assert config.llm_profiles["coder"].wire_api == "responses"
    assert config.responses_tools.enable_web_search is True
    assert config.responses_tools.skills["enabled"] == ["pdebench"]


def test_load_config_rejects_non_responses_router_wire_api(tmp_path):
    project_root = tmp_path
    agent_runner_dir = project_root / "agent_runner"
    agent_runner_dir.mkdir(parents=True, exist_ok=True)
    config_path = agent_runner_dir / "config.yaml"
    _write_config(config_path, router_wire_api="chat_completions")

    with pytest.raises(ValueError, match="responses"):
        load_config(config_path)


def test_load_config_rejects_non_responses_profile_wire_api(tmp_path):
    project_root = tmp_path
    agent_runner_dir = project_root / "agent_runner"
    agent_runner_dir.mkdir(parents=True, exist_ok=True)
    config_path = agent_runner_dir / "config.yaml"
    _write_config(config_path, coder_wire_api="chat_completions")

    with pytest.raises(ValueError, match="responses"):
        load_config(config_path)
