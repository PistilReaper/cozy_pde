from __future__ import annotations

from pathlib import Path

from agent_runner.config import ResponsesToolConfig


def test_installed_default_skills_exist():
    project_root = Path(__file__).resolve().parent.parent
    config = ResponsesToolConfig()
    skill_dirs = [project_root / path for path in config.skills["local_skill_dirs"]]

    for skill_name in config.skills["enabled"]:
        assert any((skill_dir / skill_name / "SKILL.md").exists() for skill_dir in skill_dirs), skill_name
