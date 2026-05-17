from __future__ import annotations

from pathlib import Path

import pytest

from agent_runner.skills import build_skill_catalog, load_local_skills


def _write_skill(root: Path, name: str, description: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f'description: "{description}"',
                "---",
                "",
                f"# {name}",
                "",
                "Follow the local workflow.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_local_skills_reads_manifest_and_builds_catalog(tmp_path):
    skill_root = tmp_path / "skills"
    _write_skill(skill_root, "pdebench", "Inspect PDEBench dataset assumptions.")

    skills = load_local_skills([skill_root], ["pdebench"])
    catalog = build_skill_catalog(skills)

    assert [skill.name for skill in skills] == ["pdebench"]
    assert "Inspect PDEBench dataset assumptions." in catalog


def test_load_local_skills_rejects_missing_description(tmp_path):
    skill_dir = tmp_path / "skills" / "broken"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: broken",
                "---",
                "",
                "# broken",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="description"):
        load_local_skills([tmp_path / "skills"], ["broken"])
