from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class InstalledSkill:
    name: str
    description: str
    path: Path
    instructions: str


def _parse_skill_file(path: Path) -> InstalledSkill:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"Skill file is missing front matter: {path}")
    _, _, remainder = text.partition("---\n")
    front_matter_text, separator, body = remainder.partition("\n---\n")
    if not separator:
        raise ValueError(f"Skill file is missing closing front matter marker: {path}")
    front_matter = yaml.safe_load(front_matter_text) or {}
    name = front_matter.get("name")
    description = front_matter.get("description")
    if not name:
        raise ValueError(f"Skill file missing name: {path}")
    if not description:
        raise ValueError(f"Skill file missing description: {path}")
    return InstalledSkill(
        name=str(name),
        description=str(description),
        path=path,
        instructions=body.strip(),
    )


def load_local_skills(skill_dirs: list[Path], enabled: list[str]) -> list[InstalledSkill]:
    enabled_set = set(enabled)
    loaded: list[InstalledSkill] = []
    missing: list[str] = []

    for skill_name in enabled:
        found_path: Path | None = None
        for skill_dir in skill_dirs:
            candidate = skill_dir / skill_name / "SKILL.md"
            if candidate.exists():
                found_path = candidate
                break
        if found_path is None:
            missing.append(skill_name)
            continue
        loaded.append(_parse_skill_file(found_path))

    if missing:
        raise FileNotFoundError(f"Missing enabled skills: {', '.join(missing)}")

    return loaded


def build_skill_catalog(skills: list[InstalledSkill]) -> str:
    lines: list[str] = []
    for skill in skills:
        lines.append(f"- {skill.name}: {skill.description} ({skill.path})")
        if skill.instructions:
            lines.append(f"  {skill.instructions}")
    return "\n".join(lines)
