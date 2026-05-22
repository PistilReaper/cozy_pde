from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResearchProviderFlags:
    arxiv_enabled: bool = True
    github_enabled: bool = True
    allow_unauthenticated_github: bool = True
