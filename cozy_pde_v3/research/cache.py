from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def canonicalize_url(url: str) -> str:
    split = urlsplit(url.strip())
    path = split.path or "/"
    normalized = (
        split.scheme.lower(),
        split.netloc.lower(),
        path,
        split.query,
        "",
    )
    return urlunsplit(normalized)


@dataclass(slots=True)
class ResearchCache:
    cache_dir: Path

    @property
    def index_path(self) -> Path:
        return self.cache_dir / "research_sources.jsonl"

    @property
    def raw_dir(self) -> Path:
        return self.cache_dir / "raw"

    def initialize(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text("", encoding="utf-8")

    def append_source(self, record: dict[str, object]) -> None:
        self.initialize()
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


__all__ = ["ResearchCache", "canonicalize_url"]
