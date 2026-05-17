from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .config import ResearchConfig


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    normalized_path = parts.path or "/"
    if normalized_path != "/" and normalized_path.endswith("/"):
        normalized_path = normalized_path[:-1]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), normalized_path, query, ""))


class ResearchCache:
    def __init__(self, research: ResearchConfig) -> None:
        assert research.cache_index_path is not None
        assert research.raw_cache_dir is not None
        assert research.papers_dir is not None
        self.research = research
        self.path = research.cache_index_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        research.raw_cache_dir.mkdir(parents=True, exist_ok=True)
        research.papers_dir.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def _load(self) -> list[dict]:
        records: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))
        return records

    def _write_all(self, records: list[dict]) -> None:
        self.path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )

    def _normalize(self, record: dict) -> dict:
        normalized = dict(record)
        normalized["url"] = canonicalize_url(str(record.get("url", "")))
        normalized["raw_url"] = canonicalize_url(str(record.get("raw_url", ""))) if record.get("raw_url") else ""
        normalized["source_id"] = str(record.get("source_id") or f"{record.get('source_type', 'source')}:{normalized['url']}")
        normalized["source_type"] = str(record.get("source_type", "web_page"))
        normalized["title"] = str(record.get("title", ""))
        normalized["retrieved_at"] = str(record.get("retrieved_at") or datetime.now(timezone.utc).isoformat())
        normalized["query"] = str(record.get("query", ""))
        normalized["summary"] = str(record.get("summary", ""))
        normalized["content_sha256"] = str(record.get("content_sha256", ""))
        normalized["raw_cache_path"] = str(record.get("raw_cache_path", ""))
        normalized["license_hint"] = str(record.get("license_hint", ""))
        normalized["risk_flags"] = list(record.get("risk_flags", []))
        normalized["allowed_for_submission_code_reference"] = bool(
            record.get("allowed_for_submission_code_reference", True)
        )
        normalized["allowed_for_training_data"] = False

        blocked_extensions = tuple(self.research.blocked_extensions)
        blocked_targets = [normalized["url"], normalized["raw_url"], normalized["raw_cache_path"]]
        if any(target.lower().endswith(blocked_extensions) for target in blocked_targets if target):
            raise ValueError("Research cache cannot store external data/checkpoint artifacts")
        if "submission/code" in normalized["raw_cache_path"].replace("\\", "/"):
            raise ValueError("Research cache must not write to submission/code")
        return normalized

    def write(self, record: dict) -> dict:
        normalized = self._normalize(record)
        existing = self._load()
        deduped: list[dict] = []
        replacement_written = False

        for current in existing:
            same_url = current.get("url") == normalized["url"]
            same_hash = normalized["content_sha256"] and current.get("content_sha256") == normalized["content_sha256"]
            if same_url or same_hash:
                if not replacement_written:
                    merged = dict(current)
                    merged.update(normalized)
                    deduped.append(merged)
                    replacement_written = True
                continue
            deduped.append(current)

        if not replacement_written:
            deduped.append(normalized)

        self._write_all(deduped)
        return next(record for record in deduped if record.get("source_id") == normalized["source_id"])

    def read(self, *, source_id: str | None = None, url: str | None = None) -> list[dict]:
        records = self._load()
        if source_id is None and url is None:
            return records

        normalized_url = canonicalize_url(url) if url else None
        return [
            record
            for record in records
            if (source_id is not None and record.get("source_id") == source_id)
            or (normalized_url is not None and record.get("url") == normalized_url)
        ]

    def search(self, query: str, *, max_results: int = 10) -> list[dict]:
        lowered = query.lower()
        matches = []
        for record in self._load():
            haystack = " ".join(
                [
                    str(record.get("source_type", "")),
                    str(record.get("title", "")),
                    str(record.get("url", "")),
                    str(record.get("summary", "")),
                ]
            ).lower()
            if lowered in haystack:
                matches.append(record)
        return matches[:max_results]
