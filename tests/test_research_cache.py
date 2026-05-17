from __future__ import annotations

import json

from agent_runner.config import RunnerConfig
from agent_runner.research_cache import ResearchCache


def test_research_cache_deduplicates_by_url_and_hash(workspace):
    config = RunnerConfig.from_workspace(workspace)
    cache = ResearchCache(config.research)
    record = {
        "source_id": "arxiv:2010.08895v1",
        "source_type": "arxiv",
        "title": "Fourier Neural Operator",
        "url": "https://arxiv.org/abs/2010.08895v1",
        "raw_url": "",
        "retrieved_at": "2026-05-17T00:00:00+00:00",
        "query": "Fourier Neural Operator Burgers PDEBench",
        "summary": "Foundational neural operator paper.",
        "content_sha256": "sha-1",
        "raw_cache_path": "workspace/research/cache/raw/arxiv.txt",
        "license_hint": "",
        "risk_flags": [],
        "allowed_for_submission_code_reference": True,
        "allowed_for_training_data": False,
    }

    first = cache.write(record)
    second = cache.write({**record, "summary": "Duplicate write should not add a line."})

    assert first["source_id"] == second["source_id"]
    lines = cache.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["url"] == "https://arxiv.org/abs/2010.08895v1"
    assert payload["content_sha256"] == "sha-1"
