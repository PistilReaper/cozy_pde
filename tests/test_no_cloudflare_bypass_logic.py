from __future__ import annotations

from pathlib import Path


FORBIDDEN_STRINGS = [
    "cloudscraper",
    "cf_clearance",
    "undetected_chromedriver",
    "turnstile",
    "bypass cloudflare",
]


def test_no_cloudflare_bypass_logic():
    package_root = Path(__file__).resolve().parent.parent / "agent_runner"
    python_files = sorted(package_root.rglob("*.py"))
    combined = "\n".join(path.read_text(encoding="utf-8") for path in python_files)

    for forbidden in FORBIDDEN_STRINGS:
        assert forbidden not in combined.lower()
