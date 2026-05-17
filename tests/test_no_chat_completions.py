from __future__ import annotations

from pathlib import Path


FORBIDDEN_STRINGS = [
    "client.responses.create",
    "tool_choice=\"auto\"",
    "tool_choice='auto'",
]


def test_agent_runner_no_longer_contains_responses_tool_call_path():
    package_root = Path(__file__).resolve().parent.parent / "agent_runner"
    python_files = sorted(package_root.rglob("*.py"))
    combined = "\n".join(path.read_text(encoding="utf-8") for path in python_files)

    for forbidden in FORBIDDEN_STRINGS:
        assert forbidden not in combined
