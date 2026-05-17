from __future__ import annotations

from pathlib import Path


FORBIDDEN_STRINGS = [
    "chat.completions.create",
    "messages_to_responses_input",
    "build_assistant_message",
    "build_tool_result_message",
]


def test_agent_runner_no_longer_contains_chat_completions_compatibility_layer():
    package_root = Path(__file__).resolve().parent.parent / "agent_runner"
    python_files = sorted(package_root.rglob("*.py"))
    combined = "\n".join(path.read_text(encoding="utf-8") for path in python_files)

    for forbidden in FORBIDDEN_STRINGS:
        assert forbidden not in combined

    assert not (package_root / "llm_client.py").exists()
