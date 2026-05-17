from __future__ import annotations

from agent_runner.safety import WorkspaceSafety
from agent_runner.tools.fs_tools import list_files, read_file, write_file


def test_write_read_and_list_files_are_limited_to_workspace(workspace):
    safety = WorkspaceSafety(workspace)

    write_result = write_file(
        path="submission/code/hello.py",
        content="print('hello')\n",
        safety=safety,
    )
    assert write_result["ok"] is True
    assert write_result["data"]["sha256"]

    read_result = read_file(path="submission/code/hello.py", safety=safety)
    assert read_result["ok"] is True
    assert "print('hello')" in read_result["data"]["content"]

    list_result = list_files(path="submission", safety=safety, recursive=True)
    assert list_result["ok"] is True
    assert any(entry["path"].endswith("submission/code/hello.py") for entry in list_result["data"]["entries"])


def test_write_file_rejects_read_only_workspace_roots(workspace):
    safety = WorkspaceSafety(workspace)

    result = write_file(path="data/blocked.txt", content="nope", safety=safety)

    assert result["ok"] is False
    assert "not allowed" in result["error"].lower()


def test_read_file_rejects_sensitive_paths(workspace):
    safety = WorkspaceSafety(workspace)
    (workspace / ".env").write_text("LLM_API_KEY=test\n", encoding="utf-8")

    result = read_file(path=".env", safety=safety)

    assert result["ok"] is False
    assert "sensitive" in result["error"].lower()


def test_workspace_prefixed_paths_are_normalized(workspace):
    safety = WorkspaceSafety(workspace)

    write_result = write_file(
        path="workspace/submission/code/normalized.py",
        content="print('normalized')\n",
        safety=safety,
    )
    assert write_result["ok"] is True
    assert write_result["data"]["path"].endswith("submission/code/normalized.py")

    read_result = read_file(path="workspace/submission/code/normalized.py", safety=safety)
    assert read_result["ok"] is True
    assert "normalized" in read_result["data"]["content"]
