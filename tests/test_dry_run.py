from __future__ import annotations

from agent_runner.config import RunnerConfig
from agent_runner.logger import ToolCallLogger
from agent_runner.tool_registry import build_tool_registry


def test_dry_run_allows_docs_reads_and_runs_writes_but_blocks_submission_and_shell(tmp_path):
    project_root = tmp_path
    workspace = tmp_path / "workspace"
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    for relative in ["runs/autonomous_dry_run", "internal_logs", "llm_logs", "submission/code"]:
        (workspace / relative).mkdir(parents=True, exist_ok=True)

    (docs_dir / "plan.md").write_text("dry run docs", encoding="utf-8")
    config = RunnerConfig.from_workspace(workspace, project_root=project_root)
    registry = build_tool_registry(
        config,
        ToolCallLogger(workspace / "internal_logs" / "tool_calls.jsonl"),
        allow_run_shell=False,
        allow_submission_writes=False,
        extra_read_roots=[docs_dir],
    )

    read_relative_parent = registry.execute("read_file", {"path": "../docs/plan.md"})
    assert read_relative_parent["ok"] is True

    read_docs_prefix = registry.execute("read_file", {"path": "docs/plan.md"})
    assert read_docs_prefix["ok"] is True

    write_ok = registry.execute(
        "write_file",
        {"path": "runs/autonomous_dry_run/plan.md", "content": "dry run plan"},
    )
    assert write_ok["ok"] is True

    write_blocked = registry.execute(
        "write_file",
        {"path": "submission/code/blocked.py", "content": "print('blocked')"},
    )
    assert write_blocked["ok"] is False

    shell_blocked = registry.execute("run_shell", {"command": "echo hi"})
    assert shell_blocked["ok"] is False
