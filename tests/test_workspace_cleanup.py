from __future__ import annotations

from agent_runner.config import RunnerConfig
from agent_runner.main import prepare_run_workspace


def test_prepare_run_workspace_archives_and_cleans_output_directories(workspace):
    config = RunnerConfig.from_workspace(workspace)

    (workspace / "data" / "keep.txt").write_text("keep-data\n", encoding="utf-8")
    (workspace / "baselines" / "keep.txt").write_text("keep-baseline\n", encoding="utf-8")
    (workspace / "checkpoints" / "keep.txt").write_text("keep-checkpoint\n", encoding="utf-8")

    (workspace / "llm_logs" / "all_llm_calls.jsonl").write_text('{"old":"llm"}\n', encoding="utf-8")
    (workspace / "internal_logs" / "tool_calls.jsonl").write_text('{"old":"tool"}\n', encoding="utf-8")
    (workspace / "submission" / "old.txt").write_text("old-submission\n", encoding="utf-8")
    (workspace / "submission" / "code" / "old.py").write_text("print('old')\n", encoding="utf-8")
    (workspace / "research" / "cache" / "old.json").write_text("old-research\n", encoding="utf-8")
    (workspace / "research" / "papers" / "old.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (workspace / "runs" / "scratch" / "old.txt").write_text("old-scratch\n", encoding="utf-8")
    (workspace / "runs" / "rehearsal").mkdir(parents=True, exist_ok=True)
    (workspace / "runs" / "rehearsal" / "old.md").write_text("old-rehearsal\n", encoding="utf-8")
    archive_keep = workspace / "runs" / "archive" / "existing" / "keep.txt"
    archive_keep.parent.mkdir(parents=True, exist_ok=True)
    archive_keep.write_text("existing-archive\n", encoding="utf-8")

    archive_dir = prepare_run_workspace(config, run_label="autonomous_rehearsal")

    assert (workspace / "data" / "keep.txt").read_text(encoding="utf-8") == "keep-data\n"
    assert (workspace / "baselines" / "keep.txt").read_text(encoding="utf-8") == "keep-baseline\n"
    assert (workspace / "checkpoints" / "keep.txt").read_text(encoding="utf-8") == "keep-checkpoint\n"

    assert not (workspace / "submission" / "old.txt").exists()
    assert not (workspace / "submission" / "code" / "old.py").exists()
    assert not (workspace / "research" / "cache" / "old.json").exists()
    assert not (workspace / "research" / "papers" / "old.pdf").exists()
    assert not (workspace / "runs" / "scratch" / "old.txt").exists()
    assert not (workspace / "runs" / "rehearsal" / "old.md").exists()
    assert archive_keep.exists()

    assert archive_dir.exists()
    assert archive_dir.parent == workspace / "runs" / "archive"
    assert (archive_dir / "llm_logs" / "all_llm_calls.jsonl").read_text(encoding="utf-8") == '{"old":"llm"}\n'
    assert (archive_dir / "internal_logs" / "tool_calls.jsonl").read_text(encoding="utf-8") == '{"old":"tool"}\n'
    assert (archive_dir / "submission" / "old.txt").read_text(encoding="utf-8") == "old-submission\n"
    assert (archive_dir / "submission" / "code" / "old.py").read_text(encoding="utf-8") == "print('old')\n"
    assert (archive_dir / "research" / "cache" / "old.json").read_text(encoding="utf-8") == "old-research\n"
    assert (archive_dir / "runs" / "scratch" / "old.txt").read_text(encoding="utf-8") == "old-scratch\n"
    assert (archive_dir / "runs" / "rehearsal" / "old.md").read_text(encoding="utf-8") == "old-rehearsal\n"

    assert (workspace / "llm_logs").is_dir()
    assert (workspace / "internal_logs").is_dir()
    assert (workspace / "submission" / "code").is_dir()
    assert (workspace / "research" / "cache" / "raw").is_dir()
    assert (workspace / "research" / "papers").is_dir()
    assert (workspace / "runs" / "scratch").is_dir()
    assert (workspace / "runs" / "logs").is_dir()
    assert (workspace / "runs" / "snapshots").is_dir()


def test_prepare_run_workspace_returns_none_when_output_state_is_empty(workspace):
    config = RunnerConfig.from_workspace(workspace)

    archive_dir = prepare_run_workspace(config, run_label="autonomous")

    assert archive_dir is None
    assert (workspace / "runs" / "archive").is_dir()
