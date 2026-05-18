from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from agent_runner.config import RunnerConfig
from agent_runner.main import (
    _build_deterministic_methodology_text,
    _prepare_session_logs,
    export_task_logs,
    parse_args,
    run_package_final,
)
from agent_runner.prompts import SYSTEM_PROMPT, build_task_instruction_block


def _write_task_test(path, *, samples: int, total_steps: int, offset: float = 0.0) -> None:
    array = (
        np.arange(samples * total_steps * 256, dtype=np.float32).reshape(samples, total_steps, 256) / 1000.0
    ) + offset
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=array)


def _write_task_bundle(workspace, task: str, *, samples: int, total_steps: int, input_steps: int, offset: float = 0.0) -> None:
    submission_dir = workspace / "submission"
    test_hdf5 = workspace / "data" / f"{task}_test.hdf5"
    _write_task_test(test_hdf5, samples=samples, total_steps=total_steps, offset=offset)
    with h5py.File(test_hdf5, "r") as source:
        tensor = source["tensor"][:]
    pred = tensor.copy()
    pred[:, input_steps:, :] = pred[:, input_steps:, :] + 0.01
    with h5py.File(submission_dir / f"{task}_pred.hdf5", "w") as handle:
        handle.create_dataset("pred", data=pred)
    (submission_dir / f"{task}_time.csv").write_text("train_time,inference_time\n1.0,0.2\n", encoding="utf-8")
    (submission_dir / f"{task}_logs.log").write_text(
        json.dumps(
            {
                "timestamp": "2026-05-18T00:00:00+00:00",
                "elapsed_seconds": 0.1,
                "response": f"{task} ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_parse_args_accepts_single_task_autonomous():
    args = parse_args(["--mode", "autonomous", "--config", "config.yaml", "--tasks", "task1"])

    assert args.tasks == ["task1"]
    assert args.allow_multi_task_session is False


def test_parse_args_rejects_multi_task_autonomous_without_override(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--mode", "autonomous", "--config", "config.yaml", "--tasks", "task1,task2"])

    captured = capsys.readouterr()
    assert "independent task sessions" in captured.err


def test_parse_args_allows_multi_task_autonomous_with_override():
    args = parse_args(
        [
            "--mode",
            "autonomous",
            "--config",
            "config.yaml",
            "--tasks",
            "task1,task2",
            "--allow-multi-task-session",
        ]
    )

    assert args.tasks == ["task1", "task2"]
    assert args.allow_multi_task_session is True


def test_runner_config_task_scoped_log_paths_are_distinct(workspace):
    config = RunnerConfig.from_workspace(workspace)

    task1_config = config.with_session("task1")
    task2_config = config.with_session("task2")

    assert task1_config.llm_log_path == workspace / "llm_logs" / "task1_all_llm_calls.jsonl"
    assert task2_config.llm_log_path == workspace / "llm_logs" / "task2_all_llm_calls.jsonl"
    assert task1_config.tool_log_path == workspace / "internal_logs" / "task1_tool_calls.jsonl"
    assert task2_config.tool_log_path == workspace / "internal_logs" / "task2_tool_calls.jsonl"
    assert task1_config.llm_log_path != task2_config.llm_log_path
    assert task1_config.tool_log_path != task2_config.tool_log_path


def test_starting_task2_session_does_not_delete_task1_logs(workspace):
    config = RunnerConfig.from_workspace(workspace)
    task1_config = config.with_session("task1")
    task2_config = config.with_session("task2")

    _prepare_session_logs(task1_config)
    task1_config.llm_log_path.write_text('{"task":"task1"}\n', encoding="utf-8")
    task1_config.tool_log_path.write_text('{"task":"task1"}\n', encoding="utf-8")

    _prepare_session_logs(task2_config)

    assert task1_config.llm_log_path.read_text(encoding="utf-8") == '{"task":"task1"}\n'
    assert task1_config.tool_log_path.read_text(encoding="utf-8") == '{"task":"task1"}\n'
    assert task2_config.llm_log_path.read_text(encoding="utf-8") == ""
    assert task2_config.tool_log_path.read_text(encoding="utf-8") == ""


def test_export_task_logs_writes_only_requested_task_log(workspace):
    task1_log = workspace / "llm_logs" / "task1_all_llm_calls.jsonl"
    task2_log = workspace / "llm_logs" / "task2_all_llm_calls.jsonl"
    task1_log.write_text(
        json.dumps({"timestamp": "2026-05-18T00:00:00+00:00", "elapsed_seconds": 0.1, "response": "task1"})
        + "\n",
        encoding="utf-8",
    )
    task2_log.write_text(
        json.dumps({"timestamp": "2026-05-18T00:00:01+00:00", "elapsed_seconds": 0.2, "response": "task2"})
        + "\n",
        encoding="utf-8",
    )

    result = export_task_logs(workspace=workspace, tasks=["task2"])

    assert result["ok"] is True
    assert (workspace / "submission" / "task1_logs.log").exists() is False
    assert (workspace / "submission" / "task2_logs.log").read_text(encoding="utf-8") == task2_log.read_text(encoding="utf-8")


def test_export_task_logs_copies_shared_multi_task_session_when_allowed(workspace):
    shared_log = workspace / "llm_logs" / "task1_task2_all_llm_calls.jsonl"
    shared_log.write_text(
        json.dumps({"timestamp": "2026-05-18T00:00:00+00:00", "elapsed_seconds": 0.1, "response": "shared"})
        + "\n",
        encoding="utf-8",
    )

    result = export_task_logs(workspace=workspace, tasks=["task1", "task2"], allow_multi_task_session=True)

    assert result["ok"] is True
    task1_export = (workspace / "submission" / "task1_logs.log").read_text(encoding="utf-8")
    task2_export = (workspace / "submission" / "task2_logs.log").read_text(encoding="utf-8")
    assert task1_export == shared_log.read_text(encoding="utf-8")
    assert task2_export == shared_log.read_text(encoding="utf-8")


def test_package_final_validates_selected_tasks_and_creates_submission_zip(workspace):
    config = RunnerConfig.from_workspace(workspace)
    _write_task_bundle(workspace, "task1", samples=2, total_steps=200, input_steps=10)
    _write_task_bundle(workspace, "task3", samples=1000, total_steps=400, input_steps=20, offset=10.0)
    (workspace / "submission" / "code" / "task1").mkdir(parents=True, exist_ok=True)
    (workspace / "submission" / "code" / "task1" / "generated.py").write_text("print('task1')\n", encoding="utf-8")
    (workspace / "submission" / "code" / "task3").mkdir(parents=True, exist_ok=True)
    (workspace / "submission" / "code" / "task3" / "generated.py").write_text("print('task3')\n", encoding="utf-8")
    (workspace / "submission" / "README.md").write_text("# Submission\n\nTask outputs are packaged here.\n", encoding="utf-8")

    exit_code = run_package_final(config, tasks=["task1", "task3"])

    assert exit_code == 0
    assert (workspace / "submission" / "submission.json").exists()
    assert (workspace / "submission" / "methodology.pdf").exists()
    assert (workspace / "submission" / "submission.zip").exists()


def test_task_prompt_blocks_include_required_hard_rules():
    task1_block = build_task_instruction_block("task1")
    task2_block = build_task_instruction_block("task2")
    task3_block = build_task_instruction_block("task3")

    assert "official Task 1 PDEBench checkpoints may be used" in task1_block
    assert "workspace/checkpoints/task1_official/" in task1_block
    assert "workspace/submission/task1_pred.hdf5" in task1_block
    assert "workspace/submission/code/task1/" in task1_block

    assert "trained from scratch" in task2_block
    assert "Do not use Task 1 data, Task 1 checkpoint, or Task 1 fine-tuned weights" in task2_block
    assert "workspace/submission/task2_pred.hdf5" in task2_block
    assert "workspace/submission/code/task2/" in task2_block

    assert "Kuramoto-Sivashinsky" in task3_block
    assert "unknown `lambda2` at test time" in task3_block
    assert "train from scratch" in task3_block
    assert "workspace/submission/task3_pred.hdf5" in task3_block
    assert "parameter inference in logs" in task3_block
    assert "workspace/submission/code/task3/" in task3_block
    assert "Do not use Task 1 or Task 2 weights for Task 3" in SYSTEM_PROMPT


def test_deterministic_methodology_text_describes_runner_architecture_and_no_fake_experiments():
    text = _build_deterministic_methodology_text(["task1", "task2", "task3"])

    assert "CozyPDE Deterministic Methodology" in text
    assert "task-isolated formal sessions" in text
    assert "tool-mediated file generation" in text
    assert "logging and provenance" in text
    assert "experiment loop" in text
    assert "This document does not claim task-specific model results" in text
    assert "Task 1, Task 2, and Task 3" in text


def test_package_final_rejects_task2_code_that_references_task1_checkpoint(workspace):
    config = RunnerConfig.from_workspace(workspace)
    _write_task_bundle(workspace, "task2", samples=2, total_steps=200, input_steps=10)
    (workspace / "submission" / "code" / "task2").mkdir(parents=True, exist_ok=True)
    (workspace / "submission" / "code" / "task2" / "train.py").write_text(
        "CKPT = 'workspace/checkpoints/task1_official/1D_Burgers_Sols_Nu0.001_FNO.pt'\n",
        encoding="utf-8",
    )

    exit_code = run_package_final(config, tasks=["task2"])

    assert exit_code == 1
    assert not (workspace / "submission" / "submission.zip").exists()


def test_package_final_rejects_task3_code_that_references_task2_weights(workspace):
    config = RunnerConfig.from_workspace(workspace)
    _write_task_bundle(workspace, "task3", samples=1000, total_steps=400, input_steps=20)
    (workspace / "submission" / "code" / "task3").mkdir(parents=True, exist_ok=True)
    (workspace / "submission" / "code" / "task3" / "infer.py").write_text(
        "resume_path = 'workspace/runs/task2/checkpoints/best_task2_model.pth'\n",
        encoding="utf-8",
    )

    exit_code = run_package_final(config, tasks=["task3"])

    assert exit_code == 1
    assert not (workspace / "submission" / "submission.zip").exists()


def test_package_final_cli_builds_zip_with_only_expected_submission_files(workspace):
    _write_task_bundle(workspace, "task1", samples=2, total_steps=200, input_steps=10)
    _write_task_bundle(workspace, "task2", samples=2, total_steps=200, input_steps=10, offset=10.0)
    _write_task_bundle(workspace, "task3", samples=1000, total_steps=400, input_steps=20, offset=20.0)
    for task in ("task1", "task2", "task3"):
        task_dir = workspace / "submission" / "code" / task
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "generated.py").write_text(f"print('{task}')\n", encoding="utf-8")
    (workspace / "submission" / "code" / "train.py").write_text("print('train')\n", encoding="utf-8")
    (workspace / "submission" / "code" / "infer.py").write_text("print('infer')\n", encoding="utf-8")
    (workspace / "submission" / "code" / "README.md").write_text("# code\n", encoding="utf-8")
    (workspace / "submission" / "sample_submission.txt").write_text("do not package\n", encoding="utf-8")
    (workspace / "submission" / "old.log").write_text("old\n", encoding="utf-8")
    (workspace / "data" / "do_not_package.txt").write_text("data\n", encoding="utf-8")
    (workspace / "runs" / "do_not_package.txt").write_text("runs\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_runner.main",
            "--mode",
            "package_final",
            "--config",
            "agent_runner/config.example.yaml",
            "--workspace",
            str(workspace),
            "--tasks",
            "task1,task2,task3",
        ],
        cwd="/home/cty/cozy_pde",
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    zip_path = workspace / "submission" / "submission.zip"
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = sorted(archive.namelist())

    assert "submission.json" in names
    assert "methodology.pdf" in names
    assert "manifest.json" in names
    assert "task1_pred.hdf5" in names
    assert "task2_pred.hdf5" in names
    assert "task3_pred.hdf5" in names
    assert "code/task1/generated.py" in names
    assert "code/task2/generated.py" in names
    assert "code/task3/generated.py" in names
    assert "code/train.py" in names
    assert "code/infer.py" in names
    assert "code/README.md" in names
    assert "sample_submission.txt" not in names
    assert "old.log" not in names
    assert all(not name.startswith("data/") for name in names)
    assert all(not name.startswith("checkpoints/") for name in names)
    assert all(not name.startswith("runs/") for name in names)


def test_pytest_ini_limits_collection_to_repo_tests():
    text = Path("/home/cty/cozy_pde/pytest.ini").read_text(encoding="utf-8")

    assert "testpaths = tests" in text
    assert "norecursedirs =" in text
    assert "workspace/baselines" in text
    assert ".venv" in text
