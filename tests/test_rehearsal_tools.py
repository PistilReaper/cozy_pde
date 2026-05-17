from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from agent_runner.config import RunnerConfig
from agent_runner.logger import ToolCallLogger
from agent_runner.safety import WorkspaceSafety
from agent_runner.tool_registry import build_tool_registry
from agent_runner.tools.log_tools import analyze_log
from agent_runner.tools.validate_tools import validate_submission


def test_rehearsal_profile_rejects_unbounded_train_command(workspace):
    config = RunnerConfig.from_workspace(workspace)
    registry = build_tool_registry(
        config,
        ToolCallLogger(workspace / "internal_logs" / "tool_calls.jsonl"),
    )

    result = registry.execute(
        "run_shell",
        {
            "command": "python runs/train.py --task task1",
            "profile": "rehearsal",
        },
    )

    assert result["ok"] is False
    assert "smoke" in result["error"].lower()


def test_rehearsal_profile_accepts_smoke_train_command(workspace):
    config = RunnerConfig.from_workspace(workspace)
    registry = build_tool_registry(
        config,
        ToolCallLogger(workspace / "internal_logs" / "tool_calls.jsonl"),
    )
    train_script = workspace / "runs" / "train.py"
    train_script.write_text("print('loss=0.1 smoke success')\n", encoding="utf-8")

    result = registry.execute(
        "run_shell",
        {
            "command": "python3 runs/train.py --smoke --max-batches 1",
            "profile": "rehearsal",
            "timeout_seconds": 30,
        },
    )

    assert result["ok"] is True
    assert result["data"]["returncode"] == 0


def test_run_shell_rejects_submission_code_mutation_and_restores_tree(workspace):
    config = RunnerConfig.from_workspace(workspace)
    registry = build_tool_registry(
        config,
        ToolCallLogger(workspace / "internal_logs" / "tool_calls.jsonl"),
    )
    original_path = workspace / "submission" / "code" / "safe.py"
    original_path.write_text("print('safe')\n", encoding="utf-8")
    script_path = workspace / "runs" / "mutate_code.py"
    script_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "target = Path('submission/code/unsafe.py')",
                "target.write_text(\"print('unsafe')\\n\", encoding='utf-8')",
                "print('mutated submission code')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = registry.execute(
        "run_shell",
        {
            "command": "python3 runs/mutate_code.py",
            "timeout_seconds": 30,
        },
    )

    assert result["ok"] is False
    assert "submission/code" in result["error"].lower()
    assert not (workspace / "submission" / "code" / "unsafe.py").exists()
    assert original_path.read_text(encoding="utf-8") == "print('safe')\n"


def test_run_shell_sanitizes_secret_env_and_redacts_output(workspace, monkeypatch):
    config = RunnerConfig.from_workspace(workspace)
    registry = build_tool_registry(
        config,
        ToolCallLogger(workspace / "internal_logs" / "tool_calls.jsonl"),
    )
    monkeypatch.setenv("MY_API_KEY", "sk-secretvalue123456789012345")
    script_path = workspace / "runs" / "print_env.py"
    script_path.write_text(
        "\n".join(
            [
                "import os",
                "print(os.getenv('MY_API_KEY', 'missing'))",
                "print('Authorization: Bearer sk-secretvalue123456789012345')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = registry.execute(
        "run_shell",
        {
            "command": "python3 runs/print_env.py",
            "timeout_seconds": 30,
        },
    )

    assert result["ok"] is True
    stdout_tail = result["data"]["stdout_tail"]
    assert "missing" in stdout_tail
    assert "sk-secretvalue123456789012345" not in stdout_tail
    assert "REDACTED" in stdout_tail
    log_text = (Path(result["data"]["log_path"])).read_text(encoding="utf-8")
    assert "sk-secretvalue123456789012345" not in log_text


def test_run_shell_blocks_direct_download_paths(workspace):
    config = RunnerConfig.from_workspace(workspace)
    registry = build_tool_registry(
        config,
        ToolCallLogger(workspace / "internal_logs" / "tool_calls.jsonl"),
    )

    commands = [
        "git clone https://github.com/example/repo.git",
        "huggingface-cli download repo/model",
        "kaggle datasets download foo/bar",
        "gdown https://drive.google.com/file/d/123/view",
        "aria2c https://example.com/model.pt",
        "pip install https://example.com/pkg.whl",
        "pip install git+https://github.com/example/repo.git",
        "python -c \"import requests; requests.get('https://example.com')\"",
        "python -c \"import urllib.request; urllib.request.urlopen('https://example.com')\"",
        "python -c \"import torch; torch.hub.load('repo', 'model')\"",
    ]

    for command in commands:
        result = registry.execute("run_shell", {"command": command})
        assert result["ok"] is False, command


def test_write_file_updates_code_manifest_for_submission_code(workspace):
    config = RunnerConfig.from_workspace(workspace)
    registry = build_tool_registry(
        config,
        ToolCallLogger(workspace / "internal_logs" / "tool_calls.jsonl"),
    )
    registry.set_context(task_id="task1", step_id="step-001")

    content = "print('generated by llm')\n"
    result = registry.execute(
        "write_file",
        {
            "path": "submission/code/foo.py",
            "content": content,
        },
    )

    assert result["ok"] is True
    manifest_path = workspace / "submission" / "code_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest) == 1
    assert manifest[0]["path"] == "submission/code/foo.py"
    assert manifest[0]["step_id"] == "step-001"
    assert manifest[0]["task_id"] == "task1"
    assert manifest[0]["size"] == len(content.encode("utf-8"))
    assert manifest[0]["sha256"] == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert "timestamp" in manifest[0]


def test_analyze_log_detects_failure_modes_and_smoke_success(workspace):
    safety = WorkspaceSafety(workspace)
    log_path = workspace / "runs" / "train.log"

    log_path.write_text("loss: 0.42\nval_loss: 0.30\nsmoke success\n", encoding="utf-8")
    result = analyze_log(path="runs/train.log", safety=safety)
    assert result["ok"] is True
    assert result["data"]["recommendation"] == "finalize_rehearsal"
    assert result["data"]["smoke_success"] is True

    log_path.write_text("RuntimeError: CUDA out of memory\n", encoding="utf-8")
    result = analyze_log(path="runs/train.log", safety=safety)
    assert result["data"]["recommendation"] == "reduce_model"
    assert result["data"]["oom_detected"] is True

    log_path.write_text("shape mismatch: got [8, 199, 256]\n", encoding="utf-8")
    result = analyze_log(path="runs/train.log", safety=safety)
    assert result["data"]["recommendation"] == "fix_code"
    assert result["data"]["shape_mismatch_detected"] is True

    log_path.write_text("loss=nan\n", encoding="utf-8")
    result = analyze_log(path="runs/train.log", safety=safety)
    assert result["data"]["recommendation"] == "rollback"
    assert result["data"]["nan_detected"] is True


def test_validate_submission_allows_rehearsal_subset_predictions(workspace, fake_test_hdf5):
    rehearsal_dir = workspace / "runs" / "rehearsal"
    rehearsal_dir.mkdir(parents=True, exist_ok=True)
    (workspace / "submission" / "code" / "generated.py").write_text("print('ok')\n", encoding="utf-8")

    with h5py.File(fake_test_hdf5, "r") as source:
        test_tensor = source["tensor"][:1]

    pred = test_tensor.copy()
    pred[:, 10:, :] = pred[:, 10:, :] + 0.05
    with h5py.File(rehearsal_dir / "pred.hdf5", "w") as handle:
        handle.create_dataset("pred", data=pred)
    (rehearsal_dir / "time.csv").write_text("train_time,inference_time\n1.0,0.1\n", encoding="utf-8")
    (rehearsal_dir / "logs.log").write_text(
        json.dumps(
            {
                "timestamp": "2026-05-17T00:00:00+00:00",
                "elapsed_seconds": 0.1,
                "response": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rehearsal_result = validate_submission(
        submission_dir=rehearsal_dir,
        test_hdf5=fake_test_hdf5,
        code_dir=workspace / "submission" / "code",
        rehearsal_mode=True,
    )
    assert rehearsal_result["ok"] is True
    assert rehearsal_result["data"]["rehearsal_only"] is True

    formal_result = validate_submission(
        submission_dir=rehearsal_dir,
        test_hdf5=fake_test_hdf5,
        code_dir=workspace / "submission" / "code",
        rehearsal_mode=False,
    )
    assert formal_result["ok"] is False
    assert "incompatible" in formal_result["error"].lower()
