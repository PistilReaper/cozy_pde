from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from agent_runner.config import RunnerConfig
from agent_runner.main import _scan_for_secret_leaks, run_final_check
from agent_runner.tools.package_tools import package_submission


def _write_task_test(path: Path, *, offset: float) -> None:
    array = (np.arange(2 * 200 * 256, dtype=np.float32).reshape(2, 200, 256) / 1000.0) + offset
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=array)


def _write_task_bundle(workspace: Path, task: str, test_hdf5: Path) -> None:
    submission_dir = workspace / "submission"
    with h5py.File(test_hdf5, "r") as source:
        tensor = source["tensor"][:]
    pred = tensor.copy()
    pred[:, 10:, :] = pred[:, 10:, :] + 0.01
    with h5py.File(submission_dir / f"{task}_pred.hdf5", "w") as handle:
        handle.create_dataset("pred", data=pred)
    (submission_dir / f"{task}_time.csv").write_text("train_time,inference_time\n1.0,0.2\n", encoding="utf-8")
    (submission_dir / f"{task}_logs.log").write_text(
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


def _prepare_strict_submission_workspace(workspace: Path) -> None:
    (workspace / "submission" / "code" / "placeholder.py").write_text("print('ok')\n", encoding="utf-8")
    (workspace / "llm_logs" / "all_llm_calls.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-05-16T12:00:00+00:00",
                "elapsed_seconds": 0.1,
                "model": "gpt-5.4",
                "profile": "coder",
                "phase": "implementation",
                "raw_response": {"id": "resp_1"},
                "tool_calls": [
                    {
                        "name": "write_file",
                        "call_id": "call_1",
                        "arguments": {
                            "path": "submission/code/placeholder.py",
                            "content": "print('ok')\n",
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "submission" / "submission.json").write_text("{}", encoding="utf-8")
    _write_task_test(workspace / "data" / "task1_test.hdf5", offset=0.0)
    _write_task_test(workspace / "data" / "task2_test.hdf5", offset=10.0)
    _write_task_bundle(workspace, "task1", workspace / "data" / "task1_test.hdf5")
    _write_task_bundle(workspace, "task2", workspace / "data" / "task2_test.hdf5")


def test_final_check_warns_on_missing_artifacts_without_strict(workspace, capsys):
    (workspace / "submission" / "code" / "placeholder.py").write_text("print('ok')\n", encoding="utf-8")
    (workspace / "llm_logs" / "all_llm_calls.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-05-16T12:00:00+00:00",
                "elapsed_seconds": 0.1,
                "response": "hello",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = RunnerConfig.from_workspace(workspace)
    exit_code = run_final_check(config, strict=False)
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "WARN" in captured


def test_final_check_fails_on_missing_artifacts_with_strict(workspace):
    (workspace / "submission" / "code" / "placeholder.py").write_text("print('ok')\n", encoding="utf-8")
    (workspace / "llm_logs" / "all_llm_calls.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-05-16T12:00:00+00:00",
                "elapsed_seconds": 0.1,
                "response": "hello",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = RunnerConfig.from_workspace(workspace)
    exit_code = run_final_check(config, strict=True)

    assert exit_code == 1


def test_final_check_strict_requires_methodology_pdf(workspace):
    _prepare_strict_submission_workspace(workspace)
    config = RunnerConfig.from_workspace(workspace)

    exit_code = run_final_check(config, strict=True)

    assert exit_code == 1


def test_final_check_strict_uses_task_specific_test_hdf5(workspace):
    _prepare_strict_submission_workspace(workspace)
    (workspace / "submission" / "methodology.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    config = RunnerConfig.from_workspace(workspace)

    exit_code = run_final_check(config, strict=True)

    assert exit_code == 0


def test_package_submission_requires_methodology_and_task_specific_bundles(workspace):
    _prepare_strict_submission_workspace(workspace)

    missing_methodology = package_submission(submission_dir=workspace / "submission")
    assert missing_methodology["ok"] is False
    assert "methodology" in missing_methodology["error"].lower()

    (workspace / "submission" / "methodology.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    packaged = package_submission(submission_dir=workspace / "submission")
    assert packaged["ok"] is True
    assert packaged["data"]["bundles"] == ["task1", "task2"]


def test_secret_leak_scan_ignores_task_specific_plain_text(workspace):
    log_path = workspace / "submission" / "task1_logs.log"
    log_path.write_text("task-specific planning note\n", encoding="utf-8")

    hits = _scan_for_secret_leaks([workspace / "submission"])

    assert hits == []


def test_readiness_check_reports_failure_when_any_stage_fails(workspace, monkeypatch, capsys):
    from agent_runner import main as main_module

    config = RunnerConfig.from_workspace(workspace)
    monkeypatch.setattr(main_module, "run_preflight", lambda cfg: 0)
    monkeypatch.setattr(main_module, "run_provider_health_check", lambda cfg: 0)
    monkeypatch.setattr(main_module, "run_autonomous_dry_run", lambda cfg, tasks, max_steps: 0)
    monkeypatch.setattr(main_module, "run_autonomous_rehearsal", lambda cfg, tasks, max_steps, max_train_seconds_per_task: 1)
    monkeypatch.setattr(main_module, "run_final_check", lambda cfg, strict=False: 0)

    class FakeCompleted:
        returncode = 0
        stdout = "55 passed\n"
        stderr = ""

    monkeypatch.setattr(main_module.subprocess, "run", lambda *args, **kwargs: FakeCompleted())

    exit_code = main_module.run_readiness_check(config, tasks=["task1", "task2"], max_steps=3, max_train_seconds_per_task=10)
    captured = capsys.readouterr().out

    assert exit_code == 1
    assert "autonomous_rehearsal" in captured
