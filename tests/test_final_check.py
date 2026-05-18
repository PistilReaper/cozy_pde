from __future__ import annotations

import json
from pathlib import Path
import hashlib

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


def _write_code_manifest(workspace: Path) -> None:
    content = (workspace / "submission" / "code" / "placeholder.py").read_bytes()
    (workspace / "submission" / "code_manifest.json").write_text(
        json.dumps(
            [
                {
                    "path": "submission/code/placeholder.py",
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size": len(content),
                    "step_id": "step-001",
                    "task_id": "task1",
                    "timestamp": "2026-05-17T00:00:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )


def _write_repeated_code_manifest(workspace: Path) -> None:
    code_path = workspace / "submission" / "code" / "placeholder.py"
    code_path.write_text("print('final')\n", encoding="utf-8")
    final_content = code_path.read_bytes()
    (workspace / "submission" / "code_manifest.json").write_text(
        json.dumps(
            [
                {
                    "path": "submission/code/placeholder.py",
                    "sha256": hashlib.sha256(b"print('old')\n").hexdigest(),
                    "size": len(b"print('old')\n"),
                    "step_id": "step-000",
                    "task_id": "task1",
                    "timestamp": "2026-05-16T00:00:00+00:00",
                },
                {
                    "path": "submission/code/placeholder.py",
                    "sha256": hashlib.sha256(final_content).hexdigest(),
                    "size": len(final_content),
                    "step_id": "step-001",
                    "task_id": "task1",
                    "timestamp": "2026-05-17T00:00:00+00:00",
                },
            ]
        ),
        encoding="utf-8",
    )


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
    _write_code_manifest(workspace)
    (workspace / "submission" / "methodology.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    config = RunnerConfig.from_workspace(workspace)

    exit_code = run_final_check(config, strict=True)

    assert exit_code == 0


def test_final_check_strict_requires_code_manifest_json(workspace):
    _prepare_strict_submission_workspace(workspace)
    (workspace / "submission" / "methodology.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    config = RunnerConfig.from_workspace(workspace)

    exit_code = run_final_check(config, strict=True)

    assert exit_code == 1


def test_final_check_strict_fails_when_validate_responses_logs_fails(workspace, monkeypatch):
    _prepare_strict_submission_workspace(workspace)
    (workspace / "submission" / "methodology.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    _write_code_manifest(workspace)
    config = RunnerConfig.from_workspace(workspace)

    def fake_validate_responses_logs(path, *, workspace_root=None):
        assert Path(path) == config.llm_log_path
        assert Path(workspace_root) == config.workspace_root
        return {
            "ok": False,
            "error": "synthetic responses log failure",
        }

    monkeypatch.setattr("agent_runner.main.validate_responses_logs", fake_validate_responses_logs)

    exit_code = run_final_check(config, strict=True)

    assert exit_code == 1


def test_final_check_strict_uses_final_code_manifest_entry_per_path(workspace):
    _prepare_strict_submission_workspace(workspace)
    (workspace / "submission" / "code" / "placeholder.py").write_text("print('final')\n", encoding="utf-8")
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
                            "content": "print('old')\n",
                        },
                    },
                    {
                        "name": "write_file",
                        "call_id": "call_2",
                        "arguments": {
                            "path": "submission/code/placeholder.py",
                            "content": "print('final')\n",
                        },
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_repeated_code_manifest(workspace)
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
    monkeypatch.setattr(main_module, "run_startup_readiness", lambda cfg: 1)

    exit_code = main_module.run_readiness_check(config, tasks=["task1", "task2"], max_steps=3, max_train_seconds_per_task=10)
    captured = capsys.readouterr().out

    assert exit_code == 1
    assert "startup_readiness" in captured


def test_readiness_check_does_not_require_final_submission_artifacts(workspace, monkeypatch, capsys):
    from agent_runner import main as main_module

    config = RunnerConfig.from_workspace(workspace)
    monkeypatch.setattr(main_module, "run_startup_readiness", lambda cfg: 0)

    exit_code = main_module.run_readiness_check(config, tasks=["task1", "task2"], max_steps=3, max_train_seconds_per_task=10)
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "PASS pytest" not in captured
    assert "PASS provider_health_check" not in captured
    assert "FAIL provider_health_check" not in captured
    assert "PASS final_check" not in captured
    assert "FAIL final_check" not in captured
    assert "PASS autonomous_rehearsal" not in captured
    assert "FAIL autonomous_rehearsal" not in captured


def test_readiness_check_reports_stage_exception_as_failure(workspace, monkeypatch, capsys):
    from agent_runner import main as main_module

    config = RunnerConfig.from_workspace(workspace)
    def boom(cfg):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(main_module, "run_startup_readiness", boom)

    exit_code = main_module.run_readiness_check(config, tasks=["task1", "task2"], max_steps=3, max_train_seconds_per_task=10)
    captured = capsys.readouterr().out

    assert exit_code == 1
    assert "startup_readiness" in captured
    assert "network unavailable" in captured
