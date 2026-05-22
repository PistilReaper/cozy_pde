from __future__ import annotations

import json
from pathlib import Path

import pytest

from cozy_pde_v3 import cli as cli_module
from cozy_pde_v3.cli import build_parser
from cozy_pde_v3.config import load_config
from cozy_pde_v3.task_specs import DEFAULT_TASK_SPECS


def test_run_accepts_single_known_task() -> None:
    args = build_parser().parse_args(["run", "--config", "config.yaml", "--task", "task1"])

    assert args.command == "run"
    assert args.config == "config.yaml"
    assert args.task == "task1"


def test_run_rejects_comma_separated_tasks() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--config", "config.yaml", "--task", "task1,task2"])


def test_run_rejects_unknown_task() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--config", "config.yaml", "--task", "task4"])


def test_run_requires_config_and_task() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--task", "task1"])

    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--config", "config.yaml"])


def test_check_commands_require_only_config() -> None:
    provider_args = build_parser().parse_args(["check-provider", "--config", "config.yaml"])
    research_args = build_parser().parse_args(["check-research", "--config", "config.yaml"])

    assert provider_args.command == "check-provider"
    assert provider_args.config == "config.yaml"
    assert not hasattr(provider_args, "task")

    assert research_args.command == "check-research"
    assert research_args.config == "config.yaml"
    assert not hasattr(research_args, "task")

    with pytest.raises(SystemExit):
        build_parser().parse_args(["check-provider", "--task", "task1"])

    with pytest.raises(SystemExit):
        build_parser().parse_args(["check-research", "--task", "task1"])


@pytest.mark.parametrize("command_name", ["validate", "package", "status"])
def test_task_commands_require_config_and_single_task(command_name: str) -> None:
    args = build_parser().parse_args([command_name, "--config", "config.yaml", "--task", "task2"])

    assert args.command == command_name
    assert args.config == "config.yaml"
    assert args.task == "task2"

    with pytest.raises(SystemExit):
        build_parser().parse_args([command_name, "--config", "config.yaml"])

    with pytest.raises(SystemExit):
        build_parser().parse_args([command_name, "--task", "task2"])


def test_task_specs_expose_approved_hard_rule_fields() -> None:
    task1 = DEFAULT_TASK_SPECS["task1"]
    task2 = DEFAULT_TASK_SPECS["task2"]
    task3 = DEFAULT_TASK_SPECS["task3"]

    assert task1.input_steps == 10
    assert task1.output_steps == 200
    assert task1.total_steps == 200
    assert task1.spatial_points == 256
    assert task1.pred_shape == (0, 200, 256)
    assert task1.first_steps_must_match == 10
    assert task1.inference_time_limit_sec == 120.0
    assert task1.must_train_from_scratch is False

    assert task2.input_steps == 10
    assert task2.output_steps == 200
    assert task2.total_steps == 200
    assert task2.spatial_points == 256
    assert task2.pred_shape == (0, 200, 256)
    assert task2.first_steps_must_match == 10
    assert task2.inference_time_limit_sec == 120.0
    assert task2.must_train_from_scratch is True

    assert task3.input_steps == 20
    assert task3.output_steps == 400
    assert task3.total_steps == 400
    assert task3.spatial_points == 256
    assert task3.pred_shape == (1000, 400, 256)
    assert task3.first_steps_must_match == 20
    assert task3.inference_time_limit_sec == 120.0
    assert task3.must_train_from_scratch is True
    assert task3.allow_public_pretrained_weights is False


def test_task_specs_expose_configurable_default_filenames() -> None:
    for spec in DEFAULT_TASK_SPECS.values():
        assert spec.default_train_filenames
        assert spec.default_validation_filenames
        assert spec.default_test_filenames
        assert all(isinstance(name, str) for name in spec.default_train_filenames)
        assert all(isinstance(name, str) for name in spec.default_validation_filenames)
        assert all(isinstance(name, str) for name in spec.default_test_filenames)

    task1 = DEFAULT_TASK_SPECS["task1"]
    forbidden_names = {"task1_train.hdf5", "task2_val.h5", "KS_val.hdf5"}

    assert forbidden_names.isdisjoint(task1.default_train_filenames)
    assert forbidden_names.isdisjoint(task1.default_validation_filenames)
    assert forbidden_names.isdisjoint(task1.default_test_filenames)


def test_load_config_tracks_config_path_and_workspace_root() -> None:
    config = load_config("configs/task1.yaml")

    assert config.config_path == Path("configs/task1.yaml")
    assert config.workspace_root == Path.cwd()
    assert config.task_specs["task3"].pred_shape == (1000, 400, 256)


def test_load_config_allows_workspace_override() -> None:
    config = load_config("configs/task1.yaml", workspace_root="/tmp/cozy-workspace")

    assert config.workspace_root == Path("/tmp/cozy-workspace")


@pytest.mark.parametrize(
    ("argv", "expected_command"),
    [
        (["run", "--config", "config.yaml", "--task", "task1"], "run"),
        (["check-provider", "--config", "config.yaml"], "check-provider"),
        (["check-research", "--config", "config.yaml"], "check-research"),
        (["validate", "--config", "config.yaml", "--task", "task2"], "validate"),
        (["package", "--config", "config.yaml", "--task", "task2"], "package"),
        (["status", "--config", "config.yaml", "--task", "task3"], "status"),
    ],
)
def test_main_dispatches_each_v3_command_path(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected_command: str,
) -> None:
    seen: list[tuple[str, object, str | None]] = []

    monkeypatch.setattr(cli_module, "load_config", lambda path: {"config_path": path})

    def record(name: str):
        def _handler(config, task: str | None = None) -> int:
            seen.append((name, config, task))
            return 17

        return _handler

    monkeypatch.setattr(cli_module, "run_command", record("run"))
    monkeypatch.setattr(cli_module, "check_provider_command", record("check-provider"))
    monkeypatch.setattr(cli_module, "check_research_command", record("check-research"))
    monkeypatch.setattr(cli_module, "validate_command", record("validate"))
    monkeypatch.setattr(cli_module, "package_command", record("package"))
    monkeypatch.setattr(cli_module, "status_command", record("status"))

    exit_code = cli_module.main(argv)

    assert exit_code == 17
    assert seen == [
        (
            expected_command,
            {"config_path": "config.yaml"},
            next((value for index, value in enumerate(argv) if argv[index - 1] == "--task"), None),
        )
    ]


def test_validate_command_uses_v3_submission_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[object, list[str], bool]] = []

    def fake_validate_submission_bundle_v3(*, workspace_root, tasks, strict, **kwargs):
        seen.append((workspace_root, tasks, strict))
        return {"ok": True, "data": {"finalize_gate": {"overall_ok": True}}}

    monkeypatch.setattr(cli_module, "validate_submission_bundle_v3", fake_validate_submission_bundle_v3)

    config = load_config("config.yaml", workspace_root="/tmp/cozy-v3-cli")
    exit_code = cli_module.validate_command(config, "task2")

    assert exit_code == 0
    assert seen == [(config.workspace_root, ["task2"], True)]


def test_package_command_uses_v3_package_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[Path, list[str], list[Path], bool]] = []

    def fake_package_submission_v3(*, submission_dir, tasks, test_data_roots, strict, **kwargs):
        seen.append((submission_dir, tasks, test_data_roots, strict))
        return {"ok": True, "data": {"zip_path": str(submission_dir / "submission.zip")}}

    monkeypatch.setattr(cli_module, "package_submission_v3", fake_package_submission_v3)

    config = load_config("config.yaml", workspace_root="/tmp/cozy-v3-cli")
    exit_code = cli_module.package_command(config, "task3")

    assert exit_code == 0
    assert seen == [
        (
            config.workspace_root / "submission",
            ["task3"],
            [config.workspace_root / "data"],
            True,
        )
    ]


def test_status_command_reports_finalize_gate_failures_from_v3_validation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli_module,
        "validate_submission_bundle_v3",
        lambda **kwargs: {
            "ok": False,
            "error": "submission validation failed",
            "data": {
                "finalize_gate": {
                    "overall_ok": False,
                    "failures": ["missing code provenance linkage: submission/code/train.py"],
                    "warnings": [],
                }
            },
        },
    )

    config = load_config("config.yaml", workspace_root="/tmp/cozy-v3-cli")
    exit_code = cli_module.status_command(config, "task1")
    captured = capsys.readouterr().out

    assert exit_code == 1
    assert "missing code provenance linkage: submission/code/train.py" in captured


def test_run_command_returns_preflight_failure_when_provider_report_is_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        workspace_root: ./workspace
        provider:
          wire_api: responses
          primary:
            base_url: https://primary.example.com
            api_key_env: PRIMARY_API_KEY
            model_id: gpt-5.4
        artifacts:
          provider_report_path: ./workspace/capabilities/provider_report.json
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_config(config_path)

    exit_code = cli_module.run_command(config, "task1")
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "provider readiness report is missing"


def test_main_loads_config_with_cli_workspace_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[object, str | None]] = []

    def fake_load_config(path, workspace_root=None):
        seen.append((path, workspace_root))
        return {"config_path": path, "workspace_root": workspace_root}

    monkeypatch.setattr(cli_module, "load_config", fake_load_config)
    monkeypatch.setattr(cli_module, "run_command", lambda config, task: 0)

    exit_code = cli_module.main(
        ["run", "--config", "config.yaml", "--workspace-root", "/tmp/v3-workspace", "--task", "task1"]
    )

    assert exit_code == 0
    assert seen == [("config.yaml", "/tmp/v3-workspace")]
