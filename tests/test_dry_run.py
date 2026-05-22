from __future__ import annotations

import pytest

from cozy_pde_v3.cli import build_parser


def _removed_mode_name() -> str:
    return "_".join(["autonomous", "dry", "run"])


def test_v3_cli_rejects_removed_dry_run_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["dry-run", "--config", "config.yaml", "--task", "task1"])


def test_v3_cli_rejects_removed_mode_flag_value() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "run",
                "--config",
                "config.yaml",
                "--task",
                "task1",
                "--mode",
                _removed_mode_name(),
            ]
        )
