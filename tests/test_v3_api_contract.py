from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from textwrap import dedent

from cozy_pde_v3.validation.submission import _validate_submission_api_contract


def _write_script(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def _submission_task_code_path(task: str, filename: str) -> str:
    return "/".join(("submission", "code", task, filename))


def test_api_contract_accepts_shared_train_and_infer_entrypoints(workspace: Path) -> None:
    _write_script(
        workspace / "submission" / "code" / "train.py",
        """
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True)
        parser.add_argument("--config", required=True)
        parser.add_argument("--data_dir", required=True)
        parser.add_argument("--output_dir", required=True)
        """,
    )
    _write_script(
        workspace / "submission" / "code" / "infer.py",
        """
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True)
        parser.add_argument("--config", required=True)
        parser.add_argument("--data_dir", required=True)
        parser.add_argument("--output", required=True)
        """,
    )

    result = _validate_submission_api_contract(
        submission_dir=workspace / "submission",
        final_code_paths=[
            "submission/code/train.py",
            "submission/code/infer.py",
        ],
    )

    assert result["ok"] is True
    assert result["required_files"] == ["submission/code/infer.py", "submission/code/train.py"]


def test_api_contract_rejects_missing_required_cli_flags(workspace: Path) -> None:
    _write_script(
        workspace / "submission" / "code" / "train.py",
        """
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True)
        parser.add_argument("--config", required=True)
        """,
    )
    _write_script(
        workspace / "submission" / "code" / "infer.py",
        """
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True)
        parser.add_argument("--config", required=True)
        parser.add_argument("--data_dir", required=True)
        parser.add_argument("--output", required=True)
        """,
    )

    result = _validate_submission_api_contract(
        submission_dir=workspace / "submission",
        final_code_paths=[
            "submission/code/train.py",
            "submission/code/infer.py",
        ],
    )

    assert result["ok"] is False
    assert "submission/code/train.py missing required CLI flags" in result["failures"][0]


def test_api_contract_rejects_task_specific_code_fork_dirs(workspace: Path) -> None:
    _write_script(
        workspace / "submission" / "code" / "train.py",
        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--task")
        parser.add_argument("--config")
        parser.add_argument("--data_dir")
        parser.add_argument("--output_dir")
        """,
    )
    _write_script(
        workspace / "submission" / "code" / "infer.py",
        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--task")
        parser.add_argument("--config")
        parser.add_argument("--data_dir")
        parser.add_argument("--output")
        """,
    )
    _write_script(workspace / "submission" / "code" / "task2" / "train.py", "print('fork')\n")

    result = _validate_submission_api_contract(
        submission_dir=workspace / "submission",
        final_code_paths=[
            "submission/code/infer.py",
            _submission_task_code_path("task2", "train.py"),
            "submission/code/train.py",
        ],
    )

    assert result["ok"] is False
    assert "task-specific code fork detected" in "\n".join(result["failures"])
