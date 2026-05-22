from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from cozy_pde_v3.validation.submission import _validate_task_policy_rules


def _write_policy_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def test_task_policy_allows_task1_official_checkpoint_reference(workspace: Path) -> None:
    _write_policy_file(
        workspace / "submission" / "code" / "train.py",
        """
        TASK1_CHECKPOINT = "workspace/checkpoints/task1_official/model.pt"
        """,
    )
    _write_policy_file(
        workspace / "submission" / "code" / "infer.py",
        """
        DATA_ROOT = "workspace/data"
        """,
    )

    result = _validate_task_policy_rules(
        workspace_root=workspace,
        submission_dir=workspace / "submission",
        tasks=["task1"],
    )

    assert result["ok"] is True


def test_task_policy_rejects_task2_reuse_of_task1_checkpoint_or_data(workspace: Path) -> None:
    _write_policy_file(
        workspace / "submission" / "code" / "train.py",
        """
        TASK2_BOOTSTRAP = "workspace/checkpoints/task1_official/model.pt"
        TASK2_DATA = "workspace/data/task1_train.hdf5"
        """,
    )
    _write_policy_file(workspace / "submission" / "code" / "infer.py", "print('infer')\n")

    result = _validate_task_policy_rules(
        workspace_root=workspace,
        submission_dir=workspace / "submission",
        tasks=["task2"],
    )

    assert result["ok"] is False
    joined = "\n".join(result["failures"])
    assert "task2 must be trained from scratch" in joined
    assert "task2 references forbidden Task 1 source" in joined


def test_task_policy_rejects_task3_public_pretrained_weights_and_validation_leakage(workspace: Path) -> None:
    _write_policy_file(
        workspace / "submission" / "code" / "train.py",
        """
        import torch

        MODEL = torch.hub.load("owner/repo", "model")
        VALIDATION_FILE = "KS_val.hdf5"
        """,
    )
    _write_policy_file(workspace / "submission" / "code" / "infer.py", "print('infer')\n")

    result = _validate_task_policy_rules(
        workspace_root=workspace,
        submission_dir=workspace / "submission",
        tasks=["task3"],
    )

    assert result["ok"] is False
    joined = "\n".join(result["failures"])
    assert "task3 forbids public pretrained weights" in joined
    assert "hardcoded validation/test reference" in joined
