from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..config import RunnerConfig
from ..safety import WorkspaceSafety
from . import failure, success


def snapshot(*, config: RunnerConfig, safety: WorkspaceSafety, label: str | None = None) -> dict:
    snapshot_root = config.workspace_root / "runs" / "snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    stamp = label or datetime.now(timezone.utc).strftime("step_%Y%m%dT%H%M%SZ")
    target = snapshot_root / stamp
    submission_target = target / "submission"
    if target.exists():
        return failure("snapshot", "Snapshot label already exists", snapshot=str(target))

    submission_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(config.submission_dir, submission_target, dirs_exist_ok=False)
    return success("snapshot", f"Created snapshot {stamp}", snapshot_path=str(target))


def rollback(*, config: RunnerConfig, safety: WorkspaceSafety, snapshot_path: str) -> dict:
    snapshot_root = Path(snapshot_path)
    if not snapshot_root.exists():
        return failure("rollback", "Snapshot does not exist", snapshot_path=str(snapshot_root))
    submission_source = snapshot_root / "submission"
    if not submission_source.exists():
        return failure("rollback", "Snapshot does not contain submission state", snapshot_path=str(snapshot_root))

    for source in submission_source.rglob("*"):
        if source.is_dir():
            continue
        relative = source.relative_to(submission_source)
        target = config.submission_dir / relative
        check = safety.validate_write_path(target)
        if not check.ok:
            return failure("rollback", check.error or "rollback path rejected", path=str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    return success("rollback", "Rolled back submission files from snapshot", snapshot_path=str(snapshot_root))

