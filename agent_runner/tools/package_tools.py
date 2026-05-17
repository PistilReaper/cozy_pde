from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from ..config import SubmissionTaskConfig, _default_submission_tasks
from . import failure, success
from .validate_tools import validate_task_submission_bundles


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def package_submission(
    *,
    submission_dir: str | Path,
    workspace_root: str | Path | None = None,
    task_configs: list[SubmissionTaskConfig] | None = None,
    code_dir: str | Path | None = None,
) -> dict:
    submission_dir = Path(submission_dir)
    if not submission_dir.exists():
        return failure("package_submission", "Submission directory does not exist", submission_dir=str(submission_dir))

    submission_json = submission_dir / "submission.json"
    if not submission_json.exists():
        return failure("package_submission", "submission.json does not exist", path=str(submission_json))

    methodology_path = submission_dir / "methodology.pdf"
    if not methodology_path.exists():
        return failure("package_submission", "methodology.pdf is required", path=str(methodology_path))

    resolved_workspace_root = Path(workspace_root) if workspace_root is not None else submission_dir.parent
    resolved_task_configs = task_configs or list(_default_submission_tasks().values())
    validations = validate_task_submission_bundles(
        submission_dir=submission_dir,
        task_configs=resolved_task_configs,
        workspace_root=resolved_workspace_root,
        code_dir=code_dir,
        rehearsal_mode=False,
    )
    for result in validations:
        if not result["ok"]:
            return failure("package_submission", result["error"], validation=result)

    manifest_entries = []
    for file_path in sorted(path for path in submission_dir.rglob("*") if path.is_file() and path.name != "submission.zip"):
        manifest_entries.append(
            {
                "path": str(file_path.relative_to(submission_dir)),
                "size": file_path.stat().st_size,
                "sha256": _sha256_file(file_path),
            }
        )

    manifest_path = submission_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_entries, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = submission_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(path for path in submission_dir.rglob("*") if path.is_file() and path != zip_path):
            archive.write(file_path, arcname=str(file_path.relative_to(submission_dir)))

    return success(
        "package_submission",
        f"Packaged submission with {len(manifest_entries)} files",
        zip_path=str(zip_path),
        manifest_path=str(manifest_path),
        warnings=[],
        bundles=[task_config.name for task_config in resolved_task_configs],
        validations=validations,
    )
