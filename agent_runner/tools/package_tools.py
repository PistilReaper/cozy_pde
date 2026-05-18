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


def _detect_task_configs(
    *,
    submission_dir: Path,
    task_configs: list[SubmissionTaskConfig],
) -> list[SubmissionTaskConfig]:
    detected = [
        task_config
        for task_config in task_configs
        if any(
            (submission_dir / filename).exists()
            for filename in (
                task_config.pred_filename,
                task_config.time_filename,
                task_config.logs_filename,
            )
        )
    ]
    return detected or task_configs


def _package_file_paths(
    *,
    submission_dir: Path,
    task_configs: list[SubmissionTaskConfig],
    code_dir: Path,
    include_manifest: bool,
) -> list[Path]:
    allowed: list[Path] = []
    for name in ("submission.json", "methodology.pdf", "README.md", "code_manifest.json"):
        candidate = submission_dir / name
        if candidate.exists():
            allowed.append(candidate)
    if include_manifest:
        manifest_path = submission_dir / "manifest.json"
        if manifest_path.exists():
            allowed.append(manifest_path)
    for task_config in task_configs:
        for filename in (
            task_config.pred_filename,
            task_config.time_filename,
            task_config.logs_filename,
        ):
            candidate = submission_dir / filename
            if candidate.exists():
                allowed.append(candidate)
    if code_dir.exists():
        allowed.extend(path for path in sorted(code_dir.rglob("*")) if path.is_file())
    return sorted(dict.fromkeys(allowed))


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
    configured_task_configs = task_configs or list(_default_submission_tasks().values())
    resolved_task_configs = _detect_task_configs(
        submission_dir=submission_dir,
        task_configs=configured_task_configs,
    )
    resolved_code_dir = Path(code_dir) if code_dir is not None else submission_dir / "code"
    validations = validate_task_submission_bundles(
        submission_dir=submission_dir,
        task_configs=resolved_task_configs,
        workspace_root=resolved_workspace_root,
        code_dir=resolved_code_dir,
        rehearsal_mode=False,
    )
    for result in validations:
        if not result["ok"]:
            return failure("package_submission", result["error"], validation=result)

    manifest_entries = []
    packaged_files = _package_file_paths(
        submission_dir=submission_dir,
        task_configs=resolved_task_configs,
        code_dir=resolved_code_dir,
        include_manifest=False,
    )
    for file_path in packaged_files:
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
        zipped_files = _package_file_paths(
            submission_dir=submission_dir,
            task_configs=resolved_task_configs,
            code_dir=resolved_code_dir,
            include_manifest=True,
        )
        for file_path in zipped_files:
            if file_path == zip_path:
                continue
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
