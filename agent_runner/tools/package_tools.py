from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from . import failure, success
from .validate_tools import validate_submission


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _discover_bundles(submission_dir: Path) -> list[tuple[str, str, str]]:
    bundles: list[tuple[str, str, str]] = []
    if (submission_dir / "pred.hdf5").exists():
        bundles.append(("pred.hdf5", "time.csv", "logs.log"))
    for pred in sorted(submission_dir.glob("task*_pred.hdf5")):
        prefix = pred.name.removesuffix("_pred.hdf5")
        bundles.append((pred.name, f"{prefix}_time.csv", f"{prefix}_logs.log"))
    return bundles


def package_submission(*, submission_dir: str | Path, test_hdf5: str | Path | None = None) -> dict:
    submission_dir = Path(submission_dir)
    if not submission_dir.exists():
        return failure("package_submission", "Submission directory does not exist", submission_dir=str(submission_dir))

    submission_json = submission_dir / "submission.json"
    if not submission_json.exists():
        return failure("package_submission", "submission.json does not exist", path=str(submission_json))

    bundles = _discover_bundles(submission_dir)
    if not bundles:
        return failure("package_submission", "No prediction bundles found", submission_dir=str(submission_dir))

    validations = []
    for pred_name, time_name, log_name in bundles:
        result = validate_submission(
            submission_dir=submission_dir,
            test_hdf5=test_hdf5,
            pred_filename=pred_name,
            time_filename=time_name,
            logs_filename=log_name,
        )
        validations.append(result)
        if not result["ok"]:
            return failure("package_submission", result["error"], validation=result)

    warnings: list[str] = []
    if not (submission_dir / "methodology.pdf").exists():
        warnings.append("methodology.pdf is missing")

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
        warnings=warnings,
        bundles=[bundle[0] for bundle in bundles],
        validations=validations,
    )

