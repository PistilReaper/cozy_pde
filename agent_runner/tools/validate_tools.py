from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from ..config import SubmissionTaskConfig
from . import failure, success


def _normalize_logged_path(path_value: object, workspace_root: Path | None) -> str | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None

    candidate = Path(path_value)
    if candidate.parts and candidate.parts[0] == "workspace":
        candidate = Path(*candidate.parts[1:])
    if workspace_root is None:
        return candidate.as_posix()

    try:
        if candidate.is_absolute():
            return candidate.resolve().relative_to(workspace_root).as_posix()
        return (workspace_root / candidate).resolve().relative_to(workspace_root).as_posix()
    except ValueError:
        return candidate.as_posix()


def validate_jsonl_logs(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return failure("validate_jsonl_logs", "Log file does not exist", path=str(path))

    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            return failure("validate_jsonl_logs", f"Empty line at {index}", path=str(path), line=index)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            return failure("validate_jsonl_logs", f"Invalid JSON at line {index}: {exc}", path=str(path), line=index)
        for required in ("timestamp", "elapsed_seconds"):
            if required not in payload:
                return failure("validate_jsonl_logs", f"Missing {required} at line {index}", path=str(path), line=index)
        if "response" not in payload and "tool_calls" not in payload:
            return failure("validate_jsonl_logs", f"Missing response/tool_calls at line {index}", path=str(path), line=index)
        try:
            datetime.fromisoformat(payload["timestamp"])
        except ValueError as exc:
            return failure("validate_jsonl_logs", f"Invalid timestamp at line {index}: {exc}", path=str(path), line=index)
        if not isinstance(payload["elapsed_seconds"], (int, float)):
            return failure("validate_jsonl_logs", f"elapsed_seconds must be numeric at line {index}", path=str(path), line=index)

    return success("validate_jsonl_logs", f"Validated {len(lines)} JSONL lines", path=str(path), lines=len(lines))


def validate_responses_logs(path: str | Path, *, workspace_root: str | Path | None = None) -> dict:
    path = Path(path)
    if not path.exists():
        return failure("validate_responses_logs", "Log file does not exist", path=str(path))

    workspace_root = Path(workspace_root).resolve() if workspace_root is not None else None
    lines = path.read_text(encoding="utf-8").splitlines()
    write_file_calls_by_path: dict[str, list[str]] = {}
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            return failure("validate_responses_logs", f"Empty line at {index}", path=str(path), line=index)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            return failure("validate_responses_logs", f"Invalid JSON at line {index}: {exc}", path=str(path), line=index)

        for required in ("timestamp", "elapsed_seconds", "model", "profile", "phase", "raw_response"):
            if required not in payload:
                return failure("validate_responses_logs", f"Missing {required} at line {index}", path=str(path), line=index)

        tool_calls = payload.get("tool_calls", [])
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                return failure("validate_responses_logs", f"tool_call must be an object at line {index}", path=str(path), line=index)
            for required in ("name", "arguments", "call_id"):
                if required not in tool_call:
                    return failure(
                        "validate_responses_logs",
                        f"tool_call missing {required} at line {index}",
                        path=str(path),
                        line=index,
                    )
            arguments = tool_call.get("arguments", {})
            if tool_call.get("name") == "write_file":
                if not isinstance(arguments, dict) or "content" not in arguments:
                    return failure(
                        "validate_responses_logs",
                        f"write_file call missing content at line {index}",
                        path=str(path),
                        line=index,
                    )
                normalized_path = _normalize_logged_path(arguments.get("path"), workspace_root)
                if normalized_path is None:
                    return failure(
                        "validate_responses_logs",
                        f"write_file call missing path at line {index}",
                        path=str(path),
                        line=index,
                    )
                content = arguments.get("content")
                if not isinstance(content, str):
                    return failure(
                        "validate_responses_logs",
                        f"write_file content must be a string at line {index}",
                        path=str(path),
                        line=index,
                    )
                write_file_calls_by_path.setdefault(normalized_path, []).append(content)

    traced_write_paths: list[str] = []
    if workspace_root is not None:
        submission_code_dir = workspace_root / "submission" / "code"
        if submission_code_dir.exists():
            untraced_files: list[str] = []
            content_mismatch_files: list[str] = []
            for file_path in sorted(candidate for candidate in submission_code_dir.rglob("*") if candidate.is_file()):
                relative = file_path.relative_to(workspace_root).as_posix()
                actual_content = file_path.read_text(encoding="utf-8")
                logged_contents = write_file_calls_by_path.get(relative, [])
                if not logged_contents:
                    untraced_files.append(relative)
                    continue
                if actual_content not in logged_contents:
                    content_mismatch_files.append(relative)
                    continue
                traced_write_paths.append(relative)

            if untraced_files or content_mismatch_files:
                problems: list[str] = []
                if untraced_files:
                    problems.append(f"Untraced submission/code files: {', '.join(untraced_files)}")
                if content_mismatch_files:
                    problems.append(f"write_file content mismatch for: {', '.join(content_mismatch_files)}")
                return failure(
                    "validate_responses_logs",
                    "; ".join(problems),
                    path=str(path),
                    lines=len(lines),
                    traced_write_paths=traced_write_paths,
                    untraced_files=untraced_files,
                    content_mismatch_files=content_mismatch_files,
                    logged_write_paths=sorted(write_file_calls_by_path),
                )

    return success(
        "validate_responses_logs",
        f"Validated {len(lines)} Responses log lines",
        path=str(path),
        lines=len(lines),
        traced_write_paths=traced_write_paths,
        logged_write_paths=sorted(write_file_calls_by_path),
    )


def _first_dataset(handle: h5py.File) -> h5py.Dataset:
    datasets: list[h5py.Dataset] = []

    def collect(_: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets.append(obj)

    handle.visititems(collect)
    if not datasets:
        raise ValueError("No dataset found in HDF5 file")
    return datasets[0]


def _validate_time_csv(path: Path) -> tuple[bool, str, dict[str, float] | None]:
    if not path.exists():
        return False, "time.csv does not exist", None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return False, "time.csv has no data rows", None
    row = rows[0]
    for key in ("train_time", "inference_time"):
        if key not in row:
            return False, f"time.csv missing column {key}", None
        try:
            row[key] = float(row[key])
        except ValueError:
            return False, f"time.csv column {key} is not numeric", None
    return True, "", {"train_time": row["train_time"], "inference_time": row["inference_time"]}


def validate_submission(
    *,
    submission_dir: str | Path,
    test_hdf5: str | Path | None = None,
    pred_filename: str = "pred.hdf5",
    time_filename: str = "time.csv",
    logs_filename: str = "logs.log",
    code_dir: str | Path | None = None,
    rehearsal_mode: bool = False,
) -> dict:
    submission_dir = Path(submission_dir)
    pred_path = submission_dir / pred_filename
    time_path = submission_dir / time_filename
    logs_path = submission_dir / logs_filename
    code_dir = Path(code_dir) if code_dir is not None else submission_dir / "code"

    if not pred_path.exists():
        return failure("validate_submission", "pred.hdf5 does not exist", pred_path=str(pred_path))
    if not code_dir.exists() or not any(code_dir.iterdir()):
        return failure("validate_submission", "submission/code is empty", code_dir=str(code_dir))

    try:
        with h5py.File(pred_path, "r") as handle:
            pred_dataset = _first_dataset(handle)
            pred = np.asarray(pred_dataset[...])
    except Exception as exc:  # noqa: BLE001
        return failure("validate_submission", f"Failed to read prediction HDF5: {exc}", pred_path=str(pred_path))

    if pred.ndim != 3 or pred.shape[1:] != (200, 256):
        return failure("validate_submission", f"Prediction shape must be (N, 200, 256), got {pred.shape}", pred_shape=list(pred.shape))
    if np.isnan(pred).any() or np.isinf(pred).any():
        return failure("validate_submission", "Prediction contains NaN or Inf", pred_shape=list(pred.shape))

    rehearsal_only = False
    if test_hdf5 is not None:
        try:
            with h5py.File(test_hdf5, "r") as handle:
                test = np.asarray(_first_dataset(handle)[...])
        except Exception as exc:  # noqa: BLE001
            return failure("validate_submission", f"Failed to read test HDF5: {exc}", test_hdf5=str(test_hdf5))

        if test.shape[1] < 10 or test.shape[2] != 256:
            return failure("validate_submission", "Test HDF5 shape is incompatible with prediction", test_shape=list(test.shape), pred_shape=list(pred.shape))
        if rehearsal_mode and pred.shape[0] <= test.shape[0]:
            rehearsal_only = pred.shape[0] < test.shape[0]
            if not np.allclose(pred[:, :10, :], test[: pred.shape[0], :10, :], atol=1e-3, rtol=0.0):
                return failure("validate_submission", "Prediction first 10 steps do not match test input", pred_shape=list(pred.shape), rehearsal_only=True)
        elif test.shape[0] != pred.shape[0]:
            return failure("validate_submission", "Test HDF5 shape is incompatible with prediction", test_shape=list(test.shape), pred_shape=list(pred.shape))
        elif not np.allclose(pred[:, :10, :], test[:, :10, :], atol=1e-3, rtol=0.0):
            return failure("validate_submission", "Prediction first 10 steps do not match test input", pred_shape=list(pred.shape))

    time_ok, time_error, time_values = _validate_time_csv(time_path)
    if not time_ok:
        return failure("validate_submission", time_error, time_path=str(time_path))

    logs_result = validate_jsonl_logs(logs_path)
    if not logs_result["ok"]:
        return failure("validate_submission", logs_result["error"], logs_validation=logs_result)

    return success(
        "validate_submission",
        "Submission bundle passed validation",
        pred_shape=list(pred.shape),
        pred_path=str(pred_path),
        time_csv=time_values,
        logs_path=str(logs_path),
        rehearsal_only=rehearsal_only,
    )


def validate_task_submission_bundles(
    *,
    submission_dir: str | Path,
    task_configs: list[SubmissionTaskConfig],
    workspace_root: str | Path,
    code_dir: str | Path | None = None,
    rehearsal_mode: bool = False,
) -> list[dict]:
    submission_dir = Path(submission_dir)
    workspace_root = Path(workspace_root)
    validations: list[dict] = []
    for task_config in task_configs:
        test_hdf5 = workspace_root / task_config.test_hdf5
        validations.append(
            validate_submission(
                submission_dir=submission_dir,
                test_hdf5=test_hdf5,
                pred_filename=task_config.pred_filename,
                time_filename=task_config.time_filename,
                logs_filename=task_config.logs_filename,
                code_dir=code_dir,
                rehearsal_mode=rehearsal_mode,
            )
        )
    return validations
