from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from cozy_pde_v3.memory_store import MemoryStore
from cozy_pde_v3.task_specs import TASK_IDS
from cozy_pde_v3.task_specs import DEFAULT_TASK_SPECS, TaskSpec
from cozy_pde_v3.validation.logs import validate_task_log_jsonl
from cozy_pde_v3.validation.provenance import build_shared_code_union, detect_task_specific_code_forks

_ALLOWED_METHODOLOGY_SOURCES = {
    "agent_state_snapshots",
    "decision_records",
    "experiment_cards",
    "validation_reports",
    "artifact_metadata",
    "final_package_snapshot",
    "code_snapshots",
    "code_patch_records",
}
_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b", re.IGNORECASE),
    re.compile(r"\bapi[_-]?key\b", re.IGNORECASE),
    re.compile(r"\bauthorization\b", re.IGNORECASE),
    re.compile(r"\bbearer\b", re.IGNORECASE),
]
_TEXT_SCAN_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".csv", ".md", ".py", ".yaml", ".yml"}
_CLI_FLAG_PATTERN = re.compile(r"add_argument\(\s*[\"'](?P<flag>--[a-zA-Z0-9_-]+)[\"']")
_TASK1_FORBIDDEN_REFERENCE_PATTERN = re.compile(
    r"(task1_official|checkpoints/task1|data/task1_|artifacts/task1|submission/task1_(pred|time|logs))",
    re.IGNORECASE,
)
_TASK2_FORBIDDEN_REFERENCE_PATTERN = re.compile(
    r"(artifacts/task2|submission/task2_(pred|time|logs)|checkpoints/task2)",
    re.IGNORECASE,
)
_PUBLIC_PRETRAINED_PATTERN = re.compile(
    r"(torch\.hub\.load|hf_hub_download|from_pretrained\s*\(|pretrained\s*=\s*True|timm\.create_model\s*\([^)]*pretrained\s*=\s*True)",
    re.IGNORECASE,
)
_EXTRA_DATA_PATTERN = re.compile(
    r"(https?://|urllib\.request|requests\.(get|post)|kaggle|huggingface|load_dataset\s*\(|gdown)",
    re.IGNORECASE,
)
_SOLVER_PATTERN = re.compile(
    r"(solve_ivp|odeint|fenics|dedalus|pdeint|scipy\.integrate)",
    re.IGNORECASE,
)
_HARDCODED_LEAK_PATTERN = re.compile(
    r"[\"'](?:task1_val\.hdf5|task2_val\.h5|task2_val\.hdf5|ks_val\.hdf5|validation\.h5|validation\.hdf5|val\.h5|val\.hdf5|task1_test\.hdf5|task2_test\.h5|task2_test\.hdf5|task3_test\.hdf5)[\"']",
    re.IGNORECASE,
)


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_submission_code_texts(submission_dir: Path) -> dict[str, str]:
    code_dir = submission_dir / "code"
    texts: dict[str, str] = {}
    if not code_dir.exists():
        return texts
    for file_path in sorted(path for path in code_dir.rglob("*") if path.is_file()):
        relative_path = str(file_path.relative_to(submission_dir.parent)).replace("\\", "/")
        texts[relative_path] = _read_text_file(file_path)
    return texts


def _extract_cli_flags(script_text: str) -> set[str]:
    return {match.group("flag") for match in _CLI_FLAG_PATTERN.finditer(script_text)}


def _cli_flag_failures(path: str, flags: set[str], required_flags: set[str], optional_group: set[str] | None = None) -> list[str]:
    failures: list[str] = []
    missing = sorted(flag for flag in required_flags if flag not in flags)
    if missing:
        failures.append(f"{path} missing required CLI flags: {', '.join(missing)}")
    if optional_group is not None and not any(flag in flags for flag in optional_group):
        failures.append(f"{path} must support one of: {', '.join(sorted(optional_group))}")
    return failures


def _memory_store_paths(workspace_root: Path) -> list[Path]:
    internal_logs = workspace_root / "internal_logs"
    if not internal_logs.exists():
        return []
    preferred = internal_logs / "memory.db"
    if preferred.exists():
        return [preferred]
    return sorted(path for path in internal_logs.glob("*.db") if path.is_file())


def _load_memory_store_records(workspace_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    issues: list[str] = []
    for db_path in _memory_store_paths(workspace_root):
        try:
            store = MemoryStore(db_path)
            return store.list_code_snapshots(), store.list_patch_records(), []
        except Exception as exc:  # noqa: BLE001
            issues.append(f"failed to read memory store {db_path}: {exc}")
    if not issues:
        issues.append("memory store missing under workspace/internal_logs")
    return [], [], issues


def _validate_submission_api_contract(
    *,
    submission_dir: Path,
    final_code_paths: list[str],
) -> dict[str, Any]:
    failures: list[str] = []
    required_files = ["submission/code/infer.py", "submission/code/train.py"]
    task_specific_dir_present = any((submission_dir / "code" / task).exists() for task in TASK_IDS)
    shared_code_ok = not detect_task_specific_code_forks(final_code_paths) and not task_specific_dir_present

    for path in detect_task_specific_code_forks(final_code_paths):
        failures.append(f"task-specific code fork detected: {path}")
    for task in TASK_IDS:
        if (submission_dir / "code" / task).exists():
            failures.append(f"task-specific code fork detected: submission/code/{task}")

    code_texts = _load_submission_code_texts(submission_dir)
    for required_file in required_files:
        if required_file not in code_texts:
            failures.append(f"missing required shared entrypoint: {required_file}")

    train_text = code_texts.get("submission/code/train.py", "")
    infer_text = code_texts.get("submission/code/infer.py", "")
    if train_text:
        failures.extend(
            _cli_flag_failures(
                "submission/code/train.py",
                _extract_cli_flags(train_text),
                {"--task", "--config", "--data_dir", "--output_dir"},
            )
        )
    if infer_text:
        failures.extend(
            _cli_flag_failures(
                "submission/code/infer.py",
                _extract_cli_flags(infer_text),
                {"--task", "--config", "--data_dir"},
                {"--output_dir", "--output"},
            )
        )

    return {
        "ok": not failures,
        "failures": failures,
        "required_files": required_files,
        "shared_code_ok": shared_code_ok,
    }


def _validate_task_policy_rules(
    *,
    workspace_root: Path,
    submission_dir: Path,
    tasks: list[str],
) -> dict[str, Any]:
    failures: list[str] = []
    code_texts = _load_submission_code_texts(submission_dir)
    train_infer_texts = {
        path: text
        for path, text in code_texts.items()
        if path.endswith("/train.py") or path.endswith("/infer.py")
    }
    all_code_text = "\n".join(code_texts.values())

    if _EXTRA_DATA_PATTERN.search(all_code_text):
        failures.append("submission code references forbidden extra data source")
    if _SOLVER_PATTERN.search(all_code_text):
        failures.append("submission code references solver-generated data path")

    for path, text in sorted(train_infer_texts.items()):
        if _HARDCODED_LEAK_PATTERN.search(text):
            failures.append(f"{path} contains hardcoded validation/test reference")

    for task in tasks:
        task_text = all_code_text
        if task == "task1":
            if _PUBLIC_PRETRAINED_PATTERN.search(task_text) and "task1_official" not in task_text.lower():
                failures.append("task1 may only use official Task 1 checkpoints")
        if task == "task2":
            if _TASK1_FORBIDDEN_REFERENCE_PATTERN.search(task_text):
                failures.append("task2 must be trained from scratch")
                failures.append("task2 references forbidden Task 1 source")
            if _PUBLIC_PRETRAINED_PATTERN.search(task_text):
                failures.append("task2 must be trained from scratch")
        if task == "task3":
            if _TASK1_FORBIDDEN_REFERENCE_PATTERN.search(task_text) or _TASK2_FORBIDDEN_REFERENCE_PATTERN.search(task_text):
                failures.append("task3 references forbidden Task 1/Task 2 artifact")
            if _PUBLIC_PRETRAINED_PATTERN.search(task_text):
                failures.append("task3 forbids public pretrained weights")

    return {
        "ok": not failures,
        "failures": failures,
        "workspace_root": str(workspace_root),
    }


def _default_subprocess_runner(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def run_submission_cli_smoke(
    *,
    workspace_root: str | Path,
    submission_dir: str | Path,
    tasks: list[str],
    runner: Any | None = None,
) -> dict[str, Any]:
    workspace_path = Path(workspace_root)
    submission_path = Path(submission_dir)
    code_dir = submission_path / "code"
    train_path = code_dir / "train.py"
    infer_path = code_dir / "infer.py"
    command_runner = runner or _default_subprocess_runner
    train_flags = _extract_cli_flags(_read_text_file(train_path)) if train_path.exists() else set()
    infer_flags = _extract_cli_flags(_read_text_file(infer_path)) if infer_path.exists() else set()
    status: dict[str, bool] = {}
    details: dict[str, dict[str, bool]] = {}
    failures: list[str] = []

    if not train_path.exists() or not infer_path.exists():
        missing = []
        if not train_path.exists():
            missing.append("submission/code/train.py")
        if not infer_path.exists():
            missing.append("submission/code/infer.py")
        return {
            "ok": False,
            "status": {task: False for task in tasks},
            "details": {
                task: {
                    "cli_parse_ok": False,
                    "train_smoke_ok": False,
                    "infer_smoke_ok": False,
                    "checkpoint_load_ok": False,
                }
                for task in tasks
            },
            "failures": [f"cli smoke missing required entrypoints: {', '.join(missing)}"],
        }

    for task in tasks:
        task_details = {
            "cli_parse_ok": False,
            "train_smoke_ok": False,
            "infer_smoke_ok": False,
            "checkpoint_load_ok": False,
        }
        with tempfile.TemporaryDirectory(prefix=f"v3-cli-smoke-{task}-", dir=str(workspace_path)) as temp_dir:
            smoke_root = Path(temp_dir)
            output_dir = smoke_root / "artifacts"
            output_dir.mkdir(parents=True, exist_ok=True)
            config_path = output_dir / "smoke-config.json"
            config_path.write_text(json.dumps({"task": task}), encoding="utf-8")
            output_path = smoke_root / f"{task}_pred.hdf5"
            data_dir = workspace_path / "data"

            help_results = [
                command_runner([sys.executable, str(train_path), "--help"], cwd=workspace_path),
                command_runner([sys.executable, str(infer_path), "--help"], cwd=workspace_path),
            ]
            task_details["cli_parse_ok"] = all(result.returncode == 0 for result in help_results)
            if not task_details["cli_parse_ok"]:
                failures.append(f"{task}: CLI parse smoke failed")
                details[task] = task_details
                status[task] = False
                continue

            train_command = [
                sys.executable,
                str(train_path),
                "--task",
                task,
                "--config",
                str(config_path),
                "--data_dir",
                str(data_dir),
            ]
            if "--output_dir" in train_flags:
                train_command.extend(["--output_dir", str(output_dir)])
            train_result = command_runner(train_command, cwd=workspace_path)
            task_details["train_smoke_ok"] = train_result.returncode == 0
            checkpoint_exists = any(output_dir.iterdir()) if output_dir.exists() else False
            task_details["checkpoint_load_ok"] = task_details["train_smoke_ok"] and checkpoint_exists
            if not task_details["train_smoke_ok"]:
                failures.append(f"{task}: train smoke failed")
                details[task] = task_details
                status[task] = False
                continue

            infer_command = [
                sys.executable,
                str(infer_path),
                "--task",
                task,
                "--config",
                str(config_path),
                "--data_dir",
                str(data_dir),
            ]
            candidate_output_paths: list[Path] = []
            if "--output_dir" in infer_flags:
                infer_command.extend(["--output_dir", str(output_dir)])
                candidate_output_paths.extend(
                    [
                        output_dir / f"{task}_pred.hdf5",
                        output_dir / "pred.hdf5",
                        output_dir / "prediction.hdf5",
                    ]
                )
            if "--output" in infer_flags:
                infer_command.extend(["--output", str(output_path)])
                candidate_output_paths.insert(0, output_path)
            infer_result = command_runner(infer_command, cwd=workspace_path)
            resolved_output_path = next((path for path in candidate_output_paths if path.exists()), None)
            if resolved_output_path is None and output_dir.exists():
                resolved_output_path = next(
                    (path for path in sorted(output_dir.rglob("*")) if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}),
                    None,
                )
            if infer_result.returncode == 0 and resolved_output_path is not None:
                try:
                    with h5py.File(resolved_output_path, "r") as handle:
                        pred_shape = tuple(np.asarray(_first_dataset(handle)[...]).shape)
                except Exception:  # noqa: BLE001
                    pred_shape = ()
                task_details["infer_smoke_ok"] = pred_shape == DEFAULT_TASK_SPECS[task].pred_shape or (
                    DEFAULT_TASK_SPECS[task].pred_shape[0] == 0
                    and len(pred_shape) == 3
                    and pred_shape[1:] == DEFAULT_TASK_SPECS[task].pred_shape[1:]
                )
            else:
                task_details["infer_smoke_ok"] = False
            if infer_result.returncode != 0:
                task_details["checkpoint_load_ok"] = False
                failures.append(f"{task}: infer smoke failed")
            elif not task_details["infer_smoke_ok"]:
                failures.append(f"{task}: infer smoke produced incompatible shape")

            details[task] = task_details
            status[task] = all(task_details.values())

    return {
        "ok": not failures and all(status.values()),
        "status": status,
        "details": details,
        "failures": failures,
    }


def _validate_incremental_patch_records(
    *,
    workspace_root: Path,
    final_code_paths: list[str],
    supported_tasks: list[str],
) -> dict[str, Any]:
    snapshots, patch_records, issues = _load_memory_store_records(workspace_root)
    failures = list(issues)
    if not snapshots:
        return {"ok": False, "failures": failures or ["code snapshots missing"], "shared_code_union": {"shared_code_versions": []}}

    sorted_snapshots = sorted(snapshots, key=lambda snapshot: int(snapshot.get("id", 0)))
    latest_snapshot = sorted_snapshots[-1]
    latest_version = str(latest_snapshot.get("code_version", "")).strip()
    latest_supported_tasks = [str(task).strip() for task in latest_snapshot.get("supported_tasks", []) if str(task).strip()]
    if any(task not in latest_supported_tasks for task in supported_tasks):
        failures.append("latest code snapshot missing supported task coverage")

    previous_hash_by_task: dict[str, str] = {}
    for index, snapshot in enumerate(sorted_snapshots):
        version = str(snapshot.get("code_version", "")).strip()
        parent = snapshot.get("parent_version")
        if index > 0 and not parent:
            failures.append(f"snapshot {version} missing parent_version")
        supported = [str(task).strip() for task in snapshot.get("supported_tasks", []) if str(task).strip()]
        api_contract_hash = str(snapshot.get("api_contract_hash", "")).strip()
        for task in supported:
            if task in previous_hash_by_task and previous_hash_by_task[task] != api_contract_hash:
                failures.append(f"later patch breaks earlier task API contract: {task}")
            previous_hash_by_task.setdefault(task, api_contract_hash)

    patch_union: set[str] = set()
    patch_by_new_version = {
        str(record.get("new_code_version", "")).strip(): record
        for record in patch_records
        if str(record.get("new_code_version", "")).strip()
    }
    for snapshot in sorted_snapshots[1:]:
        version = str(snapshot.get("code_version", "")).strip()
        patch = patch_by_new_version.get(version)
        if patch is None:
            failures.append(f"missing patch record for snapshot {version}")
            continue
        changed_files = [str(path).strip() for path in patch.get("changed_files", []) if str(path).strip()]
        if not changed_files:
            failures.append(f"patch {patch.get('patch_id', version)} missing changed_files")
        patch_union.update(changed_files)
        llm_call_ids = [str(call_id).strip() for call_id in patch.get("llm_call_ids", []) if str(call_id).strip()]
        if not llm_call_ids:
            failures.append(f"patch {patch.get('patch_id', version)} missing llm_call_ids")
        validation_results = patch.get("validation_results", {})
        task_compatibility = validation_results.get("task_compatibility", {}) if isinstance(validation_results, dict) else {}
        for task in supported_tasks:
            if task_compatibility.get(task) is not True:
                failures.append(f"patch {patch.get('patch_id', version)} missing supported-task compatibility for {task}")

    missing_changed_files = sorted(path for path in final_code_paths if path not in patch_union)
    for path in missing_changed_files:
        failures.append(f"patch history missing changed_file coverage for {path}")

    union_records = []
    for snapshot in sorted_snapshots:
        version = str(snapshot.get("code_version", "")).strip()
        matching_patch = patch_by_new_version.get(version, {})
        union_records.append(
            {
                "version": version,
                "parent": str(snapshot.get("parent_version", "")).strip() or None,
                "changed_files": [
                    str(path).strip()
                    for path in matching_patch.get("changed_files", [])
                    if str(path).strip()
                ],
                "validated_tasks": [
                    str(task).strip()
                    for task in snapshot.get("supported_tasks", [])
                    if str(task).strip()
                ],
                "llm_call_ids": [
                    str(call_id).strip()
                    for call_id in matching_patch.get("llm_call_ids", [])
                    if str(call_id).strip()
                ],
            }
        )
    shared_code_union = build_shared_code_union(union_records)

    return {
        "ok": not failures,
        "failures": failures,
        "shared_code_union": shared_code_union,
    }


def _success(summary: str, **data: Any) -> dict[str, Any]:
    payload = {"ok": True, "summary": summary}
    if data:
        payload["data"] = data
    return payload


def _failure(message: str, **data: Any) -> dict[str, Any]:
    payload = {"ok": False, "error": message}
    if data:
        payload["data"] = data
    return payload


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _first_dataset(handle: h5py.File) -> h5py.Dataset:
    datasets: list[h5py.Dataset] = []

    def collect(_: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets.append(obj)

    handle.visititems(collect)
    if not datasets:
        raise ValueError("No dataset found in HDF5 file")
    return datasets[0]


def _find_test_hdf5(workspace_root: Path, task: str, spec: TaskSpec) -> Path | None:
    candidates = [workspace_root / "data" / f"{task}_test.hdf5", workspace_root / "data" / f"{task}_test.h5"]
    candidates.extend(workspace_root / "data" / filename for filename in spec.default_test_filenames)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _validate_time_csv(path: Path, *, strict: bool, limit_seconds: float) -> tuple[bool, str, dict[str, float] | None]:
    if not path.exists():
        return False, f"{path.name} does not exist", None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return False, f"{path.name} has no rows", None
    row = rows[0]
    values: dict[str, float] = {}
    for key in ("train_time", "inference_time"):
        if key not in row:
            return False, f"{path.name} missing column {key}", None
        try:
            values[key] = float(row[key])
        except ValueError:
            return False, f"{path.name} column {key} is not numeric", None
    if strict and values["inference_time"] > limit_seconds:
        return False, f"{path.name} inference_time exceeds {limit_seconds:g} seconds", values
    return True, "", values


def _validate_manifest(manifest_path: Path, submission_dir: Path) -> tuple[bool, list[str]]:
    issues: list[str] = []
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, [f"Failed to read manifest.json: {exc}"]
    if not isinstance(entries, list):
        return False, ["manifest.json must contain a list of file metadata"]
    for entry in entries:
        relative = entry.get("path")
        expected_size = entry.get("size")
        expected_sha = entry.get("sha256")
        if not isinstance(relative, str):
            issues.append("manifest entry missing path")
            continue
        file_path = submission_dir / relative
        if not file_path.exists():
            issues.append(f"manifest entry missing file: {relative}")
            continue
        if file_path.stat().st_size != expected_size:
            issues.append(f"manifest size mismatch: {relative}")
        if _sha256_file(file_path) != expected_sha:
            issues.append(f"manifest sha256 mismatch: {relative}")
    return len(issues) == 0, issues


def _validate_code_manifest(manifest_path: Path, workspace_root: Path) -> tuple[bool, list[str], list[dict[str, Any]] | None]:
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, [f"Failed to read code_manifest.json: {exc}"], None
    return _validate_code_manifest_entries(entries, workspace_root)


def _validate_code_manifest_entries(
    entries: object,
    workspace_root: Path,
) -> tuple[bool, list[str], list[dict[str, Any]] | None]:
    issues: list[str] = []
    if not isinstance(entries, list):
        return False, ["code_manifest.json must contain a list of file metadata"], None
    final_entries_by_path: dict[str, dict[str, Any]] = {}
    ordered_paths: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            issues.append("code manifest entry must be an object")
            continue
        relative = entry.get("path")
        if not isinstance(relative, str):
            issues.append("code manifest entry missing path")
            continue
        if relative in final_entries_by_path:
            ordered_paths.remove(relative)
        ordered_paths.append(relative)
        final_entries_by_path[relative] = entry
    final_entries = [final_entries_by_path[path] for path in ordered_paths]
    for entry in final_entries:
        relative = str(entry["path"])
        for key in (
            "path",
            "sha256",
            "size",
            "code_version",
            "originating_task",
            "patch_id",
            "step_id",
            "task_id",
            "timestamp",
            "llm_call_ids",
        ):
            if key not in entry:
                issues.append(f"code manifest entry missing {key}")
        if not str(entry.get("code_version", "")).strip():
            issues.append(f"code manifest entry missing code_version: {relative}")
        if not str(entry.get("originating_task", "")).strip():
            issues.append(f"code manifest entry missing originating_task: {relative}")
        if not str(entry.get("patch_id", "")).strip():
            issues.append(f"code manifest entry missing patch_id: {relative}")
        llm_call_ids = entry.get("llm_call_ids", [])
        if not isinstance(llm_call_ids, list) or not [
            str(call_id).strip() for call_id in llm_call_ids if str(call_id).strip()
        ]:
            issues.append(f"code manifest entry missing llm_call_ids: {relative}")
        file_path = workspace_root / relative
        if not file_path.exists():
            issues.append(f"code manifest entry missing file: {relative}")
            continue
        if file_path.stat().st_size != entry.get("size"):
            issues.append(f"code manifest size mismatch: {relative}")
        if _sha256_file(file_path) != entry.get("sha256"):
            issues.append(f"code manifest sha256 mismatch: {relative}")
    return len(issues) == 0, issues, final_entries


def _scan_for_secret_leaks(paths: list[Path]) -> list[str]:
    hits: list[str] = []
    for root in paths:
        if not root.exists():
            continue
        for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
            if file_path.suffix.lower() not in _TEXT_SCAN_SUFFIXES:
                continue
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            for pattern in _SECRET_PATTERNS:
                if pattern.search(text):
                    hits.append(f"{file_path}: matched {pattern.pattern}")
                    break
    return hits


def _code_paths(submission_dir: Path) -> list[str]:
    code_dir = submission_dir / "code"
    if not code_dir.exists():
        return []
    return [
        str(path.relative_to(submission_dir.parent)).replace("\\", "/")
        for path in sorted(code_dir.rglob("*"))
        if path.is_file()
    ]


def _validate_single_task_bundle(
    *,
    submission_dir: Path,
    workspace_root: Path,
    task: str,
    spec: TaskSpec,
    strict: bool,
) -> tuple[bool, list[str], dict[str, Any]]:
    failures: list[str] = []
    pred_path = submission_dir / f"{task}_pred.hdf5"
    time_path = submission_dir / f"{task}_time.csv"
    logs_path = submission_dir / f"{task}_logs.log"
    details: dict[str, Any] = {}

    if not pred_path.exists():
        failures.append(f"{task}: missing {pred_path.name}")
        return False, failures, details
    try:
        with h5py.File(pred_path, "r") as handle:
            pred = np.asarray(_first_dataset(handle)[...])
    except Exception as exc:  # noqa: BLE001
        failures.append(f"{task}: failed to read prediction HDF5: {exc}")
        return False, failures, details

    expected_shape = spec.pred_shape
    if expected_shape[0] == 0:
        if pred.ndim != 3 or pred.shape[1:] != expected_shape[1:]:
            failures.append(f"{task}: prediction shape must be (N, {expected_shape[1]}, {expected_shape[2]}), got {tuple(pred.shape)}")
    elif tuple(pred.shape) != expected_shape:
        failures.append(f"{task}: prediction shape must be {expected_shape}, got {tuple(pred.shape)}")
    if np.isnan(pred).any() or np.isinf(pred).any():
        failures.append(f"{task}: prediction contains NaN or Inf")

    test_hdf5 = _find_test_hdf5(workspace_root, task, spec)
    if test_hdf5 is None:
        failures.append(f"{task}: missing test HDF5")
    else:
        try:
            with h5py.File(test_hdf5, "r") as handle:
                test = np.asarray(_first_dataset(handle)[...])
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{task}: failed to read test HDF5: {exc}")
        else:
            if test.ndim != 3 or test.shape[2] != spec.spatial_points or test.shape[1] < spec.first_steps_must_match:
                failures.append(f"{task}: test HDF5 shape is incompatible with prediction")
            elif test.shape[0] != pred.shape[0]:
                failures.append(f"{task}: test HDF5 shape is incompatible with prediction")
            elif not np.allclose(
                pred[:, : spec.first_steps_must_match, :],
                test[:, : spec.first_steps_must_match, :],
                atol=1e-3,
                rtol=0.0,
            ):
                failures.append(f"{task}: prediction first {spec.first_steps_must_match} steps do not match test input")

    time_ok, time_error, time_values = _validate_time_csv(
        time_path,
        strict=strict,
        limit_seconds=spec.inference_time_limit_sec,
    )
    if not time_ok:
        failures.append(f"{task}: {time_error}")
    else:
        details[f"{task}_time_csv"] = time_values

    logs_result = validate_task_log_jsonl(logs_path)
    if not logs_result["ok"]:
        failures.append(f"{task}: {logs_result['error']}")
    details[f"{task}_pred_shape"] = list(pred.shape)
    return len(failures) == 0, failures, details


def validate_submission_bundle_v3(
    *,
    workspace_root: str | Path,
    tasks: list[str],
    strict: bool = True,
    code_manifest_entries: list[dict[str, Any]] | None = None,
    methodology_sources: list[str] | None = None,
) -> dict[str, Any]:
    workspace_path = Path(workspace_root)
    submission_dir = workspace_path / "submission"
    selected_tasks = [task for task in tasks if task in TASK_IDS]
    if not selected_tasks:
        return _failure("at least one known task is required")

    if not submission_dir.exists():
        return _failure("submission directory does not exist", submission_dir=str(submission_dir))

    code_dir = submission_dir / "code"
    code_files = [path for path in sorted(code_dir.rglob("*")) if path.is_file()] if code_dir.exists() else []
    if not code_files:
        return _failure("submission/code is empty", submission_dir=str(submission_dir))

    manifest_path = submission_dir / "manifest.json"
    code_manifest_path = submission_dir / "code_manifest.json"
    resolved_manifest_entries = code_manifest_entries
    resolved_code_manifest_ok = False
    failures: list[str] = []
    warnings: list[str] = []

    prediction_ok = True
    time_csv_ok = True
    logs_ok = True
    inference_time_ok = True
    task_details: dict[str, Any] = {}
    for task in selected_tasks:
        ok, task_failures, details = _validate_single_task_bundle(
            submission_dir=submission_dir,
            workspace_root=workspace_path,
            task=task,
            spec=DEFAULT_TASK_SPECS[task],
            strict=strict,
        )
        task_details.update(details)
        for failure in task_failures:
            failures.append(failure)
            if "prediction shape" in failure or "prediction contains" in failure or "test HDF5" in failure:
                prediction_ok = False
            if "time.csv" in failure:
                time_csv_ok = False
            if "inference_time exceeds" in failure:
                inference_time_ok = False
            if "logs.log" in failure or "response or tool_calls" in failure or "valid JSON" in failure:
                logs_ok = False

    if manifest_path.exists():
        manifest_ok, manifest_issues = _validate_manifest(manifest_path, submission_dir)
        if not manifest_ok:
            failures.extend(manifest_issues)
    if code_manifest_entries is not None:
        code_manifest_ok, code_manifest_issues, manifest_entries = _validate_code_manifest_entries(
            code_manifest_entries,
            workspace_path,
        )
        resolved_code_manifest_ok = code_manifest_ok
        resolved_manifest_entries = manifest_entries
        if not code_manifest_ok:
            failures.extend(code_manifest_issues)
    elif code_manifest_path.exists():
        code_manifest_ok, code_manifest_issues, manifest_entries = _validate_code_manifest(code_manifest_path, workspace_path)
        resolved_code_manifest_ok = code_manifest_ok
        resolved_manifest_entries = manifest_entries
        if not code_manifest_ok:
            failures.extend(code_manifest_issues)
    elif strict:
        failures.append("code_manifest.json missing")

    methodology_path = submission_dir / "methodology.pdf"
    methodology_artifact_ok = methodology_path.exists()
    if methodology_sources and not methodology_artifact_ok:
        failures.append("methodology.pdf missing")

    final_code_paths = _code_paths(submission_dir)
    provenance_links: dict[str, list[str]] = {}
    if resolved_manifest_entries is not None:
        for entry in resolved_manifest_entries:
            path = str(entry.get("path", "")).strip()
            call_ids = entry.get("llm_call_ids") or entry.get("llm_call_id") or []
            if isinstance(call_ids, str):
                provenance_links[path] = [call_ids]
            elif isinstance(call_ids, list):
                provenance_links[path] = [str(call_id) for call_id in call_ids if str(call_id)]

    task_policy_result = _validate_task_policy_rules(
        workspace_root=workspace_path,
        submission_dir=submission_dir,
        tasks=selected_tasks,
    )
    failures.extend(task_policy_result["failures"])

    api_contract_result = _validate_submission_api_contract(
        submission_dir=submission_dir,
        final_code_paths=final_code_paths,
    )
    failures.extend(api_contract_result["failures"])

    cli_smoke_result = run_submission_cli_smoke(
        workspace_root=workspace_path,
        submission_dir=submission_dir,
        tasks=selected_tasks,
    )
    failures.extend(cli_smoke_result["failures"])

    incremental_result = _validate_incremental_patch_records(
        workspace_root=workspace_path,
        final_code_paths=final_code_paths,
        supported_tasks=selected_tasks,
    )
    failures.extend(incremental_result["failures"])

    secret_scan_hits = _scan_for_secret_leaks(
        [
            submission_dir,
            workspace_path / "llm_logs",
            workspace_path / "proxy_logs",
            workspace_path / "internal_logs",
        ]
    )
    if secret_scan_hits:
        failures.extend(secret_scan_hits)

    finalize_gate = build_finalize_gate_status(
        prediction_ok=prediction_ok,
        time_csv_ok=time_csv_ok,
        logs_ok=logs_ok,
        provenance_log_ok=bool(resolved_manifest_entries) and not bool(_missing_provenance_paths(final_code_paths, provenance_links)),
        provenance_ok=bool(resolved_manifest_entries) and not bool(_missing_provenance_paths(final_code_paths, provenance_links)),
        inference_time_ok=inference_time_ok,
        package_ok=not failures,
        code_manifest_ok=resolved_code_manifest_ok,
        methodology_records_only_ok=bool(methodology_sources),
        methodology_ok=bool(methodology_sources) and methodology_artifact_ok,
        secret_scan_ok=not secret_scan_hits,
        task_rule_ok=task_policy_result["ok"],
        final_code_paths=final_code_paths,
        provenance_links=provenance_links,
        code_manifest_entries=resolved_manifest_entries,
        methodology_sources=methodology_sources,
        cli_smoke_status=cli_smoke_result["status"],
        cli_smoke_details=cli_smoke_result["details"],
        api_contract_ok=api_contract_result["ok"],
        incremental_patch_ok=incremental_result["ok"],
        shared_code_ok=api_contract_result["shared_code_ok"],
        supported_tasks=selected_tasks,
        failures=failures,
        warnings=warnings,
    )
    if finalize_gate["overall_ok"]:
        return _success(
            "submission bundle passed deterministic validation",
            finalize_gate=finalize_gate,
            task_details=task_details,
        )
    return _failure(
        finalize_gate["failures"][0] if finalize_gate["failures"] else "submission validation failed",
        finalize_gate=finalize_gate,
        task_details=task_details,
    )


def _missing_provenance_paths(
    final_code_paths: list[str],
    provenance_links: dict[str, list[str]],
) -> list[str]:
    missing: list[str] = []
    for path in sorted(final_code_paths):
        call_ids = [str(call_id) for call_id in provenance_links.get(path, []) if str(call_id)]
        if not call_ids:
            missing.append(path)
    return missing


def _stable_task_order(task: str) -> tuple[int, str]:
    suffix = task[4:]
    if task.startswith("task") and suffix.isdigit():
        return int(suffix), task
    return 10**9, task


def _normalize_supported_tasks(
    cli_smoke_status: dict[str, bool],
    supported_tasks: list[str] | None,
) -> list[str]:
    if supported_tasks is not None:
        candidates = supported_tasks
    else:
        candidates = [task for task in cli_smoke_status if task in TASK_IDS]
        if not candidates:
            candidates = list(TASK_IDS)
    normalized = {str(task).strip() for task in candidates if str(task).strip() in TASK_IDS}
    return sorted(normalized, key=_stable_task_order)


def _normalize_cli_smoke_details(
    cli_smoke_details: dict[str, dict[str, bool]] | None,
) -> dict[str, dict[str, bool]]:
    normalized: dict[str, dict[str, bool]] = {}
    for task in TASK_IDS:
        raw_details = (cli_smoke_details or {}).get(task, {})
        normalized[task] = {
            "cli_parse_ok": bool(raw_details.get("cli_parse_ok", False)),
            "train_smoke_ok": bool(raw_details.get("train_smoke_ok", False)),
            "infer_smoke_ok": bool(raw_details.get("infer_smoke_ok", False)),
            "checkpoint_load_ok": bool(raw_details.get("checkpoint_load_ok", raw_details.get("infer_smoke_ok", False))),
        }
    return normalized


def _coerce_call_id_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return sorted({str(item).strip() for item in value if str(item).strip()})
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _merge_provenance_links(
    first: dict[str, list[str]],
    second: dict[str, list[str]],
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for path in sorted(set(first) | set(second)):
        call_ids = _coerce_call_id_list(first.get(path, [])) + _coerce_call_id_list(second.get(path, []))
        merged[path] = sorted({call_id for call_id in call_ids if call_id})
    return merged


def _derive_manifest_provenance(
    code_manifest_entries: list[dict[str, Any]] | None,
) -> tuple[set[str], dict[str, list[str]], list[str]]:
    if code_manifest_entries is None:
        return set(), {}, []

    manifest_paths: set[str] = set()
    provenance_links: dict[str, list[str]] = {}
    issues: list[str] = []

    for entry in code_manifest_entries:
        if not isinstance(entry, dict):
            issues.append("code manifest entry must be an object")
            continue
        path = str(entry.get("path", "")).strip()
        if not path:
            issues.append("code manifest entry missing path")
            continue
        manifest_paths.add(path)
        call_ids = _coerce_call_id_list(entry.get("llm_call_ids"))
        if not call_ids:
            call_ids = _coerce_call_id_list(entry.get("llm_call_id"))
        provenance_links[path] = call_ids

    return manifest_paths, provenance_links, issues


def _normalize_methodology_sources(
    methodology_sources: list[str] | None,
) -> tuple[list[str], list[str]]:
    if methodology_sources is None:
        return [], []
    normalized = sorted({str(source).strip() for source in methodology_sources if str(source).strip()})
    invalid = [source for source in normalized if source not in _ALLOWED_METHODOLOGY_SOURCES]
    return normalized, invalid


def build_finalize_gate_status(
    *,
    prediction_ok: bool,
    logs_ok: bool,
    provenance_log_ok: bool,
    package_ok: bool,
    methodology_records_only_ok: bool,
    final_code_paths: list[str],
    provenance_links: dict[str, list[str]],
    cli_smoke_status: dict[str, bool],
    api_contract_ok: bool,
    incremental_patch_ok: bool,
    shared_code_ok: bool,
    time_csv_ok: bool = False,
    provenance_ok: bool | None = None,
    inference_time_ok: bool = False,
    code_manifest_ok: bool = False,
    methodology_ok: bool | None = None,
    secret_scan_ok: bool = False,
    task_rule_ok: bool = False,
    supported_tasks: list[str] | None = None,
    failures: list[str] | None = None,
    warnings: list[str] | None = None,
    code_manifest_entries: list[dict[str, Any]] | None = None,
    methodology_sources: list[str] | None = None,
    cli_smoke_details: dict[str, dict[str, bool]] | None = None,
) -> dict[str, Any]:
    manifest_paths, manifest_provenance_links, manifest_issues = _derive_manifest_provenance(code_manifest_entries)
    combined_provenance_links = _merge_provenance_links(provenance_links, manifest_provenance_links)
    task_fork_violations = detect_task_specific_code_forks(final_code_paths)
    missing_provenance_paths = _missing_provenance_paths(final_code_paths, combined_provenance_links)
    normalized_supported_tasks = _normalize_supported_tasks(cli_smoke_status, supported_tasks)
    normalized_cli_smoke_details = _normalize_cli_smoke_details(cli_smoke_details)
    resolved_provenance_ok = bool(provenance_log_ok) if provenance_ok is None else bool(provenance_ok)
    normalized_methodology_sources, invalid_methodology_sources = _normalize_methodology_sources(methodology_sources)
    if methodology_sources is not None:
        resolved_methodology_records_only_ok = (
            bool(methodology_records_only_ok) and bool(normalized_methodology_sources) and not invalid_methodology_sources
        )
        resolved_methodology_ok = bool(methodology_ok) and resolved_methodology_records_only_ok
    else:
        resolved_methodology_records_only_ok = False
        resolved_methodology_ok = False
    resolved_code_manifest_ok = False
    if code_manifest_entries is not None:
        missing_manifest_paths = sorted(path for path in final_code_paths if path not in manifest_paths)
        resolved_code_manifest_ok = bool(code_manifest_ok) and not manifest_issues and not missing_manifest_paths
    else:
        missing_manifest_paths = []
    gate_failures = list(failures or [])
    gate_warnings = list(warnings or [])

    for violation in task_fork_violations:
        gate_failures.append(f"task-specific code fork detected: {violation}")
    if code_manifest_entries is None:
        gate_failures.append("code manifest entries missing")
    for issue in manifest_issues:
        gate_failures.append(f"code manifest issue: {issue}")
    for missing_path in missing_manifest_paths:
        gate_failures.append(f"code manifest missing final path: {missing_path}")
    for missing_path in missing_provenance_paths:
        gate_failures.append(f"missing code provenance linkage: {missing_path}")
    if methodology_sources is None:
        gate_failures.append("methodology structured sources missing")
    for invalid_source in invalid_methodology_sources:
        gate_failures.append(f"methodology uses non-structured source: {invalid_source}")

    task_compatibility: dict[str, bool] = {}
    for task in normalized_supported_tasks:
        if task not in (cli_smoke_details or {}):
            gate_failures.append(f"cli smoke details missing for supported task: {task}")
        details = normalized_cli_smoke_details[task]
        task_compatibility[task] = all(details.values())
        legacy_summary = cli_smoke_status.get(task)
        if legacy_summary is not None and bool(legacy_summary) != task_compatibility[task]:
            gate_warnings.append(f"cli smoke summary mismatch for {task}")

    status: dict[str, Any] = {
        "prediction_ok": bool(prediction_ok),
        "time_csv_ok": bool(time_csv_ok),
        "logs_ok": bool(logs_ok),
        "provenance_ok": resolved_provenance_ok,
        "provenance_log_ok": bool(provenance_log_ok),
        "inference_time_ok": bool(inference_time_ok),
        "package_ok": bool(package_ok),
        "code_manifest_ok": resolved_code_manifest_ok,
        "methodology_ok": resolved_methodology_ok,
        "methodology_records_only_ok": resolved_methodology_records_only_ok,
        "secret_scan_ok": bool(secret_scan_ok),
        "task_rule_ok": bool(task_rule_ok),
        "shared_code_ok": bool(shared_code_ok),
        "code_provenance_ok": not missing_provenance_paths,
        "api_contract_ok": bool(api_contract_ok),
        "task1_compat_ok": task_compatibility.get("task1", False),
        "task2_compat_ok": task_compatibility.get("task2", False),
        "task3_compat_ok": task_compatibility.get("task3", False),
        "incremental_patch_ok": bool(incremental_patch_ok),
        "no_task_specific_code_fork_ok": not task_fork_violations,
        "task_specific_code_fork_violations": task_fork_violations,
        "missing_code_provenance_paths": missing_provenance_paths,
        "missing_code_manifest_paths": missing_manifest_paths,
        "code_manifest_issues": manifest_issues,
        "methodology_sources": normalized_methodology_sources,
        "invalid_methodology_sources": invalid_methodology_sources,
        "supported_tasks": normalized_supported_tasks,
        "failures": gate_failures,
        "warnings": gate_warnings,
    }
    for task in TASK_IDS:
        details = normalized_cli_smoke_details[task]
        status[f"{task}_cli_parse_ok"] = details["cli_parse_ok"]
        status[f"{task}_train_smoke_ok"] = details["train_smoke_ok"]
        status[f"{task}_infer_smoke_ok"] = details["infer_smoke_ok"]
        status[f"{task}_checkpoint_load_ok"] = details["checkpoint_load_ok"]

    required_ok_fields = [
        "prediction_ok",
        "time_csv_ok",
        "logs_ok",
        "provenance_ok",
        "inference_time_ok",
        "package_ok",
        "code_manifest_ok",
        "methodology_ok",
        "methodology_records_only_ok",
        "secret_scan_ok",
        "task_rule_ok",
        "shared_code_ok",
        "code_provenance_ok",
        "api_contract_ok",
        "incremental_patch_ok",
        "no_task_specific_code_fork_ok",
    ]
    required_ok_fields.extend(f"{task}_compat_ok" for task in normalized_supported_tasks)
    status["overall_ok"] = all(bool(status[name]) for name in required_ok_fields)
    return status
