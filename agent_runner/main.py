from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .config import RunnerConfig, load_config
from .logger import LLMCallLogger, ToolCallLogger
from .research_cache import ResearchCache
from .prompts import (
    REHEARSAL_PROMPT,
    SYSTEM_PROMPT,
    TEST_TOOL_LOOP_PROMPT,
    build_autonomous_dry_run_prompt,
    build_autonomous_rehearsal_prompt,
    build_autonomous_user_prompt,
)
from .json_action_client import JsonActionClient
from .safety import WorkspaceSafety
from .responses_items import (
    extract_final_output_text,
    extract_function_calls,
    extract_output_text,
    function_call_output,
    response_to_ledger_items,
    system_text,
    user_text,
)
from .router import RouteDecision, Router
from .skills import build_skill_catalog, load_local_skills
from .state import AgentState
from .tool_registry import ToolDefinition, ToolRegistry, build_tool_registry
from .tools import failure, success
from .tools.document_tools import generate_methodology_pdf
from .tools.package_tools import package_submission
from .tools.research_tools import fetch_pdf, fetch_url, parse_pdf, search_arxiv, search_github
from .tools.validate_tools import validate_jsonl_logs, validate_responses_logs, validate_submission

SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b", re.IGNORECASE),
    re.compile(r"\bapi[_-]?key\b", re.IGNORECASE),
    re.compile(r"\bauthorization\b", re.IGNORECASE),
    re.compile(r"\bbearer\b", re.IGNORECASE),
]
TEXT_SCAN_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".csv", ".md", ".py", ".yaml", ".yml"}
CODE_POLICY_SCAN_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".sh", ".txt"}
ACTION_PROTOCOL_RULE = (
    "You do not execute tools directly. "
    "Respond with exactly one JSON object and nothing else. "
    'Use {"type":"action","tool_name":"<tool>","arguments":{...}} when one local tool is needed. '
    'Use {"type":"final","message":"RUNNER_FINALIZED"} only when the task is actually complete. '
    "Never emit multiple actions in one response and never wrap JSON in markdown."
)
ACTION_PROTOCOL_RETRY_RULE = (
    "Retry now with exactly one valid JSON object in the action protocol format. "
    "Do not emit prose, markdown, or multiple actions."
)


def _load_docs_context(project_root: Path, max_chars_per_file: int = 12000) -> str:
    chunks = []
    docs_dir = project_root / "docs"
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.append(f"# {path.name}\n{text[:max_chars_per_file]}")
    return "\n\n".join(chunks)


def _summarize_workspace(config: RunnerConfig) -> str:
    lines = []
    for relative in ["data", "checkpoints", "baselines", "submission", "runs"]:
        path = config.workspace_root / relative
        if not path.exists():
            continue
        count = sum(1 for _ in path.rglob("*"))
        lines.append(f"{relative}: {count} entries")
    return "\n".join(lines)


def _directory_listing(root: Path, max_entries: int = 200) -> str:
    if not root.exists():
        return f"{root.name}: missing"
    entries = []
    for index, path in enumerate(sorted(root.rglob("*"))):
        if index >= max_entries:
            break
        entries.append(str(path.relative_to(root)))
    return "\n".join(entries) if entries else f"{root.name}: empty"


def _parse_task_names(raw_tasks: str) -> list[str]:
    tasks = [task.strip() for task in raw_tasks.split(",") if task.strip()]
    if not tasks:
        raise ValueError("At least one task must be provided.")
    return tasks


def _session_label_for_tasks(tasks: list[str]) -> str:
    return "_".join(tasks)


def _validate_known_tasks(config: RunnerConfig, tasks: list[str]) -> list[str]:
    unknown = [task for task in tasks if task not in config.submission_tasks]
    if unknown:
        raise ValueError(f"Unknown task(s): {', '.join(unknown)}")
    return tasks


def _autodetect_submission_tasks(config: RunnerConfig) -> list[str]:
    detected = [
        task_config.name
        for task_config in config.submission_task_list
        if any(
            (config.submission_dir / filename).exists()
            for filename in (
                task_config.pred_filename,
                task_config.time_filename,
                task_config.logs_filename,
            )
        )
    ]
    return detected or [task_config.name for task_config in config.submission_task_list]


def _selected_task_configs(config: RunnerConfig, tasks: list[str] | None = None) -> list[Any]:
    selected_tasks = tasks or _autodetect_submission_tasks(config)
    return [config.task_config(task) for task in _validate_known_tasks(config, selected_tasks)]


def _format_task_labels(tasks: list[str]) -> str:
    labels = [f"Task {task.removeprefix('task')}" for task in tasks]
    if not labels:
        return "no tasks"
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _build_deterministic_methodology_text(tasks: list[str]) -> str:
    named_tasks = _format_task_labels(tasks)
    return f"""# CozyPDE Deterministic Methodology

## Scope

This deterministic methodology document was generated by the `package_final` utility for {named_tasks}.
It describes CozyPDE's runner architecture and formal session process. This document does not claim task-specific model results, tuned hyperparameters, or experiment outcomes that were not explicitly recorded elsewhere by the Agent.

## Agent architecture

- CozyPDE is a tool-using research Agent that operates through tool-mediated file generation, shell execution, validation, and packaging.
- Final source code is expected to be written through allowed file-writing tools under `workspace/submission/code/`.
- Task-specific implementation files should live under `workspace/submission/code/taskN/`, while optional shared wrappers may live at the code root.

## Task-isolated formal sessions

- Formal autonomous execution should use task-isolated formal sessions, one task per session by default.
- Each task session writes its own LLM timeline, tool-call log, run directory, prediction file, time CSV, and exported task log.
- If a multi-task session is explicitly allowed, the same complete session log must be exported for every task handled in that shared session.

## logging and provenance

- CozyPDE records LLM responses, tool calls, generated files, and validation steps so that final artifacts remain traceable.
- Code provenance is checked against generated files under `workspace/submission/code/`.
- Packaging is deterministic and should preserve task-specific logs rather than rewriting scientific history.

## experiment loop

- The intended Agent loop is observe, plan, implement, validate, experiment, reflect, and finalize.
- Formal packaging should happen only after required task artifacts already exist.
- `package_final` validates existing outputs, writes deterministic metadata, and creates `submission.zip` without rerunning modeling.
"""


def _code_policy_patterns_for_task(task: str) -> list[tuple[re.Pattern[str], str]]:
    common_task1_checkpoint_rules = [
        (re.compile(r"task1_official", re.IGNORECASE), "references Task 1 official checkpoint directory"),
        (
            re.compile(r"1D_Burgers_Sols_Nu0\.001_FNO\.pt", re.IGNORECASE),
            "references the official Task 1 FNO checkpoint",
        ),
        (
            re.compile(r"1D_Burgers_Sols_Nu0\.001_Unet-PF-20\.pt", re.IGNORECASE),
            "references the official Task 1 U-Net checkpoint",
        ),
        (re.compile(r"Unet-PF", re.IGNORECASE), "references a Task 1 official U-Net checkpoint family"),
    ]
    if task == "task2":
        return common_task1_checkpoint_rules
    if task == "task3":
        return common_task1_checkpoint_rules + [
            (
                re.compile(r"(task1|task2).*\.(pt|pth|ckpt)\b", re.IGNORECASE),
                "references Task 1 or Task 2 weight files",
            ),
            (
                re.compile(r"(task1|task2).*(checkpoint|weights?)", re.IGNORECASE),
                "references Task 1 or Task 2 checkpoints or weights",
            ),
            (
                re.compile(r"(checkpoint|weights?).*(task1|task2)", re.IGNORECASE),
                "references Task 1 or Task 2 checkpoints or weights",
            ),
        ]
    return []


def _scan_task_code_policy_risks(config: RunnerConfig, tasks: list[str]) -> list[str]:
    issues: list[str] = []
    for task in tasks:
        task_dir = config.task_submission_code_dir(task)
        if not task_dir.exists():
            continue
        patterns = _code_policy_patterns_for_task(task)
        if not patterns:
            continue
        for file_path in sorted(candidate for candidate in task_dir.rglob("*") if candidate.is_file()):
            if file_path.name.lower() == "readme.md":
                continue
            if file_path.suffix.lower() not in CODE_POLICY_SCAN_SUFFIXES:
                continue
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            for pattern, reason in patterns:
                if pattern.search(text):
                    relative = file_path.relative_to(config.workspace_root).as_posix()
                    issues.append(f"{task}: {relative} {reason}")
    return issues


def _write_submission_metadata(config: RunnerConfig, tasks: list[str]) -> Path:
    selected_task_configs = _selected_task_configs(config, tasks)
    payload = {
        "code_dir": "code",
        "tasks": {
            task_config.name: {
                "prediction": task_config.pred_filename,
                "time_csv": task_config.time_filename,
                "logs": task_config.logs_filename,
                "code_dir": f"code/{task_config.name}",
            }
            for task_config in selected_task_configs
        },
    }
    path = config.submission_dir / "submission.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _ensure_methodology_pdf(config: RunnerConfig, tasks: list[str] | None = None) -> dict[str, Any]:
    methodology_path = config.submission_dir / "methodology.pdf"
    if methodology_path.exists():
        return success("ensure_methodology_pdf", "methodology.pdf already exists", path=str(methodology_path))

    safety = WorkspaceSafety(
        config.workspace_root,
        allowed_write_roots=[config.submission_dir],
        extra_read_roots=[config.project_root / "docs"],
    )
    return generate_methodology_pdf(
        content=_build_deterministic_methodology_text(
            [task_config.name for task_config in _selected_task_configs(config, tasks)]
        ),
        path="submission/methodology.pdf",
        safety=safety,
    )


def _validate_selected_task_outputs(config: RunnerConfig, tasks: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for task_config in _selected_task_configs(config, tasks):
        results.append(
            validate_submission(
                submission_dir=config.submission_dir,
                test_hdf5=config.workspace_root / task_config.test_hdf5,
                pred_filename=task_config.pred_filename,
                time_filename=task_config.time_filename,
                logs_filename=task_config.logs_filename,
                code_dir=config.submission_code_dir,
                rehearsal_mode=False,
                expected_total_steps=task_config.total_steps,
                expected_spatial_points=task_config.spatial_points,
                input_steps=task_config.input_steps,
            )
        )
    return results


def _has_any_file(path: Path) -> bool:
    return path.exists() and any(candidate.is_file() for candidate in path.rglob("*"))


def _resolve_export_source_log(
    *,
    workspace: Path,
    tasks: list[str],
    allow_multi_task_session: bool = False,
) -> Path | None:
    if len(tasks) == 1:
        candidates = [
            workspace / "llm_logs" / f"{tasks[0]}_all_llm_calls.jsonl",
            workspace / "llm_logs" / "all_llm_calls.jsonl",
        ]
    else:
        candidates = [
            workspace / "llm_logs" / f"{_session_label_for_tasks(tasks)}_all_llm_calls.jsonl",
            workspace / "llm_logs" / "all_llm_calls.jsonl",
        ]
        if not allow_multi_task_session and not any(candidate.exists() for candidate in candidates):
            return None
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def prepare_task_session_workspace(config: RunnerConfig, *, tasks: list[str]) -> RunnerConfig:
    session_label = _session_label_for_tasks(tasks)
    session_config = config.with_session(session_label)
    for task in tasks:
        session_config.task_run_dir(task).mkdir(parents=True, exist_ok=True)
        session_config.task_submission_code_dir(task).mkdir(parents=True, exist_ok=True)
    _prepare_session_logs(session_config)
    return session_config


def _print_lines(lines: list[str]) -> None:
    for line in lines:
        print(line)


def _validate_tool_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "error": f"{path} does not exist"}
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            return {"ok": False, "error": f"Empty line at {index}"}
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"Invalid JSON at line {index}: {exc}"}
        for key in ("timestamp", "elapsed_seconds", "tool_name", "arguments", "result"):
            if key not in payload:
                return {"ok": False, "error": f"Missing {key} at line {index}"}
    return {"ok": True}


def _read_tool_log_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))
    return entries


def prepare_run_workspace(config: RunnerConfig, *, run_label: str) -> Path | None:
    config.ensure_workspace_dirs()

    def has_files(path: Path) -> bool:
        return path.exists() and any(candidate.is_file() for candidate in path.rglob("*"))

    output_roots = [
        config.workspace_root / "llm_logs",
        config.workspace_root / "internal_logs",
        config.workspace_root / "submission",
        config.workspace_root / "research",
    ]
    runs_dir = config.workspace_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    archive_root = runs_dir / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    runs_children_to_archive = [path for path in runs_dir.iterdir() if path.name != "archive" and has_files(path)]
    has_archivable_output = any(has_files(path) for path in output_roots) or bool(runs_children_to_archive)

    archive_dir: Path | None = None
    if has_archivable_output:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_dir = archive_root / f"{timestamp}_{run_label}"
        archive_dir.mkdir(parents=True, exist_ok=False)

        for source in output_roots:
            if has_files(source):
                shutil.move(str(source), str(archive_dir / source.name))
        archived_runs_dir = archive_dir / "runs"
        for source in runs_children_to_archive:
            shutil.move(str(source), str(archived_runs_dir / source.name))

    config.ensure_workspace_dirs()
    archive_root.mkdir(parents=True, exist_ok=True)
    return archive_dir


def _write_rehearsal_report_if_missing(rehearsal_config: RunnerConfig, *, last_text: str, ok: bool) -> Path:
    report_path = rehearsal_config.workspace_root / "runs" / "rehearsal" / "rehearsal_report.md"
    if report_path.exists():
        return report_path

    code_files = sorted(
        path.relative_to(rehearsal_config.workspace_root).as_posix()
        for path in rehearsal_config.submission_code_dir.rglob("*")
        if path.is_file()
    ) if rehearsal_config.submission_code_dir.exists() else []
    report_lines = [
        "# Rehearsal Report",
        "",
        f"- status: {'ok' if ok else 'failed'}",
        f"- final_message: {last_text or '(empty)'}",
        f"- submission_code_files: {len(code_files)}",
    ]
    report_lines.extend(f"- code_file: {path}" for path in code_files)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")
    return report_path


def _prepare_session_logs(config: RunnerConfig) -> None:
    config.llm_log_path.parent.mkdir(parents=True, exist_ok=True)
    config.tool_log_path.parent.mkdir(parents=True, exist_ok=True)
    config.llm_log_path.write_text("", encoding="utf-8")
    config.tool_log_path.write_text("", encoding="utf-8")


def _instruction_with_tool_guardrail(
    *,
    instructions: str,
    tools: list[dict[str, Any]],
    stronger: bool = False,
    narrowed_tool_names: list[str] | None = None,
) -> str:
    rule = ACTION_PROTOCOL_RETRY_RULE if stronger else ACTION_PROTOCOL_RULE
    if narrowed_tool_names and len(narrowed_tool_names) == 1:
        rule = f"{rule} The only allowed tool in this response is {narrowed_tool_names[0]}."
    if not tools:
        return f"{instructions}\n\n{rule}"
    tool_manifest = json.dumps(
        [
            {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("parameters"),
            }
            for tool in tools
            if tool.get("type") == "function"
        ],
        ensure_ascii=False,
    )
    return f"{instructions}\n\n{rule}\n\nAllowed tools for this response:\n{tool_manifest}"


def _tool_names_from_ledger(ledger: list[dict[str, Any]], tool_names: list[str]) -> list[str]:
    texts: list[str] = []
    for item in ledger:
        for content_item in item.get("content", []):
            text = content_item.get("text")
            if text:
                texts.append(str(text))
    combined = "\n".join(texts)
    return [tool_name for tool_name in tool_names if tool_name in combined]


def _narrow_tools_for_retry(tools: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_names = [tool["name"] for tool in tools if tool.get("type") == "function" and tool.get("name")]
    matches = _tool_names_from_ledger(ledger, tool_names)
    if len(matches) != 1:
        return tools
    return [tool for tool in tools if tool.get("name") == matches[0]]


def _serialize_function_calls(response: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": call.name,
            "arguments": call.arguments,
            "call_id": call.call_id,
        }
        for call in extract_function_calls(response)
    ]


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _find_test_hdf5(config: RunnerConfig) -> Path | None:
    data_dir = config.workspace_root / "data"
    preferred = data_dir / "test.hdf5"
    if preferred.exists():
        return preferred
    candidates = sorted(data_dir.glob("*test*.hdf5")) + sorted(data_dir.glob("*.hdf5"))
    return candidates[0] if candidates else None


def _read_prediction_shape(path: Path) -> tuple[bool, str, list[int] | None]:
    try:
        with h5py.File(path, "r") as handle:
            datasets = []

            def collect(_: str, obj: object) -> None:
                if isinstance(obj, h5py.Dataset):
                    datasets.append(obj)

            handle.visititems(collect)
            if not datasets:
                return False, "No dataset found in prediction HDF5", None
            pred = np.asarray(datasets[0][...])
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to read prediction HDF5: {exc}", None

    if pred.ndim != 3 or pred.shape[1:] != (200, 256):
        return False, f"Prediction shape must be (N, 200, 256), got {pred.shape}", list(pred.shape)
    if np.isnan(pred).any() or np.isinf(pred).any():
        return False, "Prediction contains NaN or Inf", list(pred.shape)
    return True, "", list(pred.shape)


def _check_first_ten_steps(pred_path: Path, test_hdf5: Path | None) -> tuple[bool, str]:
    if test_hdf5 is None or not test_hdf5.exists():
        return True, ""
    try:
        with h5py.File(pred_path, "r") as pred_handle, h5py.File(test_hdf5, "r") as test_handle:
            pred_dataset = []
            test_dataset = []

            pred_handle.visititems(lambda _name, obj: pred_dataset.append(obj) if isinstance(obj, h5py.Dataset) else None)
            test_handle.visititems(lambda _name, obj: test_dataset.append(obj) if isinstance(obj, h5py.Dataset) else None)
            pred = np.asarray(pred_dataset[0][...])
            test = np.asarray(test_dataset[0][...])
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to compare first 10 steps: {exc}"

    if test.shape[0] != pred.shape[0] or test.shape[1] < 10 or test.shape[2] != 256:
        return False, "Test HDF5 shape is incompatible with prediction"
    if not np.allclose(pred[:, :10, :], test[:, :10, :], atol=1e-3, rtol=0.0):
        return False, "Prediction first 10 steps do not match test input"
    return True, ""


def _validate_time_csv(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "time.csv does not exist"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return False, "time.csv has no rows"
    row = rows[0]
    for key in ("train_time", "inference_time"):
        if key not in row:
            return False, f"time.csv missing column {key}"
        try:
            float(row[key])
        except ValueError:
            return False, f"time.csv column {key} is not numeric"
    return True, ""


def _scan_for_secret_leaks(paths: list[Path]) -> list[str]:
    hits: list[str] = []
    for root in paths:
        if not root.exists():
            continue
        for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
            if file_path.suffix.lower() not in TEXT_SCAN_SUFFIXES:
                continue
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    hits.append(f"{file_path}: matched {pattern.pattern}")
                    break
    return hits


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


def _validate_code_manifest(manifest_path: Path, workspace_root: Path) -> tuple[bool, list[str]]:
    issues: list[str] = []
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, [f"Failed to read code_manifest.json: {exc}"]
    if not isinstance(entries, list):
        return False, ["code_manifest.json must contain a list of file metadata"]
    final_entries_by_path: dict[str, dict] = {}
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
    for relative in ordered_paths:
        entry = final_entries_by_path[relative]
        for key in ("path", "sha256", "size", "step_id", "task_id", "timestamp"):
            if key not in entry:
                issues.append(f"code manifest entry missing {key}")
                continue
        file_path = workspace_root / relative
        if not file_path.exists():
            issues.append(f"code manifest entry missing file: {relative}")
            continue
        if file_path.stat().st_size != entry.get("size"):
            issues.append(f"code manifest size mismatch: {relative}")
        if _sha256_file(file_path) != entry.get("sha256"):
            issues.append(f"code manifest sha256 mismatch: {relative}")
    return len(issues) == 0, issues


def _load_skill_catalog(config: RunnerConfig) -> str:
    if not config.responses_tools.enable_skills:
        return ""
    skill_dirs = [config.project_root / path for path in config.responses_tools.skills.get("local_skill_dirs", [])]
    enabled = list(config.responses_tools.skills.get("enabled", []))
    if not skill_dirs or not enabled:
        return ""
    try:
        skills = load_local_skills(skill_dirs, enabled)
    except FileNotFoundError:
        return ""
    return build_skill_catalog(skills)


def run_preflight(config: RunnerConfig) -> int:
    checks: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    config.ensure_workspace_dirs()
    checks.append(f"PASS workspace: {config.workspace_root}")

    for path in [config.workspace_root / "llm_logs", config.workspace_root / "internal_logs", config.workspace_root / "runs" / "scratch"]:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append(f"PASS writable: {path}")

    if sys.version_info >= (3, 10):
        checks.append(f"PASS python: {sys.version.split()[0]}")
    else:
        errors.append(f"FAIL python: unsupported version {sys.version.split()[0]}")

    try:
        import openai  # noqa: F401

        checks.append("PASS openai_sdk: importable")
    except ImportError as exc:
        errors.append(f"FAIL openai_sdk: {exc}")

    try:
        import torch

        if torch.cuda.is_available():
            checks.append(f"PASS torch_cuda: {torch.cuda.get_device_name(0)}")
        else:
            warnings.append("WARN torch_cuda: CUDA unavailable")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"WARN torch: {exc}")

    checks.append(
        "PASS config_fields: "
        f"router={config.router.model}/{config.router.wire_api} "
        f"profiles={','.join(sorted(config.llm_profiles))}"
    )

    if config.endpoint.api_key:
        checks.append(f"PASS api_key_env: {config.endpoint.api_key_env} is set")
    else:
        warnings.append(f"WARN api_key_env: {config.endpoint.api_key_env} is not set")

    scratch_log = config.workspace_root / "runs" / "scratch" / "preflight.jsonl"
    scratch_log.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-16T00:00:00+00:00",
                "elapsed_seconds": 0.0,
                "response": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    log_result = validate_jsonl_logs(scratch_log)
    if log_result["ok"]:
        checks.append("PASS validate_jsonl_logs: runnable")
    else:
        errors.append(f"FAIL validate_jsonl_logs: {log_result['error']}")

    _print_lines(checks + warnings + errors)
    return 0 if not errors else 1


def _call_model(
    *,
    client: JsonActionClient,
    llm_logger: LLMCallLogger,
    ledger: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    instructions: str,
    task_id: str,
    step_id: str,
    state: AgentState,
    profile_name: str,
    phase: str,
    profile_override=None,
) -> Any:
    profile = profile_override
    if profile is None:
        raise ValueError("profile_override is required for JSON-action calls")
    started = time.perf_counter()
    response = client.create(
        profile=profile,
        input_items=ledger,
        tools=tools,
        instructions=instructions,
        metadata={
            "task_id": task_id,
            "step_id": step_id,
            "profile": profile_name,
            "phase": phase,
        },
    )
    elapsed = time.perf_counter() - started
    tool_calls = _serialize_function_calls(response)
    content = extract_output_text(response)
    logged_model = str(response.get("model") or profile.model) if isinstance(response, dict) else profile.model
    llm_logger.log_call(
        step_id=step_id,
        task_id=task_id,
        model=logged_model,
        profile=profile_name,
        phase=phase,
        elapsed_seconds=elapsed,
        response=content or None,
        tool_calls=tool_calls or None,
        raw_response=response.get("raw_response", response) if isinstance(response, dict) else response,
    )
    state.record_llm_call()
    return response


def execute_agent_loop(
    *,
    config: RunnerConfig,
    initial_items: list[dict[str, Any]],
    task_id: str,
    max_steps: int,
    registry: ToolRegistry | None = None,
    client: JsonActionClient | None = None,
    completion_token: str = "RUNNER_FINALIZED",
    continue_instruction: str = "继续 autonomous loop。需要具体动作时必须调用工具；只有在校验和打包完成后才能输出 RUNNER_FINALIZED。",
    system_prompt: str = SYSTEM_PROMPT,
    phase_hint: str | None = None,
    fixed_route: RouteDecision | None = None,
    exposed_tool_names: set[str] | None = None,
) -> tuple[bool, list[dict[str, Any]], str]:
    client = client or JsonActionClient(config.endpoint, config.responses, config.fallback_provider)
    llm_logger = LLMCallLogger(config.llm_log_path)
    tool_logger = ToolCallLogger(config.tool_log_path)
    registry = registry or build_tool_registry(
        config,
        tool_logger,
        extra_read_roots=[config.project_root / "docs"],
    )
    state = AgentState(config.budget)
    router = Router(client=client, config=config, llm_logger=llm_logger)
    ledger = list(initial_items)
    last_text = ""
    skill_catalog = _load_skill_catalog(config)

    instructions = system_prompt
    if skill_catalog:
        instructions = f"{system_prompt}\n\nLocal skill catalog:\n{skill_catalog}"

    for step_index in range(1, max_steps + 1):
        state.record_step()
        step_id = f"step-{step_index:03d}"
        registry.set_context(task_id=task_id, step_id=step_id, exposed_tool_names=exposed_tool_names)
        if state.should_finalize():
            ledger.append(
                system_text("预算接近上限。停止新实验，优先 validate、导出 task logs、生成提交文件并调用 package_submission。")
            )

        route = fixed_route or router.choose(
            summary=last_text or continue_instruction,
            task_id=task_id,
            step_id=step_id,
            phase_hint=phase_hint,
        )
        profile = config.llm_profiles[route.profile]
        registry.set_context(phase=route.phase, exposed_tool_names=exposed_tool_names)
        local_tools = registry.response_function_tools()
        active_tools = local_tools
        retry_count = 0

        while True:
            narrowed_tool_names = [tool["name"] for tool in active_tools if tool.get("type") == "function" and tool.get("name")]
            active_instructions = _instruction_with_tool_guardrail(
                instructions=instructions,
                tools=active_tools,
                stronger=retry_count > 0,
                narrowed_tool_names=narrowed_tool_names,
            )
            try:
                response = _call_model(
                    client=client,
                    llm_logger=llm_logger,
                    ledger=ledger,
                    tools=active_tools,
                    instructions=active_instructions,
                    task_id=task_id,
                    step_id=step_id,
                    state=state,
                    profile_name=profile.name,
                    phase=route.phase,
                    profile_override=profile,
                )
            except Exception as exc:  # noqa: BLE001
                return False, ledger, f"provider_multi_tool_or_gateway_error: {exc}"

            function_calls = extract_function_calls(response)
            if len(function_calls) > config.responses.max_tool_calls_per_turn:
                tool_logger.log_call(
                    tool_name="multi_tool_call_violation",
                    elapsed_seconds=0.0,
                    arguments={
                        "task_id": task_id,
                        "step_id": step_id,
                        "profile": profile.name,
                        "phase": route.phase,
                        "tool_calls": _serialize_function_calls(response),
                    },
                    result=failure(
                        "multi_tool_call_violation",
                        f"Model emitted {len(function_calls)} tool calls in one response.",
                    ),
                )
                if config.responses.retry_on_multi_tool_failure and retry_count < 1:
                    retry_count += 1
                    active_tools = _narrow_tools_for_retry(active_tools, ledger)
                    ledger.append(user_text("You emitted multiple actions. Retry with exactly one action JSON object."))
                    continue
                return False, ledger, "multi_tool_call_violation: model emitted multiple tool calls"

            final_text = extract_final_output_text(response)
            if not function_calls and final_text is None and active_tools and retry_count < 1:
                retry_count += 1
                ledger.extend(response_to_ledger_items(response))
                ledger.append(
                    user_text(
                        "Your reply was not a valid action JSON object. Retry with exactly one JSON action or final object."
                    )
                )
                continue
            break

        ledger.extend(response_to_ledger_items(response))
        function_calls = extract_function_calls(response)
        final_text = extract_final_output_text(response)
        last_text = final_text or extract_output_text(response)

        if len(function_calls) == 1:
            call = function_calls[0]
            if call.name in registry:
                result = registry.execute_response_function_call(call)
                state.record_tool_call()
                ledger.append(function_call_output(call.call_id, result))
            continue

        if final_text is not None and completion_token in final_text:
            return True, ledger, last_text

        ledger.append(user_text(continue_instruction))

    return False, ledger, last_text


def _make_echo_tool() -> ToolDefinition:
    return ToolDefinition(
        name="echo_tool",
        description="Echo the provided text exactly.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=lambda text: success("echo_tool", f"Echoed {text!r}", text=text),
    )


def _run_tool_round(
    *,
    config: RunnerConfig,
    task_id: str,
    items: list[dict[str, Any]],
    registry: ToolRegistry,
    client: JsonActionClient,
    completion_token: str,
    max_steps: int = 4,
    system_prompt: str = SYSTEM_PROMPT,
    phase_hint: str | None = "implementation",
    fixed_route: RouteDecision | None = None,
    exposed_tool_names: set[str] | None = None,
) -> tuple[bool, str]:
    ok, _, last_text = execute_agent_loop(
        config=config,
        initial_items=items,
        task_id=task_id,
        max_steps=max_steps,
        registry=registry,
        client=client,
        completion_token=completion_token,
        continue_instruction=f"如果还未完成，请继续并在完成后输出 {completion_token}。",
        system_prompt=system_prompt,
        phase_hint=phase_hint,
        fixed_route=fixed_route,
        exposed_tool_names=exposed_tool_names,
    )
    return ok, last_text


def run_test_tool_loop(config: RunnerConfig) -> int:
    if not config.endpoint.api_key:
        print(f"LLM API key is not configured. Set {config.endpoint.api_key_env} before running test_tool_loop.")
        return 1

    items = [
        system_text(SYSTEM_PROMPT),
        user_text(TEST_TOOL_LOOP_PROMPT),
    ]
    ok, _, last_text = execute_agent_loop(
        config=config,
        initial_items=items,
        task_id="test_tool_loop",
        max_steps=4,
        phase_hint="implementation",
    )

    if not (config.submission_code_dir / "hello.py").exists():
        print("FAIL hello.py was not created under submission/code")
        return 1

    llm_log_result = validate_jsonl_logs(config.llm_log_path)
    tool_log_result = _validate_tool_log(config.tool_log_path)
    if not llm_log_result["ok"]:
        print(f"FAIL llm_log: {llm_log_result['error']}")
        return 1
    if not tool_log_result["ok"]:
        print(f"FAIL tool_log: {tool_log_result['error']}")
        return 1

    print("PASS hello.py created")
    print("PASS llm log valid")
    print("PASS tool log valid")
    if last_text:
        print(last_text)
    return 0 if ok else 0


def run_provider_health_check(config: RunnerConfig) -> int:
    if not config.endpoint.api_key:
        print(f"LLM API key is not configured. Set {config.endpoint.api_key_env} before running provider_health_check.")
        return 1

    _prepare_session_logs(config)
    client = JsonActionClient(config.endpoint, config.responses, config.fallback_provider)
    llm_logger = LLMCallLogger(config.llm_log_path)
    tool_logger = ToolCallLogger(config.tool_log_path)
    state = AgentState(config.budget)
    checks: list[str] = []

    simple_response = _call_model(
        client=client,
        llm_logger=llm_logger,
        ledger=[user_text("This is provider_health_check step 1. Reply with a short confirmation sentence.")],
        tools=[],
        instructions="Reply with one short sentence only.",
        task_id="provider_health_check",
        step_id="simple_response",
        state=state,
        profile_name="coder",
        phase="implementation",
        profile_override=config.llm_profiles["coder"],
    )
    if extract_output_text(simple_response).strip():
        checks.append("PASS simple_response")
    else:
        print("FAIL simple_response: empty response")
        return 1

    live_registry = build_tool_registry(
        config,
        tool_logger,
        allow_run_shell=False,
        allow_submission_writes=False,
        extra_tools=[_make_echo_tool()],
    )
    tool_log_count = len(_read_tool_log_entries(config.tool_log_path))
    echo_items = [
        system_text("You are validating function tool-calling. You must call echo_tool exactly once."),
        user_text(
            "Call echo_tool with text 'hello-tool'. After receiving the tool result, summarize it briefly and include PROVIDER_HEALTH_CHECK_COMPLETE."
        ),
    ]
    echo_ok, echo_text = _run_tool_round(
        config=config,
        task_id="provider_health_echo_tool",
        items=echo_items,
        registry=live_registry,
        client=client,
        completion_token="PROVIDER_HEALTH_CHECK_COMPLETE",
        fixed_route=RouteDecision(
            profile="coder",
            phase="implementation",
            enable_hosted_tools=False,
            reason="provider health check fixed single-tool route",
        ),
        exposed_tool_names={"echo_tool"},
    )
    if not echo_ok:
        print(f"FAIL echo_tool_call: {echo_text}")
        return 1
    echo_entries = _read_tool_log_entries(config.tool_log_path)[tool_log_count:]
    if not any(
        entry.get("tool_name") == "echo_tool"
        and entry.get("arguments", {}).get("text") == "hello-tool"
        and entry.get("result", {}).get("ok") is True
        for entry in echo_entries
    ):
        print("FAIL echo_tool_call: echo_tool was not actually executed")
        return 1
    tool_log_count += len(echo_entries)
    checks.append("PASS echo_tool_call")

    write_items = [
        system_text("You are validating write_file tool-calling. Do not use submission/code."),
        user_text(
            "Call write_file and write one line of text to runs/scratch/provider_health_check.txt. "
            "After the tool result, reply with PROVIDER_HEALTH_CHECK_COMPLETE and a short summary."
        ),
    ]
    write_ok, write_text = _run_tool_round(
        config=config,
        task_id="provider_health_write_file",
        items=write_items,
        registry=live_registry,
        client=client,
        completion_token="PROVIDER_HEALTH_CHECK_COMPLETE",
        fixed_route=RouteDecision(
            profile="coder",
            phase="implementation",
            enable_hosted_tools=False,
            reason="provider health check fixed single-tool route",
        ),
        exposed_tool_names={"write_file"},
    )
    if not write_ok:
        print(f"FAIL write_file_tool_call: {write_text}")
        return 1
    write_entries = _read_tool_log_entries(config.tool_log_path)[tool_log_count:]
    if not any(
        entry.get("tool_name") == "write_file"
        and entry.get("arguments", {}).get("path") == "runs/scratch/provider_health_check.txt"
        and entry.get("result", {}).get("ok") is True
        for entry in write_entries
    ):
        print("FAIL write_file_tool_call: write_file was not actually executed")
        return 1
    if not (config.workspace_root / "runs" / "scratch" / "provider_health_check.txt").exists():
        print("FAIL write_file_tool_call: provider_health_check.txt was not created")
        return 1
    checks.append("PASS write_file_tool_call")

    llm_log_result = validate_jsonl_logs(config.llm_log_path)
    tool_log_result = _validate_tool_log(config.tool_log_path)
    if not llm_log_result["ok"]:
        print(f"FAIL llm log validation: {llm_log_result['error']}")
        return 1
    if not tool_log_result["ok"]:
        print(f"FAIL tool log validation: {tool_log_result['error']}")
        return 1
    checks.append("PASS validate_jsonl_logs")

    _print_lines(checks)
    return 0


def run_local_research_check(config: RunnerConfig, http_client=None) -> int:
    if not config.research.enabled:
        print("FAIL local_research_check: research config is disabled")
        return 1

    config.ensure_workspace_dirs()
    cache = ResearchCache(config.research)
    report_lines = [
        "# Local Research Check",
        "",
        "This report verifies local research tools without requiring hosted Responses web_search.",
        "",
    ]

    arxiv_result = search_arxiv(
        query="Fourier Neural Operator Burgers PDEBench",
        research=config.research,
        http_client=http_client,
    )
    if not arxiv_result["ok"] or not arxiv_result["data"]["results"]:
        print(f"FAIL local_research_check arXiv: {arxiv_result.get('error', 'no results')}")
        return 1
    arxiv_record = arxiv_result["data"]["results"][0]
    cache.write(
        {
            "source_id": f"arxiv:{arxiv_record['arxiv_id']}",
            "source_type": "arxiv",
            "title": arxiv_record["title"],
            "url": arxiv_record["abs_url"],
            "raw_url": arxiv_record["pdf_url"],
            "query": arxiv_result["data"]["query"],
            "summary": arxiv_record["abstract"][:400],
            "content_sha256": "",
            "raw_cache_path": "",
            "license_hint": "",
            "risk_flags": [],
            "allowed_for_submission_code_reference": True,
            "allowed_for_training_data": False,
        }
    )
    report_lines.extend(
        [
            "## arXiv",
            "",
            f"- title: {arxiv_record['title']}",
            f"- abs_url: {arxiv_record['abs_url']}",
            f"- pdf_url: {arxiv_record['pdf_url']}",
            "",
        ]
    )

    github_queries = [
        "neuraloperator Fourier neural operator Burgers",
        "neuraloperator",
    ]
    github_result = None
    github_query_used = github_queries[0]
    for github_query in github_queries:
        github_result = search_github(
            query=github_query,
            research=config.research,
            http_client=http_client,
        )
        if github_result["ok"] and github_result["data"]["results"]:
            github_query_used = github_query
            break
    if not github_result["ok"] or not github_result["data"]["results"]:
        print(f"FAIL local_research_check GitHub: {github_result.get('error', 'no results')}")
        return 1
    github_record = github_result["data"]["results"][0]
    github_source_id = f"github:{github_record['owner']}/{github_record['repo']}"
    if github_record["path"]:
        github_source_id = f"{github_source_id}:{github_record['path']}"
    cache.write(
        {
            "source_id": github_source_id,
            "source_type": github_record["source_type"],
            "title": github_record["path"] or github_record["repo"],
            "url": github_record["url"],
            "raw_url": github_record["raw_url"],
            "query": github_query_used,
            "summary": github_record["summary"],
            "content_sha256": "",
            "raw_cache_path": "",
            "license_hint": github_record["license_hint"],
            "risk_flags": [],
            "allowed_for_submission_code_reference": True,
            "allowed_for_training_data": False,
        }
    )
    report_lines.extend(
        [
            "## GitHub",
            "",
            f"- query_used: {github_query_used}",
            f"- repo: {github_record['owner']}/{github_record['repo']}",
            f"- url: {github_record['url']}",
            f"- raw_url: {github_record['raw_url'] or '(not available for repository search)'}",
            "",
        ]
    )

    fetch_target = github_record["raw_url"] or arxiv_record["abs_url"]
    fetch_result = fetch_url(url=fetch_target, research=config.research, http_client=http_client)
    if not fetch_result["ok"]:
        print(f"FAIL local_research_check fetch_url: {fetch_result['error']}")
        return 1
    report_lines.extend(
        [
            "## fetch_url",
            "",
            f"- url: {fetch_result['data']['url']}",
            f"- content_type: {fetch_result['data']['content_type']}",
            f"- cache_path: {fetch_result['data']['cache_path']}",
            "",
        ]
    )

    try:
        import pypdf  # noqa: F401

        pdf_parser_available = True
    except ImportError:
        pdf_parser_available = False

    if pdf_parser_available and arxiv_record["pdf_url"]:
        pdf_result = fetch_pdf(url=arxiv_record["pdf_url"], research=config.research, http_client=http_client)
        if pdf_result["ok"]:
            parsed_pdf = parse_pdf(path=pdf_result["data"]["local_path"])
            if parsed_pdf["ok"]:
                report_lines.extend(
                    [
                        "## PDF Parse",
                        "",
                        f"- page_count: {parsed_pdf['data']['page_count']}",
                        "",
                    ]
                )
            else:
                report_lines.extend(["## PDF Parse", "", f"- skipped: {parsed_pdf['error']}", ""])
        else:
            report_lines.extend(["## PDF Parse", "", f"- skipped: {pdf_result['error']}", ""])
    else:
        report_lines.extend(["## PDF Parse", "", "- skipped: local PDF parser is unavailable", ""])

    report_path = config.workspace_root / "runs" / "local_research_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    print("PASS local_research_check")
    return 0


def run_startup_readiness(config: RunnerConfig) -> int:
    checks: list[str] = []
    failures: list[str] = []

    try:
        config.ensure_workspace_dirs()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL startup_readiness config: {exc}")
        return 1
    checks.append("PASS config_loaded")

    if config.endpoint.api_key:
        checks.append(f"PASS api_key: {config.endpoint.api_key_env}")
    else:
        failures.append(f"FAIL api_key: set {config.endpoint.api_key_env}")

    for path in [config.llm_log_path.parent, config.tool_log_path.parent, config.workspace_root / "runs" / "scratch"]:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".startup_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append(f"PASS writable: {path}")

    registry = build_tool_registry(
        config,
        ToolCallLogger(config.tool_log_path),
    )
    registry.set_context(task_id="startup_readiness", step_id="step-001", phase="planning")
    schemas = registry.response_function_tools()
    if schemas:
        checks.append(f"PASS tool_schema: {len(schemas)} tools")
    else:
        failures.append("FAIL tool_schema: no tools available")

    readme_path = config.workspace_root / "runs" / "scratch" / "startup_probe.txt"
    readme_path.write_text("startup probe\n", encoding="utf-8")
    tool_result = registry.execute("read_file", {"path": "runs/scratch/startup_probe.txt"})
    if tool_result["ok"]:
        checks.append("PASS single_tool_call")
    else:
        failures.append(f"FAIL single_tool_call: {tool_result['error']}")

    registry.set_context(task_id="startup_readiness", step_id="step-002", phase="research")
    research_tool_names = {schema["name"] for schema in registry.response_function_tools()}
    if config.research.enabled and config.research.providers.arxiv.enabled:
        if "search_arxiv" in research_tool_names:
            checks.append("PASS research_arxiv: available")
        else:
            failures.append("FAIL research_arxiv: tool not registered")
    else:
        checks.append("PASS research_arxiv: skipped")
    if config.research.enabled and config.research.providers.github.enabled:
        if "search_github" in research_tool_names:
            if config.research.providers.github.api_key or config.research.providers.github.allow_unauthenticated:
                checks.append("PASS research_github: available")
            else:
                checks.append("PASS research_github: skipped")
        else:
            failures.append("FAIL research_github: tool not registered")
    else:
        checks.append("PASS research_github: skipped")

    safety = WorkspaceSafety(config.workspace_root)
    safety_check = safety.validate_write_path("workspace/submission/code/startup.py")
    if safety_check.ok and safety_check.resolved_path is not None:
        normalized = safety_check.resolved_path.relative_to(config.workspace_root).as_posix()
        if normalized == "submission/code/startup.py":
            checks.append("PASS workspace_path_normalization")
        else:
            failures.append(f"FAIL workspace_path_normalization: resolved to {normalized}")
    else:
        failures.append(f"FAIL workspace_path_normalization: {safety_check.error or 'unknown error'}")

    _print_lines(checks + failures)
    return 0 if not failures else 1


def run_autonomous(config: RunnerConfig, tasks: list[str], max_steps: int | None = None) -> int:
    session_config = prepare_task_session_workspace(config, tasks=tasks)
    docs_context = _load_docs_context(session_config.project_root)
    workspace_listing = _summarize_workspace(session_config)
    baseline_listing = _directory_listing(session_config.workspace_root / "baselines")
    registry = build_tool_registry(
        session_config,
        ToolCallLogger(session_config.tool_log_path),
        extra_read_roots=[session_config.project_root / "docs"],
    )
    items = [
        system_text(SYSTEM_PROMPT),
        user_text(
            build_autonomous_user_prompt(
                tasks=tasks,
                docs_context=docs_context,
                baseline_listing=baseline_listing,
                workspace_listing=workspace_listing,
            )
        ),
    ]
    ok, _, last_text = execute_agent_loop(
        config=session_config,
        initial_items=items,
        task_id=_session_label_for_tasks(tasks),
        max_steps=max_steps or session_config.budget.max_agent_steps,
        registry=registry,
    )
    print(last_text or ("RUNNER_FINALIZED" if ok else "Autonomous loop stopped without finalization."))
    return 0 if ok else 1


def run_autonomous_dry_run(config: RunnerConfig, tasks: list[str], max_steps: int) -> int:
    if not config.endpoint.api_key:
        print(f"LLM API key is not configured. Set {config.endpoint.api_key_env} before running autonomous_dry_run.")
        return 1

    prepare_run_workspace(config, run_label="autonomous_dry_run")
    _prepare_session_logs(config)
    docs_context = _load_docs_context(config.project_root)
    workspace_listing = _summarize_workspace(config)
    baseline_listing = _directory_listing(config.workspace_root / "baselines")
    registry = build_tool_registry(
        config,
        ToolCallLogger(config.tool_log_path),
        allow_run_shell=False,
        allow_submission_writes=False,
        extra_read_roots=[config.project_root / "docs"],
    )
    items = [
        system_text(SYSTEM_PROMPT),
        user_text(
            build_autonomous_dry_run_prompt(
                tasks=tasks,
                docs_context=docs_context,
                baseline_listing=baseline_listing,
                workspace_listing=workspace_listing,
            )
        ),
    ]
    ok, _, last_text = execute_agent_loop(
        config=config,
        initial_items=items,
        task_id="autonomous_dry_run",
        max_steps=max_steps,
        registry=registry,
        completion_token="DRY_RUN_COMPLETE",
        continue_instruction=(
            "继续 dry-run。只能读取 docs 和 workspace 内容，允许把计划写入 runs/autonomous_dry_run/plan.md，"
            "禁止 run_shell 和 submission/code 写入。完成后输出 DRY_RUN_COMPLETE。"
        ),
        phase_hint="planning",
    )

    plan_path = config.workspace_root / "runs" / "autonomous_dry_run" / "plan.md"
    llm_log_result = validate_jsonl_logs(config.llm_log_path)
    tool_log_result = _validate_tool_log(config.tool_log_path)
    if not plan_path.exists():
        print("FAIL autonomous_dry_run: plan.md was not created")
        return 1
    if not llm_log_result["ok"]:
        print(f"FAIL autonomous_dry_run llm log: {llm_log_result['error']}")
        return 1
    if not tool_log_result["ok"]:
        print(f"FAIL autonomous_dry_run tool log: {tool_log_result['error']}")
        return 1

    print("PASS autonomous_dry_run plan generated")
    print("PASS autonomous_dry_run llm log valid")
    print("PASS autonomous_dry_run tool log valid")
    if last_text:
        print(last_text)
    return 0 if ok else 0


def run_autonomous_rehearsal(
    config: RunnerConfig,
    tasks: list[str],
    max_steps: int,
    max_train_seconds_per_task: int,
) -> int:
    if not config.endpoint.api_key:
        print(f"LLM API key is not configured. Set {config.endpoint.api_key_env} before running autonomous_rehearsal.")
        return 1

    prepare_run_workspace(config, run_label="autonomous_rehearsal")
    _prepare_session_logs(config)
    rehearsal_budget = replace(
        config.budget,
        max_agent_steps=min(max_steps, 20),
        max_single_shell_seconds=min(config.budget.max_single_shell_seconds, 900, max_train_seconds_per_task),
    )
    rehearsal_profiles = {
        name: replace(profile, max_tokens=min(profile.max_tokens, 4096))
        for name, profile in config.llm_profiles.items()
    }
    rehearsal_config = RunnerConfig(
        project_root=config.project_root,
        workspace_root=config.workspace_root,
        endpoint=config.endpoint,
        router=config.router,
        llm_profiles=rehearsal_profiles,
        responses_tools=config.responses_tools,
        budget=rehearsal_budget,
    )
    rehearsal_config.ensure_workspace_dirs()
    (rehearsal_config.workspace_root / "runs" / "rehearsal").mkdir(parents=True, exist_ok=True)

    docs_context = _load_docs_context(rehearsal_config.project_root)
    workspace_listing = _summarize_workspace(rehearsal_config)
    baseline_listing = _directory_listing(rehearsal_config.workspace_root / "baselines")
    registry = build_tool_registry(
        rehearsal_config,
        ToolCallLogger(rehearsal_config.tool_log_path),
        run_shell_profile="rehearsal",
        rehearsal_validation=True,
        extra_read_roots=[rehearsal_config.project_root / "docs"],
    )
    items = [
        system_text(SYSTEM_PROMPT),
        system_text(REHEARSAL_PROMPT),
        user_text(
            build_autonomous_rehearsal_prompt(
                tasks=tasks,
                docs_context=docs_context,
                baseline_listing=baseline_listing,
                workspace_listing=workspace_listing,
                max_train_seconds_per_task=max_train_seconds_per_task,
            )
        ),
    ]
    ok, _, last_text = execute_agent_loop(
        config=rehearsal_config,
        initial_items=items,
        task_id="autonomous_rehearsal",
        max_steps=min(max_steps, 20),
        registry=registry,
        completion_token="REHEARSAL_COMPLETE",
        continue_instruction=(
            "继续 rehearsal loop。你必须优先用工具完成读取、生成代码、smoke validate、日志分析和必要修复。"
            "禁止长训练；run_shell 只能做 smoke 动作。完成后输出 REHEARSAL_COMPLETE。"
        ),
        phase_hint="implementation",
    )

    report_path = _write_rehearsal_report_if_missing(rehearsal_config, last_text=last_text, ok=ok)
    llm_log_result = validate_jsonl_logs(rehearsal_config.llm_log_path)
    tool_log_result = _validate_tool_log(rehearsal_config.tool_log_path)
    if not report_path.exists():
        print("FAIL autonomous_rehearsal: rehearsal_report.md was not created")
        return 1
    if not llm_log_result["ok"]:
        print(f"FAIL autonomous_rehearsal llm log: {llm_log_result['error']}")
        return 1
    if not tool_log_result["ok"]:
        print(f"FAIL autonomous_rehearsal tool log: {tool_log_result['error']}")
        return 1

    code_manifest_path = rehearsal_config.submission_dir / "code_manifest.json"
    if rehearsal_config.submission_code_dir.exists() and any(rehearsal_config.submission_code_dir.iterdir()):
        if not code_manifest_path.exists():
            print("FAIL autonomous_rehearsal: submission/code was written but code_manifest.json is missing")
            return 1
        manifest_ok, manifest_issues = _validate_code_manifest(code_manifest_path, rehearsal_config.workspace_root)
        if not manifest_ok:
            for issue in manifest_issues:
                print(f"FAIL autonomous_rehearsal code_manifest: {issue}")
            return 1

    print("PASS autonomous_rehearsal report generated")
    print("PASS autonomous_rehearsal llm log valid")
    print("PASS autonomous_rehearsal tool log valid")
    if code_manifest_path.exists():
        print("PASS autonomous_rehearsal code_manifest valid")
    if last_text:
        print(last_text)
    return 0 if ok else 1


def export_task_logs(
    *,
    workspace: str | Path,
    tasks: list[str],
    allow_multi_task_session: bool = False,
) -> dict[str, Any]:
    workspace = Path(workspace)
    llm_log = _resolve_export_source_log(
        workspace=workspace,
        tasks=tasks,
        allow_multi_task_session=allow_multi_task_session,
    )
    if llm_log is None:
        return failure(
            "export_task_logs",
            "Formal autonomous runs should use independent task sessions. Export one task at a time, or pass --allow-multi-task-session for an intentional shared session.",
            tasks=tasks,
        )
    if not llm_log.exists():
        return failure("export_task_logs", f"{llm_log} does not exist", path=str(llm_log))

    content = llm_log.read_text(encoding="utf-8")
    if not content.strip():
        return failure("export_task_logs", f"{llm_log} is empty", path=str(llm_log))

    source_validation = validate_jsonl_logs(llm_log)
    if not source_validation["ok"]:
        return failure("export_task_logs", source_validation["error"], validation=source_validation)

    exported: list[str] = []
    for task in tasks:
        path = workspace / "submission" / f"{task}_logs.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        validation = validate_jsonl_logs(path)
        if not validation["ok"]:
            return failure("export_task_logs", validation["error"], path=str(path), validation=validation)
        exported.append(str(path))

    return success("export_task_logs", f"Exported logs for {', '.join(tasks)}", exported=exported)


def run_export_task_logs(config: RunnerConfig, tasks: list[str], *, allow_multi_task_session: bool = False) -> int:
    result = export_task_logs(
        workspace=config.workspace_root,
        tasks=tasks,
        allow_multi_task_session=allow_multi_task_session,
    )
    if not result["ok"]:
        print(f"FAIL export_task_logs: {result['error']}")
        return 1
    print(result["summary"])
    for path in result["data"]["exported"]:
        print(f"PASS {path}")
    return 0


def run_final_check(config: RunnerConfig, strict: bool = False, tasks: list[str] | None = None) -> int:
    checks: list[str] = []
    warnings: list[str] = []
    failures: list[str] = []

    llm_log_result = validate_jsonl_logs(config.llm_log_path)
    if llm_log_result["ok"]:
        checks.append("PASS logs JSONL")
    elif strict:
        failures.append(f"FAIL logs JSONL: {llm_log_result['error']}")
    else:
        warnings.append(f"WARN logs JSONL: {llm_log_result['error']}")

    if strict:
        responses_log_result = validate_responses_logs(
            config.llm_log_path,
            workspace_root=config.workspace_root,
        )
        if responses_log_result["ok"]:
            checks.append("PASS responses logs trace submission/code")
        else:
            failures.append(f"FAIL responses logs: {responses_log_result['error']}")

    if config.submission_code_dir.exists():
        checks.append("PASS submission/code exists")
    elif strict:
        failures.append("FAIL submission/code does not exist")
    else:
        warnings.append("WARN submission/code does not exist")

    code_files = (
        sorted(
            path
            for path in config.submission_code_dir.glob("**/*")
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
        )
        if config.submission_code_dir.exists()
        else []
    )
    if code_files:
        checks.append("PASS submission/code non-empty")
        for file_path in code_files:
            checks.append(f"CODE {file_path.name} sha256={_sha256_file(file_path)}")
    elif strict:
        failures.append("FAIL submission/code is empty")
    else:
        warnings.append("WARN submission/code is empty")

    methodology_path = config.submission_dir / "methodology.pdf"
    if methodology_path.exists():
        checks.append("PASS methodology.pdf exists")
    elif strict:
        failures.append("FAIL methodology.pdf missing")
    else:
        warnings.append("WARN methodology.pdf missing")

    selected_task_configs = _selected_task_configs(config, tasks)
    if strict:
        policy_issues = _scan_task_code_policy_risks(
            config,
            [task_config.name for task_config in selected_task_configs],
        )
        if policy_issues:
            for issue in policy_issues:
                failures.append(f"FAIL code policy risk: {issue}")
    for task_config in selected_task_configs:
        pred_path = config.submission_dir / task_config.pred_filename
        time_path = config.submission_dir / task_config.time_filename
        logs_path = config.submission_dir / task_config.logs_filename
        any_exists = any(path.exists() for path in [pred_path, time_path, logs_path])
        if not any_exists:
            if strict:
                failures.append(f"FAIL {task_config.name} bundle missing")
            else:
                warnings.append(f"WARN {task_config.name} bundle missing")
            continue

        result = validate_submission(
            submission_dir=config.submission_dir,
            test_hdf5=config.workspace_root / task_config.test_hdf5,
            pred_filename=task_config.pred_filename,
            time_filename=task_config.time_filename,
            logs_filename=task_config.logs_filename,
            code_dir=config.submission_code_dir,
            rehearsal_mode=False,
            expected_total_steps=task_config.total_steps,
            expected_spatial_points=task_config.spatial_points,
            input_steps=task_config.input_steps,
        )
        if result["ok"]:
            checks.append(f"PASS {task_config.name} bundle")
        else:
            failures.append(f"FAIL {task_config.name} bundle: {result['error']}")

    manifest_path = config.submission_dir / "manifest.json"
    if manifest_path.exists():
        manifest_ok, manifest_issues = _validate_manifest(manifest_path, config.submission_dir)
        if manifest_ok:
            checks.append("PASS manifest.json sha256")
        else:
            for issue in manifest_issues:
                failures.append(f"FAIL manifest.json: {issue}")

    code_manifest_path = config.submission_dir / "code_manifest.json"
    if code_manifest_path.exists():
        code_manifest_ok, code_manifest_issues = _validate_code_manifest(code_manifest_path, config.workspace_root)
        if code_manifest_ok:
            checks.append("PASS code_manifest.json sha256")
        else:
            for issue in code_manifest_issues:
                failures.append(f"FAIL code_manifest.json: {issue}")
    elif strict:
        failures.append("FAIL code_manifest.json missing")

    rehearsal_report = config.workspace_root / "runs" / "rehearsal" / "rehearsal_report.md"
    if rehearsal_report.exists() and not any((config.submission_dir / task.pred_filename).exists() for task in selected_task_configs):
        warnings.append("WARN rehearsal artifacts exist but formal task predictions are not present")

    leak_hits = _scan_for_secret_leaks([config.submission_dir, config.workspace_root / "llm_logs"])
    if leak_hits:
        for hit in leak_hits:
            failures.append(f"FAIL secret leak scan: {hit}")
    else:
        checks.append("PASS secret leak scan")

    _print_lines(checks + warnings + failures)
    return 0 if not failures else 1


def run_package_final(config: RunnerConfig, tasks: list[str]) -> int:
    selected_task_configs = _selected_task_configs(config, tasks)
    if not _has_any_file(config.submission_code_dir):
        print("FAIL package_final: workspace/submission/code is empty")
        return 1
    for task_config in selected_task_configs:
        task_code_dir = config.task_submission_code_dir(task_config.name)
        if not _has_any_file(task_code_dir):
            print(f"FAIL package_final: {task_code_dir} is missing or empty")
            return 1

    policy_issues = _scan_task_code_policy_risks(
        config,
        [task_config.name for task_config in selected_task_configs],
    )
    if policy_issues:
        for issue in policy_issues:
            print(f"FAIL package_final policy: {issue}")
        return 1

    submission_json_path = _write_submission_metadata(config, tasks)
    methodology_result = _ensure_methodology_pdf(config, tasks)
    if not methodology_result["ok"]:
        print(f"FAIL package_final: {methodology_result['error']}")
        return 1

    validations = _validate_selected_task_outputs(config, tasks)
    for task_config, result in zip(selected_task_configs, validations, strict=True):
        if not result["ok"]:
            print(f"FAIL package_final {task_config.name}: {result['error']}")
            return 1

    packaged = package_submission(
        submission_dir=config.submission_dir,
        workspace_root=config.workspace_root,
        task_configs=selected_task_configs,
        code_dir=config.submission_code_dir,
    )
    if not packaged["ok"]:
        print(f"FAIL package_final: {packaged['error']}")
        return 1

    print(f"PASS package_final submission_json={submission_json_path}")
    print(f"PASS package_final methodology={config.submission_dir / 'methodology.pdf'}")
    print(f"PASS package_final zip={packaged['data']['zip_path']}")
    return 0


def run_readiness_check(
    config: RunnerConfig,
    *,
    tasks: list[str],
    max_steps: int | None,
    max_train_seconds_per_task: int,
) -> int:
    report_dir = config.workspace_root / "runs" / "readiness_check"
    report_dir.mkdir(parents=True, exist_ok=True)
    stage_results: list[dict[str, object]] = []

    def record(name: str, exit_code: int, detail: str | None = None) -> None:
        stage_results.append(
            {
                "name": name,
                "exit_code": exit_code,
                "ok": exit_code == 0,
                "detail": detail or "",
            }
        )

    def run_stage(name: str, func, *args) -> None:
        try:
            result = func(*args)
        except Exception as exc:  # noqa: BLE001
            record(name, 1, f"{type(exc).__name__}: {exc}")
            return
        if isinstance(result, int):
            record(name, result)
        else:
            record(name, 1, f"unexpected return value: {result!r}")

    run_stage("startup_readiness", run_startup_readiness, config)

    safe_to_start = all(bool(stage["ok"]) for stage in stage_results)
    report_path = report_dir / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "safe_to_start_full_autonomous_run": safe_to_start,
                "stages": stage_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for stage in stage_results:
        status = "PASS" if stage["ok"] else "FAIL"
        detail = f" {stage['detail']}" if stage["detail"] else ""
        print(f"{status} {stage['name']}{detail}")
    print(f"READINESS safe_to_start_full_autonomous_run={str(safe_to_start).lower()}")
    print(f"READINESS report={report_path}")
    return 0 if safe_to_start else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDE competition autonomous agent runner")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "preflight",
            "test_tool_loop",
            "provider_health_check",
            "local_research_check",
            "autonomous",
            "autonomous_dry_run",
            "autonomous_rehearsal",
            "export_task_logs",
            "package_final",
            "final_check",
            "readiness_check",
        ],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", default="task1,task2,task3")
    parser.add_argument("--workspace")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-seconds-per-task", type=int, default=600)
    parser.add_argument("--allow-multi-task-session", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.tasks = _parse_task_names(args.tasks)
    except ValueError as exc:
        parser.error(str(exc))
    if args.mode == "autonomous" and len(args.tasks) != 1 and not args.allow_multi_task_session:
        parser.error(
            "Formal autonomous runs should use independent task sessions. Pass exactly one task via --tasks taskN, or add --allow-multi-task-session for an intentional shared session."
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, workspace_override=args.workspace)
    try:
        tasks = _validate_known_tasks(config, args.tasks)
    except ValueError as exc:
        print(f"FAIL task selection: {exc}")
        return 1
    if args.mode == "preflight":
        return run_preflight(config)
    if args.mode == "test_tool_loop":
        return run_test_tool_loop(config)
    if args.mode == "provider_health_check":
        return run_provider_health_check(config)
    if args.mode == "local_research_check":
        return run_local_research_check(config)
    if args.mode == "autonomous":
        return run_autonomous(config, tasks, args.max_steps)
    if args.mode == "autonomous_dry_run":
        return run_autonomous_dry_run(config, tasks, args.max_steps or 6)
    if args.mode == "autonomous_rehearsal":
        return run_autonomous_rehearsal(config, tasks, args.max_steps or 20, args.max_train_seconds_per_task)
    if args.mode == "export_task_logs":
        return run_export_task_logs(config, tasks, allow_multi_task_session=args.allow_multi_task_session)
    if args.mode == "package_final":
        return run_package_final(config, tasks)
    if args.mode == "readiness_check":
        return run_readiness_check(
            config,
            tasks=tasks,
            max_steps=args.max_steps,
            max_train_seconds_per_task=args.max_train_seconds_per_task,
        )
    return run_final_check(config, strict=args.strict, tasks=tasks)


if __name__ == "__main__":
    raise SystemExit(main())
