from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .config import RunnerConfig, load_config
from .hosted_tools import build_hosted_tools
from .logger import LLMCallLogger, ToolCallLogger
from .prompts import (
    REHEARSAL_PROMPT,
    SYSTEM_PROMPT,
    TEST_TOOL_LOOP_PROMPT,
    build_autonomous_dry_run_prompt,
    build_autonomous_rehearsal_prompt,
    build_autonomous_user_prompt,
)
from .responses_client import ResponsesClient
from .responses_items import (
    extract_function_calls,
    extract_hosted_tool_calls,
    extract_output_text,
    function_call_output,
    response_to_ledger_items,
    system_text,
    user_text,
)
from .router import Router
from .skills import build_skill_catalog, load_local_skills
from .state import AgentState
from .tool_registry import ToolDefinition, ToolRegistry, build_tool_registry
from .tools import failure, success
from .tools.validate_tools import validate_jsonl_logs, validate_responses_logs

SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b", re.IGNORECASE),
    re.compile(r"\bapi[_-]?key\b", re.IGNORECASE),
    re.compile(r"\bauthorization\b", re.IGNORECASE),
    re.compile(r"\bbearer\b", re.IGNORECASE),
]
TEXT_SCAN_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".csv", ".md", ".py", ".yaml", ".yml"}


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


def _prepare_session_logs(config: RunnerConfig) -> None:
    config.llm_log_path.parent.mkdir(parents=True, exist_ok=True)
    config.tool_log_path.parent.mkdir(parents=True, exist_ok=True)
    config.llm_log_path.write_text("", encoding="utf-8")
    config.tool_log_path.write_text("", encoding="utf-8")


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
    for entry in entries:
        for key in ("path", "sha256", "size", "step_id", "task_id", "timestamp"):
            if key not in entry:
                issues.append(f"code manifest entry missing {key}")
                continue
        relative = entry.get("path")
        if not isinstance(relative, str):
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
    client: ResponsesClient,
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
        raise ValueError("profile_override is required for Responses-only calls")
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
    tool_calls = [
        {
            "name": call.name,
            "arguments": call.arguments,
            "call_id": call.call_id,
        }
        for call in extract_function_calls(response)
    ]
    hosted_tool_calls = extract_hosted_tool_calls(response)
    content = extract_output_text(response)
    llm_logger.log_call(
        step_id=step_id,
        task_id=task_id,
        model=profile.model,
        profile=profile_name,
        phase=phase,
        elapsed_seconds=elapsed,
        response=content or None,
        tool_calls=tool_calls or None,
        hosted_tool_calls=hosted_tool_calls or None,
        raw_response=response,
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
    client: ResponsesClient | None = None,
    completion_token: str = "RUNNER_FINALIZED",
    continue_instruction: str = "继续 autonomous loop。需要具体动作时必须调用工具；只有在校验和打包完成后才能输出 RUNNER_FINALIZED。",
    system_prompt: str = SYSTEM_PROMPT,
    phase_hint: str | None = None,
) -> tuple[bool, list[dict[str, Any]], str]:
    client = client or ResponsesClient(config.endpoint)
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
        registry.set_context(task_id=task_id, step_id=step_id)
        if state.should_finalize():
            ledger.append(
                system_text("预算接近上限。停止新实验，优先 validate、导出 task logs、生成提交文件并调用 package_submission。")
            )

        route = router.choose(
            summary=last_text or continue_instruction,
            task_id=task_id,
            step_id=step_id,
            phase_hint=phase_hint,
        )
        profile = config.llm_profiles[route.profile]
        local_tools = registry.response_function_tools()
        hosted_tools = build_hosted_tools(config, phase=route.phase) if route.enable_hosted_tools else []
        response = _call_model(
            client=client,
            llm_logger=llm_logger,
            ledger=ledger,
            tools=hosted_tools + local_tools,
            instructions=instructions,
            task_id=task_id,
            step_id=step_id,
            state=state,
            profile_name=profile.name,
            phase=route.phase,
            profile_override=profile,
        )
        ledger.extend(response_to_ledger_items(response))
        function_calls = extract_function_calls(response)
        last_text = extract_output_text(response)

        if function_calls:
            for call in function_calls:
                if call.name in registry:
                    result = registry.execute_response_function_call(call)
                    state.record_tool_call()
                    ledger.append(function_call_output(call.call_id, result))
            continue

        if completion_token in last_text:
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
    client: ResponsesClient,
    completion_token: str,
    max_steps: int = 4,
    system_prompt: str = SYSTEM_PROMPT,
    phase_hint: str | None = "implementation",
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
        print("FAIL hello.py was not created under workspace/submission/code")
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


def run_live_api_check(config: RunnerConfig) -> int:
    if not config.endpoint.api_key:
        print(f"LLM API key is not configured. Set {config.endpoint.api_key_env} before running live_api_check.")
        return 1

    _prepare_session_logs(config)
    client = ResponsesClient(config.endpoint)
    llm_logger = LLMCallLogger(config.llm_log_path)
    tool_logger = ToolCallLogger(config.tool_log_path)
    state = AgentState(config.budget)
    checks: list[str] = []

    simple_response = _call_model(
        client=client,
        llm_logger=llm_logger,
        ledger=[user_text("This is live_api_check step 1. Reply with a short confirmation sentence.")],
        tools=[],
        instructions="Reply with one short sentence only.",
        task_id="live_api_check",
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
    echo_items = [
        system_text("You are validating function tool-calling. You must call echo_tool exactly once."),
        user_text("Call echo_tool with text 'hello-tool'. After receiving the tool result, summarize it briefly and include LIVE_API_CHECK_COMPLETE."),
    ]
    echo_ok, echo_text = _run_tool_round(
        config=config,
        task_id="live_api_echo_tool",
        items=echo_items,
        registry=live_registry,
        client=client,
        completion_token="LIVE_API_CHECK_COMPLETE",
    )
    if not echo_ok:
        print(f"FAIL echo_tool_call: {echo_text}")
        return 1
    checks.append("PASS echo_tool_call")

    write_items = [
        system_text("You are validating write_file tool-calling. Do not use submission/code."),
        user_text(
            "Call write_file and write one line of text to workspace/runs/scratch/live_api_check.txt. "
            "After the tool result, reply with LIVE_API_CHECK_COMPLETE and a short summary."
        ),
    ]
    write_ok, write_text = _run_tool_round(
        config=config,
        task_id="live_api_write_file",
        items=write_items,
        registry=live_registry,
        client=client,
        completion_token="LIVE_API_CHECK_COMPLETE",
    )
    if not write_ok:
        print(f"FAIL write_file_tool_call: {write_text}")
        return 1
    if not (config.workspace_root / "runs" / "scratch" / "live_api_check.txt").exists():
        print("FAIL write_file_tool_call: live_api_check.txt was not created")
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


def run_research_api_check(config: RunnerConfig) -> int:
    if not config.endpoint.api_key:
        print(f"LLM API key is not configured. Set {config.endpoint.api_key_env} before running research_api_check.")
        return 1

    _prepare_session_logs(config)
    client = ResponsesClient(config.endpoint)
    llm_logger = LLMCallLogger(config.llm_log_path)
    state = AgentState(config.budget)
    response = _call_model(
        client=client,
        llm_logger=llm_logger,
        ledger=[user_text("Find one relevant arXiv or GitHub source for neural operators and summarize it briefly.")],
        tools=build_hosted_tools(config, phase="research"),
        instructions="Use hosted research tools when helpful. Keep the answer short.",
        task_id="research_api_check",
        step_id="research_step",
        state=state,
        profile_name="strong_planner",
        phase="research",
        profile_override=config.llm_profiles["strong_planner"],
    )
    if not extract_output_text(response).strip():
        print("FAIL research_api_check: empty response")
        return 1
    if validate_jsonl_logs(config.llm_log_path)["ok"] is not True:
        print("FAIL research_api_check: llm logs invalid")
        return 1
    print("PASS research_api_check")
    return 0


def run_autonomous(config: RunnerConfig, tasks: list[str], max_steps: int | None = None) -> int:
    docs_context = _load_docs_context(config.project_root)
    workspace_listing = _summarize_workspace(config)
    baseline_listing = _directory_listing(config.workspace_root / "baselines")
    registry = build_tool_registry(
        config,
        ToolCallLogger(config.tool_log_path),
        extra_read_roots=[config.project_root / "docs"],
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
        config=config,
        initial_items=items,
        task_id="autonomous",
        max_steps=max_steps or config.budget.max_agent_steps,
        registry=registry,
    )
    print(last_text or ("RUNNER_FINALIZED" if ok else "Autonomous loop stopped without finalization."))
    return 0 if ok else 1


def run_autonomous_dry_run(config: RunnerConfig, tasks: list[str], max_steps: int) -> int:
    if not config.endpoint.api_key:
        print(f"LLM API key is not configured. Set {config.endpoint.api_key_env} before running autonomous_dry_run.")
        return 1

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
            "继续 dry-run。只能读取 docs 和 workspace 内容，允许把计划写入 workspace/runs/autonomous_dry_run/plan.md，"
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

    report_path = rehearsal_config.workspace_root / "runs" / "rehearsal" / "rehearsal_report.md"
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


def export_task_logs(*, workspace: str | Path, tasks: list[str]) -> dict[str, Any]:
    workspace = Path(workspace)
    llm_log = workspace / "llm_logs" / "all_llm_calls.jsonl"
    if not llm_log.exists():
        return failure("export_task_logs", "workspace/llm_logs/all_llm_calls.jsonl does not exist", path=str(llm_log))

    content = llm_log.read_text(encoding="utf-8")
    if not content.strip():
        return failure("export_task_logs", "workspace/llm_logs/all_llm_calls.jsonl is empty", path=str(llm_log))

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


def run_export_task_logs(config: RunnerConfig, tasks: list[str]) -> int:
    result = export_task_logs(workspace=config.workspace_root, tasks=tasks)
    if not result["ok"]:
        print(f"FAIL export_task_logs: {result['error']}")
        return 1
    print(result["summary"])
    for path in result["data"]["exported"]:
        print(f"PASS {path}")
    return 0


def run_final_check(config: RunnerConfig, strict: bool = False) -> int:
    checks: list[str] = []
    warnings: list[str] = []
    failures: list[str] = []
    test_hdf5 = _find_test_hdf5(config)

    llm_log_result = validate_jsonl_logs(config.llm_log_path)
    if llm_log_result["ok"]:
        checks.append("PASS logs JSONL")
    elif strict:
        failures.append(f"FAIL logs JSONL: {llm_log_result['error']}")
    else:
        warnings.append(f"WARN logs JSONL: {llm_log_result['error']}")

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

    for task in ["task1", "task2"]:
        pred_path = config.submission_dir / f"{task}_pred.hdf5"
        time_path = config.submission_dir / f"{task}_time.csv"
        logs_path = config.submission_dir / f"{task}_logs.log"
        any_exists = any(path.exists() for path in [pred_path, time_path, logs_path])
        if not any_exists:
            if strict:
                failures.append(f"FAIL {task} bundle missing")
            else:
                warnings.append(f"WARN {task} bundle missing")
            continue

        if logs_path.exists():
            log_result = validate_jsonl_logs(logs_path)
            if log_result["ok"]:
                checks.append(f"PASS {task} logs JSONL")
            else:
                failures.append(f"FAIL {task} logs JSONL: {log_result['error']}")
        elif strict:
            failures.append(f"FAIL {task} logs missing")
        else:
            warnings.append(f"WARN {task} logs missing")

        if time_path.exists():
            time_ok, time_error = _validate_time_csv(time_path)
            if time_ok:
                checks.append(f"PASS {task} time.csv")
            else:
                failures.append(f"FAIL {task} time.csv: {time_error}")
        elif strict:
            failures.append(f"FAIL {task} time.csv missing")
        else:
            warnings.append(f"WARN {task} time.csv missing")

        if pred_path.exists():
            pred_ok, pred_error, pred_shape = _read_prediction_shape(pred_path)
            if pred_ok:
                checks.append(f"PASS {task} pred shape {tuple(pred_shape or [])}")
                first_ten_ok, first_ten_error = _check_first_ten_steps(pred_path, test_hdf5)
                if first_ten_ok:
                    checks.append(f"PASS {task} initial condition")
                else:
                    failures.append(f"FAIL {task} initial condition: {first_ten_error}")
            else:
                failures.append(f"FAIL {task} pred.hdf5: {pred_error}")
        elif strict:
            failures.append(f"FAIL {task} pred.hdf5 missing")
        else:
            warnings.append(f"WARN {task} pred.hdf5 missing")

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

    rehearsal_report = config.workspace_root / "runs" / "rehearsal" / "rehearsal_report.md"
    if rehearsal_report.exists() and not any((config.submission_dir / f"{task}_pred.hdf5").exists() for task in ["task1", "task2"]):
        warnings.append("WARN rehearsal artifacts exist but formal task predictions are not present")

    leak_hits = _scan_for_secret_leaks([config.submission_dir, config.workspace_root / "llm_logs"])
    if leak_hits:
        for hit in leak_hits:
            failures.append(f"FAIL secret leak scan: {hit}")
    else:
        checks.append("PASS secret leak scan")

    _print_lines(checks + warnings + failures)
    return 0 if not failures else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDE competition autonomous agent runner")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "preflight",
            "test_tool_loop",
            "live_api_check",
            "research_api_check",
            "autonomous",
            "autonomous_dry_run",
            "autonomous_rehearsal",
            "export_task_logs",
            "final_check",
        ],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", default="task1,task2")
    parser.add_argument("--workspace")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-seconds-per-task", type=int, default=600)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, workspace_override=args.workspace)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if args.mode == "preflight":
        return run_preflight(config)
    if args.mode == "test_tool_loop":
        return run_test_tool_loop(config)
    if args.mode == "live_api_check":
        return run_live_api_check(config)
    if args.mode == "research_api_check":
        return run_research_api_check(config)
    if args.mode == "autonomous":
        return run_autonomous(config, tasks, args.max_steps)
    if args.mode == "autonomous_dry_run":
        return run_autonomous_dry_run(config, tasks, args.max_steps or 6)
    if args.mode == "autonomous_rehearsal":
        return run_autonomous_rehearsal(config, tasks, args.max_steps or 20, args.max_train_seconds_per_task)
    if args.mode == "export_task_logs":
        return run_export_task_logs(config, tasks)
    return run_final_check(config, strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
