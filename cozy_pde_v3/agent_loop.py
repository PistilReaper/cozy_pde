from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from cozy_pde_v3.code_evolution import CodePatchRecord, snapshot_shared_code_directory, read_shared_code_directory_content
from cozy_pde_v3.config import V3Config
from cozy_pde_v3.context_packer import ContextPacker
from cozy_pde_v3.deterministic_router import DeterministicRouter, RouteDecision
from cozy_pde_v3.experiment_engine import patch_compatibility_gate
from cozy_pde_v3.logging import append_jsonl, build_llm_log_entry, utc_timestamp
from cozy_pde_v3.memory_store import MemoryStore
from cozy_pde_v3.package import package_submission_v3
from cozy_pde_v3.state import AgentState
from cozy_pde_v3.validation.submission import validate_submission_bundle_v3


def should_start_formal_run(
    *,
    primary_ready: bool,
    fallback_ready: bool,
    require_fallback: bool,
) -> bool:
    if not primary_ready:
        return False
    if require_fallback and not fallback_ready:
        return False
    return True


def should_allow_finalize(finalize_gate_status: dict[str, Any]) -> tuple[bool, str]:
    if bool(finalize_gate_status.get("overall_ok", False)):
        return True, "finalize gate ready"

    failures = [str(item) for item in finalize_gate_status.get("failures", []) if str(item)]
    if failures:
        return False, "; ".join(failures)
    return False, "finalize gate is not ready"


def _tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a UTF-8 text file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "write_file",
            "description": "Write a UTF-8 text file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "compatibility": {
                        "type": "object",
                        "properties": {
                            "cli_ok": {"type": "boolean"},
                            "smoke_ok": {"type": "boolean"},
                            "infer_shape_ok": {"type": "boolean"},
                        },
                        "additionalProperties": False,
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "patch_file",
            "description": "Replace old text with new text in a UTF-8 file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "compatibility": {
                        "type": "object",
                        "properties": {
                            "cli_ok": {"type": "boolean"},
                            "smoke_ok": {"type": "boolean"},
                            "infer_shape_ok": {"type": "boolean"},
                        },
                        "additionalProperties": False,
                    },
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "run_python",
            "description": "Run a short Python snippet locally.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "run_shell",
            "description": "Run a shell command locally in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "inspect_hdf5",
            "description": "Inspect the datasets inside an HDF5 file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "validate_submission",
            "description": "Run deterministic submission validation for the current task.",
            "parameters": {
                "type": "object",
                "properties": {"strict": {"type": "boolean"}},
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "package_submission",
            "description": "Package the current submission bundle for the current task.",
            "parameters": {
                "type": "object",
                "properties": {"strict": {"type": "boolean"}},
                "additionalProperties": False,
            },
        },
    ]


def _tool_schemas_for_names(allowed_names: list[str]) -> list[dict[str, Any]]:
    allowed = set(allowed_names)
    return [schema for schema in _tool_schemas() if schema["name"] in allowed]


def _load_provider_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"provider capability report does not exist: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _provider_ready_flags(report: dict[str, Any]) -> tuple[bool, bool, bool]:
    primary = report.get("primary", {})
    fallback = report.get("fallback", {})
    primary_ready = bool(primary.get("formal_ready", report.get("formal_ready", False)))
    fallback_ready = bool(fallback.get("formal_ready", False))
    require_fallback = bool(report.get("forced_failover", {}).get("required", False))
    return primary_ready, fallback_ready, require_fallback


def _normalize_workspace_relative_path(
    workspace_root: Path,
    raw_path: str,
) -> tuple[Path, str]:
    normalized = str(raw_path).replace("\\", "/").strip()
    if normalized.startswith("workspace/"):
        normalized = normalized[len("workspace/") :]
    absolute = (workspace_root / normalized).resolve()
    workspace_resolved = workspace_root.resolve()
    if workspace_resolved not in absolute.parents and absolute != workspace_resolved:
        raise ValueError(f"path escapes workspace: {raw_path}")
    relative = str(absolute.relative_to(workspace_resolved)).replace("\\", "/")
    return absolute, relative


def _input_message_from_text(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "input_text", "text": text}]}


def _function_call_output_item(call_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(result, ensure_ascii=False, sort_keys=True),
    }


def _state_hash(state: AgentState) -> str:
    payload = json.dumps(asdict(state), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{sha256(payload.encode('utf-8')).hexdigest()}"


def _normalize_turn(turn: Any) -> dict[str, Any]:
    if isinstance(turn, dict):
        raw_response = turn.get("raw_response", turn)
        standard_output_items = turn.get("standard_output_items")
        if standard_output_items is None:
            standard_output_items = raw_response.get("output", []) if isinstance(raw_response, dict) else []
        provider_output_items = turn.get("provider_output_items", standard_output_items)
        return {
            "provider": str(turn.get("provider", "unknown")),
            "model": str(turn.get("model", raw_response.get("model", ""))) if isinstance(raw_response, dict) else "",
            "raw_response": raw_response if isinstance(raw_response, dict) else {"value": raw_response},
            "standard_output_items": list(standard_output_items),
            "provider_output_items": list(provider_output_items),
            "usage": dict(turn.get("usage", {})),
        }
    return {
        "provider": str(getattr(turn, "provider", "unknown")),
        "model": str(getattr(turn, "model", "")),
        "raw_response": dict(getattr(turn, "raw_response", {})),
        "standard_output_items": list(getattr(turn, "standard_output_items", [])),
        "provider_output_items": list(getattr(turn, "provider_output_items", [])),
        "usage": dict(getattr(turn, "usage", {})),
    }


def _assistant_output(standard_output_items: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    texts: list[str] = []
    function_calls: list[dict[str, Any]] = []
    for item in standard_output_items:
        item_type = str(item.get("type", ""))
        if item_type == "function_call":
            function_calls.append(item)
            continue
        if item_type != "message":
            continue
        for chunk in item.get("content", []):
            if chunk.get("type") == "output_text":
                texts.append(str(chunk.get("text", "")))
    return texts, function_calls


def _task_spec_payload(config: V3Config, task: str) -> dict[str, Any]:
    spec = config.task_specs[task]
    return {
        "task_id": spec.task_id,
        "equation": spec.equation,
        "input_steps": spec.input_steps,
        "output_steps": spec.output_steps,
        "total_steps": spec.total_steps,
        "spatial_points": spec.spatial_points,
        "pred_shape": list(spec.pred_shape),
        "first_steps_must_match": spec.first_steps_must_match,
        "inference_time_limit_sec": spec.inference_time_limit_sec,
        "must_train_from_scratch": spec.must_train_from_scratch,
        "allow_public_pretrained_weights": spec.allow_public_pretrained_weights,
    }


def _api_contract_payload(config: V3Config, task: str) -> dict[str, Any]:
    return {
        "task_spec": _task_spec_payload(config, task),
        "tool_schemas": _tool_schemas(),
        "mode": "formal_single_task_v3",
    }


def _inspect_data_directory(data_dir: Path) -> dict[str, Any]:
    files = [path for path in sorted(data_dir.rglob("*")) if path.is_file()] if data_dir.exists() else []
    return {
        "file_count": len(files),
        "hdf5_files": [
            str(path.relative_to(data_dir.parent)).replace("\\", "/")
            for path in files
            if path.suffix.lower() in {".hdf5", ".h5"}
        ],
    }


def _build_context_text(
    *,
    packer: ContextPacker,
    config: V3Config,
    task: str,
    state: AgentState,
    decision: RouteDecision,
    provider_report: dict[str, Any],
    memory: MemoryStore,
) -> str:
    recent_decisions = memory.list_decision_records()[-5:]
    retrieved_memory = json.dumps(recent_decisions, ensure_ascii=False, sort_keys=True)
    compact_state = json.dumps(
        {
            "task": state.task,
            "run_id": state.run_id,
            "phase": state.current_phase,
            "shared_code_version": state.shared_code_version,
            "supported_tasks": state.supported_tasks,
            "latest_error_type": state.latest_error_type,
            "preflight_complete": state.preflight_complete,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    sections = packer.build(
        developer_contract=json.dumps(
            {
                "provider": provider_report.get("primary", {}).get("provider", "unknown"),
                "formal_ready": provider_report.get("formal_ready", False),
                "single_task_only": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        task_spec=json.dumps(_task_spec_payload(config, task), ensure_ascii=False, sort_keys=True),
        phase_tool_policy=json.dumps(
            {
                "phase": decision.phase,
                "profile": decision.profile,
                "allowed_tools": decision.allowed_tools,
                "deterministic_action": decision.deterministic_action,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        compact_state=compact_state,
        retrieved_memory=retrieved_memory,
        current_request=state.current_objective or f"Complete the formal run for {task}.",
    )
    return packer.render_text(sections)


def _sha256_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_code_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    manifest: dict[str, dict[str, Any]] = {}
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                manifest[str(entry["path"])] = dict(entry)
    return manifest


def _write_code_manifest(path: Path, manifest_by_path: dict[str, dict[str, Any]]) -> None:
    ordered = [manifest_by_path[key] for key in sorted(manifest_by_path)]
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")


def _backup_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "content": None}
    return {"exists": True, "content": path.read_text(encoding="utf-8")}


def _restore_file(path: Path, backup: dict[str, Any]) -> None:
    if not backup["exists"]:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(backup["content"]), encoding="utf-8")


def _tool_execution_result(
    *,
    name: str,
    arguments: dict[str, Any],
    workspace_root: Path,
    task: str,
) -> dict[str, Any]:
    if name == "read_file":
        absolute, relative = _normalize_workspace_relative_path(workspace_root, str(arguments["path"]))
        return {
            "ok": True,
            "path": relative,
            "content": absolute.read_text(encoding="utf-8"),
        }

    if name == "write_file":
        absolute, relative = _normalize_workspace_relative_path(workspace_root, str(arguments["path"]))
        absolute.parent.mkdir(parents=True, exist_ok=True)
        absolute.write_text(str(arguments["content"]), encoding="utf-8")
        return {
            "ok": True,
            "path": relative,
            "changed_files": [relative],
        }

    if name == "patch_file":
        absolute, relative = _normalize_workspace_relative_path(workspace_root, str(arguments["path"]))
        original = absolute.read_text(encoding="utf-8")
        old_text = str(arguments["old_text"])
        if old_text not in original:
            return {
                "ok": False,
                "path": relative,
                "error": "old_text not found in file",
            }
        updated = original.replace(old_text, str(arguments["new_text"]), 1)
        absolute.write_text(updated, encoding="utf-8")
        return {
            "ok": True,
            "path": relative,
            "changed_files": [relative],
        }

    if name == "run_python":
        interpreter = Path.cwd() / ".venv" / "bin" / "python"
        if not interpreter.exists():
            interpreter = Path(sys.executable)
        completed = subprocess.run(
            [str(interpreter), "-c", str(arguments["code"])],
            cwd=workspace_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    if name == "run_shell":
        completed = subprocess.run(
            shlex.split(str(arguments["command"])),
            cwd=workspace_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    if name == "inspect_hdf5":
        absolute, relative = _normalize_workspace_relative_path(workspace_root, str(arguments["path"]))
        import h5py

        datasets: list[dict[str, Any]] = []
        with h5py.File(absolute, "r") as handle:
            def collect(dataset_path: str, obj: object) -> None:
                if isinstance(obj, h5py.Dataset):
                    datasets.append({"path": dataset_path, "shape": list(obj.shape)})

            handle.visititems(collect)
        return {"ok": True, "path": relative, "datasets": datasets}

    if name == "validate_submission":
        return validate_submission_bundle_v3(
            workspace_root=workspace_root,
            tasks=[task],
            strict=bool(arguments.get("strict", False)),
        )

    if name == "package_submission":
        return package_submission_v3(
            submission_dir=workspace_root / "submission",
            tasks=[task],
            test_data_roots=[workspace_root / "data"],
            strict=bool(arguments.get("strict", False)),
        )

    return {"ok": False, "error": f"unknown tool {name}"}


def _finalize_gate_status(state: AgentState, final_text: str) -> dict[str, Any]:
    failures: list[str] = []
    if not state.shared_code_version:
        failures.append("shared code baseline missing")
    if not final_text.strip():
        failures.append("missing assistant final text")
    if state.latest_error_type:
        failures.append(state.latest_error_summary or state.latest_error_type)
    return {
        "overall_ok": not failures,
        "shared_code_ok": bool(state.shared_code_version),
        "failures": failures,
        "warnings": [],
    }


def run_formal_task_session(
    *,
    config: V3Config,
    task: str,
    provider_report_path: str | Path | None = None,
    responses_client: Any | None = None,
    memory_db_path: str | Path | None = None,
    max_steps: int = 6,
) -> dict[str, Any]:
    if "," in task:
        return {"ok": False, "error": "expected exactly one task id"}
    if task not in config.task_specs:
        return {"ok": False, "error": f"unknown task {task!r}"}

    workspace_root = config.workspace_root
    code_dir = workspace_root / "submission" / "code"
    llm_log_path = workspace_root / "llm_logs" / "all_llm_calls.jsonl"
    code_manifest_path = workspace_root / "submission" / "code_manifest.json"
    code_dir.mkdir(parents=True, exist_ok=True)
    llm_log_path.parent.mkdir(parents=True, exist_ok=True)
    (workspace_root / "internal_logs").mkdir(parents=True, exist_ok=True)

    run_id = f"run-{uuid.uuid4().hex[:12]}"
    memory = MemoryStore(memory_db_path or workspace_root / "internal_logs" / "v3_memory.db")
    memory.initialize()
    state = AgentState(
        task=task,
        run_id=run_id,
        current_objective=f"Complete the formal single-task run for {task}.",
    )

    latest_snapshot = memory.latest_code_snapshot()
    if latest_snapshot is not None:
        current_content_hash = snapshot_shared_code_directory(
            code_dir=code_dir,
            api_contract_payload=_api_contract_payload(config, task),
            parent_version=None,
            supported_tasks=list(latest_snapshot.get("supported_tasks", [])),
            task_support_matrix={},
            created_by_run_id=run_id,
            created_at=utc_timestamp(),
        ).content_hash
        state.supported_tasks = list(latest_snapshot.get("supported_tasks", []))
        if current_content_hash == latest_snapshot.get("content_hash"):
            state.shared_code_version = str(latest_snapshot["code_version"])

    resolved_provider_report_path = provider_report_path
    if resolved_provider_report_path is None:
        artifacts = getattr(config, "artifacts", None)
        if artifacts is not None and hasattr(artifacts, "provider_report_path"):
            resolved_provider_report_path = artifacts.provider_report_path
        else:
            resolved_provider_report_path = workspace_root / "provider_report.json"

    provider_report = _load_provider_report(resolved_provider_report_path)
    primary_ready, fallback_ready, require_fallback = _provider_ready_flags(provider_report)
    packer = ContextPacker()
    router = DeterministicRouter()
    pending_items: list[dict[str, Any]] = []
    manifest_by_path = _load_code_manifest(code_manifest_path)
    llm_steps = 0
    final_text = ""

    for _ in range(max_steps + 4):
        capability_ready = should_start_formal_run(
            primary_ready=primary_ready,
            fallback_ready=fallback_ready,
            require_fallback=require_fallback,
        )
        decision = router.choose(
            state,
            capability_ready=capability_ready,
            preflight_pending=not state.preflight_complete,
        )
        state.current_phase = decision.phase
        decision_created_at = utc_timestamp()

        if decision.phase == "capability_readiness":
            state.finalize_gate_status = {
                "overall_ok": False,
                "shared_code_ok": False,
                "failures": ["provider capability report is not formal-ready"],
                "warnings": [],
            }
            memory.record_decision(
                state_hash=_state_hash(state),
                reason_code=decision.reason_code,
                route=decision.deterministic_action,
                selected_profile=decision.profile,
                selected_phase=decision.phase,
                selected_tools=decision.allowed_tools,
                outcome="blocked",
                created_at=decision_created_at,
            )
            return {
                "ok": False,
                "error": "provider capability report is not formal-ready",
                "run_id": run_id,
                "memory_db_path": str(memory.db_path),
                "state": asdict(state),
            }

        if decision.phase == "preflight":
            state.data_inspection_summary = _inspect_data_directory(workspace_root / "data")
            state.preflight_complete = True
            memory.record_decision(
                state_hash=_state_hash(state),
                reason_code=decision.reason_code,
                route=decision.deterministic_action,
                selected_profile=decision.profile,
                selected_phase=decision.phase,
                selected_tools=decision.allowed_tools,
                outcome="completed",
                created_at=decision_created_at,
            )
            continue

        if decision.phase == "baseline_guard":
            created_at = utc_timestamp()
            snapshot = snapshot_shared_code_directory(
                code_dir=code_dir,
                api_contract_payload=_api_contract_payload(config, task),
                parent_version=state.shared_code_version,
                supported_tasks=sorted(set(state.supported_tasks + [task])),
                task_support_matrix={task: {"status": "baseline"}},
                created_by_run_id=run_id,
                created_at=created_at,
            )
            memory.record_code_snapshot(snapshot)
            state.shared_code_version = snapshot.code_version
            state.best_artifact_version = snapshot.code_version
            state.best_artifact_path = "submission/code"
            state.supported_tasks = snapshot.supported_tasks
            memory.record_decision(
                state_hash=_state_hash(state),
                reason_code=decision.reason_code,
                route=decision.deterministic_action,
                selected_profile=decision.profile,
                selected_phase=decision.phase,
                selected_tools=decision.allowed_tools,
                outcome="completed",
                created_at=decision_created_at,
            )
            continue

        if responses_client is None:
            provider_settings = getattr(config, "provider", None)
            if provider_settings is not None:
                from cozy_pde_v3.responses_client import ResponsesClient

                responses_client = ResponsesClient.from_config(config)

        if responses_client is None:
            state.finalize_gate_status = {
                "overall_ok": False,
                "shared_code_ok": bool(state.shared_code_version),
                "failures": ["responses client is required for formal loop execution"],
                "warnings": [],
            }
            memory.record_decision(
                state_hash=_state_hash(state),
                reason_code=decision.reason_code,
                route=decision.deterministic_action,
                selected_profile=decision.profile,
                selected_phase=decision.phase,
                selected_tools=decision.allowed_tools,
                outcome="blocked_no_client",
                created_at=decision_created_at,
            )
            return {
                "ok": False,
                "error": "responses client is required for formal loop execution",
                "run_id": run_id,
                "memory_db_path": str(memory.db_path),
                "state": asdict(state),
            }

        llm_steps += 1
        if llm_steps > max_steps:
            break

        prompt_text = _build_context_text(
            packer=packer,
            config=config,
            task=task,
            state=state,
            decision=decision,
            provider_report=provider_report,
            memory=memory,
        )
        step_id = f"{run_id}-step-{llm_steps}"
        input_items = [_input_message_from_text(prompt_text), *pending_items]
        pending_items = []
        start_time = time.monotonic()
        raw_turn = responses_client.create(
            input=input_items,
            tools=_tool_schemas_for_names(decision.allowed_tools),
            metadata={
                "task_id": task,
                "run_id": run_id,
                "step_id": step_id,
                "profile": decision.profile,
                "phase": decision.phase,
            },
            parallel_tool_calls=False,
        )
        elapsed_seconds = time.monotonic() - start_time
        turn = _normalize_turn(raw_turn)
        raw_response = turn["raw_response"]
        state.last_llm_call_id = str(raw_response.get("id") or step_id)
        append_jsonl(
            llm_log_path,
            build_llm_log_entry(
                elapsed_seconds=elapsed_seconds,
                provider=str(turn["provider"]),
                model=str(turn["model"]),
                profile=decision.profile,
                phase=decision.phase,
                raw_response=raw_response,
                standard_output_items=list(turn["standard_output_items"]),
                task_id=task,
                run_id=run_id,
                step_id=step_id,
            ),
        )

        texts, function_calls = _assistant_output(list(turn["standard_output_items"]))
        if function_calls:
            if len(function_calls) > 1:
                state.latest_error_type = "multi_tool_turn_disallowed"
                state.latest_error_summary = "assistant returned multiple function calls in one turn"
                pending_items = [
                    _input_message_from_text(
                        "Return at most one function call in the next turn. The previous turn contained multiple tool calls and none were executed."
                    )
                ]
                memory.record_decision(
                    state_hash=_state_hash(state),
                    reason_code=decision.reason_code,
                    route=decision.deterministic_action,
                    selected_profile=decision.profile,
                    selected_phase=decision.phase,
                    selected_tools=decision.allowed_tools,
                    outcome="multi_tool_rejected",
                    created_at=decision_created_at,
                )
                continue
            outcome = "tool_calls_executed"
            for item in function_calls:
                tool_name = str(item.get("name", ""))
                state.latest_tool_name = tool_name
                state.last_tool_call_id = str(item.get("call_id", ""))
                arguments = item.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                backup: dict[str, Any] | None = None
                target_path = ""
                if tool_name in {"write_file", "patch_file"}:
                    absolute_path, target_path = _normalize_workspace_relative_path(
                        workspace_root,
                        str(arguments["path"]),
                    )
                    if target_path.startswith("submission/code/"):
                        backup = _backup_file(absolute_path)

                result = _tool_execution_result(
                    name=tool_name,
                    arguments=dict(arguments),
                    workspace_root=workspace_root,
                    task=task,
                )

                if result.get("ok") and target_path.startswith("submission/code/"):
                    gate = patch_compatibility_gate(
                        supported_tasks=state.supported_tasks,
                        current_task=task,
                        validation=arguments.get("compatibility")
                        if isinstance(arguments.get("compatibility"), dict)
                        else None,
                    )
                    if not bool(gate["accepted"]):
                        if backup is not None:
                            _restore_file(absolute_path, backup)
                        result = {
                            "ok": False,
                            "path": target_path,
                            "error": gate["reason"],
                            "compatibility": gate["compatibility"],
                        }
                        state.latest_error_type = "compatibility_guard_failed"
                        state.latest_error_summary = str(gate["reason"])
                        outcome = "compatibility_rejected"
                    else:
                        created_at = utc_timestamp()
                        next_supported_tasks = sorted(set(state.supported_tasks + [task]))
                        snapshot = snapshot_shared_code_directory(
                            code_dir=code_dir,
                            api_contract_payload=_api_contract_payload(config, task),
                            parent_version=state.shared_code_version,
                            supported_tasks=next_supported_tasks,
                            task_support_matrix={task: {"status": "accepted_patch"}},
                            created_by_run_id=run_id,
                            created_at=created_at,
                        )
                        if snapshot.code_version != state.shared_code_version:
                            patch_id = f"patch-{state.last_tool_call_id or step_id}"
                            memory.record_patch(
                                CodePatchRecord(
                                    patch_id=patch_id,
                                    base_code_version=state.shared_code_version or "",
                                    new_code_version=snapshot.code_version,
                                    task_context=task,
                                    changed_files=list(result.get("changed_files", [])),
                                    change_intent=f"{tool_name} accepted for {task}",
                                    backward_compatibility_claim=str(gate["reason"]),
                                    affected_interfaces=list(result.get("changed_files", [])),
                                    llm_call_ids=[state.last_llm_call_id] if state.last_llm_call_id else [],
                                    validation_results=dict(gate),
                                )
                            )
                            memory.record_code_snapshot(snapshot)
                            state.shared_code_version = snapshot.code_version
                            state.best_artifact_version = snapshot.code_version
                            state.best_artifact_path = "submission/code"
                            state.supported_tasks = snapshot.supported_tasks
                            manifest_by_path[target_path] = {
                                "path": target_path,
                                "sha256": _sha256_file(workspace_root / target_path),
                                "size": (workspace_root / target_path).stat().st_size,
                                "code_version": snapshot.code_version,
                                "originating_task": task,
                                "patch_id": patch_id,
                                "step_id": patch_id,
                                "task_id": task,
                                "timestamp": created_at,
                                "llm_call_ids": [state.last_llm_call_id],
                            }
                            _write_code_manifest(code_manifest_path, manifest_by_path)
                        state.latest_error_type = None
                        state.latest_error_summary = None

                state.latest_tool_result_ok = bool(result.get("ok"))
                if not result.get("ok") and state.latest_error_type is None:
                    state.latest_error_type = "tool_execution_failed"
                    state.latest_error_summary = str(result.get("error", f"{tool_name} failed"))
                    outcome = "tool_failed"
                pending_items.append(_function_call_output_item(state.last_tool_call_id or "", result))

            memory.record_decision(
                state_hash=_state_hash(state),
                reason_code=decision.reason_code,
                route=decision.deterministic_action,
                selected_profile=decision.profile,
                selected_phase=decision.phase,
                selected_tools=decision.allowed_tools,
                outcome=outcome,
                created_at=decision_created_at,
            )
            continue

        if texts:
            final_text = "\n".join(text for text in texts if text).strip()
            state.finalize_gate_status = _finalize_gate_status(state, final_text)
            ok, message = should_allow_finalize(state.finalize_gate_status)
            memory.record_decision(
                state_hash=_state_hash(state),
                reason_code=decision.reason_code,
                route=decision.deterministic_action,
                selected_profile=decision.profile,
                selected_phase=decision.phase,
                selected_tools=decision.allowed_tools,
                outcome="finalized" if ok else "finalize_blocked",
                created_at=decision_created_at,
            )
            result: dict[str, Any] = {
                "ok": ok,
                "run_id": run_id,
                "final_text": final_text,
                "memory_db_path": str(memory.db_path),
                "llm_log_path": str(llm_log_path),
                "state": asdict(state),
                "decision_records": memory.list_decision_records(),
            }
            if not ok:
                result["error"] = message
            return result

        state.latest_error_type = "empty_response"
        state.latest_error_summary = "assistant produced no message text or function calls"
        memory.record_decision(
            state_hash=_state_hash(state),
            reason_code=decision.reason_code,
            route=decision.deterministic_action,
            selected_profile=decision.profile,
            selected_phase=decision.phase,
            selected_tools=decision.allowed_tools,
            outcome="empty_response",
            created_at=decision_created_at,
        )

    state.finalize_gate_status = {
        "overall_ok": False,
        "shared_code_ok": bool(state.shared_code_version),
        "failures": ["max steps exceeded"],
        "warnings": [],
    }
    return {
        "ok": False,
        "error": "max steps exceeded",
        "run_id": run_id,
        "memory_db_path": str(memory.db_path),
        "llm_log_path": str(llm_log_path),
        "state": asdict(state),
        "decision_records": memory.list_decision_records(),
    }
