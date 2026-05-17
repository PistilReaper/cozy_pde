from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import RunnerConfig
from .logger import ToolCallLogger
from .safety import WorkspaceSafety
from .responses_items import ResponsesFunctionCall
from .tools import failure
from .tools.fs_tools import list_files, read_file, write_file
from .tools.hdf5_tools import inspect_hdf5
from .tools.log_tools import analyze_log
from .tools.package_tools import package_submission
from .tools.python_tools import run_python
from .tools.shell_tools import run_shell
from .tools.snapshot_tools import rollback, snapshot
from .tools.validate_tools import validate_jsonl_logs, validate_submission


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict[str, Any]]


class ToolRegistry:
    def __init__(self, tools: list[ToolDefinition], logger: ToolCallLogger) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._logger = logger
        self._context: dict[str, Any] = {}

    def set_context(self, *, task_id: str | None = None, step_id: str | None = None, mode: str | None = None) -> None:
        if task_id is not None:
            self._context["task_id"] = task_id
        if step_id is not None:
            self._context["step_id"] = step_id
        if mode is not None:
            self._context["mode"] = mode

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def schemas(self) -> list[dict[str, Any]]:
        return self.response_function_tools()

    def response_function_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": True,
            }
            for tool in self._tools.values()
        ]

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in self._tools:
            result = failure(tool_name, f"Unknown tool: {tool_name}")
            self._logger.log_call(tool_name=tool_name, elapsed_seconds=0.0, arguments=arguments, result=result)
            return result

        tool = self._tools[tool_name]
        started = time.perf_counter()
        try:
            signature = inspect.signature(tool.handler)
            if "runner_context" in signature.parameters:
                result = tool.handler(**arguments, runner_context=dict(self._context))
            else:
                result = tool.handler(**arguments)
        except Exception as exc:  # noqa: BLE001
            result = failure(tool_name, f"{type(exc).__name__}: {exc}")
        elapsed = time.perf_counter() - started
        self._logger.log_call(tool_name=tool_name, elapsed_seconds=elapsed, arguments=arguments, result=result)
        return result

    def execute_response_function_call(self, call: ResponsesFunctionCall) -> dict[str, Any]:
        return self.execute(call.name, call.arguments)


def build_tool_registry(
    config: RunnerConfig,
    logger: ToolCallLogger,
    *,
    allow_run_shell: bool = True,
    allow_submission_writes: bool = True,
    run_shell_profile: str = "default",
    rehearsal_validation: bool = False,
    extra_read_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
    extra_tools: list[ToolDefinition] | None = None,
) -> ToolRegistry:
    allowed_write_roots = (
        [
            config.workspace_root / "submission",
            config.workspace_root / "submission" / "code",
            config.workspace_root / "runs",
        ]
        if allow_submission_writes
        else [config.workspace_root / "runs"]
    )
    safety = WorkspaceSafety(
        config.workspace_root,
        allowed_write_roots=allowed_write_roots,
        extra_read_roots=extra_read_roots,
    )

    def validate_jsonl_logs_tool(path: str) -> dict[str, Any]:
        check = safety.validate_read_path(path)
        if not check.ok:
            return failure("validate_jsonl_logs", check.error or "read check failed", path=path)
        assert check.resolved_path is not None
        return validate_jsonl_logs(check.resolved_path)

    def validate_submission_tool(
        submission_dir: str = "submission",
        test_hdf5: str | None = None,
        pred_filename: str = "pred.hdf5",
        time_filename: str = "time.csv",
        logs_filename: str = "logs.log",
        code_dir: str | None = None,
        rehearsal_mode: bool = rehearsal_validation,
    ) -> dict[str, Any]:
        dir_check = safety.validate_read_path(submission_dir)
        if not dir_check.ok:
            return failure("validate_submission", dir_check.error or "submission path rejected", submission_dir=submission_dir)
        assert dir_check.resolved_path is not None
        test_path: str | Path | None = None
        if test_hdf5 is not None:
            test_check = safety.validate_read_path(test_hdf5)
            if not test_check.ok:
                return failure("validate_submission", test_check.error or "test HDF5 rejected", test_hdf5=test_hdf5)
            assert test_check.resolved_path is not None
            test_path = test_check.resolved_path
        code_path: str | Path | None = None
        if code_dir is not None:
            code_check = safety.validate_read_path(code_dir)
            if not code_check.ok:
                return failure("validate_submission", code_check.error or "code_dir rejected", code_dir=code_dir)
            assert code_check.resolved_path is not None
            code_path = code_check.resolved_path
        return validate_submission(
            submission_dir=dir_check.resolved_path,
            test_hdf5=test_path,
            pred_filename=pred_filename,
            time_filename=time_filename,
            logs_filename=logs_filename,
            code_dir=code_path,
            rehearsal_mode=rehearsal_mode,
        )

    def package_submission_tool(submission_dir: str = "submission", test_hdf5: str | None = None) -> dict[str, Any]:
        dir_check = safety.validate_read_path(submission_dir)
        if not dir_check.ok:
            return failure("package_submission", dir_check.error or "submission path rejected", submission_dir=submission_dir)
        assert dir_check.resolved_path is not None
        test_path: str | Path | None = None
        if test_hdf5 is not None:
            test_check = safety.validate_read_path(test_hdf5)
            if not test_check.ok:
                return failure("package_submission", test_check.error or "test HDF5 rejected", test_hdf5=test_hdf5)
            assert test_check.resolved_path is not None
            test_path = test_check.resolved_path
        return package_submission(submission_dir=dir_check.resolved_path, test_hdf5=test_path)

    tools = [
        ToolDefinition(
            name="read_file",
            description="Read a UTF-8 text file inside workspace.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}, "max_chars": {"type": "integer", "default": 20000}},
                "required": ["path"],
            },
            handler=lambda path, max_chars=20000: (
                failure("read_file", "Rehearsal mode does not read prior submission log files", path=path)
                if rehearsal_validation and str(path).startswith("submission/") and str(path).lower().endswith((".log", ".jsonl"))
                else read_file(path=path, max_chars=max_chars, safety=safety)
            ),
        ),
        ToolDefinition(
            name="write_file",
            description="Write a file under workspace/submission, workspace/submission/code, or workspace/runs.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
            handler=lambda path, content, runner_context=None: write_file(
                path=path,
                content=content,
                safety=safety,
                runner_context=runner_context,
            ),
        ),
        ToolDefinition(
            name="list_files",
            description="List files under a workspace directory.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean", "default": False},
                    "max_entries": {"type": "integer", "default": 200},
                },
                "required": ["path"],
            },
            handler=lambda path, recursive=False, max_entries=200: list_files(
                path=path,
                recursive=recursive,
                max_entries=max_entries,
                safety=safety,
            ),
        ),
        ToolDefinition(
            name="run_shell",
            description="Run a shell command inside workspace with timeout and stdout/stderr capture.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string", "default": "."},
                    "timeout_seconds": {"type": "integer"},
                    "profile": {"type": "string", "default": run_shell_profile},
                },
                "required": ["command"],
            },
            handler=(
                lambda command, cwd=".", timeout_seconds=None, profile=run_shell_profile: run_shell(
                    command=command,
                    cwd=cwd,
                    timeout_seconds=timeout_seconds,
                    profile=profile,
                    safety=safety,
                    config=config,
                )
                if allow_run_shell
                else failure("run_shell", "run_shell is disabled in this mode", command=command)
            ),
        ),
        ToolDefinition(
            name="run_python",
            description="Run a short scratch Python snippet under workspace/runs/scratch.",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}, "timeout_seconds": {"type": "integer", "default": 120}},
                "required": ["code"],
            },
            handler=lambda code, timeout_seconds=120: run_python(code=code, timeout_seconds=timeout_seconds, safety=safety, config=config),
        ),
        ToolDefinition(
            name="inspect_hdf5",
            description="Inspect an HDF5 file inside workspace and report keys, shapes, dtypes, and numeric statistics.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            handler=lambda path: inspect_hdf5(path=path, safety=safety),
        ),
        ToolDefinition(
            name="validate_jsonl_logs",
            description="Validate that a JSONL log file contains ISO timestamps, elapsed_seconds, and response or tool_calls.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            handler=validate_jsonl_logs_tool,
        ),
        ToolDefinition(
            name="validate_submission",
            description="Validate prediction files, time.csv, logs.log, and generated code under workspace/submission.",
            parameters={
                "type": "object",
                "properties": {
                    "submission_dir": {"type": "string", "default": "submission"},
                    "test_hdf5": {"type": "string"},
                    "pred_filename": {"type": "string", "default": "pred.hdf5"},
                    "time_filename": {"type": "string", "default": "time.csv"},
                    "logs_filename": {"type": "string", "default": "logs.log"},
                    "code_dir": {"type": "string"},
                    "rehearsal_mode": {"type": "boolean", "default": rehearsal_validation},
                },
            },
            handler=validate_submission_tool,
        ),
        ToolDefinition(
            name="analyze_log",
            description="Analyze a training or inference log for loss trends, NaN, and OOM signals.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            handler=lambda path: analyze_log(path=path, safety=safety),
        ),
        ToolDefinition(
            name="snapshot",
            description="Snapshot the current workspace/submission state into workspace/runs/snapshots.",
            parameters={"type": "object", "properties": {"label": {"type": "string"}}},
            handler=lambda label=None: snapshot(config=config, safety=safety, label=label),
        ),
        ToolDefinition(
            name="rollback",
            description="Restore workspace/submission files from a previous snapshot path.",
            parameters={
                "type": "object",
                "properties": {"snapshot_path": {"type": "string"}},
                "required": ["snapshot_path"],
            },
            handler=lambda snapshot_path: rollback(config=config, safety=safety, snapshot_path=snapshot_path),
        ),
        ToolDefinition(
            name="package_submission",
            description="Validate bundle files, create manifest.json, and build submission.zip under workspace/submission.",
            parameters={
                "type": "object",
                "properties": {"submission_dir": {"type": "string", "default": "submission"}, "test_hdf5": {"type": "string"}},
            },
            handler=package_submission_tool,
        ),
    ]
    if extra_tools:
        tools.extend(extra_tools)
    return ToolRegistry(tools, logger)
