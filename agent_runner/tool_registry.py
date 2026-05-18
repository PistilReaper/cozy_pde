from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import RunnerConfig
from .logger import ToolCallLogger
from .research_cache import ResearchCache
from .safety import WorkspaceSafety
from .responses_items import ResponsesFunctionCall
from .tools import failure
from .tools.document_tools import generate_methodology_pdf
from .tools.fs_tools import list_files, read_file, write_file
from .tools.hdf5_tools import inspect_hdf5
from .tools.log_tools import analyze_log
from .tools.package_tools import package_submission
from .tools.python_tools import run_python
from .tools.research_tools import fetch_pdf, fetch_url, parse_html, parse_pdf, search_arxiv, search_github
from .tools.shell_tools import run_shell
from .tools.snapshot_tools import rollback, snapshot
from .tools.validate_tools import validate_jsonl_logs, validate_responses_logs, validate_submission


def _schema_type_matches(value: Any, expected: str, item_schema: dict[str, Any] | None = None) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        if not isinstance(value, list):
            return False
        if item_schema and item_schema.get("type") == "string":
            return all(isinstance(item, str) for item in value)
        return True
    return True


def _validate_arguments_against_schema(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    if not isinstance(arguments, dict):
        return "tool arguments must be a JSON object"

    properties = schema.get("properties", {})
    required = schema.get("required", [])
    for name in required:
        if name not in arguments:
            return f"missing required argument {name}"
    for name in arguments:
        if name not in properties:
            return f"unexpected argument {name}"
    for name, value in arguments.items():
        property_schema = properties.get(name, {})
        expected_type = property_schema.get("type")
        if isinstance(expected_type, str) and not _schema_type_matches(
            value,
            expected_type,
            property_schema.get("items") if isinstance(property_schema.get("items"), dict) else None,
        ):
            return f"argument {name} must be of type {expected_type}"
    return None


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict[str, Any]]
    allowed_phases: set[str] | None = None


class ToolRegistry:
    def __init__(self, tools: list[ToolDefinition], logger: ToolCallLogger) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._logger = logger
        self._context: dict[str, Any] = {}

    def set_context(
        self,
        *,
        task_id: str | None = None,
        step_id: str | None = None,
        mode: str | None = None,
        phase: str | None = None,
        exposed_tool_names: set[str] | None = None,
    ) -> None:
        if task_id is not None:
            self._context["task_id"] = task_id
        if step_id is not None:
            self._context["step_id"] = step_id
        if mode is not None:
            self._context["mode"] = mode
        if phase is not None:
            self._context["phase"] = phase
        if exposed_tool_names is not None:
            self._context["exposed_tool_names"] = set(exposed_tool_names)

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def schemas(self) -> list[dict[str, Any]]:
        return self.response_function_tools()

    def response_function_tools(self) -> list[dict[str, Any]]:
        active_phase = self._context.get("phase")
        exposed_tool_names = self._context.get("exposed_tool_names")
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": True,
            }
            for tool in self._tools.values()
            if (
                (tool.allowed_phases is None or active_phase is None or active_phase in tool.allowed_phases)
                and (exposed_tool_names is None or tool.name in exposed_tool_names)
            )
        ]

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in self._tools:
            result = failure(tool_name, f"Unknown tool: {tool_name}")
            self._logger.log_call(tool_name=tool_name, elapsed_seconds=0.0, arguments=arguments, result=result)
            return result

        tool = self._tools[tool_name]
        active_phase = self._context.get("phase")
        exposed_tool_names = self._context.get("exposed_tool_names")
        if tool.allowed_phases is not None and active_phase is not None and active_phase not in tool.allowed_phases:
            result = failure(
                tool_name,
                f"Tool {tool_name} is not available in phase {active_phase}",
                phase=active_phase,
            )
            self._logger.log_call(tool_name=tool_name, elapsed_seconds=0.0, arguments=arguments, result=result)
            return result
        if exposed_tool_names is not None and tool_name not in exposed_tool_names:
            result = failure(
                tool_name,
                f"Tool {tool_name} is not exposed in the current turn",
                phase=active_phase,
            )
            self._logger.log_call(tool_name=tool_name, elapsed_seconds=0.0, arguments=arguments, result=result)
            return result
        schema_error = _validate_arguments_against_schema(tool.parameters, arguments)
        if schema_error is not None:
            result = failure(tool_name, f"Schema validation failed: {schema_error}")
            self._logger.log_call(tool_name=tool_name, elapsed_seconds=0.0, arguments=arguments, result=result)
            return result
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
    research_cache = ResearchCache(config.research)
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

    def validate_responses_logs_tool(path: str, workspace_root: str | None = None) -> dict[str, Any]:
        check = safety.validate_read_path(path)
        if not check.ok:
            return failure("validate_responses_logs", check.error or "read check failed", path=path)
        assert check.resolved_path is not None
        target_workspace = workspace_root or str(config.workspace_root)
        return validate_responses_logs(check.resolved_path, workspace_root=target_workspace)

    def parse_pdf_tool(path: str) -> dict[str, Any]:
        check = safety.validate_read_path(path)
        if not check.ok:
            return failure("parse_pdf", check.error or "read check failed", path=path)
        assert check.resolved_path is not None
        return parse_pdf(path=str(check.resolved_path))

    def parse_html_tool(path_or_url: str) -> dict[str, Any]:
        if path_or_url.startswith(("http://", "https://")):
            return parse_html(path_or_url=path_or_url, research=config.research)
        check = safety.validate_read_path(path_or_url)
        if not check.ok:
            return failure("parse_html", check.error or "read check failed", path_or_url=path_or_url)
        assert check.resolved_path is not None
        return parse_html(path_or_url=str(check.resolved_path), research=config.research)

    def research_cache_write_tool(record: dict[str, Any]) -> dict[str, Any]:
        cached_record = research_cache.write(record)
        return success(
            "research_cache_write",
            f"Cached research source {cached_record['source_id']}",
            record=cached_record,
            cache_path=str(research_cache.path),
        )

    def validate_submission_tool(
        submission_dir: str = "submission",
        test_hdf5: str | None = None,
        pred_filename: str = "pred.hdf5",
        time_filename: str = "time.csv",
        logs_filename: str = "logs.log",
        code_dir: str | None = None,
        rehearsal_mode: bool = rehearsal_validation,
        expected_total_steps: int = 200,
        expected_spatial_points: int = 256,
        input_steps: int = 10,
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
            expected_total_steps=expected_total_steps,
            expected_spatial_points=expected_spatial_points,
            input_steps=input_steps,
        )

    def package_submission_tool(submission_dir: str = "submission") -> dict[str, Any]:
        dir_check = safety.validate_read_path(submission_dir)
        if not dir_check.ok:
            return failure("package_submission", dir_check.error or "submission path rejected", submission_dir=submission_dir)
        assert dir_check.resolved_path is not None
        return package_submission(
            submission_dir=dir_check.resolved_path,
            workspace_root=config.workspace_root,
            task_configs=config.submission_task_list,
            code_dir=config.submission_code_dir,
        )

    def validate_full_submission_tool(
        submission_dir: str = "submission",
        test_hdf5: str | None = None,
        responses_log_path: str = "llm_logs/all_llm_calls.jsonl",
        workspace_root: str | None = None,
        pred_filename: str = "pred.hdf5",
        time_filename: str = "time.csv",
        logs_filename: str = "logs.log",
        code_dir: str | None = None,
        rehearsal_mode: bool = rehearsal_validation,
        expected_total_steps: int = 200,
        expected_spatial_points: int = 256,
        input_steps: int = 10,
    ) -> dict[str, Any]:
        submission_result = validate_submission_tool(
            submission_dir=submission_dir,
            test_hdf5=test_hdf5,
            pred_filename=pred_filename,
            time_filename=time_filename,
            logs_filename=logs_filename,
            code_dir=code_dir,
            rehearsal_mode=rehearsal_mode,
            expected_total_steps=expected_total_steps,
            expected_spatial_points=expected_spatial_points,
            input_steps=input_steps,
        )
        responses_result = validate_responses_logs_tool(
            path=responses_log_path,
            workspace_root=workspace_root or str(config.workspace_root),
        )
        if submission_result["ok"] and responses_result["ok"]:
            return success(
                "validate_full_submission",
                "Validated submission bundle and responses provenance.",
                submission=submission_result,
                responses_logs=responses_result,
            )
        return failure(
            "validate_full_submission",
            "Validation failed for submission bundle or responses provenance.",
            submission=submission_result,
            responses_logs=responses_result,
        )

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
            allowed_phases={"research", "planning", "implementation", "debugging"},
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
            allowed_phases={"implementation", "debugging"},
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
            allowed_phases={"implementation", "debugging"},
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
            allowed_phases={"implementation", "debugging"},
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
            allowed_phases={"validation", "finalization"},
        ),
        ToolDefinition(
            name="validate_responses_logs",
            description="Validate that a Responses JSONL log file is traceable to generated submission/code files.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "workspace_root": {"type": "string"},
                },
                "required": ["path"],
            },
            handler=validate_responses_logs_tool,
            allowed_phases={"validation", "finalization"},
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
                    "expected_total_steps": {"type": "integer", "default": 200},
                    "expected_spatial_points": {"type": "integer", "default": 256},
                    "input_steps": {"type": "integer", "default": 10},
                },
            },
            handler=validate_submission_tool,
            allowed_phases={"validation", "finalization"},
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
            allowed_phases={"implementation", "debugging", "log_analysis"},
        ),
        ToolDefinition(
            name="snapshot",
            description="Snapshot the current workspace/submission state into workspace/runs/snapshots.",
            parameters={"type": "object", "properties": {"label": {"type": "string"}}},
            handler=lambda label=None: snapshot(config=config, safety=safety, label=label),
            allowed_phases={"implementation", "debugging"},
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
            allowed_phases={"implementation", "debugging"},
        ),
        ToolDefinition(
            name="package_submission",
            description="Validate bundle files, create manifest.json, and build submission.zip under workspace/submission.",
            parameters={
                "type": "object",
                "properties": {"submission_dir": {"type": "string", "default": "submission"}},
            },
            handler=package_submission_tool,
            allowed_phases={"validation", "finalization"},
        ),
        ToolDefinition(
            name="generate_methodology_pdf",
            description="Generate a simple local methodology.pdf from markdown or plain text.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "path": {"type": "string", "default": "submission/methodology.pdf"},
                },
                "required": ["content"],
            },
            handler=lambda content, path="submission/methodology.pdf": generate_methodology_pdf(
                content=content,
                path=path,
                safety=safety,
            ),
            allowed_phases={"validation", "finalization"},
        ),
        ToolDefinition(
            name="validate_full_submission",
            description="Run validate_submission and validate_responses_logs as one deterministic validation tool.",
            parameters={
                "type": "object",
                "properties": {
                    "submission_dir": {"type": "string", "default": "submission"},
                    "test_hdf5": {"type": "string"},
                    "responses_log_path": {"type": "string", "default": "llm_logs/all_llm_calls.jsonl"},
                    "workspace_root": {"type": "string"},
                    "pred_filename": {"type": "string", "default": "pred.hdf5"},
                    "time_filename": {"type": "string", "default": "time.csv"},
                    "logs_filename": {"type": "string", "default": "logs.log"},
                    "code_dir": {"type": "string"},
                    "rehearsal_mode": {"type": "boolean", "default": rehearsal_validation},
                    "expected_total_steps": {"type": "integer", "default": 200},
                    "expected_spatial_points": {"type": "integer", "default": 256},
                    "input_steps": {"type": "integer", "default": 10},
                },
            },
            handler=validate_full_submission_tool,
            allowed_phases={"validation", "finalization"},
        ),
        ToolDefinition(
            name="search_arxiv",
            description="Search the official arXiv API and return normalized metadata records without downloading PDFs.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 10},
                    "sort_by": {"type": "string", "default": "relevance"},
                },
                "required": ["query"],
            },
            handler=lambda query, max_results=10, sort_by="relevance": search_arxiv(
                query=query,
                max_results=max_results,
                sort_by=sort_by,
                research=config.research,
            ),
            allowed_phases={"research", "planning"},
        ),
        ToolDefinition(
            name="search_github",
            description="Search GitHub repositories or code via the official GitHub Search API.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "kind": {"type": "string", "default": "repositories"},
                    "max_results": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
            handler=lambda query, kind="repositories", max_results=10: search_github(
                query=query,
                kind=kind,
                max_results=max_results,
                research=config.research,
            ),
            allowed_phases={"research", "planning"},
        ),
        ToolDefinition(
            name="fetch_url",
            description="Fetch a safe text URL from an allowed domain with policy checks and local caching.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "purpose": {"type": "string", "default": "read_code_or_paper"},
                },
                "required": ["url"],
            },
            handler=lambda url, purpose="read_code_or_paper": fetch_url(url=url, purpose=purpose, research=config.research),
            allowed_phases={"research", "planning"},
        ),
        ToolDefinition(
            name="fetch_pdf",
            description="Fetch a safe PDF from an allowed domain and store it under workspace/research/papers.",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            handler=lambda url: fetch_pdf(url=url, research=config.research),
            allowed_phases={"research", "planning"},
        ),
        ToolDefinition(
            name="parse_pdf",
            description="Extract text from a locally stored PDF using a local parser when available.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            handler=parse_pdf_tool,
            allowed_phases={"research", "planning"},
        ),
        ToolDefinition(
            name="parse_html",
            description="Extract title, text, code blocks, and links from HTML content or a safe allowed URL.",
            parameters={
                "type": "object",
                "properties": {"path_or_url": {"type": "string"}},
                "required": ["path_or_url"],
            },
            handler=parse_html_tool,
            allowed_phases={"research", "planning"},
        ),
        ToolDefinition(
            name="research_cache_write",
            description="Write a normalized research record into workspace/research/cache/research_sources.jsonl.",
            parameters={
                "type": "object",
                "properties": {"record": {"type": "object"}},
                "required": ["record"],
            },
            handler=research_cache_write_tool,
            allowed_phases={"research", "planning"},
        ),
        ToolDefinition(
            name="research_cache_read",
            description="Read one or more cached research records by source_id or URL.",
            parameters={
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "url": {"type": "string"},
                },
            },
            handler=lambda source_id=None, url=None: success(
                "research_cache_read",
                "Loaded cached research records",
                records=research_cache.read(source_id=source_id, url=url),
                cache_path=str(research_cache.path),
            ),
            allowed_phases={"research", "planning", "log_analysis"},
        ),
        ToolDefinition(
            name="research_cache_search",
            description="Search cached research records by keyword.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
            handler=lambda query, max_results=10: success(
                "research_cache_search",
                f"Found cached research matches for {query!r}",
                records=research_cache.search(query, max_results=max_results),
                cache_path=str(research_cache.path),
            ),
            allowed_phases={"research", "planning"},
        ),
    ]
    if extra_tools:
        tools.extend(extra_tools)
    return ToolRegistry(tools, logger)
