from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cozy_pde_v3.config import V3Config, load_config
from cozy_pde_v3.provider_capabilities import default_provider_tool_schemas
from cozy_pde_v3.provider_capabilities import load_provider_capability_report
from cozy_pde_v3.provider_capabilities import verify_provider_capability_report
from cozy_pde_v3.provider_capabilities import write_provider_capability_report
from cozy_pde_v3.package import package_submission_v3
from cozy_pde_v3.research.cache import ResearchCache
from cozy_pde_v3.research.tools import fetch_url, parse_html, search_arxiv, search_github
from cozy_pde_v3.responses_client import ResponsesClient
from cozy_pde_v3.task_specs import TASK_IDS
from cozy_pde_v3.validation.submission import validate_submission_bundle_v3


def parse_task_id(value: str) -> str:
    if "," in value:
        raise argparse.ArgumentTypeError("expected exactly one task id")
    if value not in TASK_IDS:
        valid = ", ".join(TASK_IDS)
        raise argparse.ArgumentTypeError(f"unknown task {value!r}; expected one of: {valid}")
    return value


def _add_task_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True, type=parse_task_id)


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)


def _add_workspace_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace-root")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cozy-pde")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    _add_config_argument(run_parser)
    _add_workspace_argument(run_parser)
    _add_task_argument(run_parser)

    for command_name in ("check-provider", "check-research"):
        command_parser = subparsers.add_parser(command_name)
        _add_config_argument(command_parser)
        _add_workspace_argument(command_parser)

    for command_name in ("validate", "package", "status"):
        command_parser = subparsers.add_parser(command_name)
        _add_config_argument(command_parser)
        _add_workspace_argument(command_parser)
        _add_task_argument(command_parser)

    return parser


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def run_command(config: V3Config, task: str) -> int:
    report_path = config.artifacts.provider_report_path
    if not report_path.exists():
        _print_json({"ok": False, "error": "provider readiness report is missing"})
        return 1
    report = load_provider_capability_report(report_path)
    verification = verify_provider_capability_report(report, config=config)
    if not verification["ok"]:
        _print_json({"ok": False, "error": verification["error"]})
        return 1

    try:
        from cozy_pde_v3.agent_loop import run_formal_task_session
    except ImportError:
        _print_json({"ok": False, "error": "formal run path is not available"})
        return 1

    result = run_formal_task_session(config=config, task=task)
    _print_json(result)
    return 0 if result.get("ok") else 1


def check_provider_command(config: V3Config, task: str | None = None) -> int:
    del task
    tool_schemas = default_provider_tool_schemas()
    client = ResponsesClient.from_config(config)
    probe_result = client.probe_capabilities(
        tool_schemas=tool_schemas,
        proxy_log_dirs={
            "primary": config.proxy.primary_log_dir,
            "fallback": config.proxy.fallback_log_dir,
        },
    )
    now = datetime.now(timezone.utc)
    report = write_provider_capability_report(
        config.artifacts.provider_report_path,
        config_payload=config.payload(),
        tool_schemas=tool_schemas,
        proxy_payload=config.proxy.payload(),
        adapter_version=str(probe_result["adapter_version"]),
        sdk_version=str(probe_result["sdk_version"]),
        checked_at=now.isoformat().replace("+00:00", "Z"),
        expires_at=(now + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        forced_failover=dict(probe_result["forced_failover"]),
        primary=dict(probe_result["primary"]),
        fallback=dict(probe_result["fallback"]) if probe_result.get("fallback") is not None else None,
    )
    verification = verify_provider_capability_report(report, config=config, tool_schemas=tool_schemas)
    payload = {
        "ok": bool(verification["ok"]),
        "data": {
            "report_path": str(config.artifacts.provider_report_path),
            "formal_ready": bool(report["formal_ready"]),
            "primary": report["primary"],
            "fallback": report.get("fallback"),
            "forced_failover": report["forced_failover"],
        },
    }
    if not verification["ok"]:
        payload["error"] = verification["error"]
    _print_json(payload)
    return 0 if verification["ok"] else 1


def check_research_command(
    config: V3Config,
    task: str | None = None,
    *,
    http_client=None,
) -> int:
    del task
    cache = ResearchCache(config.research.cache_dir)
    cache.initialize()

    if not config.research.allow_network:
        payload = {
            "ok": True,
            "data": {
                "degraded": True,
                "runnable": True,
                "network_available": False,
                "cache_index_path": str(cache.index_path),
            },
        }
        _print_json(payload)
        return 0

    arxiv_results = search_arxiv("neural operator", http_client=http_client)
    github_results = search_github("neural operator", http_client=http_client)
    parsed_repo = {}
    if github_results:
        fetched = fetch_url(github_results[0]["url"], http_client=http_client)
        parsed_repo = parse_html(str(fetched["text"]))
    for item in arxiv_results + github_results:
        cache.append_source(item)

    payload = {
        "ok": True,
        "data": {
            "degraded": False,
            "runnable": True,
            "network_available": True,
            "cache_index_path": str(cache.index_path),
            "arxiv_results": len(arxiv_results),
            "github_results": len(github_results),
            "parsed_repo_title": parsed_repo.get("title", ""),
        },
    }
    _print_json(payload)
    return 0


def validate_command(config: V3Config, task: str) -> int:
    result = validate_submission_bundle_v3(
        workspace_root=config.workspace_root,
        tasks=[task],
        strict=True,
    )
    _print_json(result)
    return 0 if result["ok"] else 1


def package_command(config: V3Config, task: str) -> int:
    result = package_submission_v3(
        submission_dir=config.workspace_root / "submission",
        tasks=[task],
        test_data_roots=[config.workspace_root / "data"],
        strict=True,
    )
    _print_json(result)
    return 0 if result["ok"] else 1


def status_command(config: V3Config, task: str) -> int:
    def _validation_fallback_payload() -> dict[str, Any]:
        result = validate_submission_bundle_v3(
            workspace_root=config.workspace_root,
            tasks=[task],
            strict=True,
        )
        finalize_gate = result.get("data", {}).get("finalize_gate", {})
        return {
            "overall_ok": bool(finalize_gate.get("overall_ok", False)),
            "failures": list(finalize_gate.get("failures", [])),
            "warnings": list(finalize_gate.get("warnings", [])),
        }

    try:
        from cozy_pde_v3.status import collect_submission_status_v3
    except ImportError:
        payload = _validation_fallback_payload()
        _print_json(payload)
        return 0 if payload["overall_ok"] else 1

    payload = collect_submission_status_v3(workspace_root=config.workspace_root, tasks=[task])
    if not payload.get("finalize_gate"):
        payload = _validation_fallback_payload()
    _print_json(payload)
    overall_ok = bool(payload.get("overall_ok", payload.get("finalize_gate", {}).get("overall_ok", False)))
    missing_gates = payload.get("missing_gates")
    if not isinstance(missing_gates, list):
        missing_gates = []
    return 0 if overall_ok and not missing_gates else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace_root = getattr(args, "workspace_root", None)
    if workspace_root is None:
        config = load_config(args.config)
    else:
        config = load_config(args.config, workspace_root=workspace_root)
    if args.command == "run":
        return run_command(config, args.task)
    if args.command == "check-provider":
        return check_provider_command(config)
    if args.command == "check-research":
        return check_research_command(config)
    if args.command == "validate":
        return validate_command(config, args.task)
    if args.command == "package":
        return package_command(config, args.task)
    if args.command == "status":
        return status_command(config, args.task)
    return 1
