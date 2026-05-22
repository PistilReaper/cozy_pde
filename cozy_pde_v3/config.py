from __future__ import annotations

import os
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from textwrap import dedent
from typing import Any

import yaml

from cozy_pde_v3.task_specs import DEFAULT_TASK_SPECS, TASK_IDS, TaskSpec

_RESPONSES_WIRE_API = "responses"


def _resolve_path(root: Path, value: str | Path | None, *, default: str) -> Path:
    raw = Path(default if value is None else value)
    if raw.is_absolute():
        return raw.resolve()
    return (root / raw).resolve()


def _resolve_configured_path(
    *,
    config_root: Path,
    workspace_root: Path,
    value: str | Path | None,
    default: str,
) -> Path:
    if value is None:
        return _resolve_path(workspace_root, None, default=default)
    return _resolve_path(config_root, value, default=default)


def _required_str(data: dict[str, Any], key: str) -> str:
    value = str(data.get(key, "")).strip()
    if not value:
        raise ValueError(f"missing required config field: {key}")
    return value


def _reject_legacy_wire_api(value: str, *, field_name: str) -> str:
    wire_api = str(value).strip()
    if wire_api != _RESPONSES_WIRE_API:
        raise ValueError(f"{field_name} must be {_RESPONSES_WIRE_API!r}, got {wire_api!r}")
    return wire_api


@dataclass(frozen=True)
class DataRoots:
    primary: Path
    evaluation: list[Path] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        return {
            "primary": str(self.primary),
            "evaluation": [str(path) for path in self.evaluation],
        }


@dataclass(frozen=True)
class ProviderEndpoint:
    provider: str
    base_url: str
    api_key_env: str
    api_key: str | None
    model_id: str
    append_v1: bool = True
    supports_prompt_cache_hints: bool = True

    def payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "model_id": self.model_id,
            "append_v1": self.append_v1,
            "supports_prompt_cache_hints": self.supports_prompt_cache_hints,
        }


@dataclass(frozen=True)
class ProviderSettings:
    wire_api: str
    primary: ProviderEndpoint
    fallback: ProviderEndpoint | None = None
    require_fallback: bool = False

    def payload(self) -> dict[str, Any]:
        payload = {
            "wire_api": self.wire_api,
            "primary": self.primary.payload(),
            "require_fallback": self.require_fallback,
        }
        if self.fallback is not None:
            payload["fallback"] = self.fallback.payload()
        return payload


@dataclass(frozen=True)
class ProxySettings:
    enabled: bool = False
    primary_log_dir: Path = Path("proxy_logs/primary")
    fallback_log_dir: Path = Path("proxy_logs/fallback")
    proxy_version: str = "unknown"

    def payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "primary_log_dir": str(self.primary_log_dir),
            "fallback_log_dir": str(self.fallback_log_dir),
            "proxy_version": self.proxy_version,
        }


@dataclass(frozen=True)
class BudgetSettings:
    max_total_usd: float = 0.0
    max_steps: int = 0

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimeoutSettings:
    provider_seconds: int = 60
    formal_run_seconds: int = 3600

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchSettings:
    enabled: bool = True
    allow_network: bool = False
    cache_dir: Path = Path("research/cache")
    cache_index_path: Path = Path("research/cache/research_sources.jsonl")
    raw_cache_dir: Path = Path("research/cache/raw")
    papers_dir: Path = Path("research/papers")

    def payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "allow_network": self.allow_network,
            "cache_dir": str(self.cache_dir),
            "cache_index_path": str(self.cache_index_path),
            "raw_cache_dir": str(self.raw_cache_dir),
            "papers_dir": str(self.papers_dir),
        }


@dataclass(frozen=True)
class ArtifactSettings:
    provider_report_path: Path
    package_output_path: Path
    validation_report_path: Path
    finalize_gate_path: Path
    state_path: Path

    def payload(self) -> dict[str, Any]:
        return {
            "provider_report_path": str(self.provider_report_path),
            "package_output_path": str(self.package_output_path),
            "validation_report_path": str(self.validation_report_path),
            "finalize_gate_path": str(self.finalize_gate_path),
            "state_path": str(self.state_path),
        }


@dataclass(frozen=True)
class TaskPolicySettings:
    task_ids: list[str] = field(default_factory=lambda: list(TASK_IDS))
    strict_validation: bool = True

    def payload(self) -> dict[str, Any]:
        return {
            "task_ids": list(self.task_ids),
            "strict_validation": self.strict_validation,
        }


@dataclass(frozen=True)
class V3Config:
    config_path: Path
    workspace_root: Path
    submission_dir: Path
    shared_code_dir: Path
    data_roots: DataRoots
    provider: ProviderSettings
    proxy: ProxySettings
    budget: BudgetSettings
    timeout: TimeoutSettings
    research: ResearchSettings
    artifacts: ArtifactSettings
    task_policy: TaskPolicySettings
    task_specs: dict[str, TaskSpec] = field(default_factory=lambda: dict(DEFAULT_TASK_SPECS))

    def payload(self) -> dict[str, Any]:
        return {
            "workspace_root": str(self.workspace_root),
            "submission_dir": str(self.submission_dir),
            "shared_code_dir": str(self.shared_code_dir),
            "data_roots": self.data_roots.payload(),
            "provider": self.provider.payload(),
            "proxy": self.proxy.payload(),
            "budget": self.budget.payload(),
            "timeout": self.timeout.payload(),
            "research": self.research.payload(),
            "artifacts": self.artifacts.payload(),
            "task_policy": self.task_policy.payload(),
            "task_specs": {
                task_id: {
                    "input_steps": spec.input_steps,
                    "output_steps": spec.output_steps,
                    "total_steps": spec.total_steps,
                    "spatial_points": spec.spatial_points,
                    "pred_shape": list(spec.pred_shape),
                    "first_steps_must_match": spec.first_steps_must_match,
                    "inference_time_limit_sec": spec.inference_time_limit_sec,
                    "must_train_from_scratch": spec.must_train_from_scratch,
                    "allow_public_pretrained_weights": spec.allow_public_pretrained_weights,
                    "default_train_filenames": list(spec.default_train_filenames),
                    "default_validation_filenames": list(spec.default_validation_filenames),
                    "default_test_filenames": list(spec.default_test_filenames),
                }
                for task_id, spec in self.task_specs.items()
            },
        }


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw_text = path.read_text(encoding="utf-8")
    lines = raw_text.splitlines()
    if len(lines) > 1:
        trailing_lines = [line for line in lines[1:] if line.strip()]
        if trailing_lines:
            trailing_indent = min(len(line) - len(line.lstrip()) for line in trailing_lines)
            lines = [lines[0].lstrip(), *[line[trailing_indent:] if line.strip() else "" for line in lines[1:]]]
            raw_text = "\n".join(lines)
    raw_text = dedent(raw_text).strip()
    payload = yaml.safe_load(raw_text) if raw_text else {}
    if not isinstance(payload, dict):
        raise ValueError("config file must contain a YAML object")
    return payload


def _parse_endpoint(data: dict[str, Any]) -> ProviderEndpoint:
    return ProviderEndpoint(
        provider=str(data.get("provider", "openai_compatible")).strip() or "openai_compatible",
        base_url=_required_str(data, "base_url"),
        api_key_env=_required_str(data, "api_key_env"),
        api_key=os.getenv(_required_str(data, "api_key_env")),
        model_id=_required_str(data, "model_id"),
        append_v1=bool(data.get("append_v1", True)),
        supports_prompt_cache_hints=bool(data.get("supports_prompt_cache_hints", True)),
    )


def load_config(path: str | Path, workspace_root: str | Path | None = None) -> V3Config:
    config_path = Path(path)
    raw = _load_yaml(config_path)
    config_root = config_path.parent.resolve() if config_path.parent != Path("") else Path.cwd()

    if not raw:
        workspace_root_path = Path.cwd() if workspace_root is None else Path(workspace_root).resolve()
        primary = ProviderEndpoint(
            provider="openai_compatible",
            base_url="https://example.com",
            api_key_env="OPENAI_API_KEY",
            api_key=os.getenv("OPENAI_API_KEY"),
            model_id="gpt-5.4",
        )
        return V3Config(
            config_path=config_path,
            workspace_root=workspace_root_path,
            submission_dir=(workspace_root_path / "submission").resolve(),
            shared_code_dir=(workspace_root_path / "submission" / "code").resolve(),
            data_roots=DataRoots(
                primary=(workspace_root_path / "data").resolve(),
                evaluation=[(workspace_root_path / "data").resolve()],
            ),
            provider=ProviderSettings(wire_api=_RESPONSES_WIRE_API, primary=primary, fallback=None, require_fallback=False),
            proxy=ProxySettings(
                enabled=False,
                primary_log_dir=(workspace_root_path / "proxy_logs" / "primary").resolve(),
                fallback_log_dir=(workspace_root_path / "proxy_logs" / "fallback").resolve(),
                proxy_version="unknown",
            ),
            budget=BudgetSettings(),
            timeout=TimeoutSettings(),
            research=ResearchSettings(
                enabled=True,
                allow_network=False,
                cache_dir=(workspace_root_path / "research" / "cache").resolve(),
                cache_index_path=(workspace_root_path / "research" / "cache" / "research_sources.jsonl").resolve(),
                raw_cache_dir=(workspace_root_path / "research" / "cache" / "raw").resolve(),
                papers_dir=(workspace_root_path / "research" / "papers").resolve(),
            ),
            artifacts=ArtifactSettings(
                provider_report_path=(workspace_root_path / "capabilities" / "provider_report.json").resolve(),
                package_output_path=(workspace_root_path / "submission" / "submission.zip").resolve(),
                validation_report_path=(workspace_root_path / "submission" / "validation_report.json").resolve(),
                finalize_gate_path=(workspace_root_path / "submission" / "finalize_gate.json").resolve(),
                state_path=(workspace_root_path / "agent_state.json").resolve(),
            ),
            task_policy=TaskPolicySettings(),
        )

    if "router" in raw and isinstance(raw["router"], dict):
        router_wire_api = str(raw["router"].get("wire_api", "")).strip()
        if router_wire_api and router_wire_api != _RESPONSES_WIRE_API:
            raise ValueError(f"router.wire_api must be {_RESPONSES_WIRE_API!r}")

    if workspace_root is not None:
        workspace_root_path = Path(workspace_root).resolve()
    elif "workspace_root" in raw:
        workspace_root_path = _resolve_path(config_root, raw.get("workspace_root"), default="workspace")
    else:
        workspace_root_path = Path.cwd()

    provider_raw = raw.get("provider", {})
    if not isinstance(provider_raw, dict):
        raise ValueError("provider config must be an object")
    wire_api = _reject_legacy_wire_api(str(provider_raw.get("wire_api", _RESPONSES_WIRE_API)), field_name="provider.wire_api")
    primary_raw = provider_raw.get("primary", {})
    if not isinstance(primary_raw, dict):
        raise ValueError("provider.primary config must be an object")
    primary = _parse_endpoint(primary_raw)
    fallback_raw = provider_raw.get("fallback")
    fallback = _parse_endpoint(fallback_raw) if isinstance(fallback_raw, dict) else None
    require_fallback = bool(provider_raw.get("require_fallback", False))
    if require_fallback and fallback is None:
        raise ValueError("provider fallback is required when require_fallback=true")
    provider = ProviderSettings(
        wire_api=wire_api,
        primary=primary,
        fallback=fallback,
        require_fallback=require_fallback,
    )

    submission_dir = _resolve_configured_path(
        config_root=config_root,
        workspace_root=workspace_root_path,
        value=raw.get("submission_dir"),
        default="submission",
    )
    shared_code_dir = _resolve_configured_path(
        config_root=config_root,
        workspace_root=workspace_root_path,
        value=raw.get("shared_code_dir"),
        default="submission/code",
    )

    data_roots_raw = raw.get("data_roots", {})
    if not isinstance(data_roots_raw, dict):
        data_roots_raw = {}
    primary_data_root = _resolve_configured_path(
        config_root=config_root,
        workspace_root=workspace_root_path,
        value=data_roots_raw.get("primary"),
        default="data",
    )
    evaluation_roots_raw = data_roots_raw.get("evaluation", [primary_data_root])
    if not isinstance(evaluation_roots_raw, list):
        evaluation_roots_raw = [evaluation_roots_raw]
    evaluation_roots = [
        _resolve_configured_path(
            config_root=config_root,
            workspace_root=workspace_root_path,
            value=item,
            default="data",
        )
        for item in evaluation_roots_raw
    ]
    data_roots = DataRoots(primary=primary_data_root, evaluation=evaluation_roots)

    proxy_raw = raw.get("proxy", {})
    if not isinstance(proxy_raw, dict):
        proxy_raw = {}
    proxy = ProxySettings(
        enabled=bool(proxy_raw.get("enabled", False)),
        primary_log_dir=_resolve_configured_path(
            config_root=config_root,
            workspace_root=workspace_root_path,
            value=proxy_raw.get("primary_log_dir"),
            default="proxy_logs/primary",
        ),
        fallback_log_dir=_resolve_configured_path(
            config_root=config_root,
            workspace_root=workspace_root_path,
            value=proxy_raw.get("fallback_log_dir"),
            default="proxy_logs/fallback",
        ),
        proxy_version=str(proxy_raw.get("proxy_version", "unknown")),
    )

    budget_raw = raw.get("budget", {})
    if not isinstance(budget_raw, dict):
        budget_raw = {}
    budget = BudgetSettings(
        max_total_usd=float(budget_raw.get("max_total_usd", 0.0)),
        max_steps=int(budget_raw.get("max_steps", 0)),
    )

    timeout_raw = raw.get("timeout", {})
    if not isinstance(timeout_raw, dict):
        timeout_raw = {}
    timeout = TimeoutSettings(
        provider_seconds=int(timeout_raw.get("provider_seconds", 60)),
        formal_run_seconds=int(timeout_raw.get("formal_run_seconds", 3600)),
    )

    research_raw = raw.get("research", {})
    if not isinstance(research_raw, dict):
        research_raw = {}
    cache_dir = _resolve_configured_path(
        config_root=config_root,
        workspace_root=workspace_root_path,
        value=research_raw.get("cache_dir"),
        default="research/cache",
    )
    research = ResearchSettings(
        enabled=bool(research_raw.get("enabled", True)),
        allow_network=bool(research_raw.get("allow_network", False)),
        cache_dir=cache_dir,
        cache_index_path=(cache_dir / "research_sources.jsonl").resolve(),
        raw_cache_dir=(cache_dir / "raw").resolve(),
        papers_dir=_resolve_configured_path(
            config_root=config_root,
            workspace_root=workspace_root_path,
            value=research_raw.get("papers_dir"),
            default="research/papers",
        ),
    )

    artifacts_raw = raw.get("artifacts", {})
    if not isinstance(artifacts_raw, dict):
        artifacts_raw = {}
    artifacts = ArtifactSettings(
        provider_report_path=_resolve_path(
            config_root if artifacts_raw.get("provider_report_path") is not None else workspace_root_path,
            artifacts_raw.get("provider_report_path"),
            default="capabilities/provider_report.json",
        ),
        package_output_path=_resolve_path(
            config_root if artifacts_raw.get("package_output_path") is not None else workspace_root_path,
            artifacts_raw.get("package_output_path"),
            default="submission/submission.zip",
        ),
        validation_report_path=_resolve_path(
            config_root if artifacts_raw.get("validation_report_path") is not None else workspace_root_path,
            artifacts_raw.get("validation_report_path"),
            default="submission/validation_report.json",
        ),
        finalize_gate_path=_resolve_path(
            config_root if artifacts_raw.get("finalize_gate_path") is not None else workspace_root_path,
            artifacts_raw.get("finalize_gate_path"),
            default="submission/finalize_gate.json",
        ),
        state_path=_resolve_path(
            config_root if artifacts_raw.get("state_path") is not None else workspace_root_path,
            artifacts_raw.get("state_path"),
            default="agent_state.json",
        ),
    )

    task_policy_raw = raw.get("task_policy", {})
    if not isinstance(task_policy_raw, dict):
        task_policy_raw = {}
    task_ids = [
        str(task).strip()
        for task in task_policy_raw.get("task_ids", list(TASK_IDS))
        if str(task).strip() in TASK_IDS
    ]
    task_policy = TaskPolicySettings(
        task_ids=task_ids or list(TASK_IDS),
        strict_validation=bool(task_policy_raw.get("strict_validation", True)),
    )

    return V3Config(
        config_path=config_path,
        workspace_root=workspace_root_path,
        submission_dir=submission_dir,
        shared_code_dir=shared_code_dir,
        data_roots=data_roots,
        provider=provider,
        proxy=proxy,
        budget=budget,
        timeout=timeout,
        research=research,
        artifacts=artifacts,
        task_policy=task_policy,
    )
