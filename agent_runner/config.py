from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REQUIRED_WIRE_API = "responses"
REQUIRED_PROFILE_NAMES = ("strong_planner", "coder", "log_summarizer", "json_judge")


def _load_project_dotenv(project_root: Path) -> None:
    dotenv_path = project_root / ".env"
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _require_responses_wire_api(wire_api: str, *, field_name: str) -> str:
    if wire_api != REQUIRED_WIRE_API:
        raise ValueError(f"{field_name} must be {REQUIRED_WIRE_API!r}, got {wire_api!r}")
    return wire_api


def _default_profiles() -> dict[str, "LLMProfile"]:
    return {
        "strong_planner": LLMProfile(
            name="strong_planner",
            model="gpt-5.5",
            reasoning_effort="xhigh",
            verbosity="medium",
            temperature=0.2,
            max_tokens=8192,
        ),
        "coder": LLMProfile(
            name="coder",
            model="gpt-5.4",
            reasoning_effort="high",
            verbosity="medium",
            temperature=0.2,
            max_tokens=8192,
        ),
        "log_summarizer": LLMProfile(
            name="log_summarizer",
            model="gpt-5.2",
            reasoning_effort="medium",
            verbosity="low",
            temperature=0.0,
            max_tokens=4096,
        ),
        "json_judge": LLMProfile(
            name="json_judge",
            model="gpt-5.2",
            reasoning_effort="medium",
            verbosity="low",
            temperature=0.0,
            max_tokens=2048,
        ),
    }


@dataclass(slots=True)
class OpenAIEndpointConfig:
    provider: str = "third_party_openai_compatible"
    base_url: str = "https://example.com"
    api_key_env: str = "LLM_API_KEY"
    api_key: str | None = None
    store: bool = False
    streaming: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OpenAIEndpointConfig":
        data = data or {}
        defaults = cls()
        api_key_env = data.get("api_key_env", defaults.api_key_env)
        return cls(
            provider=data.get("provider", defaults.provider),
            base_url=data.get("base_url", defaults.base_url),
            api_key_env=api_key_env,
            api_key=os.getenv(api_key_env),
            store=bool(data.get("store", defaults.store)),
            streaming=bool(data.get("streaming", defaults.streaming)),
        )


@dataclass(slots=True)
class LLMProfile:
    name: str
    model: str
    reasoning_effort: str
    verbosity: str
    temperature: float = 0.2
    max_tokens: int = 8192
    wire_api: str = REQUIRED_WIRE_API

    def __post_init__(self) -> None:
        self.wire_api = _require_responses_wire_api(self.wire_api, field_name=f"llm_profiles.{self.name}.wire_api")

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any] | None) -> "LLMProfile":
        data = data or {}
        return cls(
            name=name,
            model=data.get("model", "gpt-5.4"),
            reasoning_effort=data.get("reasoning_effort", "medium"),
            verbosity=data.get("verbosity", "medium"),
            temperature=float(data.get("temperature", 0.2)),
            max_tokens=int(data.get("max_tokens", 8192)),
            wire_api=data.get("wire_api", REQUIRED_WIRE_API),
        )


@dataclass(slots=True)
class RouterConfig:
    model: str = "gpt-5.4"
    reasoning_effort: str = "medium"
    verbosity: str = "medium"
    temperature: float = 0.0
    max_tokens: int = 2048
    wire_api: str = REQUIRED_WIRE_API

    def __post_init__(self) -> None:
        self.wire_api = _require_responses_wire_api(self.wire_api, field_name="router.wire_api")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RouterConfig":
        data = data or {}
        return cls(
            model=data.get("model", "gpt-5.4"),
            reasoning_effort=data.get("reasoning_effort", "medium"),
            verbosity=data.get("verbosity", "medium"),
            temperature=float(data.get("temperature", 0.0)),
            max_tokens=int(data.get("max_tokens", 2048)),
            wire_api=data.get("wire_api", REQUIRED_WIRE_API),
        )


@dataclass(slots=True)
class ResponsesToolConfig:
    enable_web_search: bool = True
    enable_file_search: bool = False
    enable_skills: bool = True
    enable_tool_search: bool = True
    web_search: dict[str, Any] = field(
        default_factory=lambda: {
            "allowed_domains": [
                "arxiv.org",
                "github.com",
                "raw.githubusercontent.com",
                "openreview.net",
                "neuraloperator.github.io",
            ],
            "return_token_budget": 4096,
        }
    )
    file_search: dict[str, Any] = field(default_factory=dict)
    skills: dict[str, Any] = field(
        default_factory=lambda: {
            "local_skill_dirs": ["skills"],
            "enabled": [
                "pdebench",
                "neural_operator_research",
                "hdf5_validation",
                "rollout_training",
            ],
        }
    )
    tool_search: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ResponsesToolConfig":
        data = data or {}
        defaults = cls()
        return cls(
            enable_web_search=bool(data.get("enable_web_search", defaults.enable_web_search)),
            enable_file_search=bool(data.get("enable_file_search", defaults.enable_file_search)),
            enable_skills=bool(data.get("enable_skills", defaults.enable_skills)),
            enable_tool_search=bool(data.get("enable_tool_search", defaults.enable_tool_search)),
            web_search=dict(data.get("web_search", defaults.web_search)),
            file_search=dict(data.get("file_search", defaults.file_search)),
            skills=dict(data.get("skills", defaults.skills)),
            tool_search=dict(data.get("tool_search", defaults.tool_search)),
        )


@dataclass(slots=True)
class BudgetConfig:
    max_wall_clock_hours: float = 11.5
    reserve_finalize_seconds: int = 1800
    max_llm_calls: int = 120
    max_tool_calls: int = 500
    max_agent_steps: int = 100
    task1_preferred_train_minutes: int = 60
    task1_secondary_train_minutes: int = 120
    max_single_shell_seconds: int = 7200

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BudgetConfig":
        data = data or {}
        return cls(
            max_wall_clock_hours=float(data.get("max_wall_clock_hours", 11.5)),
            reserve_finalize_seconds=int(data.get("reserve_finalize_seconds", 1800)),
            max_llm_calls=int(data.get("max_llm_calls", 120)),
            max_tool_calls=int(data.get("max_tool_calls", 500)),
            max_agent_steps=int(data.get("max_agent_steps", 100)),
            task1_preferred_train_minutes=int(data.get("task1_preferred_train_minutes", 60)),
            task1_secondary_train_minutes=int(data.get("task1_secondary_train_minutes", 120)),
            max_single_shell_seconds=int(data.get("max_single_shell_seconds", 7200)),
        )


@dataclass(slots=True)
class RunnerConfig:
    project_root: Path
    workspace_root: Path
    endpoint: OpenAIEndpointConfig = field(default_factory=OpenAIEndpointConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    llm_profiles: dict[str, LLMProfile] = field(default_factory=_default_profiles)
    responses_tools: ResponsesToolConfig = field(default_factory=ResponsesToolConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    shell_log_dir: Path | None = None

    def __post_init__(self) -> None:
        self.project_root = self.project_root.resolve()
        self.workspace_root = self.workspace_root.resolve()
        self.llm_profiles = dict(self.llm_profiles)
        self._validate_profile_set()
        if self.shell_log_dir is None:
            self.shell_log_dir = self.workspace_root / "runs" / "logs"

    def _validate_profile_set(self) -> None:
        missing = [name for name in REQUIRED_PROFILE_NAMES if name not in self.llm_profiles]
        if missing:
            raise ValueError(f"Missing required llm_profiles: {', '.join(missing)}")

    @property
    def llm_log_path(self) -> Path:
        return self.workspace_root / "llm_logs" / "all_llm_calls.jsonl"

    @property
    def tool_log_path(self) -> Path:
        return self.workspace_root / "internal_logs" / "tool_calls.jsonl"

    @property
    def submission_dir(self) -> Path:
        return self.workspace_root / "submission"

    @property
    def submission_code_dir(self) -> Path:
        return self.submission_dir / "code"

    @property
    def router_profile(self) -> LLMProfile:
        return LLMProfile(
            name="router",
            model=self.router.model,
            reasoning_effort=self.router.reasoning_effort,
            verbosity=self.router.verbosity,
            temperature=self.router.temperature,
            max_tokens=self.router.max_tokens,
            wire_api=self.router.wire_api,
        )

    def ensure_workspace_dirs(self) -> None:
        for relative in [
            "data",
            "checkpoints",
            "baselines",
            "runs/scratch",
            "runs/logs",
            "runs/snapshots",
            "internal_logs",
            "llm_logs",
            "submission/code",
        ]:
            (self.workspace_root / relative).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_workspace(cls, workspace_root: str | Path, project_root: str | Path | None = None) -> "RunnerConfig":
        workspace_root = Path(workspace_root)
        project_root = Path(project_root) if project_root is not None else workspace_root.parent
        config = cls(project_root=project_root, workspace_root=workspace_root)
        config.ensure_workspace_dirs()
        return config


def load_config(config_path: str | Path, workspace_override: str | Path | None = None) -> RunnerConfig:
    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    project_root = path.parent.parent
    _load_project_dotenv(project_root)

    workspace_value = workspace_override or raw.get("workspace", "workspace")
    workspace_root = Path(workspace_value)
    if not workspace_root.is_absolute():
        workspace_root = (project_root / workspace_root).resolve()

    llm_profiles_raw = raw.get("llm_profiles") or {}
    default_profiles = _default_profiles()
    llm_profiles: dict[str, LLMProfile] = {}
    for name in REQUIRED_PROFILE_NAMES:
        llm_profiles[name] = LLMProfile.from_dict(name, llm_profiles_raw.get(name, asdict(default_profiles[name])))

    config = RunnerConfig(
        project_root=project_root,
        workspace_root=workspace_root,
        endpoint=OpenAIEndpointConfig.from_dict(raw.get("openai")),
        router=RouterConfig.from_dict(raw.get("router")),
        llm_profiles=llm_profiles,
        responses_tools=ResponsesToolConfig.from_dict(raw.get("responses_tools")),
        budget=BudgetConfig.from_dict(raw.get("budget")),
    )
    config.ensure_workspace_dirs()
    return config
