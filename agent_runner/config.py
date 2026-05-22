from __future__ import annotations

import os
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

REQUIRED_WIRE_API = "json_action"
REQUIRED_PROFILE_NAMES = ("strong_planner", "coder", "log_summarizer", "json_judge")
DEFAULT_RESEARCH_ALLOWED_DOMAINS = [
    "arxiv.org",
    "export.arxiv.org",
    "github.com",
    "raw.githubusercontent.com",
    "openreview.net",
    "neuraloperator.github.io",
]
DEFAULT_RESEARCH_BLOCKED_EXTENSIONS = [
    ".hdf5",
    ".h5",
    ".pt",
    ".pth",
    ".ckpt",
    ".npz",
    ".npy",
    ".tar",
    ".zip",
]
DEFAULT_FORMAL_VALIDATION_RMSE_THRESHOLD = 0.01


def _utc_run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _default_submission_tasks() -> dict[str, "SubmissionTaskConfig"]:
    return {
        "task1": SubmissionTaskConfig(
            name="task1",
            pred_filename="task1_pred.hdf5",
            time_filename="task1_time.csv",
            logs_filename="task1_logs.log",
            test_hdf5="data/task1_test.hdf5",
            validation_hdf5="data/task1_val.hdf5",
            input_steps=10,
            total_steps=200,
            spatial_points=256,
        ),
        "task2": SubmissionTaskConfig(
            name="task2",
            pred_filename="task2_pred.hdf5",
            time_filename="task2_time.csv",
            logs_filename="task2_logs.log",
            test_hdf5="data/task2_test.hdf5",
            validation_hdf5="data/task2_val.h5",
            input_steps=10,
            total_steps=200,
            spatial_points=256,
        ),
        "task3": SubmissionTaskConfig(
            name="task3",
            pred_filename="task3_pred.hdf5",
            time_filename="task3_time.csv",
            logs_filename="task3_logs.log",
            test_hdf5="data/task3_test.hdf5",
            validation_hdf5="data/KS_val.hdf5",
            input_steps=20,
            total_steps=400,
            spatial_points=256,
        ),
    }


@dataclass(slots=True)
class OpenAIEndpointConfig:
    provider: str = "third_party_openai_compatible"
    base_url: str = "https://example.com"
    api_key_env: str = "LLM_API_KEY"
    api_key: str | None = None
    append_v1: bool = True
    store: bool = False
    streaming: bool = False

    def resolve_env(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv(self.api_key_env)

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
            append_v1=bool(data.get("append_v1", defaults.append_v1)),
            store=bool(data.get("store", defaults.store)),
            streaming=bool(data.get("streaming", defaults.streaming)),
        )


@dataclass(slots=True)
class FallbackProviderConfig:
    enabled: bool = True
    provider: str = "deepseek_openai_compatible"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    api_key: str | None = None
    append_v1: bool = False
    pro_model_env: str = "DEEPSEEK_PRO_MODEL"
    pro_model: str | None = None
    flash_model_env: str = "DEEPSEEK_FLASH_MODEL"
    flash_model: str | None = None

    def resolve_env(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv(self.api_key_env)
        if self.pro_model is None:
            self.pro_model = os.getenv(self.pro_model_env)
        if self.flash_model is None:
            self.flash_model = os.getenv(self.flash_model_env)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FallbackProviderConfig":
        data = data or {}
        defaults = cls()
        api_key_env = data.get("api_key_env", defaults.api_key_env)
        pro_model_env = data.get("pro_model_env", defaults.pro_model_env)
        flash_model_env = data.get("flash_model_env", defaults.flash_model_env)
        return cls(
            enabled=bool(data.get("enabled", defaults.enabled)),
            provider=data.get("provider", defaults.provider),
            base_url=data.get("base_url", defaults.base_url),
            api_key_env=api_key_env,
            api_key=os.getenv(api_key_env),
            append_v1=bool(data.get("append_v1", defaults.append_v1)),
            pro_model_env=pro_model_env,
            pro_model=os.getenv(pro_model_env),
            flash_model_env=flash_model_env,
            flash_model=os.getenv(flash_model_env),
        )


@dataclass(slots=True)
class LogProxyConfig:
    enabled: bool = False
    base_url: str = "http://localhost:8080"
    target: str = "https://aixj.vip"
    log_dir: Path | str = "workspace/proxy_logs/aixj"
    fallback_base_url: str | None = "http://localhost:8081"
    fallback_target: str | None = "https://api.deepseek.com"
    fallback_log_dir: Path | str | None = "workspace/proxy_logs/deepseek"
    proxy_script: str = "scripts/proxy.py"

    def resolve_paths(self, workspace_root: Path, project_root: Path) -> None:
        self.log_dir = self._resolve_single_path(self.log_dir, workspace_root, project_root)
        fallback_log_dir = self.fallback_log_dir or self.log_dir
        self.fallback_log_dir = self._resolve_single_path(fallback_log_dir, workspace_root, project_root)

    @staticmethod
    def _resolve_single_path(path_value: Path | str, workspace_root: Path, project_root: Path) -> Path:
        path = Path(path_value)
        if not path.is_absolute():
            if path.parts and path.parts[0] == "workspace":
                path = (workspace_root / Path(*path.parts[1:])).resolve()
            else:
                path = (project_root / path).resolve()
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LogProxyConfig":
        data = data or {}
        defaults = cls()
        return cls(
            enabled=bool(data.get("enabled", defaults.enabled)),
            base_url=str(data.get("base_url", defaults.base_url)),
            target=str(data.get("target", defaults.target)),
            log_dir=data.get("log_dir", defaults.log_dir),
            fallback_base_url=str(data.get("fallback_base_url", data.get("base_url", defaults.base_url))),
            fallback_target=str(data.get("fallback_target", data.get("target", defaults.target))),
            fallback_log_dir=data.get("fallback_log_dir", data.get("log_dir", defaults.log_dir)),
            proxy_script=str(data.get("proxy_script", defaults.proxy_script)),
        )


@dataclass(slots=True)
class SubmissionTaskConfig:
    name: str
    pred_filename: str
    time_filename: str
    logs_filename: str
    test_hdf5: str
    validation_hdf5: str | None = None
    input_steps: int = 10
    total_steps: int = 200
    spatial_points: int = 256

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any] | None) -> "SubmissionTaskConfig":
        data = data or {}
        defaults = _default_submission_tasks()[name]
        return cls(
            name=name,
            pred_filename=str(data.get("pred_filename", defaults.pred_filename)),
            time_filename=str(data.get("time_filename", defaults.time_filename)),
            logs_filename=str(data.get("logs_filename", defaults.logs_filename)),
            test_hdf5=str(data.get("test_hdf5", defaults.test_hdf5)),
            validation_hdf5=data.get("validation_hdf5", defaults.validation_hdf5),
            input_steps=int(data.get("input_steps", defaults.input_steps)),
            total_steps=int(data.get("total_steps", defaults.total_steps)),
            spatial_points=int(data.get("spatial_points", defaults.spatial_points)),
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
    enable_web_search: bool = False
    experimental_enable_hosted_web_search: bool = False
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
            experimental_enable_hosted_web_search=bool(
                data.get("experimental_enable_hosted_web_search", defaults.experimental_enable_hosted_web_search)
            ),
            enable_file_search=bool(data.get("enable_file_search", defaults.enable_file_search)),
            enable_skills=bool(data.get("enable_skills", defaults.enable_skills)),
            enable_tool_search=bool(data.get("enable_tool_search", defaults.enable_tool_search)),
            web_search=dict(data.get("web_search", defaults.web_search)),
            file_search=dict(data.get("file_search", defaults.file_search)),
            skills=dict(data.get("skills", defaults.skills)),
            tool_search=dict(data.get("tool_search", defaults.tool_search)),
        )


@dataclass(slots=True)
class ResponsesRuntimeConfig:
    max_tool_calls_per_turn: int = 1
    parallel_tool_calls: bool = False
    retry_on_multi_tool_failure: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ResponsesRuntimeConfig":
        data = data or {}
        defaults = cls()
        return cls(
            max_tool_calls_per_turn=int(data.get("max_tool_calls_per_turn", defaults.max_tool_calls_per_turn)),
            parallel_tool_calls=bool(data.get("parallel_tool_calls", defaults.parallel_tool_calls)),
            retry_on_multi_tool_failure=bool(
                data.get("retry_on_multi_tool_failure", defaults.retry_on_multi_tool_failure)
            ),
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
class ResearchArxivProviderConfig:
    enabled: bool = True
    min_interval_seconds: float = 3.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ResearchArxivProviderConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            min_interval_seconds=float(data.get("min_interval_seconds", 3.0)),
        )


@dataclass(slots=True)
class ResearchGitHubProviderConfig:
    enabled: bool = True
    api_key_env: str = "GITHUB_TOKEN"
    api_key: str | None = None
    allow_unauthenticated: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ResearchGitHubProviderConfig":
        data = data or {}
        defaults = cls()
        api_key_env = data.get("api_key_env", defaults.api_key_env)
        return cls(
            enabled=bool(data.get("enabled", defaults.enabled)),
            api_key_env=api_key_env,
            api_key=os.getenv(api_key_env),
            allow_unauthenticated=bool(data.get("allow_unauthenticated", defaults.allow_unauthenticated)),
        )


@dataclass(slots=True)
class ResearchWebProviderConfig:
    provider_order: list[str] = field(default_factory=lambda: ["tavily", "exa", "brave", "google_cse"])
    tavily_api_key_env: str = "TAVILY_API_KEY"
    tavily_api_key: str | None = None
    exa_api_key_env: str = "EXA_API_KEY"
    exa_api_key: str | None = None
    brave_api_key_env: str = "BRAVE_SEARCH_API_KEY"
    brave_api_key: str | None = None
    google_api_key_env: str = "GOOGLE_API_KEY"
    google_api_key: str | None = None
    google_cse_id_env: str = "GOOGLE_CSE_ID"
    google_cse_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ResearchWebProviderConfig":
        data = data or {}
        defaults = cls()
        tavily_api_key_env = data.get("tavily_api_key_env", defaults.tavily_api_key_env)
        exa_api_key_env = data.get("exa_api_key_env", defaults.exa_api_key_env)
        brave_api_key_env = data.get("brave_api_key_env", defaults.brave_api_key_env)
        google_api_key_env = data.get("google_api_key_env", defaults.google_api_key_env)
        google_cse_id_env = data.get("google_cse_id_env", defaults.google_cse_id_env)
        return cls(
            provider_order=list(data.get("provider_order", defaults.provider_order)),
            tavily_api_key_env=tavily_api_key_env,
            tavily_api_key=os.getenv(tavily_api_key_env),
            exa_api_key_env=exa_api_key_env,
            exa_api_key=os.getenv(exa_api_key_env),
            brave_api_key_env=brave_api_key_env,
            brave_api_key=os.getenv(brave_api_key_env),
            google_api_key_env=google_api_key_env,
            google_api_key=os.getenv(google_api_key_env),
            google_cse_id_env=google_cse_id_env,
            google_cse_id=os.getenv(google_cse_id_env),
        )


@dataclass(slots=True)
class ResearchProvidersConfig:
    arxiv: ResearchArxivProviderConfig = field(default_factory=ResearchArxivProviderConfig)
    github: ResearchGitHubProviderConfig = field(default_factory=ResearchGitHubProviderConfig)
    web: ResearchWebProviderConfig = field(default_factory=ResearchWebProviderConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ResearchProvidersConfig":
        data = data or {}
        return cls(
            arxiv=ResearchArxivProviderConfig.from_dict(data.get("arxiv")),
            github=ResearchGitHubProviderConfig.from_dict(data.get("github")),
            web=ResearchWebProviderConfig.from_dict(data.get("web")),
        )


@dataclass(slots=True)
class ResearchConfig:
    enabled: bool = True
    cache_dir: Path | str = "workspace/research/cache"
    user_agent: str = "cozy_pde_research_agent/0.1"
    request_timeout_seconds: int = 20
    max_response_bytes: int = 5_000_000
    max_pdf_bytes: int = 30_000_000
    respect_robots_txt: bool = True
    allow_raw_github: bool = True
    providers: ResearchProvidersConfig = field(default_factory=ResearchProvidersConfig)
    allowed_domains: list[str] = field(default_factory=lambda: list(DEFAULT_RESEARCH_ALLOWED_DOMAINS))
    blocked_extensions: list[str] = field(default_factory=lambda: list(DEFAULT_RESEARCH_BLOCKED_EXTENSIONS))
    raw_cache_dir: Path | None = None
    papers_dir: Path | None = None
    cache_index_path: Path | None = None

    def resolve_paths(self, workspace_root: Path, project_root: Path) -> None:
        cache_dir = Path(self.cache_dir)
        if not cache_dir.is_absolute():
            if cache_dir.parts and cache_dir.parts[0] == "workspace":
                cache_dir = (workspace_root / Path(*cache_dir.parts[1:])).resolve()
            else:
                cache_dir = (project_root / cache_dir).resolve()
        self.cache_dir = cache_dir
        self.raw_cache_dir = cache_dir / "raw"
        self.papers_dir = cache_dir.parent / "papers"
        self.cache_index_path = cache_dir / "research_sources.jsonl"
        self.allowed_domains = [domain.lower() for domain in self.allowed_domains]
        self.blocked_extensions = [
            extension.lower() if str(extension).startswith(".") else f".{str(extension).lower()}"
            for extension in self.blocked_extensions
        ]

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ResearchConfig":
        data = data or {}
        defaults = cls()
        return cls(
            enabled=bool(data.get("enabled", defaults.enabled)),
            cache_dir=data.get("cache_dir", defaults.cache_dir),
            user_agent=str(data.get("user_agent", defaults.user_agent)),
            request_timeout_seconds=int(data.get("request_timeout_seconds", defaults.request_timeout_seconds)),
            max_response_bytes=int(data.get("max_response_bytes", defaults.max_response_bytes)),
            max_pdf_bytes=int(data.get("max_pdf_bytes", defaults.max_pdf_bytes)),
            respect_robots_txt=bool(data.get("respect_robots_txt", defaults.respect_robots_txt)),
            allow_raw_github=bool(data.get("allow_raw_github", defaults.allow_raw_github)),
            providers=ResearchProvidersConfig.from_dict(data.get("providers")),
            allowed_domains=list(data.get("allowed_domains", defaults.allowed_domains)),
            blocked_extensions=list(data.get("blocked_extensions", defaults.blocked_extensions)),
        )


@dataclass(slots=True)
class RunnerConfig:
    project_root: Path
    workspace_root: Path
    shared_workspace_root: Path | None = None
    endpoint: OpenAIEndpointConfig = field(default_factory=OpenAIEndpointConfig)
    fallback_provider: FallbackProviderConfig = field(default_factory=FallbackProviderConfig)
    log_proxy: LogProxyConfig = field(default_factory=LogProxyConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    llm_profiles: dict[str, LLMProfile] = field(default_factory=_default_profiles)
    submission_tasks: dict[str, SubmissionTaskConfig] = field(default_factory=_default_submission_tasks)
    responses: ResponsesRuntimeConfig = field(default_factory=ResponsesRuntimeConfig)
    responses_tools: ResponsesToolConfig = field(default_factory=ResponsesToolConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    shell_log_dir: Path | None = None
    session_label: str | None = None

    def __post_init__(self) -> None:
        self.project_root = self.project_root.resolve()
        self.workspace_root = self.workspace_root.resolve()
        if self.shared_workspace_root is None:
            self.shared_workspace_root = self.workspace_root
        else:
            self.shared_workspace_root = self.shared_workspace_root.resolve()
        self.endpoint.resolve_env()
        self.fallback_provider.resolve_env()
        self.llm_profiles = dict(self.llm_profiles)
        self.submission_tasks = dict(self.submission_tasks)
        self._validate_profile_set()
        self.log_proxy.resolve_paths(self.workspace_root, self.project_root)
        self.research.resolve_paths(self.workspace_root, self.project_root)
        if self.shell_log_dir is None:
            self.shell_log_dir = self.workspace_root / "runs" / "logs"

    def _validate_profile_set(self) -> None:
        missing = [name for name in REQUIRED_PROFILE_NAMES if name not in self.llm_profiles]
        if missing:
            raise ValueError(f"Missing required llm_profiles: {', '.join(missing)}")

    @property
    def llm_log_path(self) -> Path:
        if self.session_label:
            return self.workspace_root / "llm_logs" / f"{self.session_label}_all_llm_calls.jsonl"
        return self.workspace_root / "llm_logs" / "all_llm_calls.jsonl"

    @property
    def tool_log_path(self) -> Path:
        if self.session_label:
            return self.workspace_root / "internal_logs" / f"{self.session_label}_tool_calls.jsonl"
        return self.workspace_root / "internal_logs" / "tool_calls.jsonl"

    @property
    def submission_dir(self) -> Path:
        return self.workspace_root / "submission"

    @property
    def submission_code_dir(self) -> Path:
        return self.submission_dir / "code"

    @property
    def submission_task_list(self) -> list[SubmissionTaskConfig]:
        return [self.submission_tasks[name] for name in sorted(self.submission_tasks)]

    def task_config(self, name: str) -> SubmissionTaskConfig:
        return self.submission_tasks[name]

    def task_run_dir(self, name: str) -> Path:
        return self.workspace_root / "runs" / name

    def task_submission_code_dir(self, name: str) -> Path:
        return self.submission_code_dir / name

    def with_session(self, session_label: str) -> "RunnerConfig":
        run_root = self.shared_workspace_root / "runs" / f"{session_label}_{_utc_run_stamp()}"
        session_config = replace(
            self,
            workspace_root=run_root,
            shared_workspace_root=self.shared_workspace_root,
            session_label=session_label,
            shell_log_dir=run_root / "runs" / "logs",
        )
        session_config.ensure_workspace_dirs()
        return session_config

    def _ensure_linked_shared_dir(self, relative: str) -> None:
        assert self.shared_workspace_root is not None
        source = self.shared_workspace_root / relative
        target = self.workspace_root / relative
        source.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            return
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=True)

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
        if self.workspace_root == self.shared_workspace_root:
            for relative in [
                "data",
                "checkpoints",
                "baselines",
                "runs/scratch",
                "runs/logs",
                "runs/snapshots",
                "internal_logs",
                "llm_logs",
                "research/cache/raw",
                "research/papers",
                "proxy_logs",
                "submission/code",
            ]:
                (self.workspace_root / relative).mkdir(parents=True, exist_ok=True)
        else:
            for relative in [
                "runs/scratch",
                "runs/logs",
                "runs/snapshots",
                "internal_logs",
                "llm_logs",
                "research/cache/raw",
                "research/papers",
                "proxy_logs",
                "submission/code",
            ]:
                (self.workspace_root / relative).mkdir(parents=True, exist_ok=True)
            for relative in ["data", "checkpoints", "baselines"]:
                self._ensure_linked_shared_dir(relative)
        assert self.shell_log_dir is not None
        self.shell_log_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_workspace(cls, workspace_root: str | Path, project_root: str | Path | None = None) -> "RunnerConfig":
        workspace_root = Path(workspace_root)
        project_root = Path(project_root) if project_root is not None else workspace_root.parent
        _load_project_dotenv(project_root)
        config = cls(project_root=project_root, workspace_root=workspace_root, shared_workspace_root=workspace_root)
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
    submission_tasks_raw = raw.get("submission_tasks") or {}
    default_submission_tasks = _default_submission_tasks()
    submission_tasks: dict[str, SubmissionTaskConfig] = {}
    for name in default_submission_tasks:
        submission_tasks[name] = SubmissionTaskConfig.from_dict(name, submission_tasks_raw.get(name, asdict(default_submission_tasks[name])))

    config = RunnerConfig(
        project_root=project_root,
        workspace_root=workspace_root,
        shared_workspace_root=workspace_root,
        endpoint=OpenAIEndpointConfig.from_dict(raw.get("openai")),
        fallback_provider=FallbackProviderConfig.from_dict(raw.get("fallback_provider")),
        log_proxy=LogProxyConfig.from_dict(raw.get("log_proxy")),
        router=RouterConfig.from_dict(raw.get("router")),
        llm_profiles=llm_profiles,
        submission_tasks=submission_tasks,
        responses=ResponsesRuntimeConfig.from_dict(raw.get("responses")),
        responses_tools=ResponsesToolConfig.from_dict(raw.get("responses_tools")),
        research=ResearchConfig.from_dict(raw.get("research")),
        budget=BudgetConfig.from_dict(raw.get("budget")),
    )
    config.ensure_workspace_dirs()
    return config
