from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import RunnerConfig
from .logger import LLMCallLogger
from .responses_items import extract_output_text, user_text

ALLOWED_PROFILES = {"strong_planner", "coder", "log_summarizer", "json_judge"}
ALLOWED_PHASES = {
    "research",
    "planning",
    "implementation",
    "debugging",
    "log_analysis",
    "validation",
    "finalization",
}

ROUTER_PROMPT = """You are a routing controller for a Responses-only agent.
Return a single JSON object with keys:
- profile: one of strong_planner, coder, log_summarizer, json_judge
- phase: one of research, planning, implementation, debugging, log_analysis, validation, finalization
- enable_hosted_tools: boolean
- reason: short text
Return JSON only."""


@dataclass(slots=True)
class RouteDecision:
    profile: str
    phase: str
    enable_hosted_tools: bool
    reason: str


def _normalize_phase(phase: str | None, summary: str) -> str:
    if phase in ALLOWED_PHASES:
        return phase

    lowered = summary.lower()
    if any(token in lowered for token in ("research", "paper", "arxiv", "github", "baseline", "literature")):
        return "research"
    if any(token in lowered for token in ("validate", "validator", "schema", "json", "bundle", "final")):
        return "validation"
    if any(token in lowered for token in ("log", "loss", "nan", "oom", "error", "traceback")):
        return "log_analysis"
    if any(token in lowered for token in ("debug", "bug", "fix", "shape mismatch", "exception")):
        return "debugging"
    if any(token in lowered for token in ("plan", "design", "route")):
        return "planning"
    return "implementation"


def _fallback_route(*, summary: str, phase_hint: str | None) -> RouteDecision:
    phase = _normalize_phase(phase_hint, summary)
    if phase in {"research", "planning"}:
        return RouteDecision("strong_planner", phase, False, "Fallback routing for research/planning.")
    if phase in {"implementation", "debugging"}:
        return RouteDecision("coder", phase, False, "Fallback routing for code work.")
    if phase == "log_analysis":
        return RouteDecision("log_summarizer", phase, False, "Fallback routing for logs.")
    return RouteDecision("json_judge", phase, False, "Fallback routing for validation/finalization.")


class Router:
    def __init__(self, *, client: Any, config: RunnerConfig, llm_logger: LLMCallLogger) -> None:
        self.client = client
        self.config = config
        self.llm_logger = llm_logger
        self.router_profile = config.router_profile

    def choose(
        self,
        *,
        summary: str,
        task_id: str,
        step_id: str,
        phase_hint: str | None = None,
    ) -> RouteDecision:
        response = self.client.create(
            profile=self.router_profile,
            input_items=[user_text(summary)],
            tools=[],
            instructions=ROUTER_PROMPT,
            metadata={
                "task_id": f"router:{task_id}",
                "step_id": step_id,
                "profile": "router",
                "phase": "routing",
            },
        )
        text = extract_output_text(response).strip()
        self.llm_logger.log_call(
            step_id=step_id,
            task_id=f"router:{task_id}",
            model=self.router_profile.model,
            profile="router",
            phase="routing",
            elapsed_seconds=0.0,
            response=text or None,
            raw_response=response,
            tool_calls=None,
        )

        try:
            payload = json.loads(text)
        except Exception:  # noqa: BLE001
            return _fallback_route(summary=summary, phase_hint=phase_hint)

        profile = payload.get("profile")
        phase = payload.get("phase")
        enable_hosted_tools = bool(payload.get("enable_hosted_tools", False))
        reason = str(payload.get("reason", "")).strip()

        if profile not in ALLOWED_PROFILES or phase not in ALLOWED_PHASES:
            return _fallback_route(summary=summary, phase_hint=phase_hint)

        if profile in {"coder", "log_summarizer", "json_judge"} and "enable_hosted_tools" not in payload:
            enable_hosted_tools = False

        return RouteDecision(
            profile=profile,
            phase=phase,
            enable_hosted_tools=enable_hosted_tools,
            reason=reason,
        )
