from __future__ import annotations

from dataclasses import dataclass

from cozy_pde_v3.state import AgentState


@dataclass(slots=True)
class RouteDecision:
    phase: str
    profile: str
    allowed_tools: list[str]
    allow_hosted_research: bool
    requires_llm: bool
    deterministic_action: str
    reason_code: str


class DeterministicRouter:
    _IMPLEMENTATION_TOOLS = [
        "read_file",
        "write_file",
        "patch_file",
        "run_python",
        "run_shell",
        "inspect_hdf5",
        "validate_submission",
        "package_submission",
    ]
    _FAILURE_RECOVERY_TOOLS = [
        "read_file",
        "patch_file",
        "run_python",
        "run_shell",
        "validate_submission",
    ]

    def choose(
        self,
        state: AgentState,
        *,
        capability_ready: bool = True,
        preflight_pending: bool = False,
    ) -> RouteDecision:
        if not capability_ready:
            return RouteDecision(
                phase="capability_readiness",
                profile="strong_planner",
                allowed_tools=[],
                allow_hosted_research=False,
                requires_llm=False,
                deterministic_action="run_capability_checks",
                reason_code="capability_not_ready",
            )

        if preflight_pending:
            return RouteDecision(
                phase="preflight",
                profile="log_summarizer",
                allowed_tools=["inspect_hdf5", "read_file"],
                allow_hosted_research=False,
                requires_llm=True,
                deterministic_action="inspect_inputs_before_execution",
                reason_code="preflight_pending",
            )

        if state.shared_code_version is None:
            return RouteDecision(
                phase="baseline_guard",
                profile="strong_planner",
                allowed_tools=["read_file"],
                allow_hosted_research=False,
                requires_llm=True,
                deterministic_action="establish_shared_code_baseline",
                reason_code="baseline_missing",
            )

        if state.latest_error_type:
            return RouteDecision(
                phase="failure_recovery",
                profile="coder",
                allowed_tools=list(self._FAILURE_RECOVERY_TOOLS),
                allow_hosted_research=False,
                requires_llm=True,
                deterministic_action="recover_from_latest_error",
                reason_code=state.latest_error_type,
            )

        if state.current_phase == "train_validate_benchmark":
            return RouteDecision(
                phase="train_validate_benchmark",
                profile="coder",
                allowed_tools=["run_python", "run_shell", "validate_submission"],
                allow_hosted_research=False,
                requires_llm=True,
                deterministic_action="train_validate_and_benchmark",
                reason_code="post_implementation_checks",
            )

        if state.current_phase == "diagnosis":
            return RouteDecision(
                phase="diagnosis",
                profile="log_summarizer",
                allowed_tools=["read_file", "inspect_hdf5", "validate_submission"],
                allow_hosted_research=False,
                requires_llm=True,
                deterministic_action="diagnose_latest_run_state",
                reason_code="diagnosis_requested",
            )

        if state.current_phase == "reflection":
            return RouteDecision(
                phase="reflection",
                profile="strong_planner",
                allowed_tools=["read_file"],
                allow_hosted_research=False,
                requires_llm=True,
                deterministic_action="summarize_learnings",
                reason_code="reflection_requested",
            )

        if state.current_phase == "finalization":
            return RouteDecision(
                phase="finalization",
                profile="json_judge",
                allowed_tools=["validate_submission", "package_submission"],
                allow_hosted_research=False,
                requires_llm=True,
                deterministic_action="finalize_best_artifact",
                reason_code="finalization_requested",
            )

        return RouteDecision(
            phase="implementation",
            profile="coder",
            allowed_tools=list(self._IMPLEMENTATION_TOOLS),
            allow_hosted_research=False,
            requires_llm=True,
            deterministic_action="implement_current_task",
            reason_code="ready_for_implementation",
        )


def route_agent_state(
    state: AgentState,
    *,
    capability_ready: bool = True,
    preflight_pending: bool = False,
) -> RouteDecision:
    return DeterministicRouter().choose(
        state,
        capability_ready=capability_ready,
        preflight_pending=preflight_pending,
    )
