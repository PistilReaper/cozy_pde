from __future__ import annotations

from cozy_pde_v3.deterministic_router import DeterministicRouter, route_agent_state
from cozy_pde_v3.state import AgentState


def test_router_requires_capability_checks_before_implementation() -> None:
    state = AgentState(task="task1", shared_code_version="sha256:base")

    decision = route_agent_state(state, capability_ready=False, preflight_pending=False)

    assert decision.phase == "capability_readiness"
    assert decision.profile == "strong_planner"
    assert decision.allowed_tools == []
    assert decision.allow_hosted_research is False
    assert decision.requires_llm is False
    assert decision.deterministic_action == "run_capability_checks"
    assert decision.reason_code == "capability_not_ready"


def test_router_uses_preflight_research_before_formal_work() -> None:
    state = AgentState(task="task1", shared_code_version="sha256:base")

    decision = DeterministicRouter().choose(state, capability_ready=True, preflight_pending=True)

    assert decision.phase == "preflight"
    assert decision.profile == "log_summarizer"
    assert decision.allowed_tools == ["inspect_hdf5", "read_file"]
    assert decision.allow_hosted_research is False
    assert decision.requires_llm is True
    assert decision.deterministic_action == "inspect_inputs_before_execution"
    assert decision.reason_code == "preflight_pending"


def test_router_selects_coder_once_shared_code_baseline_exists() -> None:
    state = AgentState(task="task2", shared_code_version="sha256:ready")

    decision = route_agent_state(state, capability_ready=True, preflight_pending=False)

    assert decision.phase == "implementation"
    assert decision.profile == "coder"
    assert decision.allowed_tools == [
        "read_file",
        "write_file",
        "patch_file",
        "run_python",
        "run_shell",
        "inspect_hdf5",
        "validate_submission",
        "package_submission",
    ]
    assert decision.allow_hosted_research is False
    assert decision.requires_llm is True
    assert decision.deterministic_action == "implement_current_task"
    assert decision.reason_code == "ready_for_implementation"


def test_router_uses_failure_recovery_without_legacy_llm_json_router() -> None:
    state = AgentState(
        task="task3",
        shared_code_version="sha256:ready",
        latest_error_type="shape_validation_failed",
    )

    decision = DeterministicRouter().choose(state, capability_ready=True, preflight_pending=False)

    assert decision.phase == "failure_recovery"
    assert decision.profile == "coder"
    assert decision.allowed_tools == [
        "read_file",
        "patch_file",
        "run_python",
        "run_shell",
        "validate_submission",
    ]
    assert decision.requires_llm is True
    assert decision.allow_hosted_research is False
    assert decision.deterministic_action == "recover_from_latest_error"
    assert decision.reason_code == "shape_validation_failed"
