from __future__ import annotations

from cozy_pde_v3.context_packer import ContextPacker
from cozy_pde_v3.deterministic_router import DeterministicRouter, route_agent_state
from cozy_pde_v3.experiment_engine import compatibility_checks_for_patch
from cozy_pde_v3.state import AgentState


def test_failure_recovery_beats_implementation_when_latest_error_type_is_set() -> None:
    state = AgentState(
        task="task1",
        current_phase="implementation",
        latest_error_type="cuda_oom",
        shared_code_version="sha256:base",
    )

    decision = DeterministicRouter().choose(state)

    assert decision.phase == "failure_recovery"
    assert decision.profile == "coder"
    assert decision.allowed_tools == [
        "read_file",
        "patch_file",
        "run_python",
        "run_shell",
        "validate_submission",
    ]
    assert decision.allow_hosted_research is False
    assert decision.requires_llm is True
    assert decision.deterministic_action == "recover_from_latest_error"
    assert decision.reason_code == "cuda_oom"


def test_capability_readiness_wins_over_lower_priority_phases() -> None:
    state = AgentState(
        task="task1",
        current_phase="implementation",
        shared_code_version="sha256:base",
    )

    decision = DeterministicRouter().choose(
        state, capability_ready=False, preflight_pending=True
    )

    assert decision.phase == "capability_readiness"
    assert decision.profile == "strong_planner"
    assert decision.allowed_tools == []
    assert decision.allow_hosted_research is False
    assert decision.requires_llm is False
    assert decision.deterministic_action == "run_capability_checks"
    assert decision.reason_code == "capability_not_ready"


def test_preflight_wins_over_baseline_guard_when_capability_ready() -> None:
    state = AgentState(
        task="task1",
        current_phase="implementation",
        shared_code_version=None,
    )

    decision = DeterministicRouter().choose(
        state, capability_ready=True, preflight_pending=True
    )

    assert decision.phase == "preflight"
    assert decision.profile == "log_summarizer"
    assert decision.allowed_tools == ["inspect_hdf5", "read_file"]
    assert decision.allow_hosted_research is False
    assert decision.requires_llm is True
    assert decision.deterministic_action == "inspect_inputs_before_execution"
    assert decision.reason_code == "preflight_pending"


def test_baseline_guard_wins_before_implementation_without_shared_code_version() -> None:
    state = AgentState(task="task1", current_phase="implementation")

    decision = DeterministicRouter().choose(
        state, capability_ready=True, preflight_pending=False
    )

    assert decision.phase == "baseline_guard"
    assert decision.profile == "strong_planner"
    assert decision.allowed_tools == ["read_file"]
    assert decision.allow_hosted_research is False
    assert decision.requires_llm is True
    assert decision.deterministic_action == "establish_shared_code_baseline"
    assert decision.reason_code == "baseline_missing"


def test_router_output_uses_approved_profile_names_and_research_flag() -> None:
    state = AgentState(
        task="task1",
        current_phase="implementation",
        shared_code_version="sha256:base",
    )

    router = DeterministicRouter()
    decision = router.choose(state, capability_ready=True, preflight_pending=False)
    wrapped_decision = route_agent_state(
        state, capability_ready=True, preflight_pending=False
    )

    assert decision.phase == "implementation"
    assert decision.profile == "coder"
    assert decision.profile in {"strong_planner", "coder", "log_summarizer", "json_judge"}
    assert hasattr(decision, "allow_hosted_research")
    assert decision.allow_hosted_research is False
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
    assert decision.requires_llm is True
    assert decision.deterministic_action == "implement_current_task"
    assert decision.reason_code == "ready_for_implementation"
    assert wrapped_decision == decision


def test_context_packer_returns_bounded_fixed_sections() -> None:
    packer = ContextPacker(max_chars=220)

    payload = packer.build(
        developer_contract="D" * 150,
        task_spec="T" * 150,
        phase_tool_policy="P" * 150,
        compact_state="S" * 150,
        retrieved_memory="M" * 150,
        current_request="R" * 150,
    )

    assert [section["name"] for section in payload] == [
        "developer_contract",
        "task_spec",
        "phase_tool_policy",
        "compact_state",
        "retrieved_memory",
        "current_request",
    ]
    assert payload[0]["content"]
    assert payload[1]["content"]
    assert payload[2]["content"]
    assert payload[3]["content"]
    assert payload[5]["content"]
    assert sum(len(section["content"]) for section in payload) <= 220
    assert len(payload[4]["content"]) <= len("M" * 150)


def test_compatibility_helper_returns_supported_task_matrix_checks() -> None:
    compatible = compatibility_checks_for_patch(
        supported_tasks=["task1", "task2"],
        current_task="task3",
        cli_ok=True,
        smoke_ok=True,
        infer_shape_ok=True,
    )
    missing_cli = compatibility_checks_for_patch(
        supported_tasks=["task1", "task2"],
        current_task="task3",
        cli_ok=False,
        smoke_ok=True,
        infer_shape_ok=True,
    )

    assert compatible == {
        "task1_compat_ok": True,
        "task2_compat_ok": True,
        "task3_compat_ok": True,
    }
    assert missing_cli == {
        "task1_compat_ok": False,
        "task2_compat_ok": False,
        "task3_compat_ok": True,
    }
