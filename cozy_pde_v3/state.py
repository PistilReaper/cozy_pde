from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AgentState:
    task: str
    mode: str = "formal"
    run_id: str = ""
    current_phase: str = "capability_check"
    current_objective: str = ""
    latest_error_type: str | None = None
    latest_error_summary: str | None = None
    latest_tool_name: str | None = None
    latest_tool_result_ok: bool | None = None
    last_tool_call_id: str | None = None
    last_llm_call_id: str | None = None
    best_artifact_version: str | None = None
    best_artifact_path: str | None = None
    shared_code_version: str | None = None
    latest_checkpoint_path: str | None = None
    submission_snapshot_id: str | None = None
    supported_tasks: list[str] = field(default_factory=list)
    finalize_gate_status: dict[str, object] = field(default_factory=dict)
    preflight_complete: bool = False
    data_inspection_summary: dict[str, object] = field(default_factory=dict)
