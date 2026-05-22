from __future__ import annotations

PROFILE_TOOLS: dict[str, list[str]] = {
    "strong_planner": ["read_workspace", "plan_baseline"],
    "coder": ["read_workspace", "apply_patch", "run_checks"],
    "log_summarizer": ["inspect_data", "summarize_logs"],
    "json_judge": ["emit_json_report"],
}

CAPABILITY_CHECK_TOOLS = ["check_provider", "check_research"]
