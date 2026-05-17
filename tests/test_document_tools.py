from __future__ import annotations

from agent_runner.config import RunnerConfig
from agent_runner.logger import ToolCallLogger
from agent_runner.tool_registry import build_tool_registry


def test_generate_methodology_pdf_creates_pdf_under_submission(workspace):
    config = RunnerConfig.from_workspace(workspace)
    registry = build_tool_registry(
        config,
        ToolCallLogger(workspace / "internal_logs" / "tool_calls.jsonl"),
    )
    registry.set_context(task_id="task1", step_id="step-001", phase="validation")

    result = registry.execute(
        "generate_methodology_pdf",
        {
            "content": "# Method\n\nThis is a short methodology summary.",
        },
    )

    assert result["ok"] is True
    pdf_path = workspace / "submission" / "methodology.pdf"
    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF-")
