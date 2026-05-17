from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import BudgetConfig


@dataclass(slots=True)
class AgentState:
    budget: BudgetConfig
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    llm_calls: int = 0
    tool_calls: int = 0
    agent_steps: int = 0
    consecutive_nan_or_oom: int = 0
    consecutive_validation_failures: int = 0
    experiment_notes: list[str] = field(default_factory=list)

    def elapsed_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()

    def remaining_seconds(self) -> float:
        return max(0.0, self.budget.max_wall_clock_hours * 3600 - self.elapsed_seconds())

    def should_finalize(self) -> bool:
        if self.remaining_seconds() <= self.budget.reserve_finalize_seconds:
            return True
        if self.llm_calls >= self.budget.max_llm_calls:
            return True
        if self.tool_calls >= self.budget.max_tool_calls:
            return True
        if self.agent_steps >= self.budget.max_agent_steps:
            return True
        return False

    def record_llm_call(self) -> None:
        self.llm_calls += 1

    def record_tool_call(self) -> None:
        self.tool_calls += 1

    def record_step(self) -> None:
        self.agent_steps += 1

    def record_validation_failure(self) -> None:
        self.consecutive_validation_failures += 1

    def reset_validation_failures(self) -> None:
        self.consecutive_validation_failures = 0

    def record_training_failure(self, *, nan: bool = False, oom: bool = False) -> None:
        if nan or oom:
            self.consecutive_nan_or_oom += 1
        else:
            self.consecutive_nan_or_oom = 0

    def should_rollback(self) -> bool:
        return self.consecutive_nan_or_oom >= 2 or self.consecutive_validation_failures >= 3

