---
name: pdebench
description: "Guide the agent to inspect PDEBench task setup, dataset layout, and baseline assumptions before implementation."
---

# PDEBench

Use this skill when the task depends on understanding PDEBench data, checkpoints, or baseline structure.

Rules:
- Read workspace data and checkpoint inventory before proposing training changes.
- Confirm task-specific constraints separately for Task 1 and Task 2.
- Prefer existing baseline conventions over inventing new dataset layouts.
- Record any assumption about tensor shape, rollout horizon, or normalization before code changes.
