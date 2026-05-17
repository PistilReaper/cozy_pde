---
name: neural_operator_research
description: "Guide the agent to research neural-operator papers, repos, and baselines with competition-focused scope control."
---

# Neural Operator Research

Use this skill when the agent needs paper or repository context for model selection.

Rules:
- Focus on architectures that can realistically fit the current competition budget.
- Prefer evidence from official papers, benchmark repos, and reproducible baselines.
- Extract only decision-relevant findings: architecture choice, rollout strategy, stability tricks, and inference cost.
- Avoid speculative research branches that do not directly support the current implementation decision.
