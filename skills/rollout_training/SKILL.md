---
name: rollout_training
description: "Guide the agent to make minimal, competition-safe rollout training decisions for neural operator experiments."
---

# Rollout Training

Use this skill when planning or debugging training loops for rollout prediction.

Rules:
- Start from the smallest training change that can validate the hypothesis.
- Distinguish teacher-forcing behavior from multi-step rollout behavior explicitly.
- Prefer simple loss additions only when they address a specific observed failure mode.
- Keep inference-path constraints visible while changing training code.
