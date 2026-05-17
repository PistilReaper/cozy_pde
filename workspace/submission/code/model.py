from __future__ import annotations

import torch
from torch import nn


class TinyTemporalConvNet(nn.Module):
    """A minimal fast autoregressive model for smoke tests.

    The model maps a short history with 256 spatial points to the next time step.
    It is intentionally small so rehearsal smoke training and inference finish quickly.
    """

    def __init__(self, width: int = 32, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, width, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(width, 1, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, last_frame: torch.Tensor) -> torch.Tensor:
        if last_frame.ndim != 2:
            raise ValueError(f"Expected last_frame shape (batch, 256), got {tuple(last_frame.shape)}")
        x = last_frame.unsqueeze(1)
        return self.net(x).squeeze(1)


def rollout_persistence(context: torch.Tensor, total_steps: int = 200, context_steps: int = 10) -> torch.Tensor:
    if context.ndim != 3:
        raise ValueError(f"Expected context shape (batch, time, 256), got {tuple(context.shape)}")
    batch, time, width = context.shape
    if time < context_steps:
        raise ValueError(f"context has only {time} time steps, need at least {context_steps}")
    out = context.new_empty((batch, total_steps, width))
    out[:, :context_steps, :] = context[:, :context_steps, :]
    out[:, context_steps:, :] = context[:, context_steps - 1 : context_steps, :]
    return out


def rollout_model(
    model: nn.Module,
    context: torch.Tensor,
    total_steps: int = 200,
    context_steps: int = 10,
) -> torch.Tensor:
    if context.ndim != 3:
        raise ValueError(f"Expected context shape (batch, time, 256), got {tuple(context.shape)}")
    batch, time, width = context.shape
    if time < context_steps:
        raise ValueError(f"context has only {time} time steps, need at least {context_steps}")
    out = context.new_empty((batch, total_steps, width))
    out[:, :context_steps, :] = context[:, :context_steps, :]
    current = context[:, context_steps - 1, :]
    for step in range(context_steps, total_steps):
        current = model(current)
        out[:, step, :] = current
    return out
