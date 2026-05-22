"""Neural operator models for 1D Burgers forecasting.

The module provides two compatible 1D Fourier Neural Operator (FNO) variants:

* ``ResidualFNO1d``: direct baseline mapping 10 observed frames to 190 future
  frames.
* ``ChunkedFNO1d``: autoregressive/chunked model mapping 10 observed frames to a
  short future chunk and rolling out until the full 190-step horizon is reached.

The chunked model is the preferred Task-1 experiment in the current iteration:
it matches the local-in-time Markov structure of Burgers dynamics while sliding
window training greatly increases the effective number of supervised samples.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """One-dimensional Fourier convolution over a periodic spatial grid."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if modes <= 0:
            raise ValueError("modes must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x: Tensor) -> Tensor:
        """Apply spectral convolution to an input tensor ``[B, C, Nx]``."""
        if x.ndim != 3:
            raise ValueError(f"SpectralConv1d expects [B,C,Nx], got {tuple(x.shape)}")
        batch, _, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(batch, self.out_channels, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        m = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :m] = torch.einsum("bim,iom->bom", x_ft[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FNOBlock1d(nn.Module):
    """Residual FNO block combining spectral and pointwise convolutions."""

    def __init__(self, width: int, modes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=width)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        y = self.spectral(x) + self.pointwise(x)
        y = F.gelu(self.norm(y))
        return x + self.dropout(y)


class FiLM(nn.Module):
    """Feature-wise linear modulation for optional conditioning."""

    def __init__(self, cond_dim: int, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, 2 * width),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        if cond.ndim == 1:
            cond = cond[:, None]
        gamma, beta = self.net(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class FNOForecast1d(nn.Module):
    """Core residual FNO forecaster mapping ``t_in`` frames to ``t_out`` frames."""

    def __init__(
        self,
        modes: int = 24,
        width: int = 64,
        depth: int = 4,
        t_in: int = 10,
        t_out: int = 10,
        use_film: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if t_in <= 0 or t_out <= 0:
            raise ValueError("t_in and t_out must be positive")
        self.modes = int(modes)
        self.width = int(width)
        self.depth = int(depth)
        self.t_in = int(t_in)
        self.t_out = int(t_out)
        self.use_film = bool(use_film)

        self.lift = nn.Sequential(
            nn.Conv1d(self.t_in + 1, self.width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(self.width, self.width, kernel_size=1),
        )
        self.blocks = nn.ModuleList([FNOBlock1d(self.width, self.modes, dropout) for _ in range(self.depth)])
        self.films = nn.ModuleList([FiLM(1, self.width) for _ in range(self.depth)]) if self.use_film else None
        self.project = nn.Sequential(
            nn.Conv1d(self.width, 2 * self.width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(2 * self.width, self.t_out, kernel_size=1),
        )
        self.nu_estimator = nn.Sequential(
            nn.Conv1d(self.t_in, 32, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        )

    def _coord(self, x: Tensor) -> Tensor:
        b, _, n = x.shape
        return torch.linspace(0.0, 1.0, n, device=x.device, dtype=x.dtype).view(1, 1, n).expand(b, -1, -1)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """Predict a future block from normalised input ``[B,t_in,Nx]``."""
        if x.ndim != 3:
            raise ValueError(f"model input must be [B,T,Nx], got {tuple(x.shape)}")
        if x.shape[1] != self.t_in:
            raise ValueError(f"expected {self.t_in} input steps, got {x.shape[1]}")
        if self.use_film and cond is None:
            cond = self.nu_estimator(x)
        if cond is not None and cond.ndim == 1:
            cond = cond[:, None]
        h = self.lift(torch.cat([x, self._coord(x)], dim=1))
        for i, block in enumerate(self.blocks):
            h = block(h)
            if self.use_film and cond is not None and self.films is not None:
                h = self.films[i](h, cond)
        return x[:, -1:, :].expand(-1, self.t_out, -1) + self.project(h), cond


class ResidualFNO1d(FNOForecast1d):
    """Direct 10-to-190 residual FNO retained as a fast baseline."""

    def __init__(
        self,
        modes: int = 24,
        width: int = 96,
        depth: int = 5,
        t_in: int = 10,
        t_out: int = 190,
        use_film: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(modes=modes, width=width, depth=depth, t_in=t_in, t_out=t_out, use_film=use_film, dropout=dropout)
        self.model_type = "direct"

    def rollout(self, x: Tensor, horizon: int = 190, cond: Optional[Tensor] = None) -> Tensor:
        pred, _ = self.forward(x, cond)
        return pred[:, :horizon]


class ChunkedFNO1d(nn.Module):
    """Autoregressive FNO predicting short chunks and rolling out to 190 steps."""

    def __init__(
        self,
        modes: int = 24,
        width: int = 64,
        depth: int = 4,
        t_in: int = 10,
        chunk_size: int = 10,
        use_film: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.model_type = "chunked"
        self.t_in = int(t_in)
        self.chunk_size = int(chunk_size)
        self.t_out = int(chunk_size)
        self.core = FNOForecast1d(
            modes=modes,
            width=width,
            depth=depth,
            t_in=t_in,
            t_out=chunk_size,
            use_film=use_film,
            dropout=dropout,
        )

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        return self.core(x, cond)

    @torch.no_grad()
    def rollout_no_grad(self, x: Tensor, horizon: int = 190, cond: Optional[Tensor] = None) -> Tensor:
        """No-grad wrapper used by validation and inference."""
        return self.rollout(x, horizon=horizon, cond=cond)

    def rollout(
        self,
        x: Tensor,
        horizon: int = 190,
        cond: Optional[Tensor] = None,
        detach_between_chunks: bool = False,
    ) -> Tensor:
        """Roll out autoregressively from the first ``t_in`` frames."""
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        history = x
        chunks = []
        produced = 0
        while produced < horizon:
            pred, _ = self.forward(history[:, -self.t_in :, :], cond)
            take = min(pred.shape[1], horizon - produced)
            chunk = pred[:, :take, :]
            chunks.append(chunk)
            history = torch.cat([history, chunk.detach() if detach_between_chunks else chunk], dim=1)
            produced += take
        return torch.cat(chunks, dim=1)


def build_model(cfg: Any, task: str = "task1") -> nn.Module:
    """Build a model from an argparse namespace or a lightweight config object."""
    use_film = bool(getattr(cfg, "use_film", task == "task2"))
    model_type = str(getattr(cfg, "model_type", "direct")).lower()
    if model_type in {"chunk", "chunked", "rollout"}:
        return ChunkedFNO1d(
            modes=int(getattr(cfg, "modes", 24)),
            width=int(getattr(cfg, "width", 64)),
            depth=int(getattr(cfg, "depth", 4)),
            t_in=int(getattr(cfg, "t_in", 10)),
            chunk_size=int(getattr(cfg, "chunk_size", getattr(cfg, "t_out", 10))),
            use_film=use_film,
            dropout=float(getattr(cfg, "dropout", 0.0)),
        )
    return ResidualFNO1d(
        modes=int(getattr(cfg, "modes", 24)),
        width=int(getattr(cfg, "width", 96)),
        depth=int(getattr(cfg, "depth", 5)),
        t_in=int(getattr(cfg, "t_in", 10)),
        t_out=int(getattr(cfg, "t_out", 190)),
        use_film=use_film,
        dropout=float(getattr(cfg, "dropout", 0.0)),
    )


def burgers_residual(u: Tensor, dx: float = 1.0 / 256.0, dt: float = 1.0, nu: float | Tensor = 1.0e-3) -> Tensor:
    """Compute differentiable Burgers residual on a periodic grid."""
    if u.ndim != 3 or u.shape[1] < 3:
        raise ValueError("u must have shape [B,T,Nx] with T>=3")
    n = u.shape[-1]
    u_hat = torch.fft.rfft(u, dim=-1)
    k = (2.0 * math.pi * torch.fft.rfftfreq(n, d=dx)).to(u.device, u.dtype)
    ux = torch.fft.irfft(1j * k * u_hat, n=n, dim=-1)
    uxx = torch.fft.irfft(-(k**2) * u_hat, n=n, dim=-1)
    ut = (u[:, 2:] - u[:, :-2]) / (2.0 * dt)
    if torch.is_tensor(nu):
        nu_t = nu.to(u.device, u.dtype)
        if nu_t.ndim == 0:
            nu_t = nu_t.view(1, 1, 1)
        elif nu_t.ndim == 1:
            nu_t = nu_t.view(-1, 1, 1)
    else:
        nu_t = torch.tensor(float(nu), device=u.device, dtype=u.dtype).view(1, 1, 1)
    return ut + u[:, 1:-1] * ux[:, 1:-1] - nu_t * uxx[:, 1:-1]
