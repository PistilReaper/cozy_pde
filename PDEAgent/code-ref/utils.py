"""Utility functions for Burgers neural-operator experiments.

The scoring utilities mirror the local validation protocol used by ``train.py``:
the model predicts the 190 future frames, while inference/submission copies the
first 10 observed frames into the output HDF5 file.  In addition to the official
style segment scores, this module provides small diagnostics that are useful for
Burgers shocks: spatial-gradient loss, temporal-increment loss and low-order
energy-spectrum distance.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor


def set_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def compute_rel_mse(pred: Tensor, gt: Tensor, eps: float = 1.0e-10) -> float:
    """Compute mean relative MSE over the spatial dimension."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    diff_sq = ((pred - gt) ** 2).sum(dim=-1)
    gt_sq = (gt**2).sum(dim=-1).clamp_min(eps)
    return torch.clamp(diff_sq / gt_sq, max=5.0).mean().item()


def compute_rmse(pred: Tensor, gt: Tensor) -> float:
    """Compute root mean squared error."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    return torch.sqrt(torch.mean((pred - gt) ** 2)).item()


def compute_frechet_distance(u1: Tensor, u2: Tensor) -> float:
    """Lightweight Frechet-like distance on spatial mean/std statistics."""
    if u1.shape != u2.shape:
        raise ValueError(f"u1/u2 shape mismatch: {tuple(u1.shape)} vs {tuple(u2.shape)}")
    m1, m2 = u1.mean(dim=-1), u2.mean(dim=-1)
    s1, s2 = u1.std(dim=-1), u2.std(dim=-1)
    return (((m1 - m2) ** 2).mean() + ((s1 - s2) ** 2).mean()).item()


def compute_spectrum_distance(pred: Tensor, gt: Tensor, modes: int = 32) -> float:
    """Return an energy-spectrum distance for diagnostics.

    The value is the mean squared log-energy mismatch over the first ``modes``
    Fourier modes.  It is not part of the official score, but it is sensitive to
    excessive high-frequency dissipation around Burgers shocks.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    p = torch.fft.rfft(pred, dim=-1).abs().pow(2).mean(dim=(0, 1))
    g = torch.fft.rfft(gt, dim=-1).abs().pow(2).mean(dim=(0, 1))
    m = min(int(modes), p.numel(), g.numel())
    if m <= 0:
        return 0.0
    return torch.mean((torch.log1p(p[:m]) - torch.log1p(g[:m])) ** 2).item()


def _segment_stats(pred: Tensor, gt: Tensor, prefix: str) -> Dict[str, float]:
    """Return diagnostics for one temporal segment."""
    return {
        f"rel_mse_{prefix}": compute_rel_mse(pred, gt),
        f"rmse_{prefix}": compute_rmse(pred, gt),
        f"fd_{prefix}": compute_frechet_distance(pred, gt),
        f"spec_{prefix}": compute_spectrum_distance(pred, gt),
    }


def compute_segment_scores(pred: Tensor, gt: Tensor) -> Dict[str, float]:
    """Compute official-style scores and detailed segment diagnostics.

    Args:
        pred: Predicted future frames with shape ``[B, 190, Nx]``.
        gt: Ground-truth future frames with the same shape.

    Returns:
        Dictionary containing ``score1``, ``score2``, ``score3``, ``total`` and
        per-segment relative-MSE/RMSE/Frechet-like/spectrum distances. Segment 1
        covers future frames ``0:48``, segment 2 ``48:96``, and segment 3
        ``96:190``.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    if pred.ndim != 3 or pred.shape[1] != 190:
        raise ValueError(f"expected [B,190,Nx], got {tuple(pred.shape)}")
    p1, g1 = pred[:, :48], gt[:, :48]
    p2, g2 = pred[:, 48:96], gt[:, 48:96]
    p3, g3 = pred[:, 96:], gt[:, 96:]
    rel_mse_1 = compute_rel_mse(p1, g1)
    rel_mse_2 = compute_rel_mse(p2, g2)
    rmse_3 = compute_rmse(p3, g3)
    fd_3 = compute_frechet_distance(p3, g3)
    score1 = 100.0 * math.exp(-20.0 * rel_mse_1)
    score2 = 100.0 * math.exp(-10.0 * rel_mse_2)
    score3 = max(100.0 / (1.0 + 10.0 * rmse_3), 50.0 * math.exp(-fd_3))
    total = 0.25 * score1 + 0.25 * score2 + 0.5 * score3
    out: Dict[str, float] = {"score1": score1, "score2": score2, "score3": score3, "total": total}
    out.update(_segment_stats(p1, g1, "1"))
    out.update(_segment_stats(p2, g2, "2"))
    out.update(_segment_stats(p3, g3, "3"))
    return out


def spectral_gradient_loss(pred: Tensor, gt: Tensor) -> Tensor:
    """Penalise mismatch in periodic first-order spatial increments."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    dp = torch.roll(pred, shifts=-1, dims=-1) - pred
    dg = torch.roll(gt, shifts=-1, dims=-1) - gt
    return torch.mean((dp - dg) ** 2)


def temporal_difference_loss(pred: Tensor, gt: Tensor) -> Tensor:
    """Penalise mismatch in time increments within a predicted chunk."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    if pred.shape[1] < 2:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    return torch.mean(((pred[:, 1:] - pred[:, :-1]) - (gt[:, 1:] - gt[:, :-1])) ** 2)


def ensure_dir(path: str) -> None:
    """Create a directory if needed."""
    if path:
        os.makedirs(path, exist_ok=True)


def save_hdf5(pred: np.ndarray, save_path: str) -> None:
    """Save prediction array to a submission-compatible HDF5 file."""
    if pred.ndim != 3:
        raise ValueError(f"prediction must be 3D, got shape {pred.shape}")
    ensure_dir(os.path.dirname(save_path))
    with h5py.File(save_path, "w") as f:
        f.create_dataset("tensor", data=pred.astype(np.float32), compression="gzip")
    print(f"Saved prediction to {save_path}, shape={pred.shape}")


def save_metrics(metrics: Mapping[str, Any], save_path: str) -> None:
    """Write metrics dictionary as JSON."""
    ensure_dir(os.path.dirname(save_path))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(dict(metrics), f, indent=2, ensure_ascii=False)


class Timer:
    """Wall-clock timer."""

    def __init__(self) -> None:
        self.start = time.time()

    def elapsed(self) -> float:
        """Return elapsed seconds since construction or last reset."""
        return time.time() - self.start

    def reset(self) -> None:
        """Reset the timer."""
        self.start = time.time()


class Logger:
    """Simple stdout + file logger."""

    def __init__(self, log_dir: str, filename: str = "train.log") -> None:
        ensure_dir(log_dir)
        self.log_file = os.path.join(log_dir, filename)

    def log(self, msg: str) -> None:
        """Print and append one log line."""
        print(msg, flush=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def log_metrics(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        """Log numeric metrics in a compact single-line format."""
        clean: Dict[str, float] = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))}
        self.log(f"Epoch {epoch:04d}, " + ", ".join(f"{k}={v:.6f}" for k, v in clean.items()))


def to_device(batch: Sequence[Tensor], device: torch.device | str) -> Tuple[Tensor, ...]:
    """Move a batch of tensors to a device."""
    return tuple(t.to(device) for t in batch)


def chunk_indices(length: int, chunk_size: int) -> List[Tuple[int, int]]:
    """Return inclusive-exclusive chunk ranges for a 1D sequence."""
    if length <= 0:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [(i, min(i + chunk_size, length)) for i in range(0, length, chunk_size)]


def detach_to_cpu(x: Tensor) -> Tensor:
    """Detach a tensor and move it to CPU."""
    return x.detach().cpu()
