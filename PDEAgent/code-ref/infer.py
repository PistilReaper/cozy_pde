"""Inference script producing HDF5 submissions for Burgers neural-operator tasks.

The script loads a checkpoint saved by ``train.py``, normalises the official test
initial conditions with the checkpoint statistics, predicts the 190 unobserved
future frames, and writes a submission-compatible HDF5 file containing the first
10 input frames followed by the predicted future frames: ``[N, 200, 256]``.

Both long option names and short aliases are supported for compatibility with
previous experiment notes, e.g. ``--checkpoint/--ckpt`` and ``--output/--out``.
"""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch
from torch import Tensor, nn

from dataset import Normalizer, get_test_loader
from model import build_model
from utils import Timer, save_hdf5


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(description="Run Burgers model inference")
    parser.add_argument("--task", default="task1", choices=["task1", "task2"])
    parser.add_argument("--data_dir", default="./data_and_sample_submission/train_val_test_init")
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", required=True, help="Path to best_checkpoint.pt")
    parser.add_argument("--output", "--out", dest="output", required=True, help="Output HDF5 prediction path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _checkpoint_normalizer(ckpt: Dict[str, Any]) -> Normalizer:
    """Extract scalar normalisation statistics from a checkpoint."""
    n = ckpt.get("normalizer", None)
    if isinstance(n, dict):
        return Normalizer(float(n.get("mean", 0.0)), float(n.get("std", 1.0)))
    return Normalizer(0.0, 1.0)


def _load_initial_tensor(data_dir: str, task: str) -> np.ndarray:
    """Load raw official test initial conditions in physical units."""
    filename = "task1_test.hdf5" if task == "task1" else "task2_test.h5"
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"test file not found: {path}")
    with h5py.File(path, "r") as f:
        if "tensor" not in f:
            raise KeyError(f"{path} missing 'tensor'")
        tensor = f["tensor"][()].astype(np.float32)
    if tensor.ndim != 3:
        raise ValueError(f"test tensor must be 3D [N,T,Nx], got {tensor.shape}")
    return tensor


def _ensure_cfg_defaults(cfg: SimpleNamespace) -> None:
    """Fill missing historical checkpoint arguments with safe defaults."""
    defaults: Dict[str, Any] = {
        "model_type": "direct",
        "modes": 24,
        "width": 64,
        "depth": 4,
        "dropout": 0.0,
        "t_in": 10,
        "t_out": 190,
        "chunk_size": 10,
        "use_film": False,
    }
    for name, default in defaults.items():
        if not hasattr(cfg, name):
            setattr(cfg, name, default)


def _predict_future(model: nn.Module, x: Tensor, cond: Optional[Tensor], horizon: int) -> Tensor:
    """Predict future frames with either a direct model or chunked rollout model."""
    if hasattr(model, "rollout"):
        return model.rollout(x, horizon=horizon, cond=cond)  # type: ignore[attr-defined]
    pred, _ = model(x, cond)  # type: ignore[misc]
    return pred[:, :horizon]


def main() -> None:
    """Load checkpoint, predict 190 future frames, and write ``[N,200,Nx]`` HDF5."""
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = SimpleNamespace(**ckpt.get("args", {}))
    _ensure_cfg_defaults(cfg)

    model = build_model(cfg, task=args.task).to(args.device)
    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except KeyError as exc:
        raise KeyError(f"checkpoint {args.checkpoint} missing 'model_state'") from exc
    except RuntimeError as exc:
        raise RuntimeError(f"failed to load model state from {args.checkpoint}: {exc}") from exc
    model.eval()
    print(
        f"Loaded checkpoint epoch={ckpt.get('epoch', '?')}, best_score={ckpt.get('best_score', 'n/a')}, "
        f"model_type={getattr(cfg, 'model_type', 'direct')}"
    )

    normalizer = _checkpoint_normalizer(ckpt)
    loader, _ = get_test_loader(
        args.data_dir,
        task=args.task,
        batch_size=args.batch_size,
        normalizer=normalizer,
        num_workers=args.num_workers,
        t_in=int(getattr(cfg, "t_in", 10)),
    )
    raw = _load_initial_tensor(args.data_dir, args.task)
    t_in = int(getattr(cfg, "t_in", 10))
    if raw.shape[1] < t_in:
        raise ValueError(f"test tensor has only {raw.shape[1]} time steps, expected at least {t_in}")

    timer = Timer()
    preds = []
    with torch.no_grad():
        for batch in loader:
            if args.task == "task2" and len(batch) == 3:
                x, _, cond = batch
                x = x.to(args.device, non_blocking=True)
                cond = cond.to(args.device, non_blocking=True)
            else:
                x = batch[0].to(args.device, non_blocking=True)
                cond = None
            pred_norm = _predict_future(model, x, cond, horizon=190)
            preds.append(normalizer.decode(pred_norm).cpu())
    if not preds:
        raise RuntimeError("test dataloader produced no batches")
    future = torch.cat(preds, dim=0).numpy().astype(np.float32)

    n, _, nx = raw.shape
    if future.shape != (n, 190, nx):
        raise ValueError(f"future prediction shape mismatch: got {future.shape}, expected {(n, 190, nx)}")
    full = np.empty((n, 200, nx), dtype=np.float32)
    full[:, :10, :] = raw[:, :10, :]
    full[:, 10:, :] = future
    if not np.allclose(full[:, :10, :], raw[:, :10, :], atol=1.0e-6):
        raise AssertionError("first 10 steps are not identical to test input")
    save_hdf5(full, args.output)

    infer_time = timer.elapsed()
    print(f"Inference time: {infer_time:.3f}s")
    time_path = os.path.join(os.path.dirname(args.checkpoint) or ".", "time.json")
    time_data: Dict[str, float] = {"train_time": 0.0, "inference_time": infer_time}
    if os.path.exists(time_path):
        with open(time_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            time_data.update({k: float(v) for k, v in loaded.items() if isinstance(v, (int, float))})
        time_data["inference_time"] = infer_time
    with open(time_path, "w", encoding="utf-8") as f:
        json.dump(time_data, f, indent=2)


if __name__ == "__main__":
    main()
