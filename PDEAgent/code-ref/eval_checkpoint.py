"""Standalone checkpoint evaluator for Burgers neural operators.

This script fills the main diagnostic gap left by total validation score alone.
It loads a saved checkpoint, reconstructs the same validation split used by
``train.py``, performs full 190-step rollout, and prints/saves detailed segment
metrics: ``score1/2/3``, ``total``, ``rel_mse_1/2/3``, ``rmse_1/2/3``,
``fd_1/2/3`` and spectrum diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor, nn

from dataset import BurgersDataset, Normalizer, get_dataloaders, get_dataset_stats
from model import build_model
from utils import compute_segment_scores, save_metrics


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a Burgers checkpoint with detailed segment metrics")
    parser.add_argument("--task", default="task1", choices=["task1", "task2"])
    parser.add_argument("--data_dir", default="./data_and_sample_submission/train_val_test_init")
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", required=True)
    parser.add_argument("--output", default="", help="Optional JSON metrics output path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val_fraction", type=float, default=None, help="Override checkpoint validation fraction")
    parser.add_argument("--seed", type=int, default=None, help="Override checkpoint split seed")
    parser.add_argument("--use_checkpoint_normalizer", action="store_true", help="rebuild validation loader with checkpoint stats")
    return parser.parse_args()


def _checkpoint_normalizer(ckpt: Dict[str, Any]) -> Normalizer:
    """Extract normalizer saved in a checkpoint."""
    n = ckpt.get("normalizer", None)
    if isinstance(n, dict):
        return Normalizer(float(n.get("mean", 0.0)), float(n.get("std", 1.0)))
    return Normalizer(0.0, 1.0)


def _ensure_cfg_defaults(cfg: SimpleNamespace) -> None:
    """Populate missing config fields for old checkpoints."""
    defaults: Dict[str, Any] = {
        "model_type": "chunked",
        "chunk_size": 10,
        "window_stride": 1,
        "modes": 24,
        "width": 64,
        "depth": 4,
        "dropout": 0.0,
        "t_in": 10,
        "t_out": 190,
        "val_fraction": 0.2,
        "seed": 42,
        "augment_shift": False,
        "use_film": False,
        "unroll_chunks": 1,
    }
    for key, value in defaults.items():
        if not hasattr(cfg, key):
            setattr(cfg, key, value)


def _unpack_batch(batch: Tuple[Tensor, ...], task: str, device: str) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Move a validation batch to device."""
    if task == "task2" and len(batch) == 3:
        x, y, nu = batch
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True), nu.to(device, non_blocking=True)
    x, y = batch[:2]
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), None


def _predict(model: nn.Module, x: Tensor, cond: Optional[Tensor], horizon: int) -> Tensor:
    """Predict future frames for direct or chunked checkpoints."""
    if hasattr(model, "rollout"):
        return model.rollout(x, horizon=horizon, cond=cond)  # type: ignore[attr-defined]
    pred, _ = model(x, cond)  # type: ignore[misc]
    return pred[:, :horizon]


def _rebuild_task1_val_loader_with_normalizer(
    data_dir: str,
    normalizer: Normalizer,
    cfg: SimpleNamespace,
    batch_size: int,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    """Reconstruct the Task-1 validation subset encoded with checkpoint stats."""
    path = os.path.join(data_dir, "task1_val.hdf5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Task1 validation file not found: {path}")
    base = BurgersDataset(
        path,
        task="task1",
        t_in=int(getattr(cfg, "t_in", 10)),
        t_out=int(getattr(cfg, "t_out", 190)),
        normalizer=normalizer,
        compute_normalizer=False,
        augment_shift=False,
        seed=int(getattr(cfg, "seed", 42)),
    )
    n_val = max(1, int(round(len(base) * float(getattr(cfg, "val_fraction", 0.2)))))
    n_train = len(base) - n_val
    _, val_split = torch.utils.data.random_split(
        range(len(base)), [n_train, n_val], generator=torch.Generator().manual_seed(int(getattr(cfg, "seed", 42)))
    )
    subset = torch.utils.data.Subset(base, list(val_split))
    return torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, normalizer: Normalizer, task: str, device: str) -> Dict[str, float]:
    """Run full validation rollout and return detailed segment metrics."""
    model.eval()
    preds = []
    gts = []
    norm_mse_sum = 0.0
    n_batches = 0
    for batch in loader:
        x, y, cond = _unpack_batch(batch, task, device)
        pred = _predict(model, x, cond, horizon=y.shape[1])
        pred = pred[:, : y.shape[1]]
        norm_mse_sum += torch.mean((pred - y) ** 2).item()
        n_batches += 1
        preds.append(normalizer.decode(pred).cpu())
        gts.append(normalizer.decode(y).cpu())
    if not preds:
        raise RuntimeError("validation loader produced no batches")
    metrics = compute_segment_scores(torch.cat(preds, dim=0), torch.cat(gts, dim=0))
    metrics["val_mse_norm"] = norm_mse_sum / max(1, n_batches)
    metrics["num_val_samples"] = float(sum(int(p.shape[0]) for p in preds))
    return metrics


def main() -> None:
    """Entry point for checkpoint evaluation."""
    args = parse_args()
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = SimpleNamespace(**ckpt.get("args", {}))
    _ensure_cfg_defaults(cfg)
    if args.val_fraction is not None:
        cfg.val_fraction = args.val_fraction
    if args.seed is not None:
        cfg.seed = args.seed

    ckpt_normalizer = _checkpoint_normalizer(ckpt)
    if args.task == "task1" and args.use_checkpoint_normalizer:
        val_loader = _rebuild_task1_val_loader_with_normalizer(args.data_dir, ckpt_normalizer, cfg, args.batch_size, args.num_workers)
        eval_normalizer = ckpt_normalizer
    else:
        _, val_loader = get_dataloaders(
            args.data_dir,
            task=args.task,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_fraction=float(getattr(cfg, "val_fraction", 0.2)),
            seed=int(getattr(cfg, "seed", 42)),
            augment_shift=False,
            model_type=str(getattr(cfg, "model_type", "chunked")),
            chunk_size=int(getattr(cfg, "chunk_size", 10)),
            window_stride=int(getattr(cfg, "window_stride", 1)),
            t_in=int(getattr(cfg, "t_in", 10)),
            t_out=int(getattr(cfg, "t_out", 190)),
            train_target_horizon=int(getattr(cfg, "chunk_size", 10)) * max(1, int(getattr(cfg, "unroll_chunks", 1))),
        )
        eval_normalizer = get_dataset_stats(val_loader.dataset)
        if abs(eval_normalizer.mean - ckpt_normalizer.mean) > 1e-5 or abs(eval_normalizer.std - ckpt_normalizer.std) > 1e-5:
            print(
                "Warning: reconstructed split normalizer differs from checkpoint normalizer; "
                "rerun with --use_checkpoint_normalizer if evaluating a nonstandard split. "
                f"split=({eval_normalizer.mean:.6g},{eval_normalizer.std:.6g}), "
                f"ckpt=({ckpt_normalizer.mean:.6g},{ckpt_normalizer.std:.6g})"
            )

    model = build_model(cfg, task=args.task).to(args.device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    metrics = evaluate(model, val_loader, eval_normalizer, args.task, args.device)
    metrics["checkpoint_epoch"] = float(ckpt.get("epoch", -1))
    metrics["checkpoint_best_score"] = float(ckpt.get("best_score", metrics["total"]))

    print(json.dumps(metrics, indent=2, sort_keys=True))
    output = args.output or os.path.join(os.path.dirname(args.checkpoint) or ".", "eval_metrics.json")
    save_metrics(metrics, output)
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
