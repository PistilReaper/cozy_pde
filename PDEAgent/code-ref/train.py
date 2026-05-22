"""Training entry point for Burgers neural operators.

The default Task-1 experiment is a chunked rollout FNO.  It uses many sliding
windows from the 100 local complete trajectories and validates by rolling out
from the official first 10 frames to the full 190-frame horizon.  This iteration
implements the current best hypothesis for escaping the teacher-forcing plateau:
optional multi-step rollout loss with a scheduled-sampling history update, plus
spatial-gradient and temporal-increment regularisation for shock fidelity.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim

from dataset import Normalizer, get_dataloaders, get_dataset_stats
from model import build_model, burgers_residual
from utils import (
    Logger,
    Timer,
    compute_segment_scores,
    save_metrics,
    set_seed,
    spectral_gradient_loss,
    temporal_difference_loss,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a 1D FNO/Chunked-FNO for Burgers prediction")
    parser.add_argument("--task", default="task1", choices=["task1", "task2"])
    parser.add_argument("--data_dir", default="./data_and_sample_submission/train_val_test_init")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--model_type", default="chunked", choices=["direct", "chunked"])
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--window_stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--modes", type=int, default=24)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["cosine", "step"], default="cosine")
    parser.add_argument("--scheduler_step", type=int, default=50)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision")
    parser.add_argument("--compile", action="store_true", help="use torch.compile when available")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_weight", type=float, default=0.05)
    parser.add_argument("--time_diff_weight", type=float, default=0.02)
    parser.add_argument("--time_weight", type=float, default=1.0, help="late-frame weighting inside supervised windows")
    parser.add_argument("--augment_shift", action="store_true", help="random periodic spatial shifts during training")
    parser.add_argument("--use_physics_loss", action="store_true")
    parser.add_argument("--physics_weight", type=float, default=1e-5)
    parser.add_argument("--use_film", action="store_true", help="force FiLM conditioning; normally useful for Task2")
    parser.add_argument("--t_in", type=int, default=10)
    parser.add_argument("--t_out", type=int, default=190)
    parser.add_argument("--unroll_chunks", type=int, default=1, help="chunks used by optional multi-step loss")
    parser.add_argument("--multi_step_weight", type=float, default=0.0, help="try 0.25-0.5 after baseline")
    parser.add_argument("--ss_start_epoch", type=int, default=30)
    parser.add_argument("--ss_ramp_epochs", type=int, default=80)
    parser.add_argument("--ss_max_prob", type=float, default=0.3)
    parser.add_argument("--resume", default="", help="optional checkpoint path for fine-tuning")
    return parser.parse_args()


def weighted_mse(pred: Tensor, target: Tensor, late_weight: float = 1.0) -> Tensor:
    """Return MSE with optional linearly increasing temporal weight."""
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if late_weight <= 1.0 or pred.shape[1] <= 1:
        return torch.mean((pred - target) ** 2)
    weights = torch.linspace(1.0, late_weight, pred.shape[1], device=pred.device, dtype=pred.dtype).view(1, -1, 1)
    return torch.mean(weights * (pred - target) ** 2)


def scheduled_probability(epoch: int, args: argparse.Namespace) -> float:
    """Linear ramp for scheduled sampling probability."""
    if args.multi_step_weight <= 0.0 or args.unroll_chunks <= 1 or epoch < args.ss_start_epoch:
        return 0.0
    ramp = max(1, int(args.ss_ramp_epochs))
    frac = min(1.0, float(epoch - args.ss_start_epoch + 1) / float(ramp))
    return max(0.0, min(float(args.ss_max_prob), float(args.ss_max_prob) * frac))


def _unpack_batch(batch: Tuple[Tensor, ...], task: str, device: str) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Move a dataloader batch to the target device."""
    if task == "task2" and len(batch) == 3:
        x, y, nu = batch
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True), nu.to(device, non_blocking=True)
    x, y = batch[:2]
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), None


def model_predict(model: nn.Module, x: Tensor, cond: Optional[Tensor], horizon: int) -> Tensor:
    """Predict with either a direct model or a chunked rollout model."""
    if hasattr(model, "rollout"):
        return model.rollout(x, horizon=horizon, cond=cond)  # type: ignore[attr-defined]
    pred, _ = model(x, cond)  # type: ignore[misc]
    return pred[:, :horizon]


def multi_step_rollout_loss(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    cond: Optional[Tensor],
    args: argparse.Namespace,
    ss_prob: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute differentiable multi-step rollout loss with scheduled sampling.

    At each chunk boundary the next history is assembled from either the model
    prediction or the teacher chunk.  The Bernoulli probability of using the
    model prediction is ``ss_prob`` and is ramped by ``scheduled_probability``.
    The loss is always evaluated against the true rollout target, so setting
    ``ss_prob=0`` still provides a multi-step teacher-forced objective.
    """
    if not hasattr(model, "forward"):
        raise TypeError("model must be a torch module with forward")
    max_h = min(y.shape[1], int(args.chunk_size) * int(args.unroll_chunks))
    if max_h <= 0:
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        return zero, zero, zero
    history = x
    preds: List[Tensor] = []
    produced = 0
    while produced < max_h:
        pred_chunk, _ = model(history[:, -args.t_in :, :], cond)  # type: ignore[misc]
        take = min(pred_chunk.shape[1], max_h - produced)
        pred_take = pred_chunk[:, :take]
        true_take = y[:, produced : produced + take]
        preds.append(pred_take)
        use_pred = ss_prob > 0.0 and random.random() < ss_prob
        next_chunk = pred_take if use_pred else true_take
        history = torch.cat([history, next_chunk], dim=1)
        produced += take
    pred_roll = torch.cat(preds, dim=1)
    target = y[:, : pred_roll.shape[1]]
    return (
        weighted_mse(pred_roll, target, args.time_weight),
        spectral_gradient_loss(pred_roll, target),
        temporal_difference_loss(pred_roll, target),
    )


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    scaler: Optional[GradScaler],
    args: argparse.Namespace,
    normalizer: Normalizer,
    epoch: int,
) -> Dict[str, float]:
    """Run one optimisation epoch."""
    model.train()
    ss_prob = scheduled_probability(epoch, args)
    totals = {"loss": 0.0, "data_loss": 0.0, "multi_step_loss": 0.0, "grad_loss": 0.0, "time_diff_loss": 0.0, "physics_loss": 0.0}
    n_batches = 0
    for batch in loader:
        x, y, cond = _unpack_batch(batch, args.task, args.device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            pred, _ = model(x, cond)  # type: ignore[misc]
            local_h = min(pred.shape[1], y.shape[1])
            pred_local = pred[:, :local_h]
            y_local = y[:, :local_h]
            loss_data = weighted_mse(pred_local, y_local, args.time_weight)
            loss_grad = spectral_gradient_loss(pred_local, y_local)
            loss_temp = temporal_difference_loss(pred_local, y_local)
            loss_multi = torch.zeros((), device=x.device)
            if args.multi_step_weight > 0.0 and args.unroll_chunks > 1 and y.shape[1] > local_h:
                loss_multi, grad_multi, temp_multi = multi_step_rollout_loss(model, x, y, cond, args, ss_prob)
                loss_grad = 0.5 * (loss_grad + grad_multi)
                loss_temp = 0.5 * (loss_temp + temp_multi)
            loss = loss_data + args.multi_step_weight * loss_multi + args.grad_weight * loss_grad + args.time_diff_weight * loss_temp
            loss_phys = torch.zeros((), device=x.device)
            if args.use_physics_loss:
                full = torch.cat([x[:, -2:], pred_local], dim=1)
                full_phys = normalizer.decode(full)
                loss_phys = torch.mean(burgers_residual(full_phys, nu=1.0e-3) ** 2)
                loss = loss + args.physics_weight * loss_phys
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        totals["loss"] += float(loss.item())
        totals["data_loss"] += float(loss_data.item())
        totals["multi_step_loss"] += float(loss_multi.item())
        totals["grad_loss"] += float(loss_grad.item())
        totals["time_diff_loss"] += float(loss_temp.item())
        totals["physics_loss"] += float(loss_phys.item())
        n_batches += 1
    denom = max(1, n_batches)
    out = {k: v / denom for k, v in totals.items()}
    out["ss_prob"] = ss_prob
    return out


@torch.no_grad()
def validate(model: nn.Module, loader: torch.utils.data.DataLoader, args: argparse.Namespace, normalizer: Normalizer) -> Dict[str, float]:
    """Evaluate by full 190-step rollout and compute physical-unit metrics."""
    model.eval()
    preds, gts = [], []
    mse_sum, count = 0.0, 0
    for batch in loader:
        x, y, cond = _unpack_batch(batch, args.task, args.device)
        pred = model_predict(model, x, cond, horizon=args.t_out)
        if pred.shape[1] != y.shape[1]:
            pred = pred[:, : y.shape[1]]
        mse_sum += torch.mean((pred - y) ** 2).item()
        count += 1
        preds.append(normalizer.decode(pred).cpu())
        gts.append(normalizer.decode(y).cpu())
    if not preds:
        raise RuntimeError("validation loader produced no batches")
    metrics = compute_segment_scores(torch.cat(preds, dim=0), torch.cat(gts, dim=0))
    metrics["val_mse_norm"] = mse_sum / max(1, count)
    return metrics


def _load_resume(model: nn.Module, optimizer: optim.Optimizer, args: argparse.Namespace, logger: Logger) -> int:
    """Load an optional checkpoint and return the next epoch index."""
    if not args.resume:
        return 1
    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"resume checkpoint not found: {args.resume}")
    ckpt = torch.load(args.resume, map_location=args.device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    if "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except ValueError as exc:
            logger.log(f"Warning: could not load optimizer state: {exc}")
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    logger.log(f"Resumed from {args.resume}; next_epoch={start_epoch}")
    return start_epoch


def _save_checkpoint(path: str, ckpt: Dict[str, object]) -> None:
    """Save a checkpoint to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


def main() -> None:
    """Train model, validate each epoch, and save the best checkpoint."""
    args = parse_args()
    if args.epochs <= 0 or args.chunk_size <= 0 or args.unroll_chunks <= 0:
        raise ValueError("--epochs, --chunk_size and --unroll_chunks must be positive")
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    logger = Logger(args.output_dir)
    logger.log("Starting training")
    logger.log(json.dumps(vars(args), indent=2))

    train_target_horizon = args.chunk_size * max(1, args.unroll_chunks) if args.model_type == "chunked" else args.t_out
    train_loader, val_loader = get_dataloaders(
        args.data_dir,
        task=args.task,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        seed=args.seed,
        augment_shift=args.augment_shift,
        model_type=args.model_type,
        chunk_size=args.chunk_size,
        window_stride=args.window_stride,
        t_in=args.t_in,
        t_out=args.t_out,
        train_target_horizon=train_target_horizon,
    )
    normalizer = get_dataset_stats(train_loader.dataset)
    logger.log(f"normalizer: mean={normalizer.mean:.6g}, std={normalizer.std:.6g}")
    logger.log(
        f"samples/batches: train_samples={len(train_loader.dataset)}, val_samples={len(val_loader.dataset)}, "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}"
    )

    model = build_model(args, task=args.task).to(args.device)
    if args.compile and hasattr(torch, "compile"):
        logger.log("Compiling model with torch.compile")
        model = torch.compile(model)  # type: ignore[assignment]
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"model parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = _load_resume(model, optimizer, args, logger)
    if args.scheduler == "cosine":
        scheduler: optim.lr_scheduler.LRScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.02)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    for _ in range(1, start_epoch):
        scheduler.step()
    scaler = GradScaler(enabled=args.amp)

    best_score, best_epoch, patience_count = -float("inf"), 0, 0
    timer = Timer()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, scaler if args.amp else None, args, normalizer, epoch)
        val_metrics = validate(model, val_loader, args, normalizer)
        scheduler.step()
        metrics = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}, "lr": optimizer.param_groups[0]["lr"]}
        logger.log_metrics(epoch, metrics)
        score = float(val_metrics["total"])
        if score > best_score:
            best_score, best_epoch, patience_count = score, epoch, 0
            model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
            ckpt = {
                "epoch": epoch,
                "model_state": model_to_save.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "best_score": best_score,
                "normalizer": normalizer.as_dict(),
                "val_metrics": val_metrics,
            }
            _save_checkpoint(os.path.join(args.output_dir, "best_model.pt"), ckpt)
            _save_checkpoint(os.path.join(args.output_dir, "best_checkpoint.pt"), ckpt)
            logger.log(
                f"  -> saved new best checkpoint, score={best_score:.6f}, score1={val_metrics['score1']:.4f}, "
                f"score2={val_metrics['score2']:.4f}, score3={val_metrics['score3']:.4f}, "
                f"rmse3={val_metrics['rmse_3']:.6f}, fd3={val_metrics['fd_3']:.6f}"
            )
        else:
            patience_count += 1
            if patience_count >= args.patience:
                logger.log(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

    train_time = timer.elapsed()
    final = {"best_score": best_score, "best_epoch": best_epoch, "train_time": train_time, "epochs_trained": epoch}
    save_metrics(final, os.path.join(args.output_dir, "metrics.json"))
    with open(os.path.join(args.output_dir, "time.json"), "w", encoding="utf-8") as f:
        json.dump({"train_time": train_time, "inference_time": 0.0}, f, indent=2)
    logger.log(f"Training finished in {train_time:.2f}s; best_score={best_score:.6f}")


if __name__ == "__main__":
    main()
