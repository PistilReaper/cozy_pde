from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import DataNotAvailableError, load_array_from_hdf5, save_json, summarize_array
from model import TinyTemporalConvNet


def make_training_pairs(array: np.ndarray, max_samples: int = 256, context_steps: int = 10):
    if array.ndim != 3:
        raise ValueError(f"Expected train array shape (N,T,256), got {array.shape}")
    if array.shape[1] <= context_steps:
        raise ValueError(f"Need more than {context_steps} time steps, got {array.shape[1]}")
    n = min(array.shape[0], max_samples)
    x = array[:n, context_steps - 1, :].astype(np.float32)
    y = array[:n, context_steps, :].astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def run_train(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"task{args.task}_train.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    log(
        json.dumps(
            {
                "event": "hypothesis",
                "task": args.task,
                "text": "Tiny one-step conv model can overfit a small subset for smoke validation; Task 2 is trained from scratch.",
            }
        )
    )

    try:
        array, key = load_array_from_hdf5(args.train_hdf5, args.dataset_key)
    except DataNotAvailableError as exc:
        log(json.dumps({"event": "data_missing", "error": str(exc)}))
        return 2

    log(json.dumps({"event": "data_loaded", "dataset_key": key, "summary": summarize_array(array)}))
    x, y = make_training_pairs(array, max_samples=args.max_samples, context_steps=args.context_steps)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = TinyTemporalConvNet(width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    global_step = 0
    last_loss = None
    for epoch in range(args.epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            if not torch.isfinite(loss):
                log(json.dumps({"event": "non_finite_loss", "epoch": epoch, "step": global_step}))
                return 3
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            log(json.dumps({"event": "train_step", "epoch": epoch, "step": global_step, "loss": last_loss}))
            global_step += 1
            if args.max_batches and global_step >= args.max_batches:
                break
        if args.max_batches and global_step >= args.max_batches:
            break

    ckpt_path = out_dir / f"task{args.task}_tiny_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "width": args.width,
            "context_steps": args.context_steps,
            "task": args.task,
            "trained_from_scratch": True,
            "last_loss": last_loss,
        },
        ckpt_path,
    )
    elapsed = time.perf_counter() - start
    save_json(
        out_dir / f"task{args.task}_train_summary.json",
        {
            "task": args.task,
            "checkpoint": str(ckpt_path),
            "elapsed_seconds": elapsed,
            "last_loss": last_loss,
            "global_steps": global_step,
            "conclusion": "smoke training completed with finite loss" if last_loss is not None else "no train steps executed",
        },
    )
    log(json.dumps({"event": "conclusion", "elapsed_seconds": elapsed, "last_loss": last_loss, "checkpoint": str(ckpt_path)}))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal smoke trainer for PDE neural operator rehearsal.")
    parser.add_argument("--task", type=int, choices=[1, 2], required=True)
    parser.add_argument("--train-hdf5", type=str, required=True)
    parser.add_argument("--dataset-key", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="runs/rehearsal")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--context-steps", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run_train(build_arg_parser().parse_args()))
